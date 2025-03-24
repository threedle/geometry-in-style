from typing import Tuple, Literal, Optional, List, Callable, Sequence, Union, cast, Set
import os
import random
import warnings
from enum import Enum
from dataclasses import dataclass

# pytorch3d cot_laplacian still uses SparseTensor rather than sparse_coo_tensor
# Until upstream updates, there will be a deprecation warning, which we'll silence
warnings.filterwarnings("ignore", message="torch.sparse.SparseTensor")

import torch

TORCH_PLS_BE_DETERMINISTIC = os.environ.get("TORCH_PLS_BE_DETERMINISTIC")
if TORCH_PLS_BE_DETERMINISTIC:
    torch.use_deterministic_algorithms(True)
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"

import igl
import cholespy
import numpy as np
import torch.nn as nn

from pytorch3d.structures import Meshes
import pytorch3d.transforms as pt3d_transforms
import pytorch3d.ops as pt3d_ops

import thad
from thlog import (
    Thlogger,
    LOG_INFO,
    LOG_DEBUG,
    LOG_TRACE,
    LOG_NONE,
    VIZ_INFO,
    VIZ_DEBUG,
    VIZ_TRACE,
    VIZ_NONE,
    _PolyscopeRegisteredStructProxy,
)
from thronf import Thronfig, InvalidConfigError

thlog = Thlogger(LOG_INFO, VIZ_INFO, "deformations")


########################################### misc utils (sphuncs)
def normalize_to_side2_cube_inplace(meshes: Meshes):
    bounding_boxes = meshes.get_bounding_boxes()  # (n_meshes, 3, 2)
    mesh_to_verts_packed_first_idx = meshes.mesh_to_verts_packed_first_idx()

    bounding_boxes_packed = bounding_boxes[
        meshes.verts_packed_to_mesh_idx()
    ]  # (sum of all vertex counts, 3, 2)
    min_coords_packed = bounding_boxes_packed[:, :, 0]
    max_coords_packed = bounding_boxes_packed[:, :, 1]
    extent_packed, _ = (max_coords_packed - min_coords_packed).max(dim=-1)

    scale_per_mesh = 2 / extent_packed[mesh_to_verts_packed_first_idx]  # (n_meshes, )
    center_packed = (
        min_coords_packed + max_coords_packed
    ) / 2  # (sum of all vertex counts, 3)

    # normalize in-place
    meshes.offset_verts_(-center_packed)
    meshes.scale_verts_(scale_per_mesh)


def normalize_to_side2_cube_inplace_np_singlemesh(verts: np.ndarray):
    # actually just AABB
    mincoord = np.min(verts, axis=0)
    maxcoord = np.max(verts, axis=0)
    extent = (maxcoord - mincoord).max()
    scale = 2 / extent
    center = (mincoord + maxcoord) / 2
    return (verts - center) * scale


def make_sparse_diag(diag: torch.Tensor) -> torch.Tensor:
    """
    given `diag`, 1D dense tensor of shape (n,), builds a square sparse_coo
    matrix of shape (n,n) that has it as the main diagonal
    """
    idx = torch.arange((n := diag.size(0)), device=diag.device)
    return torch.sparse_coo_tensor(torch.stack((idx, idx), dim=0), diag, (n, n))


def calc_gradient_operator(
    verts: torch.Tensor,
    faces: torch.Tensor,
    face_normals_if_available: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    calculates the grad operator for a mesh using a per-vertex hat basis,
    resulting in a per-triangle linear operator.

    Returns:
    - grad, a (sparse) tensor of shape (#F * n_coords, #V) that can be
    matmul'd with any per-vertex quantity given as an (#V,*) tensor to give
    the per-face gradients (this result would have shape (#F * n_coords, *),
    which can be viewed as (#F, n_coords, *) if that's more convenient)
    - computed original-length face normals (#F, 3)
    - face doubleareas (which is also the lengths of those face normals) (#F,)
    """
    device = verts.device
    face_verts_coords = verts[faces]  # (n_faces, 3, n_coords)
    n_coords = face_verts_coords.shape[-1]
    v1 = face_verts_coords[:, 0]
    v2 = face_verts_coords[:, 1]
    v3 = face_verts_coords[:, 2]
    # edge vectors are named after the vertex they are opposite
    # verts and edges go CCW:
    #         v2
    #      e1 /|
    #     v3 / |
    #        \ | e3
    #      e2 \|
    #         v1
    # normal (u) points out from the screen
    e1 = v3 - v2
    e2 = v1 - v3
    e3 = v2 - v1

    if n_coords == 2:
        # stick onto e1, e2, e3 a z=0 coordinate and remove that later
        zeros = torch.zeros_like(e1[:, -1:])
        e1 = torch.cat((e1, zeros), dim=-1)
        e2 = torch.cat((e2, zeros), dim=-1)
        e3 = torch.cat((e3, zeros), dim=-1)

    face_normals = (
        face_normals_if_available
        if face_normals_if_available is not None
        else torch.linalg.cross(e1, e2)
    )
    face_doubleareas = torch.linalg.norm(
        face_normals, dim=-1, keepdim=True
    )  # also face normal magnitude
    u = face_normals / face_doubleareas  # face unit normals

    # edges rotated 90deg around normal (so that they point into the triangle)
    # (still on the triangle's plane), with length = original length / face doublearea
    e3perp = torch.linalg.cross(u, e3)
    e3perp /= e3perp.norm(dim=-1, keepdim=True)  # normalize,
    e3perp *= e3.norm(dim=-1, keepdim=True) / face_doubleareas

    e2perp = torch.linalg.cross(u, e2)
    e2perp /= e2perp.norm(dim=-1, keepdim=True)
    e2perp *= e2.norm(dim=-1, keepdim=True) / face_doubleareas

    e1perp = -(e3perp + e2perp)

    if n_coords == 2:
        # take out the fake zero coord (added for cross product purposes) if we
        # originally had 2d verts only
        e1perp = e1perp[:, :2]
        e2perp = e2perp[:, :2]
        e3perp = e3perp[:, :2]

    # build values and indices to fill the sparse matrix
    n_faces = faces.shape[0]
    n_verts = verts.shape[0]
    N_VERTS_PER_FACE = 3
    # indices is 2 stacked tensors, made by stacking `faces` and `vert_indices`:
    #   (faces = [0,0,0,0,0,0,0,0,0,3,3,3,3,3,3,3,3,3,6,6,6,6,6,6,6,6,6,...3*(n_faces-1),3*(n_faces-1),3*(n_faces-1)],
    #          + [0,1,2,0,1,2,0,1,2,0,1,2,0,1,2,0,1,2,0,1,2,0,1,2,0,1,2...,0,1,2],
    #    vert_indices = [f1_v1i,f1_v1i,f1_v1i, f1_v2i,f1_v2i,f1_v2i, f1_v3i,f1_v3i,f1_v3i, ..., f_nfaces_v0i, f_nfaces_v1i, f_nface_v2i])
    # all these indices__ tensors are 1D with the same length n_faces * N_COORDS * N_VERTS_PER_FACE
    #
    # values are
    #  [f1_e1perp_x, f1_e1perp_y, f1_e1perp_z, f1_e2perp_x, f1_e2perp_y, f1_e2perp_z, f1_e3perp_x, f1_e3perp_y, f1_e3perp_z, ...]
    indices__faces = torch.repeat_interleave(
        torch.arange(n_faces, device=device), n_coords * N_VERTS_PER_FACE
    )
    indices__axes = torch.arange(n_coords, device=device).repeat(n_faces * N_VERTS_PER_FACE)
    indices__vert_indices = torch.repeat_interleave(
        torch.stack((faces[:, 0], faces[:, 1], faces[:, 2]), dim=0).T.flatten(),
        n_coords,
    )
    indices = torch.stack(
        (n_coords * indices__faces + indices__axes, indices__vert_indices), dim=0
    )
    values = torch.stack((e1perp, e2perp, e3perp), dim=0).transpose(0, 1).flatten()
    grad = torch.sparse_coo_tensor(
        indices, values, size=(n_faces * n_coords, n_verts), device=device
    )
    return grad, face_normals, face_doubleareas.squeeze(-1)


def make_padded_to_packed_indexer(
    num_elements_per_mesh: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    A pytorch3d `Meshes` object provides quantities organized in packed tensors
    and padded tensors, i.e. `verts_packed` of shape `(n_verts_across_all_meshes,
    3)` and `verts_padded` of shape `(batch_size, max_n_verts_in_a_mesh, 3)`.

    Given the corresponding `meshes.num_*_per_mesh()` tensor, this will return
    indexing tensors `batch_idx`, `idx_in_batch` such that
    >>> elems_padded[batch_idx, idx_in_batch] == elems_packed

    The same indexing tensors can be used to modify elements in the padded tensor, i.e.
    >>> elems_padded[batch_idx, idx_in_batch] = new_elems_packed
    """
    batch_idx = torch.arange(
        num_elements_per_mesh.size(0),
        device=(device := num_elements_per_mesh.device),
    ).repeat_interleave(num_elements_per_mesh, dim=0)
    idx_in_batch = torch.cat(
        tuple(torch.arange(int(n), device=device) for n in num_elements_per_mesh),
        dim=0,
    )
    return batch_idx, idx_in_batch


def gather_per_face_quantity_into_per_vertex_quantity_inplace(
    face_areas: torch.Tensor,
    sum_incident_face_area_per_vertex: torch.Tensor,
    faces: torch.Tensor,
    quantity: torch.Tensor,
    out: torch.Tensor,
):
    """
    gather face quantities onto adjacent verts; around a vert, the weight of
    each face's value is its area in proportion with the sum of adjacent faces'
    areas.
    face_areas is assumed to have shape (n_faces,) and is unsqueezed accordingly
    to broadcast to quantity which should have shape (n_faces, *)
    """
    dim_expander = tuple(1 for _ in quantity.shape[1:])
    face_areas_times_quantity = quantity * face_areas.view(
        face_areas.size(0), *dim_expander
    )
    out.zero_()
    out.index_put_((faces[:, 0],), face_areas_times_quantity, accumulate=True)
    out.index_put_((faces[:, 1],), face_areas_times_quantity, accumulate=True)
    out.index_put_((faces[:, 2],), face_areas_times_quantity, accumulate=True)
    out.div_(
        sum_incident_face_area_per_vertex.view(
            sum_incident_face_area_per_vertex.size(0), *dim_expander
        )
    )


def per_vertex_packed_to_list(
    meshes: Meshes, per_vertex_quantity: torch.Tensor
) -> List[torch.Tensor]:
    # the pytorch3d.structures.packed_to_list function just calls torch.split so
    return per_vertex_quantity.split(list(meshes.num_verts_per_mesh()), dim=0)


def per_face_packed_to_list(
    meshes: Meshes, per_face_quantity: torch.Tensor
) -> List[torch.Tensor]:
    return per_face_quantity.split(list(meshes.num_faces_per_mesh()), dim=0)


def calc_barycentric_mass_matrix(
    verts: torch.Tensor, faces: torch.Tensor, return_diagonal_only=False
) -> torch.Tensor:
    """
    The barycentric mass matrix is a (sparse) diagonal tensor of shape (n_verts,
    n_verts) where the entry for vertex i is 1/3 the total area of the faces
    surrounding vertex i.
    If return_diagonal_only (default=False) is True, then return masses as a 1D
    tensor of shape (n_verts,) rather than a 2D sparse matrix.
    """

    faceverts0 = faces[:, 0]
    faceverts1 = faces[:, 1]
    faceverts2 = faces[:, 2]
    n_verts = verts.shape[0]

    face_areas = 0.5 * torch.linalg.norm(
        torch.cross(
            verts[faceverts2] - verts[faceverts0], verts[faceverts1] - verts[faceverts0]
        ),
        dim=-1,
    )

    mass_per_vertex = torch.zeros(n_verts, device=verts.device)
    mass_per_vertex.index_put_((faceverts0,), face_areas, accumulate=True)
    mass_per_vertex.index_put_((faceverts1,), face_areas, accumulate=True)
    mass_per_vertex.index_put_((faceverts2,), face_areas, accumulate=True)
    mass_per_vertex *= 1 / 3

    if return_diagonal_only:
        return mass_per_vertex
    else:
        mass_matrix = torch.sparse_coo_tensor(
            torch.tile(torch.arange(n_verts, device=verts.device), (2, 1)),
            mass_per_vertex,
            size=(n_verts, n_verts),
        ).to(verts.device)
        return mass_matrix


def calc_sum_incident_face_area_per_vertex(
    verts: torch.Tensor, faces: torch.Tensor
) -> torch.Tensor:
    return 3 * calc_barycentric_mass_matrix(
        verts, faces, return_diagonal_only=True
    ).unsqueeze(-1)


########################################### end misc utils


# if TYPE_CHECKING:
#     batched_svd = torch.linalg.svd
# else:
#     try:
#         import torch_batch_svd  # type: ignore

#         batched_svd = torch_batch_svd.svd
#     except ImportError:
#         thlog.info("couldn't import torch_batch_svd, using torch.linalg.svd")
#         batched_svd = torch.linalg.svd
# NOTE seems like torch_batch_svd doesn't behave well in sds training but torch.linalg.svd is fine...
# either that or I didn't do the solve correctly, but fewer deps is better so this is fine
batched_svd = torch.linalg.svd


# obsolete behaviors toggleable with an env var
DARAPPASTFEATURE__WRONG_ROT_AVERAGING = bool(
    os.environ.get("DARAPPASTFEATURE__WRONG_ROT_AVERAGING", False)
)
DARAPPASTFEATURE__WRONG_PROCRUSTES_NEIGHBORHOOD = bool(
    os.environ.get("DARAPPASTFEATURE__WRONG_PROCRUSTES_NEIGHBORHOOD", False)
)
if DARAPPASTFEATURE__WRONG_PROCRUSTES_NEIGHBORHOOD:
    print("using DARAPPASTFEATURE__WRONG_PROCRUSTES_NEIGHBORHOOD")


ARAPEnergyTypeName = Literal[
    "spokes_mine", "spokes_and_rims_mine", "spokes_igl", "spokes_and_rims_igl"
]

OptimizerTypeName = Literal["Adam", "VectorAdam"]


DeformOptimQuantityName = Literal[
    "verts_offsets",
    "faces_normals",
    "verts_normals",
    "faces_3x2rotations",
    "verts_3x2rotations",
    "faces_jacobians",
]
DeformSolveMethodName = Literal["arap", "poisson", "njfpoisson"]


PostprocessAfterSolveName = Literal["recenter_rescale", "recenter_only"]


@dataclass(slots=True)
class MeshesPackedIndexer:
    padded_aranges: torch.Tensor
    """ (n_meshes_in_batch, max(num_per_mesh)) """
    num_per_mesh: torch.Tensor
    """ (n_meshes_in_batch,) """
    mesh_to_packed_first_idx: torch.Tensor
    """ (n_meshes_in_batch,1) """

    @classmethod
    def from_num_per_mesh(cls, num_per_mesh: torch.Tensor):
        """
        num_per_mesh must be 1D tensor of positive ints, indicating the number
        of packed elements per mesh in the dataset
        """
        assert not num_per_mesh.is_floating_point()
        assert num_per_mesh.ndim == 1
        packed_sz = int(num_per_mesh.sum().item())
        mesh_to_packed_first_idx = torch.cumsum(num_per_mesh, dim=0) - num_per_mesh
        # there is a point to the -999999999999; it's so that it hopefully never turns
        # positive when we add mesh_to_verts_packed_first_idx to it for __call__
        return cls(
            padded_aranges=torch.stack(
                tuple(
                    nn.functional.pad(
                        torch.arange(_n := int(n.item())),
                        (0, packed_sz - _n),
                        value=-999999999999,
                    )
                    for n in num_per_mesh
                ),
                dim=0,
            ).to(num_per_mesh.device),
            num_per_mesh=num_per_mesh,
            mesh_to_packed_first_idx=mesh_to_packed_first_idx.unsqueeze(-1),
        )

    @classmethod
    def from_meshes(
        cls, meshes: Meshes, quantity_defined_on: Literal["verts", "faces", "edges"]
    ):
        if quantity_defined_on == "verts":
            num_per_mesh = meshes.num_verts_per_mesh()
        elif quantity_defined_on == "faces":
            num_per_mesh = meshes.num_faces_per_mesh()
        elif quantity_defined_on == "edges":
            num_per_mesh = meshes.num_edges_per_mesh()
        else:
            raise ValueError(
                "unknown quantity_defined_on: use 'verts' or 'faces' or 'edges'"
            )
        return cls.from_num_per_mesh(num_per_mesh)

    def __call__(self, mesh_indices: Union[int, slice, list, tuple, torch.Tensor]):
        """
        returns a tensor you can use to index dim0 of the associated packed qty.
        >>> verts_packed_idxr = MeshesPackedIndexer.from_meshes(meshes, quantity_defined_on="verts")
        >>> verts_packed[verts_packed_idxr(index)]  # will be the verts_packed of meshes[index]
        """
        picked = (self.padded_aranges + self.mesh_to_packed_first_idx)[mesh_indices]
        return picked[picked >= 0]

    def __getitem__(self, mesh_indices: Union[int, slice, list, tuple, torch.Tensor]):
        """
        returns a new indexer that works for the packed quantity associated with the
        mesh subbatch at mesh_indices
        """
        new_num_per_mesh = self.num_per_mesh[mesh_indices]
        new_mesh_to_packed_first_idx = (
            torch.cumsum(new_num_per_mesh, dim=0) - new_num_per_mesh
        )
        return __class__(
            padded_aranges=self.padded_aranges[mesh_indices],
            num_per_mesh=self.num_per_mesh[mesh_indices],
            mesh_to_packed_first_idx=new_mesh_to_packed_first_idx,
        )

    def n_meshes_in_batch(self) -> int:
        return self.num_per_mesh.size(0)


def calc_rot_matrices_axisangle(
    rot_source_vectors: torch.Tensor, rot_target_vectors: torch.Tensor, epsilon: float
) -> torch.Tensor:
    """
    rot_source_vectors and rot_target_vectors of shape (n, 3), *assumed to be unit normal*!!
    returns rotation matrices (n, 3, 3)
    """
    # this is adapted from https://iquilezles.org/articles/noacos/ this method avoids not
    # only trig (which i also avoided in my code), but also sqrt and clamp and normalize,
    # etc. Assumes rot_source_vectors and rot_target_vectors are already unit vectors (which
    # is indeed my case, of shape (n,3).
    z = rot_source_vectors
    d = rot_target_vectors

    # z (source): (n, 3), d (target): (n, 3)
    v = torch.linalg.cross(z, d)  # (n,); this is the rotation axes
    c = (z * d).sum(dim=-1)  # (n,); this is the cosine angle btwn source and target
    k = torch.reciprocal(1.0 + c + epsilon)[:, None, None]  # (n, 1, 1)
    vx = v[:, 0]  # (n,)
    vy = v[:, 1]  # (n,)
    vz = v[:, 2]  # (n,)
    rot_matrices = v.unsqueeze(1) * v.unsqueeze(2) * k + torch.stack(
        (
            torch.stack((c, -vz, vy), dim=-1),
            torch.stack((vz, c, -vx), dim=-1),
            torch.stack((-vy, vx, c), dim=-1),
        ),
        dim=1,
    )
    return rot_matrices


def cholespy_solve(
    solver: Union[cholespy.CholeskySolverF, cholespy.CholeskySolverD], rhs: torch.Tensor
) -> torch.Tensor:
    """
    rhs must be of shape (n_rows, k) where k > 0 and k <= 128 (limitation of cholespy)
    n_rows must be the same n_rows used to initialize the solver

    commented out: support for (batch, n_rows, k) where k > 0 and (batch * k) <= 128 (we
    don't need that use case because even our system matrices are batched, not just rhs, and
    each cholespy solver only corresponds to one laplacian matrix)
    """
    dtype = torch.float if isinstance(solver, cholespy.CholeskySolverF) else torch.double
    assert rhs.dtype == dtype, (
        f"rhs dtype needs to match the solver type {solver.__class__.__qualname__} (F=float, D=double)"
    )
    if rhs.ndim == 2:
        # needs contiguous() otherwise cholespy's nanobind will throw a mysterious error
        # about unsupported input types
        rhs = rhs.contiguous()
        out = torch.zeros_like(rhs)
        solver.solve(rhs, out)
        return out
    else:
        raise ValueError("rhs has an unsupported number of dimensions")
    # elif ndim == 3:
    #     batch_size, n_rows, n_features = rhs.shape
    #     rhs = rhs.permute(1, 2, 0).reshape(n_rows, -1)
    #     out = torch.zeros_like(rhs)
    #     solver.solve(rhs, out)
    #     out = out.view(n_rows, n_features, batch_size).permute(2, 0, 1)
    #     return out


class CholespySymmetricSolve_AutogradFn(torch.autograd.Function):
    """
    based on, and simplified from, Neural Jacobian Fields's SPLUSolveLayer
    """

    @staticmethod
    def forward(
        ctx,
        solver: Union[cholespy.CholeskySolverF, cholespy.CholeskySolverD],
        rhs: torch.Tensor,
    ) -> torch.Tensor:
        ctx.solver = solver
        return cholespy_solve(solver, rhs)

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        """
        forward() is intended for symmetric matrices (e.g. laplacians) so the
        backward is just this; otherwise backward() would require transposing
        the system matrix before doing the solve on the grad_output
        """
        grad = cholespy_solve(ctx.solver, grad_output)
        # because forward() needed 2 arguments, we must also return two things
        return None, grad


def remove_rowcolumn0_from_sparse_coo_matrix(x: torch.Tensor) -> torch.Tensor:
    """
    returns x with row0 and column0 removed
    """
    assert x.ndim == 2
    x = x.coalesce()
    row_indices, col_indices = x.indices()
    data = x.values()
    data_idxs_of_items_to_keep = ~((row_indices == 0).logical_or(col_indices == 0))
    result_row_indices = row_indices[data_idxs_of_items_to_keep] - 1
    result_col_indices = col_indices[data_idxs_of_items_to_keep] - 1
    result_data = data[data_idxs_of_items_to_keep]
    return torch.sparse_coo_tensor(
        torch.stack((result_row_indices, result_col_indices), dim=0),
        result_data,
        size=(x.size(0) - 1, x.size(1) - 1),
        dtype=x.dtype,
        device=x.device,
    )


def calc_cot_laplacian_for_solver(verts: torch.Tensor, faces: torch.Tensor) -> torch.Tensor:
    """
    calculate the cot laplacian suitable for solving poisson/arap equations for one mesh
    (verts and faces should not be packed from a batch of more than 1 mesh!)

    coming from pytorch3d's cot laplacian, we need to do a subtraction of the rowsum
    followed by negation in order to obtain the cot laplacian suitable for solving. This
    correct cot laplacian can also be obtained by `grad.T @ mass @ grad`, or more
    specifically, since "mass" here is a (F*3coords,F*3coords) diagonal matrix where the
    diag is filled with the corresp. face's double-areas,
    >>> grad, _, face_doubleareas = calc_gradient_operator(verts, faces)
    >>> lap = (face_doubleareas.repeat_interleave((n_coords := 3)).unsqueeze(-1) * grad).t().mm(grad)

    note also that pytorch3d's cot_laplacian gives (cota + cotb) weights, NOT 0.5*(cota +
    cotb) which is what igl.cotmatrix gives. All our runs use the (cota + cotb) weights, and
    converting to 0.5*(cota+cotb) requires adjusting the lambda hyperparam for procrustes to
    compensate; everything else should work the same (see README)
    """
    L_this_mesh, _ = pt3d_ops.cot_laplacian(verts, faces)
    L_this_mesh = (L_this_mesh - make_sparse_diag(L_this_mesh.sum(dim=0).to_dense())).neg()
    return L_this_mesh


def calc_cot_laplacian_and_cholespy_solver_until_it_works(
    verts: torch.Tensor,
    faces: torch.Tensor,
    pin_first_vertex: bool,
    max_n_attempts: int = 15,
) -> Tuple[torch.Tensor, cholespy.CholeskySolverF, Optional[torch.Tensor]]:
    """
    for some meshes, calculating the laplacian at first ends up with a non positive definite
    matrix due to very tiny or non-positive eigenvalue due to rounding / precision /
    something. because the laplacian is intrinsic, we may 'fuzz' verts with rotations and
    recompute laplacian until it passes as positive definite (i.e. initializing
    cholespy.CholeskySolverF succeeds)... the first attempt is to use the original verts;
    afterwards we rng the rotation (deterministically, with a fixed seed)

    NEW: `pin_first_vertex=True` is recommended since it seems to obviate the need for
    this rot hack and makes cholespy init much more likely to succeed without any cheesing
    """
    rng = np.random.default_rng(seed=398380)

    def __fuzz_rot_verts(_verts: torch.Tensor) -> torch.Tensor:
        rot_axisangle = torch.zeros_like(_verts)
        angles = rng.random(size=(3,), dtype=np.float32) * 360
        thlog.trace(f"fuzz rotation: {angles}")
        rot_axisangle[:] = torch.from_numpy(np.deg2rad(angles))
        rot_mats = pt3d_transforms.axis_angle_to_matrix(rot_axisangle)
        rotated_verts = rot_mats.bmm(_verts.unsqueeze(-1)).squeeze(-1)
        return rotated_verts

    for attempt_i in range(max_n_attempts):
        if attempt_i == 0:
            # on the first attempt, use the original verts, don't fuzz yet
            verts_for_lap_compute = verts
        else:
            verts_for_lap_compute = __fuzz_rot_verts(verts)
        L = calc_cot_laplacian_for_solver(verts_for_lap_compute, faces).coalesce()

        # Lpin is what we use to init the solver, we'll chop off the rhs's index 0 upon solve
        Lpin = (
            remove_rowcolumn0_from_sparse_coo_matrix(L) if pin_first_vertex else L
        ).coalesce()
        Lpin_indices = Lpin.indices()

        try:
            with thad.stdout_redirected():
                # solver is very noisy about not-posdef, which is what we're trying to catch!
                cholespy_solver = cholespy.CholeskySolverF(
                    Lpin.size(0),
                    Lpin_indices[0],
                    Lpin_indices[1],
                    Lpin.values(),
                    cholespy.MatrixType.COO,
                )
            if attempt_i > 0:
                thlog.debug(f"[cholespy solver init] okay that worked")
            # the L returned should NOT be the one with the [1:] chop though!
            maybe_removed_first_L_column = (
                L.index_select(1, torch.tensor([0], device=L.device)).to_dense()
                if pin_first_vertex
                else None
            )
            return L, cholespy_solver, maybe_removed_first_L_column
        except ValueError:
            # most likely failed with not-positive-definite error
            # continue the loop...
            thlog.debug(
                f"[cholespy solver init] failed attempt {attempt_i + 1}, retrying by rotating the mesh and recomputing laplace operator"
            )
            pass
    # if code gets here, we've exhausted attempts, give up
    raise ValueError(
        f"after {max_n_attempts}, couldn't successfully initialize cholespy's cholesky solver for this mesh"
    )


class IGL_ARAPEnergyType(Enum):
    SPOKES = 0
    SPOKES_AND_RIMS = 1


@dataclass(slots=True)
class ProcrustesPrecompute:
    padded_cell_edges_per_vertex_packed: torch.Tensor
    """
    (n_verts_packed, max_cell_neighborhood_n_edges, 2) int tensor; last dim contains edge vertex indices.
    negative ints are padding
    """
    covar_lefts_packed: torch.Tensor
    """
    (n_verts_packed, 3, max_cell_neighborhood_n_edges + 1)
    which is found by a batch matmul between
    (n_verts_packed, 3, max_cell_neighborhood_n_edges + 1) bmm (n_verts_packed, max_cell_neighborhood_n_edges + 1,max_cell_neighborhood_n_edges + 1)

    left-multiplies with a (max_cell_neighborhood_n_edges + 1, 3) matrix
    which is formed by grabbing the edge vectors corresponding to pcepv_packed, which
    would be (n_verts_packed, max_cell_neighborhood_n_edges,3) concatenated with
    the target normals (with dim1 unsqueezed so with shape (n_verts_packed, 1,
    3)) in dim1.

    then we batch_svd solve this (n_verts_packed, 3, 3) matrix to get the rotation
    """
    # bookkeeping for indexing
    _verts_packed_idxr: MeshesPackedIndexer
    _num_verts_per_mesh: torch.Tensor
    _mesh_to_verts_packed_first_idx: torch.Tensor

    @classmethod
    def from_meshes(
        cls,
        local_step_procrustes_lambda: float,
        arap_energy_type: Optional[ARAPEnergyTypeName],
        laplacians_solvers: "SparseLaplaciansSolvers",
        patient_meshes: Meshes,
    ):
        """
        (need the laplacians solvers just for the laplacian weights)
        """
        thlog.info("Calculating procrustes solve precomputation")
        verts_packed = patient_meshes.verts_packed()

        n_verts_packed = len(verts_packed)
        pcepv_packed: Tuple[Set[Tuple[int, int]], ...] = tuple(
            set() for _ in range(n_verts_packed)
        )
        need_spokes_and_rims = (
            arap_energy_type == "spokes_and_rims_mine"
            or arap_energy_type == "spokes_and_rims_igl"
        )
        for v0i_, v1i_, v2i_ in patient_meshes.faces_packed():
            v0i = int(v0i_.item())
            v1i = int(v1i_.item())
            v2i = int(v2i_.item())
            if DARAPPASTFEATURE__WRONG_PROCRUSTES_NEIGHBORHOOD:
                # i failed to distinguish between edge directions and was
                # missing the other direction of edges for spokes-and-rims
                e01i = (v0i, v1i) if v0i < v1i else (v1i, v0i)
                e12i = (v1i, v2i) if v1i < v2i else (v2i, v1i)
                e20i = (v2i, v0i) if v2i < v0i else (v0i, v2i)
                # add spokes
                pcev0_set = pcepv_packed[v0i]
                pcev1_set = pcepv_packed[v1i]
                pcev2_set = pcepv_packed[v2i]
                pcev0_set.add(e01i)
                pcev0_set.add(e20i)
                pcev1_set.add(e12i)
                pcev1_set.add(e01i)
                pcev2_set.add(e20i)
                pcev2_set.add(e12i)
                if need_spokes_and_rims:
                    pcev0_set.add(e12i)
                    pcev1_set.add(e20i)
                    pcev2_set.add(e01i)

            else:
                # correct procrustes neighborhood with directed edges, and each face
                # only contributing the edges that go in its CCW orientation
                e01i = (v0i, v1i)
                e12i = (v1i, v2i)
                e20i = (v2i, v0i)

                pcev0_set = pcepv_packed[v0i]
                pcev1_set = pcepv_packed[v1i]
                pcev2_set = pcepv_packed[v2i]

                # add spokes (radiating from vertex)
                pcev0_set.add(e01i)
                pcev1_set.add(e12i)
                pcev2_set.add(e20i)
                if need_spokes_and_rims:
                    # other face-edge pointing to vertex
                    pcev0_set.add(e20i)
                    pcev1_set.add(e01i)
                    pcev2_set.add(e12i)
                    # rims
                    pcev0_set.add(e12i)
                    pcev1_set.add(e20i)
                    pcev2_set.add(e01i)

        cell_neighborhood_n_edges = tuple(map(len, pcepv_packed))
        max_cell_neighborhood_n_edges = max(cell_neighborhood_n_edges)
        f: Callable[[Tuple[Set[Tuple[int, int]], int]], Tuple[Tuple[int, int], ...]] = (
            lambda _tup: (
                _set := _tup[0],
                _setlen := _tup[1],
                (
                    tuple(_set)
                    + tuple(
                        (-1, -1) for _ in range(max_cell_neighborhood_n_edges - _setlen)
                    )
                )
                if _setlen < max_cell_neighborhood_n_edges
                else tuple(_set),
            )[-1]
        )
        z = zip(pcepv_packed, cell_neighborhood_n_edges)
        pcepv_packed_tuples = tuple(map(f, z))
        padded_cell_edges_per_vertex_packed = torch.tensor(
            pcepv_packed_tuples, device=patient_meshes.device
        )
        thlog.debug("[procrustes precompute] done padded_cell_edges_per_vertex")
        ######################################## done computing padded_cell_edges_per_vertex

        cell_laplacian_weights_list = []
        for L, verts_packed_first_idx, n_verts in zip(
            laplacians_solvers.Ls,
            patient_meshes.mesh_to_verts_packed_first_idx(),
            patient_meshes.num_verts_per_mesh(),
        ):
            pcepv_this_mesh = (
                padded_cell_edges_per_vertex_packed[
                    verts_packed_first_idx : verts_packed_first_idx + n_verts
                ]
                - verts_packed_first_idx
            )
            pcepv_v0i_this_mesh = pcepv_this_mesh[:, :, 0]
            pcepv_v1i_this_mesh = pcepv_this_mesh[:, :, 1]
            pcepv_shape = pcepv_v1i_this_mesh.shape
            # cell_laplacian_weights_this_mesh = index_sparse_coo_matrix_rowcol(
            #     L, pcepv_v0i_this_mesh.flatten(), pcepv_v1i_this_mesh.flatten()
            # ).view(pcepv_shape)
            # ^ this runs out of mem on my laptop!
            # let's chunk this operation
            pcepv_v0i_numel = pcepv_v0i_this_mesh.numel()
            cell_laplacian_weights_this_mesh = torch.zeros(
                (pcepv_v0i_numel,), device=L.device, dtype=L.dtype
            )
            # each chunk fills the laplacian weights array for SPLITSZ edges
            SPLITSZ = 8192
            for chunk_idxs, v0idxs_this_chunk, v1idxs_this_chunk in zip(
                torch.arange(pcepv_v0i_numel).split(SPLITSZ),
                pcepv_v0i_this_mesh.flatten().split(SPLITSZ),
                pcepv_v1i_this_mesh.flatten().split(SPLITSZ),
            ):
                # dense indexing is way faster so we fetch the rows sparsely and index their columns densely
                L_v0idxs_this_chunk = L.index_select(0, v0idxs_this_chunk).to_dense()
                cell_laplacian_weights_this_mesh[chunk_idxs] = L_v0idxs_this_chunk[
                    torch.arange(v1idxs_this_chunk.size(-1)), v1idxs_this_chunk
                ]
                ## the older way:
                # cell_laplacian_weights_this_mesh[chunk_idxs] = (
                #     index_sparse_coo_matrix_rowcol(L, v0idxs_this_chunk, v1idxs_this_chunk)
                # )

            cell_laplacian_weights_this_mesh = cell_laplacian_weights_this_mesh.view(
                pcepv_shape
            ).to(patient_meshes.device)

            # wherever pcepv is negative, that's padding
            cell_laplacian_weights_this_mesh[pcepv_v1i_this_mesh < 0] = 0
            # ^ (n_verts, max_cell_neighborhood_n_edges,) float
            cell_laplacian_weights_list.append(cell_laplacian_weights_this_mesh)
        cell_laplacian_weights_packed = torch.cat(cell_laplacian_weights_list, dim=0)
        # (n_verts_packed, max_cell_neighborhood_n_edges,)
        thlog.debug("[procrustes precompute] done cotangent weights")
        ######################################## done putting cotan weights into pcepv format

        voronoi_verts_massmatrix__scipy = igl.massmatrix(
            verts_packed.cpu().detach().numpy(),
            patient_meshes.faces_packed().cpu().detach().numpy(),
        )
        # get diag of this thing
        voronoi_verts_mass_packed = (
            torch.from_numpy(voronoi_verts_massmatrix__scipy.diagonal())
            .float()
            .to(patient_meshes.device)
        ) * local_step_procrustes_lambda
        # ^ (n_verts_packed)
        assert voronoi_verts_mass_packed.shape == (n_verts_packed,)

        # form the middle neighborhood-size-by-neighborhood-size matrix
        diags_packed = torch.cat(
            (cell_laplacian_weights_packed, voronoi_verts_mass_packed.unsqueeze(-1)), dim=-1
        )
        # ^ (n_verts_packed, max_cell_neighborhood_n_edges + 1)
        diagmats_packed = torch.diag_embed(diags_packed)
        # ^ (n_verts_packed, max_cell_neighborhood_n_edges+1, max_cell_neighborhood_n_edges+1, )
        thlog.debug("[procrustes precompute] done diagonal matrix")
        if thlog.logguard(LOG_TRACE):
            torch.set_printoptions(precision=1)
            np.set_printoptions(precision=3)
            thlog.trace(f"""
            vertsmass
{voronoi_verts_mass_packed}
            diags:
{diags_packed.cpu().detach().numpy()}
            diagmats
            {diagmats_packed.cpu().detach().numpy()}
            L
            {laplacians_solvers.Ls[0].to_dense().cpu().detach().numpy()}
            pcepv
            {padded_cell_edges_per_vertex_packed}
            """)
            # assert False
        ############################################### done making middle diag matrix

        # NOTE this bit of code is also how you compute the covar_rights matrix
        # for the in-progress edge vecs and target normals (target vert normals taking the place of original_vert_normals)

        pcepv_v1i = padded_cell_edges_per_vertex_packed[:, :, 1]
        pcepv_v0i = padded_cell_edges_per_vertex_packed[:, :, 0]
        original_cell_edge_vecs_packed = verts_packed[pcepv_v1i] - verts_packed[pcepv_v0i]
        # ^ (n_verts_packed, max_cell_neighborhood_n_edges, 3)
        # zero out wherever there is padding
        original_cell_edge_vecs_packed[pcepv_v1i < 0] = 0
        original_vert_normals = patient_meshes.verts_normals_packed().unsqueeze(1)
        # ^ (n_verts_packed, 1, 3)
        covar_lefts_lefts_packed = torch.cat(
            (original_cell_edge_vecs_packed, original_vert_normals), dim=1
        )
        # ^ (n_verts_packed, max_cell_neighborhood_n_edges + 1, 3)
        covar_lefts_packed = covar_lefts_lefts_packed.transpose(-1, -2).bmm(diagmats_packed)
        # ^ (n_verts_packed, 3, max_cell_neighborhood_n_edges + 1)
        thlog.debug("[procrustes precompute] done covariance matrix")

        # make misc indexing bookkeeping
        _num_verts_per_mesh = patient_meshes.num_verts_per_mesh()
        _verts_packed_idxr = MeshesPackedIndexer.from_num_per_mesh(_num_verts_per_mesh)
        return cls(
            padded_cell_edges_per_vertex_packed=padded_cell_edges_per_vertex_packed.to(
                patient_meshes.device
            ),
            covar_lefts_packed=covar_lefts_packed,
            _verts_packed_idxr=_verts_packed_idxr,
            _num_verts_per_mesh=_num_verts_per_mesh,
            _mesh_to_verts_packed_first_idx=patient_meshes.mesh_to_verts_packed_first_idx(),
        )

    def __getitem__(self, mesh_indices: Union[int, List[int], torch.Tensor]):
        new_packed_idxr = self._verts_packed_idxr[mesh_indices]
        pcepv_packed_to_mesh_idx = torch.arange(
            new_packed_idxr.n_meshes_in_batch(),
            device=new_packed_idxr.num_per_mesh.device,
        ).repeat_interleave(new_packed_idxr.num_per_mesh)
        packed_idx = self._verts_packed_idxr(mesh_indices)
        new_pcepv_packed_noadjust = self.padded_cell_edges_per_vertex_packed[packed_idx]

        # apply offset adjustment to the indices inside new_faceadj_noadjust
        new_num_verts_per_mesh = self._num_verts_per_mesh[mesh_indices]
        new_mesh_to_verts_packed_first_idx = (
            torch.cumsum(new_num_verts_per_mesh, dim=0) - new_num_verts_per_mesh
        )
        old_mesh_to_verts_packed_first_idx = self._mesh_to_verts_packed_first_idx[
            mesh_indices
        ]
        new_pcepv_packed_adjusted = (
            new_pcepv_packed_noadjust
            - old_mesh_to_verts_packed_first_idx[pcepv_packed_to_mesh_idx, None, None]
            + new_mesh_to_verts_packed_first_idx[pcepv_packed_to_mesh_idx, None, None]
        )
        return __class__(
            padded_cell_edges_per_vertex_packed=new_pcepv_packed_adjusted,
            covar_lefts_packed=self.covar_lefts_packed[packed_idx],
            _verts_packed_idxr=new_packed_idxr,
            _num_verts_per_mesh=new_num_verts_per_mesh,
            _mesh_to_verts_packed_first_idx=new_mesh_to_verts_packed_first_idx,
        )


def calc_rot_matrices_with_procrustes(
    procrustes_precompute: ProcrustesPrecompute,
    curr_deformed_verts_packed: torch.Tensor,
    target_verts_normals_packed: torch.Tensor,
) -> torch.Tensor:
    """
    curr_deformed_verts_packed (n_verts_packed, 3)
    target_normals_packed (n_verts_packed, 3), the targeted normals
    """
    pcepv_v1i = procrustes_precompute.padded_cell_edges_per_vertex_packed[:, :, 1]
    pcepv_v0i = procrustes_precompute.padded_cell_edges_per_vertex_packed[:, :, 0]
    pcepv_v1 = curr_deformed_verts_packed[pcepv_v1i]
    pcepv_v0 = curr_deformed_verts_packed[pcepv_v0i]
    current_cell_edge_vecs_packed = pcepv_v1 - pcepv_v0
    # ^ (n_verts_packed, max_cell_neighborhood_n_edges, 3)
    if thlog.guard(VIZ_TRACE, needs_polyscope=True):
        pcepv_v1_for_vertex0 = pcepv_v1[0]
        pcepv_v1_for_vertex0 = pcepv_v1_for_vertex0[pcepv_v1i[0] >= 0]
        pcepv_v0_for_vertex0 = pcepv_v0[0]
        pcepv_v0_for_vertex0 = pcepv_v0_for_vertex0[pcepv_v0i[0] >= 0]
        cunet_pts = torch.cat((pcepv_v0_for_vertex0, pcepv_v1_for_vertex0), dim=0)
        cunet_edges = torch.stack(
            (
                torch.arange(len(pcepv_v0_for_vertex0)),
                len(pcepv_v0_for_vertex0) + torch.arange(len(pcepv_v0_for_vertex0)),
            ),
            dim=-1,
        )

        thlog.psr.register_curve_network(
            "v0 cell neigh",
            cunet_pts.cpu().detach().numpy(),
            cunet_edges.cpu().detach().numpy(),
        )
    current_cell_edge_vecs_packed[pcepv_v1i < 0] = 0
    # ^ (n_verts_packed, max_cell_neighborhood_n_edges, 3)
    target_verts_normals_packed = target_verts_normals_packed.unsqueeze(1)
    # ^ (n_verts_packed, 1, 3)
    covar_rights_packed = torch.cat(
        (current_cell_edge_vecs_packed, target_verts_normals_packed), dim=1
    )
    covar = procrustes_precompute.covar_lefts_packed.bmm(covar_rights_packed)
    # ^ (n_verts_packed, 3, 3)
    # so far that was  computed assuming vectors are column vecs, so
    # covar = covar.transpose(-1, -2)

    # batch svd this thing
    uu, ss, vvt = batched_svd(covar)
    # vvt[:, [0, 1]] *= -1  # it does not work if i don't do this
    # not inplace:
    vvt = torch.stack((-vvt[:, 0], -vvt[:, 1], vvt[:, 2]), dim=1)
    rots_packed = vvt.transpose(-1, -2).bmm(uu.transpose(-1, -2))

    # svd det correction code from https://github.com/OllieBoyne/pytorch-arap/blob/master/pytorch_arap/arap.py
    # for any det(Ri) <= 0
    entries_to_flip = torch.nonzero(rots_packed.det() <= 0, as_tuple=False).flatten()
    # ^idxs where det(R) <= 0
    if thlog.logguard(LOG_TRACE):
        thlog.trace(f"entries to flip {entries_to_flip}")
    if len(entries_to_flip) > 0:
        uumod = uu.clone()
        # minimum singular value is the last one
        uumod[entries_to_flip, :, -1] *= -1  # flip cols
        rots_packed[entries_to_flip] = (
            vvt[entries_to_flip]
            .transpose(-1, -2)
            .bmm(uumod[entries_to_flip].transpose(-1, -2))
        )

    return rots_packed.transpose(-1, -2)


@dataclass(slots=True)
class SparseLaplaciansSolvers:
    """
    Indexable dataclass holding, for a batch of meshes, a precomputed batched
    sparse Laplace operator, and cholespy solvers for those laplace operators,
    and optionally the NJF poisson system right-hand side premultiplier matrix.
    All are batched/a sequence, (first dim is batch_size; see the field docstrings.)

    All fields are immutable and are never modified inplace by any functions or
    methods. Methods that return `SparseLaplaciansSolvers` return copies
    (shallow copies: __getitem__ indexes into self's tensors and puts them in
    a new SparseLaplaciansSolvers struct).
    """

    Ls: torch.Tensor
    """
    cotangent laplacian,
    sparse_coo of shape (batch_size, max_n_verts, max_n_verts), where laplacians
    of meshes with fewer than max_n_verts verts (implicitly) get zero-padding
    """
    cholespy_solvers: Sequence[cholespy.CholeskySolverF]
    """
    list of batch_size cholespy solvers, one for each item in the batch
    (corresponding to one square laplace matrix). The only field in this struct
    that isn't a tensor. We might want to use this since this is way faster than
    torch's solve. `cholesky_decomps`, `inverses` are our caches for the torch
    solve; if we use cholespy, we don't need `cholesky_decomps` or `inverses`.
    """
    removed_first_L_column: Optional[Sequence[torch.Tensor]]
    """
    this is used when pin_first_vertex=True during from_meshes initialization. a list (of
    length batchsize) of first-columns removed from each L in the Ls batch
    """
    poisson_rhs_lefts: Optional[torch.Tensor] = None
    """
    if present, sparse of shape `(batch_size, max_n_verts, max_n_faces*3coords)`
    storing the poisson system's rhs matrix to be left-multiplied with per-face
    transform matrices to form the complete right-hand side of a system.

    for each mesh in a batch, poisson_rhs_lefts is computed via
    >>> grad, _, face_doubleareas = calc_gradient_operator(verts, faces)
        # where grad is (n_faces*3coords, n_verts)
        rhs_left = (face_doubleareas.repeat_interleave((n_coords:=3)).unsqueeze(-1) * grad)

    the system to solve is
    >>> L @ X = rhs_left @ per_face_transform_matrices

    (making sure that `per_face_transform_matrices` has been transposed to be
    `(batch, max_n_faces_per_mesh, 3coords, 3vertsperface)` and then viewed as
    `(batch, max_n_faces_per_mesh*3coords, 3vertsperface))` before doing this
    """
    igl_arap_rhs_lefts: Optional[Sequence[torch.Tensor]] = None
    """
    each one is of shape (V*3, V*3*3).
    
    For a mesh having per-vertex matrices "rot" as a (V, 3, 3) tensor, the right-hand side of
    the system to solve is computed with
    >>> rots_for_igl_rhs = rot.permute(2, 1, 0).reshape(-1, 1)
    >>> rhs = 2 * arap_rhs_left.mm(rots_for_igl_rhs).reshape(3, -1).t()
    # the 2* is because we (via pytorch3d) use (cot a + cot b) weights; IGL
    assumes 1/2(cot a + cot b) # (this correction is only needed for IGL
    prefactored solves; all others unaffected)

    then the system to solve is
    >>> L @ X = rhs

    also we are not going to batch this one because the igl matrix is very weirdly stacked
    so that it is a 2D matrix...
    """

    def __getitem__(self, mesh_indices: Union[int, List[int], torch.Tensor]):
        """
        return the batch item at `index` as its own SparseLaplaciansSolvers of batch size 1
        if `index` is a scalar (int or scalar tensor) or n if `index` is a 1D tensor of n
        index values.
        """
        if isinstance(mesh_indices, int):
            Ls = self.Ls[None, mesh_indices]
            poisson_rhs_lefts = (
                self.poisson_rhs_lefts[None, mesh_indices]
                if self.poisson_rhs_lefts is not None
                else None
            )
            cholespy_solvers = (self.cholespy_solvers[mesh_indices],)
            removed_first_L_column = (
                (self.removed_first_L_column[mesh_indices],)
                if self.removed_first_L_column is not None
                else None
            )
            igl_arap_rhs_lefts = (
                (self.igl_arap_rhs_lefts[mesh_indices],)
                if self.igl_arap_rhs_lefts is not None
                else None
            )
        else:
            if isinstance(mesh_indices, list):
                mesh_indices = torch.tensor(mesh_indices)  # turn int list into int tensor
            if mesh_indices.ndim == 0:
                # 0D tensor containing just a scalar
                Ls = self.Ls[None, mesh_indices]
                poisson_rhs_lefts = (
                    self.poisson_rhs_lefts[None, mesh_indices]
                    if self.poisson_rhs_lefts is not None
                    else None
                )
                cholespy_solvers = (self.cholespy_solvers[i := int(mesh_indices.item())],)
                removed_first_L_column = (
                    (self.removed_first_L_column[i],)
                    if self.removed_first_L_column is not None
                    else None
                )
                igl_arap_rhs_lefts = (
                    (self.igl_arap_rhs_lefts[i],)
                    if self.igl_arap_rhs_lefts is not None
                    else None
                )
            else:
                # actual tensor being used as the index; sparse tensors don't support them
                # in the usual [] indexing syntax, we have to use index_select
                index_on_dev = mesh_indices.to(self.Ls.device)
                Ls = self.Ls.index_select(0, index_on_dev)
                poisson_rhs_lefts = (
                    self.poisson_rhs_lefts.index_select(0, index_on_dev)
                    if self.poisson_rhs_lefts is not None
                    else None
                )
                cholespy_solvers = tuple(
                    self.cholespy_solvers[int(i.item())] for i in mesh_indices
                )
                removed_first_L_column = (
                    tuple(self.removed_first_L_column[int(i.item())] for i in mesh_indices)
                    if self.removed_first_L_column is not None
                    else None
                )
                igl_arap_rhs_lefts = (
                    tuple(self.igl_arap_rhs_lefts[int(i.item())] for i in mesh_indices)
                    if self.igl_arap_rhs_lefts is not None
                    else None
                )

        return __class__(
            Ls,
            removed_first_L_column=removed_first_L_column,
            cholespy_solvers=cholespy_solvers,
            poisson_rhs_lefts=poisson_rhs_lefts,
            igl_arap_rhs_lefts=igl_arap_rhs_lefts,
        )

    def to(self, device: torch.device) -> "SparseLaplaciansSolvers":
        return __class__(
            self.Ls.to(device),
            removed_first_L_column=tuple(
                col.to(device) for col in self.removed_first_L_column
            )
            if self.removed_first_L_column is not None
            else None,
            cholespy_solvers=self.cholespy_solvers,
            poisson_rhs_lefts=(
                self.poisson_rhs_lefts.to(device)
                if self.poisson_rhs_lefts is not None
                else None
            ),
            igl_arap_rhs_lefts=tuple(
                arap_rhs.to(device) for arap_rhs in self.igl_arap_rhs_lefts
            )
            if self.igl_arap_rhs_lefts is not None
            else None,
        )

    @classmethod
    def from_meshes(
        cls,
        meshes: Meshes,
        pin_first_vertex: bool,
        compute_poisson_rhs_lefts: bool,
        compute_igl_arap_rhs_lefts: Optional[IGL_ARAPEnergyType],
    ):
        """
        given a batch of meshes, compute the Laplace operator and a cholespy solver object
        with that operator as the system matrix, and optionally
        - if `compute_poisson_rhs_lefts`: the NJF poisson system right-hand side
        premultiplier (see the docstrings of the fields of this dataclass for more info)
        - if `compute_igl_arap_rhs_lefts`: the IGL ARAP right-hand-side premultipliers
        - if `pin_first_vertex`, the laplacian will have one row and column chopped off,
            and any solves using this solver will adjust the rhs to this reduced system
            (with the first vertex of each mesh in the batch subbed in) accordingly
        """
        max_n_verts_per_mesh = int(meshes.num_verts_per_mesh().max().item())
        max_n_faces_per_mesh = int(meshes.num_faces_per_mesh().max().item())
        square_shape = (max_n_verts_per_mesh, max_n_verts_per_mesh)
        Ls = []
        cholespy_solvers = []
        poisson_rhs_lefts_per_mesh = [] if compute_poisson_rhs_lefts else None
        igl_arap_rhs_lefts_per_mesh = [] if compute_igl_arap_rhs_lefts else None
        removed_first_L_column_per_mesh = [] if pin_first_vertex else None
        for verts, faces in zip(meshes.verts_list(), meshes.faces_list()):
            L_this_mesh, cholespy_solver, removed_first_L_column = (
                calc_cot_laplacian_and_cholespy_solver_until_it_works(
                    verts, faces, pin_first_vertex=pin_first_vertex
                )
            )
            cholespy_solvers.append(cholespy_solver)

            # keep the removed first column of the laplace operator for rhs adjust
            if (
                removed_first_L_column is not None
                and removed_first_L_column_per_mesh is not None
            ):
                removed_first_L_column_per_mesh.append(removed_first_L_column)

            # matrix to be left-multiplied with face transforms to form the rhs
            # of a poisson system
            if poisson_rhs_lefts_per_mesh is not None:
                grad, _, face_doubleareas = calc_gradient_operator(verts, faces)
                poisson_rhs_left_this_mesh = (
                    face_doubleareas.repeat_interleave((n_coords := 3)).unsqueeze(-1) * grad
                )
                # ^ this is sparse, with shape (n_faces_this_mesh * n_coords,
                # n_verts_this_mesh). First, we transpose it for quicker application at time
                # of use, since we'll need (V,F*3)...
                poisson_rhs_left_this_mesh = poisson_rhs_left_this_mesh.t()
                # and sparse-resize it to fit padding
                poisson_rhs_left_this_mesh.sparse_resize_(
                    (max_n_verts_per_mesh, max_n_faces_per_mesh * n_coords), 2, 0
                )
                poisson_rhs_lefts_per_mesh.append(poisson_rhs_left_this_mesh)

            if (
                igl_arap_rhs_lefts_per_mesh is not None
                and compute_igl_arap_rhs_lefts
                is not None  # not needed, but for typechecker
            ):
                igl_arap_rhs_lefts_this_mesh = igl.arap_rhs(
                    verts.cpu().detach().numpy(),
                    faces.cpu().detach().numpy(),
                    3,
                    compute_igl_arap_rhs_lefts.value,
                )
                # this is a csc_matrix, so we should turn it into a torch sparse
                igl_arap_rhs_lefts_this_mesh_scipycoo = igl_arap_rhs_lefts_this_mesh.tocoo()
                scipycoo_rows, scipycoo_cols = igl_arap_rhs_lefts_this_mesh_scipycoo.coords
                torchcoo_indices = torch.from_numpy(
                    np.stack((scipycoo_rows, scipycoo_cols), axis=0)
                ).to(verts)
                torchcoo_values = torch.from_numpy(
                    igl_arap_rhs_lefts_this_mesh_scipycoo.data
                ).to(verts)
                igl_arap_rhs_lefts_this_mesh = torch.sparse_coo_tensor(
                    torchcoo_indices,
                    torchcoo_values,
                    size=igl_arap_rhs_lefts_this_mesh_scipycoo.shape,
                )
                # the shape of this is (V*3, V*3*3), which is very funny!
                igl_arap_rhs_lefts_per_mesh.append(igl_arap_rhs_lefts_this_mesh)

            # done computing extra matrices for this mesh
            # resize the laplace operator this mesh to be the padded size
            L_this_mesh.sparse_resize_(square_shape, 2, 0)
            Ls.append(L_this_mesh)
        # done looping through meshes

        # stack the matrices into batched tensors
        padded_batched_laplacians = torch.stack(Ls, dim=0)

        poisson_rhs_lefts = (
            torch.stack(poisson_rhs_lefts_per_mesh, dim=0)
            if poisson_rhs_lefts_per_mesh is not None
            else None
        )

        return cls(
            padded_batched_laplacians,
            removed_first_L_column=removed_first_L_column_per_mesh,
            cholespy_solvers=cholespy_solvers,
            poisson_rhs_lefts=poisson_rhs_lefts,
            igl_arap_rhs_lefts=igl_arap_rhs_lefts_per_mesh,
        )


def recenter_to_centroid(meshes: Meshes, new_verts_packed: torch.Tensor) -> torch.Tensor:
    new_verts_packed_recentered = torch.zeros_like(new_verts_packed)
    for verts_packed_first_idx, n_verts in zip(
        meshes.mesh_to_verts_packed_first_idx(), meshes.num_verts_per_mesh()
    ):
        indexer = slice(verts_packed_first_idx, verts_packed_first_idx + n_verts)
        verts_this_mesh = new_verts_packed[indexer]
        verts_this_mesh = verts_this_mesh - verts_this_mesh.mean(dim=0, keepdim=True)
        new_verts_packed_recentered[indexer] = verts_this_mesh
    return new_verts_packed_recentered


def recenter_to_centroid_and_rescale_new_verts_to_fit_old_bboxes(
    meshes: Meshes, new_verts_packed: torch.Tensor
) -> torch.Tensor:
    """
    this is to match the method from textdeformer, even though it's probably not what I'd
    immediately think of if asked to preserve old bboxes
        (If I were to do this, the scale factor would be based on max axis-aligned extent,
        rather than length of the diagonal between min coord and max coord points)
    """
    old_bboxes = meshes.get_bounding_boxes()
    # (n_meshes, 3, 2) last dim is [min,max]
    old_bboxes_min = old_bboxes[:, :, 0]
    old_bboxes_max = old_bboxes[:, :, 1]
    old_sizes = (old_bboxes_max - old_bboxes_min).norm(dim=-1)  # (n_meshes,)

    new_verts_packed_scaled = torch.zeros_like(new_verts_packed)
    # old_verts_packed = meshes.verts_packed()
    for verts_packed_first_idx, n_verts, old_size in zip(
        meshes.mesh_to_verts_packed_first_idx(), meshes.num_verts_per_mesh(), old_sizes
    ):
        indexer = slice(verts_packed_first_idx, verts_packed_first_idx + n_verts)
        verts_this_mesh = new_verts_packed[indexer]

        # first, recenter to centroid
        verts_this_mesh = verts_this_mesh - verts_this_mesh.mean(dim=0, keepdim=True)
        # then scale factor is the ratio between the bounding box diagonal lengths
        bbox_diag_this_mesh = verts_this_mesh.max(dim=0)[0] - verts_this_mesh.min(dim=0)[0]
        new_size = bbox_diag_this_mesh.norm()
        new_verts_packed_scaled[indexer] = verts_this_mesh * (old_size / new_size)
    return new_verts_packed_scaled


def calc_ARAP_global_solve__better_with_rhs_lefts(
    meshes: Meshes,
    laplacians_solvers: SparseLaplaciansSolvers,
    per_vertex_rot_matrices_packed: torch.Tensor,
    postprocess: Optional[PostprocessAfterSolveName],
) -> torch.Tensor:
    """
    like calc_ARAP_global_solve but makes use of a precomputed igl_arap_rhs_lefts
    (rhs constructor to be multiplied with rotation matrices to form the rhs of
    the system). the actual correct way to do this arap solve anyway!!!
    """
    solutions = []
    assert laplacians_solvers.igl_arap_rhs_lefts is not None
    for i, (arap_rhs_left, verts, verts_packed_first_idx, n_verts) in enumerate(
        zip(
            laplacians_solvers.igl_arap_rhs_lefts,
            meshes.verts_list(),
            meshes.mesh_to_verts_packed_first_idx(),
            meshes.num_verts_per_mesh(),
        )
    ):
        # so igl's arap_rhs (= arap_rhs_left) is shape (V*3, V*9)
        # have to make the rot matrices from (V,3,3) into (V*9,1) somehow and then reshape the result from (V*3) to (V,3)
        rots_for_igl_rhs = (
            per_vertex_rot_matrices_packed[
                verts_packed_first_idx : verts_packed_first_idx + n_verts
            ]
            .permute(2, 1, 0)
            .reshape(-1, 1)
        )
        # 2x due to differing cotan weight conventions btwn pt3d and igl for the solver init
        # (result is same up to rescaling; this ensures all-identity input matrices solve to no change)
        rhs = 2 * arap_rhs_left.mm(rots_for_igl_rhs).reshape(3, -1).t()
        if thlog.logguard(LOG_TRACE):
            thlog.trace(
                f"rhs:\n{rhs}\n rotmats\n{per_vertex_rot_matrices_packed}\nigl arap rhs\n{arap_rhs_left}"
            )

        # now do the solve
        solver = laplacians_solvers.cholespy_solvers[i]
        if _haspin := (laplacians_solvers.removed_first_L_column is not None):
            removed_first_L_column = laplacians_solvers.removed_first_L_column[i]
            # removed first L column has shape (n_verts, 1)
            # verts[None, 0] has shape (1, 3)
            # multiplied has shape (n_verts, 3)
            rhs = (rhs - removed_first_L_column * verts[None, 0])[1:]

        soln = CholespySymmetricSolve_AutogradFn.apply(solver, rhs)
        assert isinstance(soln, torch.Tensor)
        # soln won't have any padding to trim because solver's system matrix is not padded
        if _haspin:
            soln = torch.cat((verts[None, 0], soln), dim=0)
        solutions.append(soln)

    soln_verts_packed = torch.cat(solutions, dim=0)  # (n_verts_packed,3)
    if postprocess == "recenter_rescale":
        soln_verts_packed = recenter_to_centroid_and_rescale_new_verts_to_fit_old_bboxes(
            meshes, soln_verts_packed
        )
    elif postprocess == "recenter_only":
        soln_verts_packed = recenter_to_centroid(meshes, soln_verts_packed)

    return soln_verts_packed


def index_sparse_coo_matrix_rowcol(
    x: torch.Tensor, row_idxs: torch.Tensor, col_idxs: torch.Tensor
) -> torch.Tensor:
    """
    indexes a 2D sparse_coo matrix with row indices and column indices behaving like
    x[row_idxs, col_idxs] as if x were a dense 2D matrix (without needing to_dense()!)
    """
    assert x.ndim == 2
    idx_selected = x.index_select(0, row_idxs).index_select(1, col_idxs).coalesce()
    idx_selected_rows, idx_selected_cols = idx_selected.indices()
    return idx_selected.values()[idx_selected_rows == idx_selected_cols]


def calc_ARAP_global_solve(
    meshes: Meshes,
    laplacians_solvers: SparseLaplaciansSolvers,
    per_vertex_rot_matrices_packed: torch.Tensor,
    arap_energy_type: ARAPEnergyTypeName,
    postprocess: Optional[PostprocessAfterSolveName],
) -> torch.Tensor:
    """
    per_vertex_rot_matrices_packed: shape (n_verts_packed, 3, 3)
    returns the ARAP global solve result, i.e. solution p' such that
    Lp' = b (equation 9 in the ARAP paper)
    where L is the cotangent laplacian, and b is the right hand side described in the paper
    """
    if laplacians_solvers.igl_arap_rhs_lefts is not None:
        assert (
            arap_energy_type == "spokes_and_rims_igl" or arap_energy_type == "spokes_igl"
        ), (
            "solver has IGL ARAP RHS constructors so must use spokes_and_rims_igl or spokes_igl for arap_energy_type"
        )
        # the rhs constructor from IGL is available, use this other function!
        # this goes with arap_energy_type == "spokes_igl" and "spokes_and_rims_igl"
        return calc_ARAP_global_solve__better_with_rhs_lefts(
            meshes,
            laplacians_solvers,
            per_vertex_rot_matrices_packed,
            postprocess,
        )
    # the rest of this function here computes rhs directly from the 2004 paper formula
    # if arap_energy_type == spokes_mine, or the rhs from the spokes-and-rims energy
    # from Chao et al 2011 (also used in normal analogies; formula described in CGAL docs)
    # the rhs to find has shape (n_verts, 3)
    solutions = []
    # can we write a batched version of this without a loop? (though we'll still end up
    # looping through the cholespy solvers anyhow...)
    for i, (L, verts_padded, faces, n_verts_this_mesh, verts_packed_first_idx) in enumerate(
        zip(
            laplacians_solvers.Ls,
            meshes.verts_padded(),
            meshes.faces_list(),
            meshes.num_verts_per_mesh(),
            meshes.mesh_to_verts_packed_first_idx(),
        )
    ):
        # it might be possible that verts_padded for this mesh has shorter dim0 length than
        # L because the meshes might have been indexed from a larger meshes batch, with
        # padding shrunken to fit just the largest mesh in the extracted batch. in this
        # case, we expand verts with padding to match the dim0 and dim1 size of the square L
        if (vp_sz0 := verts_padded.size(0)) < (L_sz0 := L.size(0)):
            verts_padded = nn.functional.pad(verts_padded, (0, 0, 0, L_sz0 - vp_sz0))

        # for each edge between a vertex i and vertex j, compute (w_ij / 2) * ((R_i
        # + R_j) @ (p_i - p_j)) (this is a 3d point)
        L = L.coalesce()

        if arap_energy_type == "spokes_mine":
            L_sp_indices = L.indices()
            # there are (2*n_edges) directed edges. we can get an array of (directed)edge
            # weights directly from the COO tensor's values array rather than going through
            # confusing index_selects (sparse tensors don't support tensor indexing). For some
            # reason the ordering has to be this way, for the i and j vert order in each index
            # in the ARAP rhs formula. If I were to do the obvious (i is indices[0]) then each
            # solve would give the right result except the y axis is flipped (?!)
            dir_edges_vi = L_sp_indices[1]
            dir_edges_vj = L_sp_indices[0]
            dir_edges_weight = L.values()

            # Ri + Rj
            rot_vi_plus_rot_vj = (
                per_vertex_rot_matrices_packed[dir_edges_vi + verts_packed_first_idx]
                + per_vertex_rot_matrices_packed[dir_edges_vj + verts_packed_first_idx]
            )

            # pi - pj
            pi_minus_pj = verts_padded[dir_edges_vi] - verts_padded[dir_edges_vj]

            # (w_ij / 2) * ((R_i + R_j) @ (p_i - p_j))
            rhs_per_dir_edge = (dir_edges_weight / 2).unsqueeze(1) * rot_vi_plus_rot_vj.bmm(
                pi_minus_pj.unsqueeze(-1)
            ).squeeze(-1)

            # the rhs vector is the same shape as verts_padded; then each slot corresponding to
            # vertex index j in the rhs vector is the sum of the values of the directed edges
            # out of vertex j. Here we use j because it corresponded to L_sp_indices[0]; if we
            # use i, then we still get the right system soln but the y axis is flipped (probably
            # a sign bug that managed to cancel out if I do this ij swap..)
            # rhs = torch.index_add(torch.zeros_like(verts), 0, dir_edges_vj, rhs_per_dir_edge)
            rhs = torch.index_put(
                torch.zeros_like(verts_padded),
                (dir_edges_vj,),
                rhs_per_dir_edge,
                accumulate=True,
            )
            # for this, index_put gives essentially the same result as index_add
            # there, but is not undefined behavior on duplicate indices, unlike index_add
        elif arap_energy_type == "spokes_and_rims_mine":
            faces_v0idx = faces[:, 0]
            faces_v1idx = faces[:, 1]
            faces_v2idx = faces[:, 2]
            # these indices won't touch the padding so verts_padded not necessary, but
            # we happen to have needed that for the above (...hopefully not? but morally,
            # the L used in the spokes_mine branch is square with max_n_verts length)
            v0 = verts_padded[faces_v0idx]
            v1 = verts_padded[faces_v1idx]
            v2 = verts_padded[faces_v2idx]
            r0 = per_vertex_rot_matrices_packed[faces_v0idx + verts_packed_first_idx]
            r1 = per_vertex_rot_matrices_packed[faces_v1idx + verts_packed_first_idx]
            r2 = per_vertex_rot_matrices_packed[faces_v2idx + verts_packed_first_idx]
            w01 = index_sparse_coo_matrix_rowcol(L, faces_v0idx, faces_v1idx)[:, None, None]
            w12 = index_sparse_coo_matrix_rowcol(L, faces_v1idx, faces_v2idx)[:, None, None]
            w20 = index_sparse_coo_matrix_rowcol(L, faces_v2idx, faces_v0idx)[:, None, None]

            e01 = (v1 - v0).unsqueeze(-1)
            e02 = (v2 - v0).unsqueeze(-1)
            e12 = (v2 - v1).unsqueeze(-1)
            e10 = -e01
            e20 = -e02
            e21 = -e12

            onethird = 1 / 3
            # v0_contrib =
            #   ((w01 * r0 / 3) / 2 + (w01 * r1 / 3) / 2 + (w10 * r0 / 3) / 2 + (w10 * r1 / 3) / 2 + (w10 * r2 / 3)) * (v1 - v0) # term for edge 01
            # + ((w02 * r0 / 3) / 2 + (w02 * r2 / 3) / 2 + (w20 * r0 / 3) / 2 + (w20 * r2 / 3) / 2 + (w02 * r1 / 3)) * (v2 - v0) # term for edge 02
            # our cot weights are symmetric so w01 == w10 and so on
            # for each edge, the terms with /2 are because the other face that borders that
            # edge will doublecount those terms

            # v0_contrib = onethird * (
            #     (w01 * (r0 + r1) + w01 * r2).bmm(e01)
            #     + (w20 * (r0 + r2) + w20 * r1).bmm(e02)
            # ).squeeze(-1)
            # v1_contrib = onethird * (
            #     (w12 * (r1 + r2) + w12 * r0).bmm(e12)
            #     + (w01 * (r1 + r0) + w01 * r2).bmm(e10)
            # ).squeeze(-1)
            # v2_contrib = onethird * (
            #     (w20 * (r2 + r0) + w20 * r1).bmm(e20)
            #     + (w12 * (r2 + r1) + w12 * r0).bmm(e21)
            # ).squeeze(-1)

            # further simplified:
            rs = r0 + r1 + r2
            # need /2 so that solution given all identity matrices would be scaled back to
            # the original size (identity matrices should lead to no change/the same mesh)
            w01rs = (w01 / 2) * rs
            w20rs = (w20 / 2) * rs
            w12rs = (w12 / 2) * rs
            v0_contrib = onethird * (w01rs.bmm(e01) + w20rs.bmm(e02)).squeeze(-1)
            v1_contrib = onethird * (w12rs.bmm(e12) + w01rs.bmm(e10)).squeeze(-1)
            v2_contrib = onethird * (w20rs.bmm(e20) + w12rs.bmm(e21)).squeeze(-1)
            v0v1v2idxs = torch.cat((faces_v0idx, faces_v1idx, faces_v2idx), dim=0)
            rhs_contribs = torch.cat((v0_contrib, v1_contrib, v2_contrib), dim=0)
            rhs = torch.index_put(
                torch.zeros_like(verts_padded), (v0v1v2idxs,), rhs_contribs, accumulate=True
            )
        else:
            raise AssertionError(
                f"shouldn't use calc_ARAP_global_solve with this arap_energy_type setting: {arap_energy_type} (if it's an igl arap energy type, maybe I forgot to init the solvers with igl_arap_rhs_lefts)"
            )

        # now do the solve
        solver = laplacians_solvers.cholespy_solvers[i]
        if _haspin := (laplacians_solvers.removed_first_L_column is not None):
            removed_first_L_column = laplacians_solvers.removed_first_L_column[i]
            # removed first L column has shape (n_verts, 1)
            # verts[None, 0] has shape (1, 3)
            # multiplied has shape (n_verts, 3)
            rhs = (rhs[:vp_sz0] - removed_first_L_column * verts_padded[None, 0])[1:vp_sz0]
        else:
            rhs = rhs[:vp_sz0]

        soln = CholespySymmetricSolve_AutogradFn.apply(solver, rhs)
        assert isinstance(soln, torch.Tensor)
        # soln won't have any padding to trim because solver's system matrix is not padded
        if _haspin:
            soln = torch.cat((verts_padded[None, 0], soln), dim=0)
        solutions.append(soln)

    soln_verts_packed = torch.cat(solutions, dim=0)  # (n_verts_packed,3)
    if postprocess == "recenter_rescale":
        soln_verts_packed = recenter_to_centroid_and_rescale_new_verts_to_fit_old_bboxes(
            meshes, soln_verts_packed
        )
    elif postprocess == "recenter_only":
        soln_verts_packed = recenter_to_centroid(meshes, soln_verts_packed)

    return soln_verts_packed


def calc_poisson_global_solve(
    meshes: Meshes,
    laplacians_solvers: SparseLaplaciansSolvers,
    faces_matrices_packed: torch.Tensor,
    postprocess: Optional[PostprocessAfterSolveName],
) -> torch.Tensor:
    """
    given per-face 3x3 transform matrices in `faces_matrices_packed` of shape
    `(len(meshes.faces_packed()), 3, 3)`, treating them as per-face jacobians of a piecewise
    linear mapping, do a poisson solve to find the best-fitting per-vertex map. (as seen in
    Neural Jacobian Fields)
    """
    poisson_rhs_lefts = laplacians_solvers.poisson_rhs_lefts
    assert poisson_rhs_lefts is not None, (
        "must have poisson_rhs_lefts precomputed in `laplacians` struct to run poisson solve. make sure to set compute_poisson_rhs_lefts=True when calling SparseLaplaciansSolvers.from_meshes"
    )
    n_verts_per_face = 3
    n_coords = 3
    # faces_matrices_packed has shape (n_faces_packed, 3vertsperface, 3coords)

    # transpose to be (n_faces, 3coords, 3vertsperface)
    faces_matrices_packed = faces_matrices_packed.transpose(-1, -2)

    # then pad it out to match (batch, max_n_faces, 3coords, 3vertsperface)
    num_faces_per_mesh = meshes.num_faces_per_mesh()
    batch_size = num_faces_per_mesh.size(0)
    max_n_faces_per_mesh = poisson_rhs_lefts.size(2) // n_coords
    # ^ this is the max_n_faces in the shape of poisson_rhs_lefts, which is at least as
    # large as max(num_faces_per_mesh). we want this shape to be compat with
    # poisson_rhs_lefts so we'll use that number.

    faces_matrices_padded = torch.zeros(
        (batch_size, max_n_faces_per_mesh, n_coords, n_verts_per_face),
        dtype=faces_matrices_packed.dtype,
        device=faces_matrices_packed.device,
    )
    batch_idx, idx_in_batch = make_padded_to_packed_indexer(num_faces_per_mesh)
    faces_matrices_padded[batch_idx, idx_in_batch] = faces_matrices_packed

    # then view it as a batch of stacked matrices for bmm with poisson_rhs_lefts
    faces_matrices_padded = faces_matrices_padded.view(
        batch_size, max_n_faces_per_mesh * n_coords, n_verts_per_face
    )

    # then do the bmm to get the right-hand side of the system
    rhs = poisson_rhs_lefts.bmm(faces_matrices_padded)
    # poisson_rhs_lefts is (batch_size, max_n_verts, max_n_faces * 3coords)
    # bmm with face_matrices_padded (batch_size, max_n_faces * 3coords, 3vertsperface)
    # to get rhs shape (batch_size, max_n_verts, 3vertsperface)

    # then solve
    # unfortunately, cholespy does not support a batched-system-matrices solve, unlike
    # torch's cholesky_solve. cholespy can only do a batched rhs (provided we cat
    # features/columns together onto one big mega rhs)).
    # also, cholespy cannot init with a padded laplacian! we have to use the unpadded
    # laplacian and the unpadded rhs for this!
    soln_verts_packed = torch.cat(
        tuple(
            (
                soln := cast(
                    torch.Tensor,
                    CholespySymmetricSolve_AutogradFn.apply(
                        solver,
                        (
                            rhs_single[: verts.size(0)]
                            - laplacians_solvers.removed_first_L_column[i] * verts[None, 0]
                        )[1 : verts.size(0)]
                        if (
                            _haspin := (
                                laplacians_solvers.removed_first_L_column is not None
                            )
                        )
                        else rhs_single[: verts.size(0)],
                    ),
                ),
                torch.cat((verts[None, 0], soln), dim=0) if _haspin else soln,
            )[-1]
            for i, (solver, rhs_single, verts) in enumerate(
                zip(laplacians_solvers.cholespy_solvers, rhs, meshes.verts_list())
            )
        ),
        dim=0,
    )

    if postprocess == "recenter_rescale":
        soln_verts_packed = recenter_to_centroid_and_rescale_new_verts_to_fit_old_bboxes(
            meshes, soln_verts_packed
        )
    elif postprocess == "recenter_only":
        soln_verts_packed = recenter_to_centroid(meshes, soln_verts_packed)

    return soln_verts_packed


def get_igl_arap_energy_type_from_cfg(
    arap_energy_type: Optional[ARAPEnergyTypeName],
) -> Optional[IGL_ARAPEnergyType]:
    # translate my arap energy type name to the igl enum
    if (
        arap_energy_type is None
        or arap_energy_type == "spokes_mine"
        or arap_energy_type == "spokes_and_rims_mine"
    ):
        # spokes_mine is my own rhs impl directly from the 2004 paper, don't use IGL
        # spokes_and_rims_mine is my attempt at reverse engineering the IGL spokes_and_rims?
        igl_arap_energy_type = None
    elif arap_energy_type == "spokes_igl":
        igl_arap_energy_type = IGL_ARAPEnergyType.SPOKES
    elif arap_energy_type == "spokes_and_rims_igl":
        igl_arap_energy_type = IGL_ARAPEnergyType.SPOKES_AND_RIMS
    else:
        raise InvalidConfigError("unknown arap_energy_type")
    return igl_arap_energy_type


def applymethod__vertex_rotations_into_ARAP_solve(
    patient_meshes: Meshes,
    laplacians_solvers: SparseLaplaciansSolvers,
    per_vertex_rot_matrices_packed: torch.Tensor,
    arap_energy_type: ARAPEnergyTypeName,
    rotations_are_3x2_repr: bool,
    return_offsets_to_solution: bool,
) -> torch.Tensor:
    """
    per_vertex_rot_matrices_packed is of shape `(len(patient_meshes.faces_packed()),3,3)`, a
    rotation matrix for each face if not rotations_are_3x2_repr ; otherwise, it's (...,3,2).
    if return_offsets_to_solution (default=True), returns the verts update
    that will take current verts to the solution, not the solution itself
    """
    soln_verts_packed = calc_ARAP_global_solve(
        patient_meshes,
        laplacians_solvers,
        per_vertex_rot_matrices_packed
        if not rotations_are_3x2_repr
        else convert_rot3x2_to_rot3x3(per_vertex_rot_matrices_packed),
        arap_energy_type,
        postprocess="recenter_rescale",
    )
    if return_offsets_to_solution:
        return soln_verts_packed - patient_meshes.verts_packed()
    else:
        return soln_verts_packed


def convert_rot3x2_to_rot3x3(rot3x2: torch.Tensor) -> torch.Tensor:
    """
    from Hao Li's paper "On the Continuity of Rotation Representations in Neural Networks"
    """
    a1 = rot3x2[:, :, 0]
    a2 = rot3x2[:, :, 1]
    b1 = nn.functional.normalize(a1, dim=-1)
    b2 = nn.functional.normalize(a2 - (b1 * a2).sum(dim=-1, keepdim=True) * b1, dim=-1)
    b3 = torch.linalg.cross(b1, b2)
    rot3x3 = torch.stack((b1, b2, b3), dim=-1)
    return rot3x3


def average_face_quaternions_onto_vertex_quaternions(
    face_areas: torch.Tensor,
    n_verts: int,
    faces: torch.Tensor,
    per_face_quaternions_packed: torch.Tensor,
) -> torch.Tensor:
    # let's do https://stackoverflow.com/a/72039849
    # which is itself from http://tbirdal.blogspot.com/2019/10/i-allocate-this-post-to-providing.html
    # which is itself from http://www.acsu.buffalo.edu/~johnc/ave_quat07.pdf Markley et al. 2007
    Q = per_face_quaternions_packed  # shape (n_faces, 4)
    oriented_Q = ((Q[:, 0:1] > 0).float() - 0.5) * 2 * Q

    # do a self-outer product. shape (n_faces, 4, 4)
    outprod_Q = torch.einsum("bi,bk->bik", (oriented_Q, oriented_Q))
    weighted_outprod_Q = outprod_Q * face_areas.view(face_areas.size(0), 1, 1)

    # gather the outer product
    out_A = torch.zeros((n_verts, 4, 4), dtype=Q.dtype, device=Q.device)
    out_A.index_put_((faceverts0 := faces[:, 0],), weighted_outprod_Q, accumulate=True)
    out_A.index_put_((faceverts1 := faces[:, 1],), weighted_outprod_Q, accumulate=True)
    out_A.index_put_((faceverts2 := faces[:, 2],), weighted_outprod_Q, accumulate=True)
    if thlog.logguard(LOG_TRACE):
        thlog.trace(f"""
        weighted_outprod_Q {weighted_outprod_Q}
        out_A {out_A}
        """)

    # gather the weights
    sumfacearea_per_vertex = torch.zeros(n_verts, device=Q.device)
    sumfacearea_per_vertex.index_put_((faceverts0,), face_areas, accumulate=True)
    sumfacearea_per_vertex.index_put_((faceverts1,), face_areas, accumulate=True)
    sumfacearea_per_vertex.index_put_((faceverts2,), face_areas, accumulate=True)

    # divide weighted sum on each vertex by sum of weights around that vertex
    out_A.div_(sumfacearea_per_vertex.view(sumfacearea_per_vertex.size(0), 1, 1))

    # see HACK below: detect bad out_A and overwrite before eigh. The intended eigvec for
    # such bad out_A are [1,0,0,0] (id quaternion), see below. We want to overwrite such
    # that the largest-eigval's eigenvector (for out_Q) comes out to be [1,0,0,0] and also
    # eigvalues are unique. One such matrix is just diag([3,2,1,0])
    out_A_has_degen_matrix = (
        (out_A[:, 1, 1] == 0)
        .logical_and_(out_A[:, 2, 2] == 0)
        .logical_and_(out_A[:, 0, 0] > 0)
    )
    out_A_has_degen_matrix_where = torch.where(out_A_has_degen_matrix)[0]
    if out_A_has_degen_matrix_where.numel() > 0:
        out_A[out_A_has_degen_matrix_where] = torch.diag(
            torch.arange(3, -1, -1, dtype=out_A.dtype, device=out_A.device)
        )

    # eigenvector corresponding to the largest eigenvalue
    eigh = torch.linalg.eigh(out_A)  # named tuple (eigenvalues, eigenvectors)
    out_Q = eigh.eigenvectors[:, :, -1]

    # if thlog.logguard(LOG_TRACE):
    #     __eigvals = eigh.eigenvalues.detach()
    #     # # there are 4 eigenvalues ordered ascending, just check pairwise isclose in order
    #     # close01 = torch.isclose(__eigvals[:, 0], __eigvals[:, 1])
    #     # close12 = torch.isclose(__eigvals[:, 1], __eigvals[:, 2])
    #     # close23 = torch.isclose(__eigvals[:, 2], __eigvals[:, 3])
    #     # if close01.any():
    #     #     argwhere = close01.argwhere()
    #     #     thlog.trace(f"eigvals 01 hit allclose! relevant eigvals rows: {__eigvals[argwhere[:, 0]]} how many? {argwhere.shape}")
    #     # if close12.any():
    #     #     argwhere = close12.argwhere()
    #     #     thlog.trace(f"eigvals 12 hit allclose! relevant eigvals rows: {__eigvals[close12.argwhere()[:, 0]]} how many? {argwhere.shape} ")
    #     # if close23.any():
    #     #     argwhere = close23.argwhere()
    #     #     thlog.trace(f"eigvals 23 hit allclose! relevant eigvals rows: {__eigvals[close23.argwhere()[:, 0]]} how many? {argwhere.shape} ")
    #     # # after doing the above, seems like all the offending eigvals is [0,0,0,1] what matrix did they correspond to?
    #     __eigvals1zero = (__eigvals[:, 1] == 0)
    #     if __eigvals1zero.any():
    #         where = torch.where(__eigvals1zero)[0]
    #         # print out the out_A matrix
    #         thlog.trace(f"found eigval1 zero at {where}, \n the out_A matrices that gave these are\n{(__outawhere := out_A[where])}\n\nare these <1e-20? {(__outawhereabs:=__outawhere.abs()) < 1e-20}\n are these <1e-15? {__outawhereabs < 1e-15 }\n are these <1e-12? {__outawhereabs < 1e-12} \nselected 'normal' out_A examples (first 10):\n{out_A[:10]}")
    #         # seems like all the cases where this happens are with this exact matrix:
    #         #[[0.2500, 0.0000, 0.0000, 0.0000],
    #         # [0.0000, 0.0000, 0.0000, 0.0000],
    #         # [0.0000, 0.0000, 0.0000, 0.0000],
    #         # [0.0000, 0.0000, 0.0000, 0.0000]]
    #         # the intended last-eigenvec for this is [1,0,0,0]
    #         # HACK so perhaps I could just check some arbitrary two elements to be exactly zero.
    #         # the main diag (1,1), (2,2) perhaps, plus making sure that (0,0) is nonzero just to be safe

    return out_Q


def applymethod__avg_face_rotations_into_ARAP_solve(
    patient_meshes: Meshes,
    laplacians_solvers: SparseLaplaciansSolvers,
    per_face_rot_matrices_packed: torch.Tensor,
    arap_energy_type: ARAPEnergyTypeName,
    rotations_are_3x2_repr: bool,
    return_offsets_to_solution: bool,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    `per_face_rot_matrices_packed` is of shape
    `(len(patient_meshes.faces_packed()),3,3)`, a rotation matrix for each face
    if not rotations_are_3x2_repr ; otherwise, it's (..., 3, 2)

    applies ARAP solve, and returns
    - if return_offsets_to_solution (default=True), returns offsets for vertices (from
        patient_meshes's current locations towards locations obtained in the ARAP solve)
      otherwise, returns the solution positions directly
    - rot matrices per vertex (avg'd from each vert's adjacent faces' rot matrices)
    """
    patient_meshes_verts = patient_meshes.verts_packed()
    patient_meshes_faces = patient_meshes.faces_packed()
    patient_meshes_faces_areas = patient_meshes.faces_areas_packed()
    do_matrix_avg = DARAPPASTFEATURE__WRONG_ROT_AVERAGING or rotations_are_3x2_repr
    last_dim_shape = 2 if rotations_are_3x2_repr else 3

    # since ARAP and normal analogies uses per-vertex rotations, we might
    # just average the rotation matrices from a vert's adjacent faces onto
    # it. we need this for the "ARAP right hand side" in eqn 8 and 9 in arap paper
    if do_matrix_avg:
        # this was the wrong rot averaging used before 2024-09-04
        per_vertex_pred_packed = torch.zeros(
            (patient_meshes_verts.size(0), 3, last_dim_shape),
            dtype=per_face_rot_matrices_packed.dtype,
            device=per_face_rot_matrices_packed.device,
        )
        gather_per_face_quantity_into_per_vertex_quantity_inplace(
            patient_meshes_faces_areas,
            calc_sum_incident_face_area_per_vertex(
                patient_meshes_verts, patient_meshes_faces
            ),
            patient_meshes_faces,
            per_face_rot_matrices_packed,
            out=per_vertex_pred_packed,
        )
        if last_dim_shape == 2:
            per_vertex_rot_matrices_packed = convert_rot3x2_to_rot3x3(
                per_vertex_pred_packed
            )
        else:
            per_vertex_rot_matrices_packed = per_vertex_pred_packed
    else:
        # use the correct quaternion averaging for 3x3 (but this gives terrible gradients)
        per_face_rot_quats_packed = pt3d_transforms.matrix_to_quaternion(
            per_face_rot_matrices_packed
        )
        per_vertex_rot_quats_packed = average_face_quaternions_onto_vertex_quaternions(
            patient_meshes_faces_areas,
            patient_meshes_verts.size(0),
            patient_meshes_faces,
            per_face_rot_quats_packed,
        )
        per_vertex_rot_matrices_packed = pt3d_transforms.quaternion_to_matrix(
            per_vertex_rot_quats_packed
        )

    soln_verts_packed = calc_ARAP_global_solve(
        patient_meshes,
        laplacians_solvers,
        per_vertex_rot_matrices_packed,
        arap_energy_type,
        postprocess="recenter_rescale",
    )

    if return_offsets_to_solution:
        return soln_verts_packed - patient_meshes_verts, per_vertex_rot_matrices_packed
    else:
        return soln_verts_packed, per_vertex_rot_matrices_packed


def applymethod__face_rotations_into_poisson_solve(
    patient_meshes: Meshes,
    laplacians_solvers: SparseLaplaciansSolvers,
    per_face_rot_matrices_packed: torch.Tensor,
    rotations_are_3x2_repr: bool,
    return_offsets_to_solution: bool,
) -> torch.Tensor:
    """
    per_vertex_rot_matrices_packed is of shape `(len(patient_meshes.faces_packed()),3,3)`, a
    rotation matrix for each face if not rotations_are_3x2_repr ; otherwise, it's (...,3,2)
    if return_offsets_to_solution (default=True), returns the verts update
    that will take current verts to the solution, not the solution itself
    """
    soln_verts_packed = calc_poisson_global_solve(
        patient_meshes,
        laplacians_solvers,
        per_face_rot_matrices_packed
        if not rotations_are_3x2_repr
        else convert_rot3x2_to_rot3x3(per_face_rot_matrices_packed),
        postprocess="recenter_rescale",
    )
    if return_offsets_to_solution:
        return soln_verts_packed - patient_meshes.verts_packed()
    else:
        return soln_verts_packed


def seed_all(torch_seed: Optional[int], numpy_seed: Optional[int]) -> Tuple[int, int]:
    if torch_seed is None:
        torch_seed = torch.initial_seed()
        thlog.info(f"torch initial seed is {torch_seed}")
    if numpy_seed is None:
        # we cannot get the actual numpy init seed, so we'll just generate a random number
        # and use that as a seed!!
        numpy_seed = np.random.randint(2**32)
        thlog.info(f"numpy initial seed is {numpy_seed}")

    torch.manual_seed(torch_seed)
    torch.cuda.manual_seed(torch_seed)
    np.random.seed(numpy_seed)
    random.seed(0)
    thlog.info(f"Set torch seed to {torch_seed} and numpy seed to {numpy_seed}")
    return torch_seed, numpy_seed


@dataclass(slots=True)
class Procrustes_Settings(Thronfig):
    lamb: float
    normalize_target_normals: bool


@dataclass(slots=True)
class ProcrustesPrecomputeAndWhetherToNormalize:
    pp: ProcrustesPrecompute
    normalize: bool
    """ whether to normalize the target normal """


#### convenience API that initializes and decides what to do based on config field values
# (the above calc functions are specialized, to be chosen depending on the config)
# mostly for use with the optimization pipeline and for applying saved deformation qty files
# saved from the optimization


@dataclass(slots=True)
class QuantityBeingOptimized:
    """
    - if this_is == "verts_offsets", tensor has shape (n_verts_packed, 3)
    - if this_is == "faces_normals", tensor has shape (n_faces_packed, 3)
    - if this_is == "verts_normals", tensor has shape (n_verts_packed, 3)
    - if this_is == "faces_jacobians", parameter has one tensor of shape (n_faces_packed, 3, 3)
    - if this_is == "faces_3x2rotations", parameter has one tensor of shape (n_faces_packed, 3, 2)
    - if this_is == "verts_3x2rotations", parameter has one tensor of shape (n_verts_packed, 3, 2)

    the __getitem__ operation returns another QuantityBeingOptimized that has an extracted
    packed quantity as a new (n_verts_packed/n_faces_packed, *) tensor, containing elements
    corresponding to the indexed meshes in the original batched QuantityBeingOptimized

    (this is also the idea for all the dataclasses that have a MeshesPackedIndexer component
    in addition to this one. This one just happens to not use MeshesPackedIndexer but has
    the same behavior on indexing.)
    """

    tensor: torch.Tensor
    """ main parameter tensor to be optimized. present for all this_is possibilities """

    this_is: DeformOptimQuantityName

    num_verts_per_mesh: List[int]
    num_faces_per_mesh: List[int]
    """ originally from the meshes where this QuantityBeingOptimized came from.
    bookkeeping to help with __getitem__ """

    procrustes_struct_if_needed: Optional[ProcrustesPrecomputeAndWhetherToNormalize]
    """
    this is only non-None when optimize_deform_via is a method supporting procrustes
    (right now, only verts_normals does, but faces_normals can also work if I do
    implement face-only ARAP) AND a procrustes lambda is specified in deform_by_csd_cfg
    (the bool indicates whether the target normal should be normalized before solving)
    """

    def __getitem__(self, index: Union[int, List[int], torch.Tensor]):
        list_idxr_fn: Callable[[Sequence], Sequence]
        if isinstance(index, int):
            list_idxr_fn = lambda xs: tuple(xs[index])
        elif isinstance(index, list):
            list_idxr_fn = lambda xs: tuple(xs[i] for i in index)
        else:
            list_idxr_fn = lambda xs: tuple(xs[int(i.item())] for i in index)

        if (
            (this_is := self.this_is) == "faces_jacobians"
            or this_is == "faces_3x2rotations"
            or this_is == "faces_normals"
        ):
            tensor = torch.cat(
                list_idxr_fn(torch.split(self.tensor, self.num_faces_per_mesh)),
                dim=0,
            )
        elif (
            this_is == "verts_offsets"
            or this_is == "verts_3x2rotations"
            or this_is == "verts_normals"
        ):
            tensor = torch.cat(
                list_idxr_fn(torch.split(self.tensor, self.num_verts_per_mesh)),
                dim=0,
            )
        else:
            raise InvalidConfigError(f"unknown quantity to optimize {this_is}")

        return __class__(
            tensor=tensor,
            this_is=this_is,
            num_verts_per_mesh=list(list_idxr_fn(self.num_verts_per_mesh)),
            num_faces_per_mesh=list(list_idxr_fn(self.num_faces_per_mesh)),
            procrustes_struct_if_needed=(
                ProcrustesPrecomputeAndWhetherToNormalize(
                    self.procrustes_struct_if_needed.pp[index],
                    self.procrustes_struct_if_needed.normalize,
                )
                if self.procrustes_struct_if_needed
                else None
            ),
        )

    @classmethod
    def init_according_to_cfg(
        cls,
        meshes: Meshes,
        optimize_deform_via: DeformOptimQuantityName,
        procrustes_cfg_and_solver_and_arap_energy_type_if_procrustes_needed: Optional[
            Tuple[
                Procrustes_Settings, SparseLaplaciansSolvers, Optional[ARAPEnergyTypeName]
            ]
        ],
    ) -> "QuantityBeingOptimized":
        procrustes_struct_if_needed = None
        if optimize_deform_via == "verts_offsets":
            tensor = torch.zeros_like(meshes.verts_packed())
        elif optimize_deform_via == "faces_normals":
            # for some reason i cannot grab the orig face normals even if i
            # detach(), clone(), clone().detach() or whatever; this is just
            # going to get a nasty autograd inplace operation error so i will
            # optimize the offsets to be added to normals instead
            tensor = torch.zeros_like(meshes.faces_normals_packed())
        elif optimize_deform_via == "verts_normals":
            tensor = meshes.verts_normals_packed()
            if procrustes_cfg_and_solver_and_arap_energy_type_if_procrustes_needed:
                procrustes_cfg, laplacians_solvers, arap_energy_type = (
                    procrustes_cfg_and_solver_and_arap_energy_type_if_procrustes_needed
                )

                procrustes_struct_if_needed = ProcrustesPrecomputeAndWhetherToNormalize(
                    pp=ProcrustesPrecompute.from_meshes(
                        local_step_procrustes_lambda=procrustes_cfg.lamb,
                        arap_energy_type=arap_energy_type,
                        laplacians_solvers=laplacians_solvers,
                        patient_meshes=meshes,
                    ),
                    normalize=procrustes_cfg.normalize_target_normals,
                )
            else:
                procrustes_struct_if_needed = None

        elif optimize_deform_via == "faces_3x2rotations":
            # from Hao Li's paper "On the Continuity of Rotation Representations in Neural Networks"
            verts = meshes.verts_packed()
            faces = meshes.faces_packed()
            tensor = torch.zeros(
                (faces.size(0), 3, 2),
                device=verts.device,
                dtype=verts.dtype,
            )
            # initialize to identity
            tensor[:, (0, 1), (0, 1)] = 1.0
        elif optimize_deform_via == "verts_3x2rotations":
            # from Hao Li's paper "On the Continuity of Rotation Representations in Neural Networks"
            verts = meshes.verts_packed()
            faces = meshes.faces_packed()
            tensor = torch.zeros(
                (verts.size(0), 3, 2),
                device=verts.device,
                dtype=verts.dtype,
            )
            # initialize to identity
            tensor[:, (0, 1), (0, 1)] = 1.0
        elif optimize_deform_via == "faces_jacobians":
            verts = meshes.verts_packed()
            faces = meshes.faces_packed()
            tensor = torch.zeros(
                (faces.size(0), n_coords := 3, n_verts_per_face := 3),
                device=verts.device,
                dtype=verts.dtype,
            )
            # initialize to identity
            tensor[:, (0, 1, 2), (0, 1, 2)] = 1.0
        else:
            raise InvalidConfigError(
                f"invalid/not yet implemented optimize_deform_via {optimize_deform_via}"
            )

        return cls(
            tensor=tensor.requires_grad_(),
            this_is=optimize_deform_via,
            num_verts_per_mesh=meshes.num_verts_per_mesh().tolist(),
            num_faces_per_mesh=meshes.num_faces_per_mesh().tolist(),
            procrustes_struct_if_needed=procrustes_struct_if_needed,
        )


@dataclass(slots=True)
class ElemNormals_IntermediateResults:
    rot_matrices: torch.Tensor


DeformationIntermediateResults = Union[None, ElemNormals_IntermediateResults]
# add other structs here into this union for the other optimize_deform_via methods


@dataclass(slots=True)
class InputsToDeformationSolveMethods:
    intermediate_results: DeformationIntermediateResults = None
    rotations_are_3x2_repr: bool = False
    face_matrices_packed: Optional[torch.Tensor] = None
    vert_matrices_packed: Optional[torch.Tensor] = None
    shortcircuit_deformation_solution: Optional[Sequence[torch.Tensor]] = None
    """
    if this field is specified, then we skip the solve method and just take this as the
    deformation solution verts. certain solve methods (namely direct vertex offset) can use
    this to directly give the solution without going through a solve as is required by
    others; can also use this to inject a different pipeline that doesn't use the usual
    poisson solve pipeline here
    """


def calc_inputs_to_solve_for_deformation_according_to_cfg(
    pt3d_batched_meshes: Meshes, quantity_being_optimized: QuantityBeingOptimized
) -> InputsToDeformationSolveMethods:
    if (quantity_is := quantity_being_optimized.this_is) == "faces_jacobians":
        return InputsToDeformationSolveMethods(
            face_matrices_packed=quantity_being_optimized.tensor
        )
    elif quantity_is == "faces_3x2rotations":
        return InputsToDeformationSolveMethods(
            face_matrices_packed=quantity_being_optimized.tensor,
            rotations_are_3x2_repr=True,
        )
        # a1 = face_rot3x2_packed[:, :, 0]
        # a2 = face_rot3x2_packed[:, :, 1]
        # b1 = torch.nn.functional.normalize(a1, dim=-1)
        # b2 = torch.nn.functional.normalize(
        #     a2 - (b1 * a2).sum(dim=-1, keepdim=True) * b1, dim=-1
        # )
        # b3 = torch.linalg.cross(b1, b2)
        # face_matrices_packed = torch.stack((b1, b2, b3), dim=-1)
    elif quantity_is == "verts_3x2rotations":
        return InputsToDeformationSolveMethods(
            vert_matrices_packed=quantity_being_optimized.tensor,
            rotations_are_3x2_repr=True,
        )
    elif quantity_is == "faces_normals":
        # because we couldn't init with faces_normals_packed itself, we have to
        # init with zeros and treat the quantity-to-optimize to be the offset
        # to add onto faces_normals_packed...ah well!
        original_faces_normals = pt3d_batched_meshes.faces_normals_packed()
        updated_faces_normals = original_faces_normals + quantity_being_optimized.tensor
        face_matrices_packed = calc_rot_matrices_axisangle(
            original_faces_normals,
            torch.nn.functional.normalize(updated_faces_normals, dim=-1),
            epsilon=1e-6,
        )
        intermediate_results = ElemNormals_IntermediateResults(face_matrices_packed)
        return InputsToDeformationSolveMethods(
            face_matrices_packed=face_matrices_packed,
            intermediate_results=intermediate_results,
        )
    elif quantity_is == "verts_normals":
        # NOTE that this procrustes is always what is "the first local step" in a typical
        # ARAP iterative optimization (i.e. current work-in-progress verts == original
        # verts.) It has no knowledge of the currently deformed verts from the upper-level
        # optimization via SDS! this is a TODO but I doubt doing that will work very well
        # However, with large lambda, one-step procrustes + one-step global solve can get us
        # most of the way there in what'd otherwise require many ARAP iters
        vert_matrices_packed = (
            calc_rot_matrices_with_procrustes(
                quantity_being_optimized.procrustes_struct_if_needed.pp,
                pt3d_batched_meshes.verts_packed(),
                (
                    torch.nn.functional.normalize(quantity_being_optimized.tensor, dim=-1)
                    if quantity_being_optimized.procrustes_struct_if_needed.normalize
                    else quantity_being_optimized.tensor
                ),
            )
            if quantity_being_optimized.procrustes_struct_if_needed is not None
            else calc_rot_matrices_axisangle(
                pt3d_batched_meshes.verts_normals_packed(),
                torch.nn.functional.normalize(quantity_being_optimized.tensor, dim=-1),
                epsilon=1e-6,
            )
        )
        intermediate_results = ElemNormals_IntermediateResults(vert_matrices_packed)
        return InputsToDeformationSolveMethods(
            vert_matrices_packed=vert_matrices_packed,
            intermediate_results=intermediate_results,
        )
    elif quantity_is == "verts_offsets":
        return InputsToDeformationSolveMethods(
            shortcircuit_deformation_solution=per_vertex_packed_to_list(
                pt3d_batched_meshes, quantity_being_optimized.tensor
            )
        )
    else:
        raise InvalidConfigError(
            f"unknown/not yet implemented quantity to optimize: {quantity_being_optimized.this_is}"
        )


def calc_deformed_verts_solution_according_to_cfg(
    pt3d_batched_meshes: Meshes,
    solve_method: DeformSolveMethodName,
    arap_energy_type: Optional[ARAPEnergyTypeName],
    maybe_my_solver: Optional[SparseLaplaciansSolvers],
    inputs_to_deformation_solve_methods: InputsToDeformationSolveMethods,
) -> Tuple[Sequence[torch.Tensor], DeformationIntermediateResults]:
    """
    returns
    - a list of solution verts tensors, each corresponding to a struct in meshes_structs
    - intermediate results, present or None depending on the quantity_being_optimized, in
        case we wish to penalize or view some intermediate result involved in a deform method

    pt3d_batched_meshes must be a pytorch3d Meshes batch with the same number of
    meshes as len(meshes_structs), and each mesh in pt3d_batched_meshes must
    match the vertex (v_pos) and face (t_pos_idx) array of the corresp. mesh struct's nvdm_loaded_mesh
    """
    rotations_are_3x2_repr = inputs_to_deformation_solve_methods.rotations_are_3x2_repr
    if inputs_to_deformation_solve_methods.shortcircuit_deformation_solution is not None:
        soln_verts_list = (
            inputs_to_deformation_solve_methods.shortcircuit_deformation_solution
        )

    elif solve_method == "poisson":
        assert inputs_to_deformation_solve_methods.face_matrices_packed is not None, (
            "this optimize_deform_via does not involve per-face transforms needed for poisson solve"
        )
        assert maybe_my_solver is not None, "need my_solver while solve_method is poisson"
        soln_verts_packed = applymethod__face_rotations_into_poisson_solve(
            pt3d_batched_meshes,
            maybe_my_solver,
            inputs_to_deformation_solve_methods.face_matrices_packed,
            rotations_are_3x2_repr=rotations_are_3x2_repr,
            return_offsets_to_solution=False,
        )
        soln_verts_list = per_vertex_packed_to_list(pt3d_batched_meshes, soln_verts_packed)

    elif solve_method == "arap":
        # arap
        assert maybe_my_solver is not None, "need my_solver while solve_method is arap"
        assert arap_energy_type is not None, (
            "need non-None arap_energy_type for solve_method arap"
        )
        if inputs_to_deformation_solve_methods.face_matrices_packed is not None:
            soln_verts_packed, _ = applymethod__avg_face_rotations_into_ARAP_solve(
                pt3d_batched_meshes,
                maybe_my_solver,
                inputs_to_deformation_solve_methods.face_matrices_packed,
                arap_energy_type,
                rotations_are_3x2_repr=rotations_are_3x2_repr,
                return_offsets_to_solution=False,
            )
        elif inputs_to_deformation_solve_methods.vert_matrices_packed is not None:
            soln_verts_packed = applymethod__vertex_rotations_into_ARAP_solve(
                pt3d_batched_meshes,
                maybe_my_solver,
                inputs_to_deformation_solve_methods.vert_matrices_packed,
                arap_energy_type=arap_energy_type,
                rotations_are_3x2_repr=rotations_are_3x2_repr,
                return_offsets_to_solution=False,
            )
        else:
            raise AssertionError(
                "did I forget to set face_matrices_packed and vert_matrices_packed"
            )
        soln_verts_list = per_vertex_packed_to_list(pt3d_batched_meshes, soln_verts_packed)

    else:
        raise InvalidConfigError(f"unknown solve method {solve_method}")

    return soln_verts_list, inputs_to_deformation_solve_methods.intermediate_results
