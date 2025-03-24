"""
## deformations

(basically deformations_dARAP but with minimal dependencies for easy inclusion
in other projects)

basic usage:
>>> (verts1, faces1) = mesh1
    (verts2, faces2) = mesh2
    # etc. These _lists store their quantity for all meshes in a batch;
    # they should have the same length (== num of meshes in the batch).
    # The per-mesh arrays don't have to have the same element count as that of all other
    meshes (i.e. this is for heterogenous batching)
    verts_list = [verts1, verts2]
    faces_list = [faces1, faces2]
    solvers = SparseLaplaciansSolvers.from_meshes(
        verts_list, faces_list,
        compute_poisson_rhs_lefts=False, compute_igl_arap_rhs_lefts=None
    )
    deformed_verts_list = calc_ARAP_global_solve(
        verts_list, faces_list, solvers, per_vertex_3x3matrices_list,
        arap_energy_type="spokes_and_rims_mine",
        postprocess=True
    )

if you ever need to extract minibatches out of verts_list, faces_list, etc, and solve a
deformation on them, make sure to also get the corresponding solvers with
`solvers[batch_item_indices]` which returns another SparseLaplaciansSolvers object
containing only the laplacians and solvers for that selection of meshes.

### Features:
- `calc_gradient_operator`, differentiable torch equivalent of igl.grad() computing the mesh's gradient operator
- `calc_ARAP_global_solve`, solves ARAP global step given per-vertex 3x3 rotation matrices
- `calc_poisson_global_solve`, solves NJF poisson solve given per-face 3x3 rotation matrices

### Note on cotangent weight convention
- The official deformations_dARAP code uses pytorch3d whose cot_laplacian happens to return
(cot a + cot b) weights, which is what we went with throughout the project.
- IGL's cotmatrix function returns 0.5 * (cot a + cot b).
- In this code, for parity with the main project code, we thus use 2*cotmatrix
- As such, USE_ACTUAL_IGL_COTMATRIX_WEIGHTS defaults to False (for parity with project code)
- Set the constant USE_ACTUAL_IGL_COTMATRIX_WEIGHTS=True to switch to 0.5*(cota+cotb) weights
- In either setting for this constant, we make sure that (with first-vertex pinning, and
without any rescale normalization), solving with all-identity matrices will yield the same
mesh back with no change, which is what should be expected.
"""

from typing import Sequence, Optional, Union, List, Tuple, Callable, Set, Literal, cast
from enum import Enum
from dataclasses import dataclass

import numpy as np
from scipy.spatial.transform import Rotation
import igl
import torch
import cholespy

ARAPEnergyTypeName = Literal[
    "spokes_mine", "spokes_and_rims_mine", "spokes_igl", "spokes_and_rims_igl"
]

PostprocessAfterSolveName = Literal["recenter_rescale", "recenter_only"]

USE_ACTUAL_IGL_COTMATRIX_WEIGHTS = False
""" 
see docstring at top of this file for notes on this. Set to True to use the 0.5*(cot a + cot
b) cotangent weights (IGL's default), and adjust solves accordingly, rather than pytorch3d's
(cot a + cot b)  weights, which the main project code uses.
"""


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


class IGL_ARAPEnergyType(Enum):
    SPOKES = 0
    SPOKES_AND_RIMS = 1


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
    verts_np = verts.cpu().detach().numpy()
    faces_np = faces.cpu().detach().numpy()

    def __fuzz_rot_verts(_verts: np.ndarray) -> np.ndarray:
        # rot_axisangle = torch.zeros_like(_verts)
        angles = rng.random(size=(3,), dtype=np.float32) * 360
        rots = Rotation.from_rotvec(np.diag(angles), degrees=True)
        rotated_verts = rots[2].apply(rots[1].apply(rots[0].apply(_verts)))
        return rotated_verts

    for attempt_i in range(max_n_attempts):
        if attempt_i == 0:
            # on the first attempt, use the original verts, don't fuzz yet
            verts_for_lap_compute = verts_np
        else:
            verts_for_lap_compute = __fuzz_rot_verts(verts_np)

        # we've been using pytorch3d's cot laplacian which has (cot a + cot b)
        # weights rather than 1/2(cot a + cot b) (IGL's convention) so we'll do
        # this also for parity. this only affects the prefactored solves that
        # assume the 1/2(cot a + cot b) weights, i.e. only IGL ARAP rhs. All the
        # other solves are either agnostic to this factor since the rhs also
        # uses the same weights as the L defined here (my ARAP solves) or
        # assumes the (cot a + cot b) weights (i.e. NJF poisson). In any case,
        # we'll always make sure that solving using all-identity input matrices
        # will give back the same verts even without rescaling
        if USE_ACTUAL_IGL_COTMATRIX_WEIGHTS:
            L = -igl.cotmatrix(verts_for_lap_compute, faces_np)
        else:
            L = -2 * igl.cotmatrix(verts_for_lap_compute, faces_np)
        Lcoo = L.tocoo()

        # Lpin is what we use to init the solver, we'll chop off the rhs's index 0 upon solve
        Lpin = L[1:, 1:].tocoo() if pin_first_vertex else Lcoo
        Lpin_rowindices_np, Lpin_colindices_np = Lpin.coords
        Lpin_values = torch.from_numpy(Lpin.data).to(verts)
        Lpin_rowindices = torch.from_numpy(Lpin_rowindices_np).to(verts.device)
        Lpin_colindices = torch.from_numpy(Lpin_colindices_np).to(verts.device)
        n_Lpin = Lpin.shape[0]  # type: ignore

        try:
            cholespy_solver = cholespy.CholeskySolverF(
                n_Lpin,
                Lpin_rowindices,
                Lpin_colindices,
                Lpin_values,
                cholespy.MatrixType.COO,
            )
            if attempt_i > 0:
                print("[cholespy solver init] okay that worked")

            # the L returned should NOT be the one with the [1:] chop though!
            maybe_removed_first_L_column = (
                torch.from_numpy(L[:, :1].todense()).to(verts) if pin_first_vertex else None
            )
            L_rowindices_np, L_colindices_np = Lcoo.coords
            L_rowindices = torch.from_numpy(L_rowindices_np).to(verts.device)
            L_colindices = torch.from_numpy(L_colindices_np).to(verts.device)
            L_values = torch.from_numpy(L.data).to(verts)
            n_L = L.shape[0]  # type: ignore
            L = torch.sparse_coo_tensor(
                torch.stack((L_rowindices, L_colindices), dim=0), L_values, size=(n_L, n_L)
            )

            return L, cholespy_solver, maybe_removed_first_L_column
        except ValueError:
            # most likely failed with not-positive-definite error
            # continue the loop...
            print(
                f"[cholespy solver init] failed attempt {attempt_i + 1}, retrying by rotating the mesh and recomputing laplace operator"
            )
            pass
    # if code gets here, we've exhausted attempts, give up
    raise ValueError(
        f"after {max_n_attempts}, couldn't successfully initialize cholespy's cholesky solver for this mesh"
    )


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


@dataclass(slots=True)
class SparseLaplaciansSolvers:
    """
    Indexable dataclass holding, for a batch of meshes, a precomputed batched
    sparse Laplace operator, and cholespy solvers for those laplace operators,
    optionally the NJF poisson system right-hand side premultiplier matrix,
    and optionally the IGL ARAP right-hand side premultiplier matrix.
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
    (each corresponding to one square laplace matrix)
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
    >>> grad, _, face_doubleareas = sphuncs.calc_gradient_operator(verts, faces)
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
        verts_list: Sequence[torch.Tensor],
        faces_list: Sequence[torch.Tensor],
        *,
        pin_first_vertex: bool,
        compute_poisson_rhs_lefts: bool,
        compute_igl_arap_rhs_lefts: Optional[IGL_ARAPEnergyType],
    ):
        """
        given a batch of meshes (given as lists of pairwise-corresponding vertex arrays and
        face arrays), compute the Laplace operator and a cholespy solver object
        with that operator as the system matrix, and optionally
        - if `compute_poisson_rhs_lefts`: the NJF poisson system right-hand side
        premultiplier (see the docstrings of the fields of this dataclass for more info)
        - if `compute_igl_arap_rhs_lefts`: the IGL ARAP right-hand-side premultipliers
        - if `pin_first_vertex`, the laplacian will have one row and column chopped off,
            and any solves using this solver will adjust the rhs to this reduced system
            (with the first vertex of each mesh in the batch subbed in) accordingly
        """
        max_n_verts_per_mesh = int(max(map(len, verts_list)))
        max_n_faces_per_mesh = int(max(map(len, faces_list)))
        square_shape = (max_n_verts_per_mesh, max_n_verts_per_mesh)
        Ls = []
        cholespy_solvers = []
        poisson_rhs_lefts_per_mesh = [] if compute_poisson_rhs_lefts else None
        igl_arap_rhs_lefts_per_mesh = [] if compute_igl_arap_rhs_lefts else None
        removed_first_L_column_per_mesh = [] if pin_first_vertex else None
        for verts, faces in zip(verts_list, faces_list):
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


def recenter_to_centroid_and_rescale_new_verts_to_fit_old_bboxes(
    old_verts_list: Sequence[torch.Tensor], new_verts_list: Sequence[torch.Tensor]
) -> Sequence[torch.Tensor]:
    """
    this is to match the method from textdeformer, even though it's probably not what I'd
    immediately think of if asked to preserve old bboxes
        (If I were to do this, the scale factor would be based on max axis-aligned extent,
        rather than length of the diagonal between min coord and max coord points)
    """
    new_verts_recentered: List[torch.Tensor] = []
    for old_verts, new_verts in zip(old_verts_list, new_verts_list):
        # scale factor is the ratio between the bounding box diagonal lengths
        new_verts = new_verts - new_verts.mean(dim=0, keepdim=True)
        new_bbox_diag = new_verts.max(dim=0)[0] - new_verts.min(dim=0)[0]
        new_size = new_bbox_diag.norm()

        old_verts = old_verts - old_verts.mean(dim=0, keepdim=True)
        old_bbox_diag = old_verts.max(dim=0)[0] - old_verts.min(dim=0)[0]
        old_size = old_bbox_diag.norm()

        new_verts_recentered.append(new_verts * (old_size / new_size))

    return new_verts_recentered


def recenter_to_centroid(verts_list: Sequence[torch.Tensor]) -> Sequence[torch.Tensor]:
    return tuple(verts - verts.mean(dim=0, keepdim=True) for verts in verts_list)


def calc_ARAP_global_solve__with_igl_rhs_lefts(
    verts_list: Sequence[torch.Tensor],
    laplacians_solvers: SparseLaplaciansSolvers,
    per_vertex_rot_matrices_list: Sequence[torch.Tensor],
    postprocess: Optional[PostprocessAfterSolveName],
) -> Sequence[torch.Tensor]:
    """
    like calc_ARAP_global_solve but makes use of a precomputed igl_arap_rhs_lefts
    (rhs constructor to be multiplied with rotation matrices to form the rhs of
    the system)
    """
    solutions = []
    assert laplacians_solvers.igl_arap_rhs_lefts is not None
    # can we write a batched version of this without a loop? (though we'll still end up
    # looping through the cholespy solvers anyhow...)
    for i, (verts, arap_rhs_left, rot) in enumerate(
        zip(verts_list, laplacians_solvers.igl_arap_rhs_lefts, per_vertex_rot_matrices_list)
    ):
        rots_for_igl_rhs = rot.permute(2, 1, 0).reshape(-1, 1)
        # 2x due to differing cotan weight conventions btwn pt3d and igl for the solver init
        # (result is same up to rescaling; this ensures all-identity input matrices solve to no change)
        if USE_ACTUAL_IGL_COTMATRIX_WEIGHTS:
            rhs = arap_rhs_left.mm(rots_for_igl_rhs).reshape(3, -1).t()
        else:
            rhs = 2 * arap_rhs_left.mm(rots_for_igl_rhs).reshape(3, -1).t()

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

    # soln_verts_packed = torch.cat(solutions, dim=0)  # (n_verts_packed,3)
    if postprocess == "recenter_rescale":
        solutions = recenter_to_centroid_and_rescale_new_verts_to_fit_old_bboxes(
            verts_list, solutions
        )
    elif postprocess == "recenter_only":
        solutions = recenter_to_centroid(solutions)

    return solutions


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
    verts_list: Sequence[torch.Tensor],
    faces_list: Sequence[torch.Tensor],
    laplacians_solvers: SparseLaplaciansSolvers,
    per_vertex_rot_matrices_list: Sequence[torch.Tensor],
    arap_energy_type: ARAPEnergyTypeName,
    postprocess: Optional[PostprocessAfterSolveName],
) -> Sequence[torch.Tensor]:
    """
    - verts_list: list of vertex arrays, each of shape (n_verts_this_mesh, 3)
    - faces_list: list of face indices arrays, each of shape (n_faces_this_mesh, 3)
    - per_vertex_rot_matrices_list: list of per-vertex matrices, each of shape (n_verts_this_mesh, 3, 3)
    returns the ARAP global solve result, i.e. solution p' such that
    Lp' = b
    where L is the cotangent laplacian, and b is the right hand side derived from
    least-squares minimizing an ARAP energy
    """
    if laplacians_solvers.igl_arap_rhs_lefts is not None:
        assert (
            arap_energy_type == "spokes_and_rims_igl" or arap_energy_type == "spokes_igl"
        ), (
            "solver has IGL ARAP RHS constructors so must use spokes_and_rims_igl or spokes_igl for arap_energy_type"
        )
        # the rhs constructor from IGL is available, use this other function!
        # this goes with arap_energy_type == "spokes_igl" and "spokes_and_rims_igl"
        return calc_ARAP_global_solve__with_igl_rhs_lefts(
            verts_list,
            laplacians_solvers,
            per_vertex_rot_matrices_list,
            postprocess,
        )
    # the rest of this function here computes rhs directly from the 2004 paper formula
    # if arap_energy_type == spokes_mine, or the rhs from the spokes-and-rims energy
    # from Chao et al 2011 (also used in normal analogies; formula described in CGAL docs)
    # the rhs to find has shape (n_verts, 3)
    solutions = []
    # can we write a batched version of this without a loop? (though we'll still end up
    # looping through the cholespy solvers anyhow...)
    for i, (L, verts, faces, rots) in enumerate(
        zip(laplacians_solvers.Ls, verts_list, faces_list, per_vertex_rot_matrices_list)
    ):
        if (vp_sz0 := verts.size(0)) < (L_sz0 := L.size(0)):
            verts = torch.nn.functional.pad(verts, (0, 0, 0, L_sz0 - vp_sz0))

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

            # for each edge between a vertex i and vertex j, compute (w_ij / 2) * ((R_i
            # + R_j) @ (p_i - p_j)) (this is a 3d point)
            # Ri + Rj
            rot_vi_plus_rot_vj = rots[dir_edges_vi] + rots[dir_edges_vj]

            # pi - pj
            pi_minus_pj = verts[dir_edges_vi] - verts[dir_edges_vj]

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
                torch.zeros_like(verts),
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
            v0 = verts[faces_v0idx]
            v1 = verts[faces_v1idx]
            v2 = verts[faces_v2idx]
            r0 = rots[faces_v0idx]
            r1 = rots[faces_v1idx]
            r2 = rots[faces_v2idx]
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
                torch.zeros_like(verts), (v0v1v2idxs,), rhs_contribs, accumulate=True
            )
        else:
            raise AssertionError(
                f"shouldn't use calc_ARAP_global_solve with this arap_energy_type setting: {arap_energy_type} (if it's an igl arap energy type, maybe you forgot to init the solvers with igl_arap_rhs_lefts)"
            )

        # now do the solve
        solver = laplacians_solvers.cholespy_solvers[i]
        if _haspin := (laplacians_solvers.removed_first_L_column is not None):
            removed_first_L_column = laplacians_solvers.removed_first_L_column[i]
            # removed first L column has shape (n_verts, 1)
            # verts[None, 0] has shape (1, 3)
            # multiplied has shape (n_verts, 3)
            rhs = (rhs[:vp_sz0] - removed_first_L_column * verts[None, 0])[1:vp_sz0]
        else:
            rhs = rhs[:vp_sz0]

        soln = CholespySymmetricSolve_AutogradFn.apply(solver, rhs)
        assert isinstance(soln, torch.Tensor)
        # soln won't have any padding to trim because solver's system matrix is not padded
        if _haspin:
            soln = torch.cat((verts[None, 0], soln), dim=0)
        solutions.append(soln)

    if postprocess == "recenter_rescale":
        solutions = recenter_to_centroid_and_rescale_new_verts_to_fit_old_bboxes(
            verts_list, solutions
        )
    elif postprocess == "recenter_only":
        solutions = recenter_to_centroid(solutions)

    return solutions


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


def calc_poisson_global_solve(
    verts_list: Sequence[torch.Tensor],
    faces_list: Sequence[torch.Tensor],
    laplacians_solvers: SparseLaplaciansSolvers,
    per_face_matrices_list: Sequence[torch.Tensor],
    postprocess: Optional[PostprocessAfterSolveName],
) -> Sequence[torch.Tensor]:
    """
    - verts_list: list of vertex arrays, each of shape (n_verts_this_mesh, 3)
    - faces_list: list of face indices arrays, each of shape (n_faces_this_mesh, 3)
    - per_face_matrices_list: list of per-face matrices, each of shape (n_faces_this_mesh, 3, 3)
    the three should have the same length (which is the number of meshes in this batch)

    given per-face 3x3 transform matrices, treating them as per-face jacobians
    of a piecewise linear mapping, do a poisson solve to find the best-fitting
    per-vertex map. (as seen in Neural Jacobian Fields)
    """
    poisson_rhs_lefts = laplacians_solvers.poisson_rhs_lefts
    assert poisson_rhs_lefts is not None, (
        "must have poisson_rhs_lefts precomputed in `laplacians` struct to run poisson solve. make sure to set compute_poisson_rhs_lefts=True when calling SparseLaplaciansSolvers.from_meshes"
    )
    n_verts_per_face = 3
    n_coords = 3
    # faces_matrices_packed has shape (n_faces_packed, 3vertsperface, 3coords)

    # transpose to be (n_faces, 3coords, 3vertsperface)
    faces_matrices_packed = torch.cat(tuple(per_face_matrices_list), dim=0)
    faces_matrices_packed = faces_matrices_packed.transpose(-1, -2)

    # then pad it out to match (batch, max_n_faces, 3coords, 3vertsperface)
    num_faces_per_mesh = torch.tensor(
        [faces.size(0) for faces in faces_list], device=faces_list[0].device
    )
    batch_size = len(num_faces_per_mesh)
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
    if USE_ACTUAL_IGL_COTMATRIX_WEIGHTS:
        rhs = 0.5 * poisson_rhs_lefts.bmm(faces_matrices_padded)
    else:
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
    solutions = [
        (
            soln := cast(
                torch.Tensor,
                CholespySymmetricSolve_AutogradFn.apply(
                    solver,
                    (
                        rhs_single[: verts.size(0)]
                        - laplacians_solvers.removed_first_L_column[i] * verts[None, 0]
                    )[1 : verts.size(0)]
                    if (_haspin := (laplacians_solvers.removed_first_L_column is not None))
                    else rhs_single[: verts.size(0)],
                ),
            ),
            torch.cat((verts[None, 0], soln), dim=0) if _haspin else soln,
        )[-1]
        for i, (solver, rhs_single, verts) in enumerate(
            zip(laplacians_solvers.cholespy_solvers, rhs, verts_list)
        )
    ]

    if postprocess == "recenter_rescale":
        solutions = recenter_to_centroid_and_rescale_new_verts_to_fit_old_bboxes(
            verts_list, solutions
        )
    elif postprocess == "recenter_only":
        solutions = recenter_to_centroid(solutions)

    return solutions


### local step stuff


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
                    torch.nn.functional.pad(
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
        cls,
        verts_list: Sequence[torch.Tensor],
        faces_list: Sequence[torch.Tensor],
        quantity_defined_on: Literal["verts", "faces", "edges"],
    ):
        if quantity_defined_on == "verts":
            num_per_mesh = torch.tensor(tuple(map(len, verts_list)))
        elif quantity_defined_on == "faces":
            num_per_mesh = torch.tensor(tuple(map(len, faces_list)))
        elif quantity_defined_on == "edges":
            raise NotImplementedError("todo: edges meshespackedindexer without pt3d")
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


@dataclass(slots=True)
class ProcrustesPrecompute:
    padded_cell_edges_per_vertex_packed: torch.Tensor
    """
    (n_verts_packed, max_cell_neighborhood_n_edges, 2) int tensor; last dim contains edge vertex indices.
    negative ints are padding
    """
    # original_cell_edge_vecs_packed: torch.Tensor
    # """
    # (n_verts_packed, max_cell_neighborhood_n_edges, 3) float tensor
    # """
    # cell_laplacian_weights_packed: torch.Tensor
    # """
    # (n_verts_packed, max_cell_neighborhood_n_edges,) float tensor
    # """
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
    _verts_packed_idxr: MeshesPackedIndexer
    _num_verts_per_mesh: torch.Tensor
    _mesh_to_verts_packed_first_idx: torch.Tensor

    @classmethod
    def from_meshes(
        cls,
        local_step_procrustes_lambda: float,
        arap_energy_type: Optional[ARAPEnergyTypeName],
        laplacians_solvers: "SparseLaplaciansSolvers",
        verts_list: Sequence[torch.Tensor],
        faces_list: Sequence[torch.Tensor],
    ):
        """
        (need the laplacians solvers just for the laplacian weights)
        """
        print("Calculating procrustes solve precomputation")
        verts_packed = torch.cat(tuple(verts_list), dim=0)
        faces_packed = torch.cat(tuple(faces_list), dim=0)
        device = verts_packed.device

        n_verts_packed = len(verts_packed)
        pcepv_packed: Tuple[Set[Tuple[int, int]], ...] = tuple(
            set() for _ in range(n_verts_packed)
        )
        need_spokes_and_rims = (
            arap_energy_type == "spokes_and_rims_mine"
            or arap_energy_type == "spokes_and_rims_igl"
        )
        for v0i_, v1i_, v2i_ in faces_packed:
            v0i = int(v0i_.item())
            v1i = int(v1i_.item())
            v2i = int(v2i_.item())
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
            pcepv_packed_tuples, device=device
        )
        print("[procrustes precompute] done padded_cell_edges_per_vertex")
        ######################################## done computing padded_cell_edges_per_vertex
        num_verts_per_mesh = torch.tensor(tuple(map(len, verts_list)))
        mesh_to_verts_packed_first_idx = (
            torch.cumsum(num_verts_per_mesh, dim=0) - num_verts_per_mesh
        )
        cell_laplacian_weights_list = []
        for L, verts_packed_first_idx, n_verts in zip(
            laplacians_solvers.Ls,
            mesh_to_verts_packed_first_idx,
            num_verts_per_mesh,
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
            ).to(device)

            # wherever pcepv is negative, that's padding
            cell_laplacian_weights_this_mesh[pcepv_v1i_this_mesh < 0] = 0
            # ^ (n_verts, max_cell_neighborhood_n_edges,) float
            cell_laplacian_weights_list.append(cell_laplacian_weights_this_mesh)
        cell_laplacian_weights_packed = torch.cat(cell_laplacian_weights_list, dim=0)
        # (n_verts_packed, max_cell_neighborhood_n_edges,)
        print("[procrustes precompute] done cotangent weights")
        ######################################## done putting cotan weights into pcepv format

        voronoi_verts_massmatrix__scipy = igl.massmatrix(
            verts_packed.cpu().detach().numpy(),
            faces_packed.cpu().detach().numpy(),
        )
        # get diag of this thing
        voronoi_verts_mass_packed = (
            torch.from_numpy(voronoi_verts_massmatrix__scipy.diagonal()).float().to(device)
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
        print("[procrustes precompute] done diagonal matrix")
        ############################################### done making middle diag matrix

        # NOTE this bit of code is also how you compute the covar_rights matrix
        # for the in-progress edge vecs and target normals (target vert normals taking the place of original_vert_normals)

        pcepv_v1i = padded_cell_edges_per_vertex_packed[:, :, 1]
        pcepv_v0i = padded_cell_edges_per_vertex_packed[:, :, 0]
        original_cell_edge_vecs_packed = verts_packed[pcepv_v1i] - verts_packed[pcepv_v0i]
        # ^ (n_verts_packed, max_cell_neighborhood_n_edges, 3)
        # zero out wherever there is padding
        original_cell_edge_vecs_packed[pcepv_v1i < 0] = 0
        # original_vert_normals = patient_meshes.verts_normals_packed().unsqueeze(1)
        original_vert_normals = torch.from_numpy(
            igl.per_vertex_normals(
                verts_packed.cpu().detach().numpy(),
                faces_packed.cpu().detach().numpy(),
                igl.PER_VERTEX_NORMALS_WEIGHTING_TYPE_AREA,
            )
        ).to(verts_packed)
        # ^ (n_verts_packed, 1, 3)
        covar_lefts_lefts_packed = torch.cat(
            (original_cell_edge_vecs_packed, original_vert_normals), dim=1
        )
        # ^ (n_verts_packed, max_cell_neighborhood_n_edges + 1, 3)
        covar_lefts_packed = covar_lefts_lefts_packed.transpose(-1, -2).bmm(diagmats_packed)
        # ^ (n_verts_packed, 3, max_cell_neighborhood_n_edges + 1)
        print("[procrustes precompute] done covariance matrix")

        # make misc indexing bookkeeping
        _verts_packed_idxr = MeshesPackedIndexer.from_num_per_mesh(num_verts_per_mesh)
        return cls(
            padded_cell_edges_per_vertex_packed=padded_cell_edges_per_vertex_packed.to(
                device
            ),
            covar_lefts_packed=covar_lefts_packed,
            _verts_packed_idxr=_verts_packed_idxr,
            _num_verts_per_mesh=num_verts_per_mesh,
            _mesh_to_verts_packed_first_idx=mesh_to_verts_packed_first_idx,
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
    # if thlog.guard(VIZ_TRACE, needs_polyscope=True):
    #     pcepv_v1_for_vertex0 = pcepv_v1[0]
    #     pcepv_v1_for_vertex0 = pcepv_v1_for_vertex0[pcepv_v1i[0] >= 0]
    #     pcepv_v0_for_vertex0 = pcepv_v0[0]
    #     pcepv_v0_for_vertex0 = pcepv_v0_for_vertex0[pcepv_v0i[0] >= 0]
    #     cunet_pts = torch.cat((pcepv_v0_for_vertex0, pcepv_v1_for_vertex0), dim=0)
    #     cunet_edges = torch.stack(
    #         (
    #             torch.arange(len(pcepv_v0_for_vertex0)),
    #             len(pcepv_v0_for_vertex0) + torch.arange(len(pcepv_v0_for_vertex0)),
    #         ),
    #         dim=-1,
    #     )

    #     thlog.psr.register_curve_network(
    #         "v0 cell neigh",
    #         cunet_pts.cpu().detach().numpy(),
    #         cunet_edges.cpu().detach().numpy(),
    #     )
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
    uu, ss, vvt = torch.linalg.svd(covar)
    # vvt[:, [0, 1]] *= -1  # huh?? it does not work if i don't do this, however
    # not inplace:
    vvt = torch.stack((-vvt[:, 0], -vvt[:, 1], vvt[:, 2]), dim=1)
    rots_packed = vvt.transpose(-1, -2).bmm(uu.transpose(-1, -2))

    # svd det correction code from https://github.com/OllieBoyne/pytorch-arap/blob/master/pytorch_arap/arap.py
    # for any det(Ri) <= 0
    entries_to_flip = torch.nonzero(rots_packed.det() <= 0, as_tuple=False).flatten()
    # ^idxs where det(R) <= 0
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
