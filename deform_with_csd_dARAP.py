from typing import Optional, Callable, Tuple, Literal, Sequence, Union, List, Dict
from typing import cast, TYPE_CHECKING
from dataclasses import dataclass, field
import os

import numpy as np
import torch
import pytorch3d.structures
import igl

import nvdiffrast.torch as dr

import nvdiffmodeling.src.mesh as nvdiffmodeling_mesh
import nvdiffmodeling.src.render as nvdiffmodeling_render
import nvdiffmodeling.src.texture as nvdiffmodeling_texture

import resize_right
import td_camera
import csd
import deformations_dARAP
import vectoradam

if TYPE_CHECKING:
    from NeuralJacobianFields.SourceMesh import SourceMesh as NJFSourceMesh
else:
    try:
        from NeuralJacobianFields.SourceMesh import SourceMesh as NJFSourceMesh
    except ImportError as e:
        NJFSourceMesh = None

from thad import parse_lr_schedule_string_into_lr_lambda, next_increment_path, expect
from thronf import Thronfig, thronfigure, InvalidConfigError
from thlog import Thlogger, LOG_INFO, VIZ_INFO, PSRSpecialArray

thlog = Thlogger(
    LOG_INFO,
    VIZ_INFO,
    "deformbycsd3",
    imports=[deformations_dARAP.thlog, csd.thlog],
)


@dataclass(slots=True)
class MeshesDatasetAsFolder_IOSettings(Thronfig):
    """not implemented yet, use MeshesDatasetAsList"""

    path: str
    prompts_file: str


@dataclass(slots=True)
class MeshesDatasetAsList_IOSettings(Thronfig):
    fnames: Sequence[str]
    """
    a list of source .obj filenames which must all exist
    """
    prompts: Sequence[str]
    prompts_negative: Sequence[Optional[str]]

    def __post_typecheck__(self):
        if not (
            (n_fnames := len(self.fnames)) == len(self.prompts)
            and n_fnames == len(self.prompts_negative)
        ):
            raise InvalidConfigError(
                "in dataset_cfg.lists, the lists fnames, prompts, prompts_negative lists must all have the same number of elements"
            )


@dataclass(slots=True)
class MeshesDataset_Settings(Thronfig):
    """
    at least one of folder or lists must be specified.
    """

    folder: Optional[MeshesDatasetAsFolder_IOSettings] = None
    lists: Optional[MeshesDatasetAsList_IOSettings] = None

    def __post_typecheck__(self):
        if self.folder is None and self.lists is None:
            raise InvalidConfigError(
                "one of folder or lists must be non-None in meshes dataset io config"
            )

    def get_dataset_size(self) -> int:
        if self.lists is not None:
            # self.lists's post typecheck already validated it to have the same number of
            # fnames, prompts
            return len(self.lists.fnames)
        elif self.folder is not None:
            raise NotImplementedError("TODO implement folder dataset config")
        else:
            return 0


@dataclass(slots=True)
class DeformByCSD_Settings(Thronfig):
    optimize_deform_via: deformations_dARAP.DeformOptimQuantityName
    """
    - verts_offsets: only optimize the vertex offsets, no other solve. simplest.
    - faces_normals: optimize normals to compute axis-angle rot matrices (taking
        face normals from curr normals to the optimized normals) and then
        solving into vertices based on the solve_method (using a face-based solve_method)
    - faces_jacobians: optimize per-face transform matrices, and then solving
        into vertices based on the solve_method (using a face-based solve_method)
    - verts_normals: optimize normals, compute neighborhood rot matrices by Procrustes
        local step then solving into vertices based on the solve_method
        (using a vert-based solve_method)
    - verts_3x2rotations: optimize per-vertex 3x2 continuous rotation representation mats,
        to be converted with a soft-diagonalization into 3x3 rotations to be solved into 
        vertices based on the solve_method (only works for a vert-based solve_method)
    - faces_3x2rotations: ditto but per-face (for face-based solve_method)
    """
    solve_method: deformations_dARAP.DeformSolveMethodName
    """
    Ignored when optimize_deform_via == "verts_offsets". Otherwise, is used to
    solve per-element transforms into vertex offsets.
    """
    arap_energy_type: Optional[deformations_dARAP.ARAPEnergyTypeName]
    """
    if solve_method has 'arap' in it, then this must be non-None, and vice versa
    """
    local_step_procrustes: Optional[deformations_dARAP.Procrustes_Settings]
    """
    if not None, use procrustes solve. only applicable for verts_normals optimize_deform_via
    Since we're always using one step of procrustes (i.e. always starts from initial verts)
    this can probably be a bit bigger than 1.0.
    """
    pin_first_vertex: bool
    """
    whether to pin the first vertex (e.g. remove the corresponding laplacian row
    and column, and adjust right-hand sides accordingly) when initializing the
    sparse laplacian solver. Makes the cholespy solver init succeed more consistently
    """

    n_iters: int
    lr: Union[float, str]
    """
    use a float for a fixed LR; use a string for a LR schedule spec string
    """

    optimizer_type: deformations_dARAP.OptimizerTypeName
    """ Adam or VectorAdam. I basically default to Adam """

    n_accum_iters: int
    """
    n of iters of SDS/CSD gradients to accumulate and do backward on per epoch.
    """
    step_after_every_backward: bool
    """
    whether to optimizer.step() after each backward() in every accum iter. 
    If False, the optimizer.step() is done only once after the accum loop, i.e. after all
    backward()s.
    """
    optimized_quantity_save_fname: str
    """
    save fname for the optimized quantity (what gets saved depends on optimize_deform_via)
    """

    view_once_every: int
    """
    how often to log to the ps recording during optimization
    """

    mesh_batch_size: int
    """
    number of patient meshes to be used in a batch
    """

    view_batch_size: int
    """
    For each of the patient meshes to be used in a batch, the number of views to render each
    one. This means the final effective batch size that goes in an iteration will be
    (mesh_batch_size * view_batch_size), where each batch istem is a pair (mesh, view).
    """

    adapt_dists: bool
    """
    if True, will apply distance multipliers which are the meshes'
    origin-centered bounding boxes' max-magnitude coord values
    """

    cams_and_lights: td_camera.CamsAndLights_Settings
    background_color: Tuple[float, float, float]

    resize_for_guidance: Optional[Tuple[int, int, resize_right.InterpMethodName]]
    """
    if not None, resize the raw render image to the desired size and with the
    specified kernel before feeding it to the CSD/SDS guidance
    """

    stage_I_weight: float
    """
    should just be 1.0
    """

    stage_II_weight_schedule: str
    """
    a ramp schedule description string (see
    parse_lr_schedule_string_into_lr_lambda in thad.py); if not None, overrides
    stage_II_weight and uses the function described in the schedule string.
    For a constant value simply write the float value in that string, e.g. "0.0"
    """

    visual_loss_weight_schedule: str
    jacobian_id_loss_weight_schedule: str
    """
    a ramp schedule description string for the weight of the jacobian identity
    regularization loss. Works for all optimize_deform_via not just faces_jacobians
    """
    guidance: csd.GuidanceConfig = field(default_factory=csd.GuidanceConfig)
    """
    defaults for CSD guidance
    """
    start_from_epoch: int = 1
    """
    this is 1-indexed, 1 is the first epoch.
    Practically useful for starting the optimization at a certain epoch with respect to the
    loss weight schedule functions
    """
    torch_seed: Optional[int] = None
    numpy_seed: Optional[int] = None

    save_at_epochs: Sequence[int] = ()
    """
    save results at the specified epochs
    """

    def __post_typecheck__(self):
        # arap_energy_type
        has_arap_energy_type = self.arap_energy_type is not None
        solvemethod_is_arap = self.solve_method == "arap"
        if has_arap_energy_type or solvemethod_is_arap:
            if not (has_arap_energy_type and solvemethod_is_arap):
                raise InvalidConfigError(
                    "if deform_by_csd.arap_energy_type is specified, then solve_method must be 'arap' and vice versa"
                )

        if self.local_step_procrustes is not None:
            if self.optimize_deform_via != "verts_normals":
                raise InvalidConfigError(
                    "can only use procrustes lambda with verts_normals optimize_deform_via"
                )


@dataclass(slots=True)
class MainConfig(Thronfig):
    dataset: MeshesDataset_Settings
    deform_by_csd: DeformByCSD_Settings
    ps_recording_save_fname: str
    device: Literal["cuda", "cpu"] = "cuda"

    def __post_typecheck__(self):
        if self.dataset.get_dataset_size() > 1:
            raise InvalidConfigError(
                """
    when optimizing multiple mesh deformations via per-element normals/rotations/jacobians
    for each mesh, and not via a shared deformer module/geocap to be trained on a dataset,
    it's usually faster and more practical to just launch runs in parallel rather than
    trying to do one run with a mesh dataset of more than one shape, since the optimizations
    of each mesh don't have anything shared between each other.
            """
            )


def make_optimizer(
    quantity_being_optimized: deformations_dARAP.QuantityBeingOptimized,
    main_init_lr: float,
    optimizer_type: deformations_dARAP.OptimizerTypeName,
) -> deformations_dARAP.torch.optim.Optimizer:
    if optimizer_type == "Adam":
        paramgroups = (
            {
                "params": (quantity_being_optimized.tensor,),
                "lr": main_init_lr,
                "this_is": "main",
            },
        )
        return torch.optim.Adam(paramgroups)
    elif optimizer_type == "VectorAdam":
        # figure out the vectoradam axis and view_before_axis for the main tensor
        if quantity_being_optimized.this_is in ("faces_normals", "verts_normals"):
            view_before_axis = None
            axis = -1
        else:
            view_before_axis = None
            axis = None
        # other shared params should not need any special axis handling in vectoradam... but
        # this might change (very highly likely NOT though)
        paramgroups = (
            {
                "params": (quantity_being_optimized.tensor,),
                "lr": main_init_lr,
                "axis": axis,
                "view_before_axis": view_before_axis,
                "this_is": "main",
            },
        )
        return vectoradam.VectorAdam(paramgroups)
    else:
        raise InvalidConfigError("unknown optimizer type, use Adam or VectorAdam")


def set_learning_rate(
    optimizer: deformations_dARAP.torch.optim.Optimizer,
    main_lr: float,
    other_shared_params_lr: float,
):
    for param_group in optimizer.param_groups:
        if (this_is := param_group["this_is"]) == "main":
            param_group["lr"] = main_lr
        elif this_is == "other":
            param_group["lr"] = other_shared_params_lr
        else:
            raise ValueError("unknown this_is value in param")


@dataclass(slots=True)
class BaseMeshGoingIntoOptimLoop:
    """
    all the information pertaining to a single patient mesh (equivalent to a
    single .obj file) to be operated upon.
    """

    nvdm_loaded_mesh: nvdiffmodeling_mesh.Mesh
    njf_solver: Optional[NJFSourceMesh]  # only present when solve_method == "njfpoisson"
    original_size: Optional[torch.Tensor]  # only present when solve_method == "njfpoisson"
    prompt: str
    prompt_negative: Optional[str]
    prompt_z: torch.Tensor
    """ embedding of prompt, of shape (1, 77, 4096) """
    prompt_negative_z: torch.Tensor
    """ embedding of negative prompt, of shape (1, 77, 4096) """


"""
these three are from meshfusion/meshup/td code.
The vertices usable here are NOT meant to be the verts_packed arrays from
batched pytorch3d meshes. Only for vertices arrays belonging to a single mesh.

These are only used for the njfpoisson method; in other solve methods, I have my
own normalization (does the same thing) built into arap/poisson solve functions
"""


def calc_bounding_box(vertices: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    min_coords = torch.min(vertices, dim=0)[0]
    max_coords = torch.max(vertices, dim=0)[0]
    return min_coords, max_coords


def calc_size(min_coords: torch.Tensor, max_coords: torch.Tensor) -> torch.Tensor:
    return torch.norm(max_coords - min_coords)


def normalize_vertices_singlemesh(
    vertices: torch.Tensor, original_size: torch.Tensor
) -> torch.Tensor:
    """
    The vertices usable here are NOT meant to be the verts_packed arrays from
    batched pytorch3d meshes. Only for vertices arrays belonging to a single mesh.
    """
    min_coords, max_coords = calc_bounding_box(vertices)
    current_size = calc_size(min_coords, max_coords)
    scale_factor = original_size / current_size
    return vertices * scale_factor


"""
end three normalization functions from meshfusion/mees
"""


def calc_deformed_verts_according_to_cfg(
    meshes_structs_if_usenjfsolver: Optional[Sequence[BaseMeshGoingIntoOptimLoop]],
    pt3d_batched_meshes: pytorch3d.structures.Meshes,
    solve_method: deformations_dARAP.DeformSolveMethodName,
    arap_energy_type: Optional[deformations_dARAP.ARAPEnergyTypeName],
    maybe_my_solver: Optional[deformations_dARAP.SparseLaplaciansSolvers],
    quantity_being_optimized: deformations_dARAP.QuantityBeingOptimized,
    this_iter_needs_viz: bool,
) -> Tuple[Sequence[torch.Tensor], deformations_dARAP.DeformationIntermediateResults]:
    """
    wraps deformations_dARAP.calc_deformed_verts_solution_according_to_cfg to
    support solve_method == njfpoisson, and handles the input-to-deformation-solve-calculating, but
    otherwise the exact same function

    returns
    - a list of solution verts tensors, each corresponding to a struct in meshes_structs
    - intermediate results, present or None depending on the quantity_being_optimized, in
        case we wish to penalize or view some intermediate result involved in a deform method

    pt3d_batched_meshes must be a pytorch3d Meshes batch with the same number of
    meshes as len(meshes_structs), and each mesh in pt3d_batched_meshes must
    match the vertex (v_pos) and face (t_pos_idx) array of the corresp. mesh struct's nvdm_loaded_mesh
    """
    inputs_to_deformation_solve_methods = (
        deformations_dARAP.calc_inputs_to_solve_for_deformation_according_to_cfg(
            pt3d_batched_meshes, quantity_being_optimized
        )
    )

    if solve_method == "njfpoisson":
        # we're only splitting this here because we don't want to import NJF
        # in deformations_dARAP, actually..
        assert inputs_to_deformation_solve_methods.face_matrices_packed is not None
        # turn packed (sum #F of all meshes in batch, 3, 3) into a list where each item is
        # of shape (#F of mesh i, 3, 3) corresponding to mesh i in the batch
        face_matrices_list = deformations_dARAP.per_face_packed_to_list(
            pt3d_batched_meshes, inputs_to_deformation_solve_methods.face_matrices_packed
        )
        soln_verts_list = []
        assert meshes_structs_if_usenjfsolver is not None, (
            "meshes_structs list needed if solve_method == 'njfpoisson'"
        )
        for face_matrices, mesh_struct in zip(
            face_matrices_list, meshes_structs_if_usenjfsolver
        ):
            assert mesh_struct.njf_solver is not None
            assert mesh_struct.original_size is not None

            soln_verts = mesh_struct.njf_solver.vertices_from_jacobians(
                face_matrices.unsqueeze(0)
            ).squeeze()
            soln_verts = normalize_vertices_singlemesh(
                soln_verts, mesh_struct.original_size
            )
            soln_verts_list.append(soln_verts)
        return soln_verts_list, inputs_to_deformation_solve_methods.intermediate_results
    else:
        # this will handle the rest of the solve_methods
        return deformations_dARAP.calc_deformed_verts_solution_according_to_cfg(
            pt3d_batched_meshes,
            solve_method,
            arap_energy_type,
            maybe_my_solver,
            inputs_to_deformation_solve_methods,
        )


def prep_nvdm_mesh_with_trivial_gray_tex(
    verts: torch.Tensor, faces: torch.Tensor
) -> nvdiffmodeling_mesh.Mesh:
    """
    create trivial UVs and init a nvdiffmodeling mesh structure for rendering
    """
    # NOTE these are trivial degenerate UVs pointing to all (0,0), which is fine since we
    # just need to color everything gray for our use.
    # If we need to learn the texture map, then these should be actual good UVs manually
    # loaded from an obj file, or from some param method.
    mega_trivial_vertex_uvs = torch.zeros(
        (verts.shape[0], 2), dtype=torch.float32, device="cuda"
    )
    mega_trivial_t_tex_idx = faces.clone().cuda()

    grayscale_color = 0.5

    # technically trainable but actually we don't update these
    texture_map = nvdiffmodeling_texture.create_trainable(
        np.full((512, 512, 3), grayscale_color, np.float32), [512] * 2, True
    )
    normal_map = nvdiffmodeling_texture.create_trainable(
        np.array([0, 0, 1]), [512] * 2, True
    )
    specular_map = nvdiffmodeling_texture.create_trainable(
        np.array([0, 0, 0]), [512] * 2, True
    )

    material = {
        "bsdf": "diffuse",
        "kd": texture_map,
        "ks": specular_map,
        "normal": normal_map,
    }
    return nvdiffmodeling_mesh.Mesh(
        v_pos=verts.float().cuda(),
        t_pos_idx=faces.cuda(),
        material=material,
        v_tex=mega_trivial_vertex_uvs,
        t_tex_idx=mega_trivial_t_tex_idx,
    )


def load_meshes_from_dataset_cfg_and_encode_prompts(
    dataset_cfg: MeshesDataset_Settings,
    solve_method: deformations_dARAP.DeformSolveMethodName,
    prompt_encoding_fn: Callable[[str, Optional[str]], Tuple[torch.Tensor, torch.Tensor]],
    device: torch.device,
) -> Tuple[Sequence[BaseMeshGoingIntoOptimLoop], pytorch3d.structures.Meshes]:
    """
    loads the dataset as a list of wrapper structs representing patient meshes
    to be operated upon, as well as those same meshes but incorporated into a
    pytorch3d.structures.Meshes batched meshes object for easy packed operations

    optionally (if shape_prep_cfg is present) apply some preprocessing, namely
    aligning shapes to principal axes plus an extra rotation or matrix for our conventions

    because the per-mesh structs also contain prompts and prompt embeddings,
    we also need a prompt_encoding_fn. In our case that's the `encode_prompt`
    method of the CSD class from `csd`.
    """
    if dataset_cfg.folder is not None:
        raise NotImplementedError(
            "TODO implement read directory of meshes and json prompt specification file"
        )
    if dataset_cfg.lists is None:
        raise InvalidConfigError(
            "only dataset_cfg.lists supported for now, so it must be present"
        )
    fnames = dataset_cfg.lists.fnames
    prompts = dataset_cfg.lists.prompts
    prompts_negative = dataset_cfg.lists.prompts_negative
    mesh_structs = []
    # we'll use a pytorch3d meshes batch to get a packed form of the quantity to optimize
    pt3d_verts_list = []
    pt3d_faces_list = []
    for fname, prompt, prompt_negative in zip(fnames, prompts, prompts_negative):
        loaded_verts_np, loaded_faces_np = igl.read_triangle_mesh(fname)
        nvdm_loaded_mesh = prep_nvdm_mesh_with_trivial_gray_tex(
            torch.from_numpy(loaded_verts_np).to(device),
            torch.from_numpy(loaded_faces_np).to(device),
        )

        # then normalize (this is actually the same normalization method as our
        # sphuncs.normalize_to_side2_cube_inplace, i.e. side-2 cube centered at origin)
        nvdm_loaded_mesh = nvdiffmodeling_mesh.unit_size(nvdm_loaded_mesh)

        assert isinstance(nvdm_loaded_mesh.v_pos, torch.Tensor)
        assert isinstance(nvdm_loaded_mesh.t_pos_idx, torch.Tensor)

        # recenter to centroid (doesn't actually matter since the solve result is what gets
        # rendered, never this source mesh, and the solve result has its own preproc config)
        nvdm_loaded_mesh.v_pos -= nvdm_loaded_mesh.v_pos.mean(dim=0)

        if solve_method == "njfpoisson":
            # poisson system solver (which will precompute laplacian and its decomp and rhs premultiplier)
            assert NJFSourceMesh is not None, (
                "NJFSourceMesh did not import successfully, solve_method njfpoisson not available"
            )
            njf_solver = NJFSourceMesh(0, None, {}, 1, ttype=torch.float)
            njf_solver.load(
                source_v=nvdm_loaded_mesh.v_pos.cpu().numpy(),
                source_f=nvdm_loaded_mesh.t_pos_idx.cpu().numpy(),
            )
            njf_solver.to(device)

            # original bounding box, for normalizing poisson solve results
            original_min, original_max = calc_bounding_box(nvdm_loaded_mesh.v_pos)
            original_size = calc_size(original_min, original_max)
        else:
            # my arap/poisson functions come with the normalization built in (see deformations_dARAP)
            njf_solver = None
            original_size = None

        # encode prompts
        prompt_z, prompt_negative_z = prompt_encoding_fn(prompt, prompt_negative)

        mesh_struct = BaseMeshGoingIntoOptimLoop(
            nvdm_loaded_mesh=nvdm_loaded_mesh,
            njf_solver=njf_solver,
            original_size=original_size,
            prompt=prompt,
            prompt_negative=prompt_negative,
            prompt_z=prompt_z,
            prompt_negative_z=prompt_negative_z,
        )
        mesh_structs.append(mesh_struct)
        pt3d_verts_list.append(nvdm_loaded_mesh.v_pos)
        pt3d_faces_list.append(nvdm_loaded_mesh.t_pos_idx)

    pt3d_batched_meshes = pytorch3d.structures.Meshes(
        verts=pt3d_verts_list, faces=pt3d_faces_list
    ).to(device)
    return mesh_structs, pt3d_batched_meshes


def get_batch_of_cameras_and_lights_with_dist_adapt(
    cams_and_lights_cfg: td_camera.CamsAndLights_Settings,
    render_device: torch.device,
    view_batch_size: int,
    adapt_dists: bool,
    verts_to_adapt_dists_to: torch.Tensor,
) -> Dict[str, torch.Tensor]:
    """
    extends td_camera.get_batch_of_cameras_and_lights with adaptation to mesh extents maybe
    """
    # adaptive distance scaling based on deformed mesh extents
    if adapt_dists:
        with torch.no_grad():
            v_pos = verts_to_adapt_dists_to
            vmin = v_pos.amin(dim=0)
            vmax = v_pos.amax(dim=0)
            v_pos = v_pos - (vmin + vmax) / 2
            adapt_dists_mult = (
                torch.cat([v_pos.amin(dim=0), v_pos.amax(dim=0)]).abs().amax().item()
            )
    else:
        adapt_dists_mult = 1.0

    # make batch of random camera parameters
    cams_and_lights_batch = td_camera.get_batch_of_cameras_and_lights(
        cams_and_lights_cfg,
        view_batch_size,
        dist_multiplier=adapt_dists_mult,
    )
    for key in cams_and_lights_batch:
        cams_and_lights_batch[key] = cams_and_lights_batch[key].to(render_device)
    return cams_and_lights_batch


def render_nvdm_mesh_with_new_verts_and_view_batch(
    glctx: Union[dr.RasterizeGLContext, dr.RasterizeCudaContext],
    cams_and_lights_cfg: td_camera.CamsAndLights_Settings,
    cams_and_lights_batch: Dict[str, torch.Tensor],
    background: torch.Tensor,
    nvdm_loaded_mesh: nvdiffmodeling_mesh.Mesh,
    new_verts: torch.Tensor,
) -> torch.Tensor:
    """
    renders a single mesh given as `nvdm_loaded_mesh`, except using `new_verts`
    rather than its verts.

    cams_and_lights_batch is from td_camera.get_batch_of_cameras_and_lights
    OR this script's get_batch_of_cameras_and_lights_with_dist_adapt

    background should be shape (3,) rgb float in range [0,1]

    returns a batch of images of shape `(view_batch_size, channels=3, h, w)`
    where h, w are equal to cams_and_lights_cfg.raster_res
    """
    nvdm_render_mesh = nvdiffmodeling_mesh.Mesh(
        new_verts,  # override only verts,
        base=nvdm_loaded_mesh,  # get everything else from nvdm_loaded_mesh
    )
    # NOTE meshfusion/td code "combines" nvdm_render_mesh into a 1-mesh scene here

    nvdm_render_mesh = nvdiffmodeling_mesh.auto_normals(nvdm_render_mesh)
    nvdm_render_mesh = nvdiffmodeling_mesh.compute_tangents(nvdm_render_mesh)
    # these functions return a lazy chain of computations on the mesh
    # which we have to eval() in order to get back a concrete nvdm Mesh struct.
    # in our case, the eval will feed the chain of computations with the camera
    # parameters

    # eval the lazy chain of computations queued on nvdm_render_mesh
    # to get back a concrete nvdm Mesh struct
    nvdm_render_mesh = nvdm_render_mesh.eval(cams_and_lights_batch)
    train_render = nvdiffmodeling_render.render_mesh(
        glctx,
        nvdm_render_mesh,
        cams_and_lights_batch["mvp"],
        cams_and_lights_batch["campos"],
        cams_and_lights_batch["lightpos"],
        cams_and_lights_cfg.light_power,
        (raster_res := cams_and_lights_cfg.raster_res),
        spp=1,
        num_layers=1,
        msaa=False,
        background=torch.broadcast_to(background, (1, raster_res, raster_res, 3)),
    )
    # ^ (view_batch_size, h, w, channels)
    assert isinstance(train_render, torch.Tensor)
    train_render = train_render.permute(0, 3, 1, 2)
    # ^ (view_batch_size, channels, h, w)
    return train_render


def save_optimized_matrices_or_normals(
    deform_by_csd_cfg: DeformByCSD_Settings,
    dataset_cfg: MeshesDataset_Settings,
    patient_mesh: pytorch3d.structures.Meshes,
    per_elem_mats_or_normals: torch.Tensor,
    save_fname: str,
) -> str:
    """
    saves the source mesh and optimized deformation quantity for 1 mesh
    (i.e. len(patient_mesh) == 1)
    (can deal with patient_mesh of more than 1 batch size too, but will fuse
    all meshes together in the same packed arrays)
    """
    if os.path.isfile(save_fname):
        fname_noext, ext = os.path.splitext(save_fname)
        save_fname = next_increment_path(fname_noext + "-({})" + ext)
    quantity_save_dict = {}
    if deform_by_csd_cfg.optimize_deform_via == "faces_jacobians":
        quantity_save_dict["faces_jacobians"] = (
            per_elem_mats_or_normals.cpu().detach().numpy()
        )
    elif deform_by_csd_cfg.optimize_deform_via == "faces_3x2rotations":
        quantity_save_dict["faces_3x2rotations"] = (
            per_elem_mats_or_normals.cpu().detach().numpy()
        )
    elif deform_by_csd_cfg.optimize_deform_via == "verts_3x2rotations":
        quantity_save_dict["verts_3x2rotations"] = (
            per_elem_mats_or_normals.cpu().detach().numpy()
        )
    elif deform_by_csd_cfg.optimize_deform_via == "faces_normals":
        # due to double backward shenanigans, we actually optimize an offset to
        # add to faces_normals rather than faces_normals itself.
        quantity_save_dict["faces_normals_offset"] = (
            per_elem_mats_or_normals.cpu().detach().numpy()
        )
    elif deform_by_csd_cfg.optimize_deform_via == "verts_normals":
        quantity_save_dict["verts_normals"] = (
            per_elem_mats_or_normals.cpu().detach().numpy()
        )
    else:
        raise ValueError(
            "cannot save optimized results for other deform_by_csd.optimize_deform_via using this function"
        )
    np.savez_compressed(
        save_fname,
        verts=patient_mesh.verts_packed().cpu().detach().numpy(),
        faces=patient_mesh.faces_packed().cpu().detach().numpy(),
        deform_by_csd_cfg=np.array(deform_by_csd_cfg.to_json_string()),
        dataset_cfg=np.array(dataset_cfg.to_json_string()),
        **quantity_save_dict,
    )
    return save_fname


def calc_jacobian_id_loss(
    optim_quantity_this_batch: deformations_dARAP.QuantityBeingOptimized,
    intermediate_results: deformations_dARAP.DeformationIntermediateResults,
) -> Tuple[torch.Tensor, float]:
    """
    compute the jacobian identity regularization loss. returns the loss tensor
    and the loss float value.

    `optim_quantity_this_batch.this_is == "faces_jacobians" or "faces_3x2rotations" or "verts_3x2rotations"` required!
    new: "faces_normals" and "verts_normals" also allowed but need intermediate_results.rot_matrices
    """
    device = optim_quantity_this_batch.tensor.device
    if (
        optim_quantity_this_batch.this_is == "faces_normals"
        or optim_quantity_this_batch.this_is == "verts_normals"
    ):
        assert isinstance(
            intermediate_results, deformations_dARAP.ElemNormals_IntermediateResults
        )
        mat3x3 = intermediate_results.rot_matrices
    elif optim_quantity_this_batch.tensor.size(-1) == 2:
        mat3x3 = deformations_dARAP.convert_rot3x2_to_rot3x3(
            optim_quantity_this_batch.tensor
        )
    else:
        mat3x3 = optim_quantity_this_batch.tensor
    jac_id_loss = (mat3x3 - torch.eye(3, device=device)).pow(2).mean()
    jac_id_loss_val = jac_id_loss.item()
    return jac_id_loss, jac_id_loss_val


def submain_deform_meshes_by_csd(
    deform_by_csd_cfg: DeformByCSD_Settings,
    dataset_cfg: MeshesDataset_Settings,
    device: torch.device,
):
    #### set seed
    deformations_dARAP.seed_all(deform_by_csd_cfg.torch_seed, deform_by_csd_cfg.numpy_seed)

    #### prep CSD modules
    if os.environ.get("CSD_DUMMY"):
        stage_I = csd.DummyCSDClass()
    else:
        stage_I = csd.CSD(deform_by_csd_cfg.guidance, stage=1)
    # stage 2 is loaded only if we're not doing a dummy run AND if the stage 2
    # weight schedule is not a constant zero
    if (not isinstance(stage_I, csd.DummyCSDClass)) and (
        not (
            (__csdsched := deform_by_csd_cfg.stage_II_weight_schedule) == "0.0"
            or __csdsched == "0"
        )
    ):
        stage_II = csd.CSD(deform_by_csd_cfg.guidance, stage=2)
    else:
        stage_II = None

    #### gl or cuda context
    try:
        glctx = dr.RasterizeGLContext()
    except RuntimeError:
        glctx = dr.RasterizeCudaContext()

    #### load meshes and structs and nvdiffmodeling structs
    meshes_structs, pt3d_batched_meshes = load_meshes_from_dataset_cfg_and_encode_prompts(
        dataset_cfg,
        deform_by_csd_cfg.solve_method,
        stage_I.encode_prompt,
        device,
    )
    n_meshes = len(meshes_structs)

    # pt3d_batched_meshes contains the same exact meshes (verts, faces) as the
    # ones in meshes_structs but allows convenient access to this whole
    # dataset's packed arrays (verts_packed (sum n verts from all meshes, 3),
    # faces_packed (sum n faces from all meshes, 3) and so on).

    #### my sparse laplacians solver
    # alongside njf_solver (SourceMesh from NJF code) which is stored per mesh
    # in its respective mesh_struct, I have my own solver which operates on
    # pytorch3d batched meshes so only one instance has to be made (and not one
    # for every item in meshes_structs and stored within it). Only needed for
    # non-njfpoisson solve methods
    solve_method = deform_by_csd_cfg.solve_method
    if solve_method == "poisson" or solve_method == "arap":
        maybe_my_solver = deformations_dARAP.SparseLaplaciansSolvers.from_meshes(
            pt3d_batched_meshes,
            pin_first_vertex=deform_by_csd_cfg.pin_first_vertex,
            compute_poisson_rhs_lefts=(deform_by_csd_cfg.solve_method == "poisson"),
            compute_igl_arap_rhs_lefts=deformations_dARAP.get_igl_arap_energy_type_from_cfg(
                deform_by_csd_cfg.arap_energy_type
            ),
        )
    else:
        maybe_my_solver = None

    #### initialize the quantity to be optimized
    quantity_being_optimized = deformations_dARAP.QuantityBeingOptimized.init_according_to_cfg(
        pt3d_batched_meshes,
        deform_by_csd_cfg.optimize_deform_via,
        (
            deform_by_csd_cfg.local_step_procrustes,
            expect(
                maybe_my_solver,
                TypeError(
                    "must use my SparseLaplaciansSolver for procrustes init, cannot be using njfpoisson and the NJF solver in the mesh structs"
                ),
            ),
            deform_by_csd_cfg.arap_energy_type,
        )
        if deform_by_csd_cfg.local_step_procrustes
        else None,
    )

    # a single lr description string or float
    main_lr_fn = parse_lr_schedule_string_into_lr_lambda(str(deform_by_csd_cfg.lr))
    other_shared_params_lr_fn = main_lr_fn
    optimizer = make_optimizer(
        quantity_being_optimized,
        main_lr_fn(1),
        deform_by_csd_cfg.optimizer_type,
    )

    #### background
    background = torch.tensor(deform_by_csd_cfg.background_color).to(device)

    #### function to resize images before putting them thru visual loss
    resize_for_guidance_fn: Callable[[torch.Tensor], torch.Tensor]
    if (resize_for_guidance := deform_by_csd_cfg.resize_for_guidance) is not None:
        resize_h, resize_w, resize_interp_method_name = resize_for_guidance
        resize_interp_fn = resize_right.get_interp_method(resize_interp_method_name)
        resize_for_guidance_fn = lambda img: resize_right.resize(
            img, out_shape=(resize_h, resize_w), interp_method=resize_interp_fn
        )
    else:
        resize_for_guidance_fn = lambda img: img

    #### parse schedule strings and get functions taking epoch and outputting weight/lr
    visual_loss_weight_fn = parse_lr_schedule_string_into_lr_lambda(
        deform_by_csd_cfg.visual_loss_weight_schedule
    )
    jacobian_id_loss_weight_fn = parse_lr_schedule_string_into_lr_lambda(
        deform_by_csd_cfg.jacobian_id_loss_weight_schedule
    )
    stage_II_loss_weight_fn = parse_lr_schedule_string_into_lr_lambda(
        deform_by_csd_cfg.stage_II_weight_schedule
    )

    #### func to save results and resumefile
    def __save_optimized_quantity(last_completed_epoch: int):
        if (
            (this_is := quantity_being_optimized.this_is) == "faces_jacobians"
            or this_is == "faces_3x2rotations"
            or this_is == "verts_3x2rotations"
            or this_is == "faces_normals"
            or this_is == "verts_normals"
        ):
            if n_meshes == 1:
                save_optimized_matrices_or_normals(
                    deform_by_csd_cfg,
                    dataset_cfg,
                    pt3d_batched_meshes,
                    quantity_being_optimized.tensor,
                    deform_by_csd_cfg.optimized_quantity_save_fname,
                )
            else:
                # TODO: save individual files for each mesh in batch with filename saying the index in batch
                raise NotImplementedError(
                    "TODO: save multiple result files for a batch of multiple meshes being optimized"
                )

    #### optimization loop
    cams_and_lights_cfg = deform_by_csd_cfg.cams_and_lights
    view_batch_size = deform_by_csd_cfg.view_batch_size
    mesh_batch_size = deform_by_csd_cfg.mesh_batch_size
    print("LOSSLOG")  # for scripts to know where the loss log starts in stdout file
    from_epoch = deform_by_csd_cfg.start_from_epoch

    detect_anomaly = bool(os.environ.get("TORCH_PLS_DETECT_ANOMALY", None))
    if detect_anomaly:
        thlog.info("enabling torch anomaly detection")
    torch.autograd.set_detect_anomaly(detect_anomaly)
    for optim_i in range(from_epoch, deform_by_csd_cfg.n_iters + 1):
        this_iter_needs_viz = (
            deform_by_csd_cfg.view_once_every > 0
            and optim_i % deform_by_csd_cfg.view_once_every == 0
        )

        # evaluate the LR schedule functions and set the current learning rates
        set_learning_rate(
            optimizer, main_lr_fn(optim_i), other_shared_params_lr_fn(optim_i)
        )

        # evaluate the schedule functions to get the current epoch's loss weights
        visual_loss_weight = visual_loss_weight_fn(optim_i)
        jac_id_loss_weight = jacobian_id_loss_weight_fn(optim_i)
        stage_II_loss_weight = stage_II_loss_weight_fn(optim_i)

        # print a line in the polyscope recording
        if this_iter_needs_viz:
            thlog.log_in_ps_recording(
                f"optim iter {optim_i}, visual loss x{visual_loss_weight:.6f} jac id loss x{jac_id_loss_weight:.6f} stage2 loss x{stage_II_loss_weight:.6f}"
            )

        # iterate thru batches in the dataset
        shuffled_mesh_indices = torch.randperm(n_meshes)
        for mesh_indices_this_batch in torch.split(shuffled_mesh_indices, mesh_batch_size):
            # gather mesh_structs and pytorch3d meshes from these indices
            mesh_indices_this_batch = cast(List[int], mesh_indices_this_batch.tolist())
            meshes_structs_this_batch = [meshes_structs[i] for i in mesh_indices_this_batch]
            pt3d_meshes_this_batch = pt3d_batched_meshes[mesh_indices_this_batch]
            optim_quantity_this_batch = quantity_being_optimized[mesh_indices_this_batch]
            if maybe_my_solver is not None:
                maybe_my_solver_this_batch = maybe_my_solver[mesh_indices_this_batch]
            else:
                maybe_my_solver_this_batch = None

            # calculate deformed verts for meshes in the batch
            new_verts_list, intermediate_results = calc_deformed_verts_according_to_cfg(
                meshes_structs_this_batch,
                pt3d_meshes_this_batch,
                deform_by_csd_cfg.solve_method,
                deform_by_csd_cfg.arap_energy_type,
                maybe_my_solver_this_batch,
                optim_quantity_this_batch,
                this_iter_needs_viz,
            )

            # render meshes using these new verts
            train_renders = []
            prompt_zs = []
            prompt_neg_zs = []
            psmeshes = []  # polyscope mesh structures (or their psrec proxies) for viz
            for idx_in_batch, (mesh_struct, new_verts) in enumerate(
                zip(meshes_structs_this_batch, new_verts_list)
            ):
                # register the deformed mesh in polyscope for viewing/logging
                if this_iter_needs_viz:
                    psmeshes.append(
                        thlog.psr.register_surface_mesh(
                            f"deformed{idx_in_batch}",
                            new_verts.cpu().detach().numpy(),
                            cast(torch.Tensor, mesh_struct.nvdm_loaded_mesh.t_pos_idx)
                            .cpu()
                            .detach()
                            .numpy(),
                        )
                    )

                # render this mesh with the new verts and a batch of views
                cams_and_lights_batch = get_batch_of_cameras_and_lights_with_dist_adapt(
                    cams_and_lights_cfg,
                    background.device,
                    view_batch_size,
                    (deform_by_csd_cfg.adapt_dists and optim_i > 1),
                    new_verts,
                )
                train_render = render_nvdm_mesh_with_new_verts_and_view_batch(
                    glctx,
                    cams_and_lights_cfg,
                    cams_and_lights_batch,
                    background,
                    mesh_struct.nvdm_loaded_mesh,
                    new_verts,
                )
                # train_render has shape (view_batch_size, channels, h, w)
                # resize h,w to the config-specified resize_for_guidance size
                train_render = resize_for_guidance_fn(train_render)
                train_renders.append(train_render)

                # while we're here looping thru individual `mesh_struct`s: NOTE
                # mesh_struct.prompt_z and prompt_negative_z are shape (1, 77, 4096)
                # so when we cat(prompt_zs) later, we get (mesh_batch_size, 77, 4096)
                # then we'll need repeat_interleave to duplicate the individual
                # zs (interleaved in original order) `view_batch_size` times
                # in order to get the same final batch size as cat(train_renders)
                prompt_zs.append(mesh_struct.prompt_z)
                prompt_neg_zs.append(mesh_struct.prompt_negative_z)
            # end per-mesh render loop

            # concat the render batches from the meshes in the batch.
            # there are mesh_batch_size meshes, each rendered with
            # view_batch_size views, so train_renders should end up with shape
            # (view_batch_size * mesh_batch_size, channels, h, w)
            train_renders = torch.cat(train_renders, dim=0)

            # cat(prompt_zs) has shape (mesh_batch_size, 77, 4096), but the
            # render batches were view_batch_size so we need repeat_interleave
            # to get (mesh_batch_size * view_batch_size, 77, 4096)
            prompt_zs = torch.cat(prompt_zs, dim=0)
            prompt_zs = prompt_zs.repeat_interleave(view_batch_size, dim=0)
            prompt_neg_zs = torch.cat(prompt_neg_zs, dim=0)
            prompt_neg_zs = prompt_neg_zs.repeat_interleave(view_batch_size, dim=0)

            # register train renders in polyscope to view
            if this_iter_needs_viz:
                # train_renders is (b, c, h, w)
                ps_images = train_renders.cpu().detach()[:8]  # just view the first 8 images
                orig_image_width = ps_images.size(3)
                # scale down to save space in the psrec recording and permute (b,c,h,w) to (b,h,w,c)
                # for polyscope (do that after resize_right which still expects (b,c,h,w))
                ps_images = resize_right.resize(
                    ps_images, scale_factors=(128 / orig_image_width)
                ).permute(0, 2, 3, 1)
                image_batch_size, image_height, image_width, image_n_channels = (
                    ps_images.shape
                )
                thlog.psr.add_color_image_quantity(
                    "renders",
                    PSRSpecialArray.image_array_as_png_bytes(
                        ps_images.reshape(
                            image_batch_size * image_height, image_width, image_n_channels
                        ).numpy()
                    ),
                    enabled=True,
                )

                thlog.psr.show()

            # calc losses and optimize
            optimizer.zero_grad()
            if isinstance(stage_I, csd.DummyCSDClass):
                raise RuntimeError("dummy csd can only go this far!")

            for accum_i in range((n_accum_iters := deform_by_csd_cfg.n_accum_iters)):
                # the first return item is the loss tensor; the rest are just python scalars
                visual_loss, stage_I_loss_val, stage_II_loss_val, total_visual_loss_val = (
                    csd.calc_csd_loss(
                        stage_I,
                        stage_II,
                        train_renders,
                        prompt_zs,
                        prompt_neg_zs,
                        deform_by_csd_cfg.stage_I_weight,
                        stage_II_loss_weight,
                    )
                )

                # add additional losses
                # jacobian identity regularization, for faces_jacobians
                this_is = optim_quantity_this_batch.this_is
                if (
                    this_is == "faces_jacobians"
                    or this_is == "faces_3x2rotations"
                    or this_is == "verts_3x2rotations"
                    or this_is == "faces_normals"
                    or this_is == "verts_normals"
                ) and jac_id_loss_weight > 0:
                    jac_id_loss, jac_id_loss_val = calc_jacobian_id_loss(
                        optim_quantity_this_batch, intermediate_results
                    )
                else:
                    jac_id_loss = None
                    jac_id_loss_val = 0.0

                # add up the losses
                loss = visual_loss_weight * visual_loss
                if jac_id_loss is not None:
                    loss = loss + jac_id_loss_weight * jac_id_loss

                # do backward and step
                is_last_accum_iter = accum_i == (n_accum_iters - 1)
                loss.backward(retain_graph=not is_last_accum_iter)
                if deform_by_csd_cfg.step_after_every_backward or is_last_accum_iter:
                    optimizer.step()

                # use the loss float values to print
                if is_last_accum_iter:
                    print(
                        f"{optim_i},{total_visual_loss_val:.6f},{stage_I_loss_val:.6f},{stage_II_loss_val:.6f},{jac_id_loss_val:.6f}",
                        flush=True,
                    )
            # end accum loop
        # end dataset batches loop

        # save if optim_i is one of the indicated save epochs, or is last epoch
        if (optim_i in deform_by_csd_cfg.save_at_epochs) or (
            optim_i == deform_by_csd_cfg.n_iters
        ):
            # our optim_i counts from 1 so the last epoch is ==n_iters
            __save_optimized_quantity(optim_i)

    # end optim loop
    print("END LOSSLOG", flush=True)


@thronfigure
def main(config: MainConfig):
    thlog.info(
        f"\nCONFIGPRINT\n{(cfg_string := config.to_json_string(pretty=True))}\nEND CONFIGPRINT"
    )
    thlog.init_polyscope(start_polyscope_recorder=True)
    submain_deform_meshes_by_csd(
        config.deform_by_csd,
        config.dataset,
        device=torch.device(config.device),
    )
    ps_recording_save_fname = config.ps_recording_save_fname
    if os.path.isfile(ps_recording_save_fname):
        # append a disambiguating, incrementing number like on windows (Copy (1), etc)
        fname_noext, ext = os.path.splitext(ps_recording_save_fname)
        ps_recording_save_fname = next_increment_path(fname_noext + "-({})" + ext)

    thlog.save_ps_recording(ps_recording_save_fname, comment=cfg_string)


if __name__ == "__main__":
    main()
