# applies and visualizes the nrmls, jcbns, rot3x2 etc files from deform_with_csd_dARAP.py
from typing import Optional, Tuple
from dataclasses import dataclass
import json
import numpy as np
import torch
import igl
import pytorch3d.structures
from thlog import *
from thronf import Thronfig, InvalidConfigError

import deformations_dARAP

thlog = Thlogger(LOG_INFO, VIZ_INFO, "apply_deform", imports=[deformations_dARAP.thlog])
thlog.init_polyscope()


@dataclass
class ApplySavedDeformSettings(Thronfig):
    fname: str
    solve_method: Optional[deformations_dARAP.DeformSolveMethodName] = None
    arap_energy_type: Optional[deformations_dARAP.ARAPEnergyTypeName] = None
    local_step_procrustes_lambda_and_normalize: Optional[Tuple[float, bool]] = None
    pin_first_vertex: bool = True
    save_fname: Optional[str] = None
    load_cfg_from_npz: bool = True

    def __post_typecheck__(self):
        if not self.load_cfg_from_npz:
            if self.solve_method is None:
                raise InvalidConfigError(
                    "if --load_cfg_from_npz false, must specify explicit --solve_method"
                )


def load_saved_deform_and_apply(
    config: ApplySavedDeformSettings, device: torch.device
) -> Tuple[np.ndarray, np.ndarray, deformations_dARAP.QuantityBeingOptimized, torch.Tensor]:
    with np.load(config.fname) as npz:
        verts = npz["verts"]
        faces = npz["faces"]
        if config.load_cfg_from_npz and "deform_by_csd_cfg" in npz:
            try:
                loaded_deform_by_csd_cfg = json.loads(str(npz["deform_by_csd_cfg"]))
            except BaseException as e:
                thlog.err(f"npz-embedded config json load failed with error:\n{e.args}")
                loaded_deform_by_csd_cfg = None
        else:
            loaded_deform_by_csd_cfg = None

        if loaded_deform_by_csd_cfg:
            thlog.info(
                "replacing command line/configured values with values loaded from deform config file"
            )
            if _v := loaded_deform_by_csd_cfg.get("arap_energy_type"):
                thlog.info(f"replacing arap_energy_type with {_v}")
                config.arap_energy_type = _v
            if _v := loaded_deform_by_csd_cfg.get("solve_method"):
                thlog.info(f"replacing solve_method with {_v}")
                config.solve_method = _v
            if _v := loaded_deform_by_csd_cfg.get("pin_first_vertex"):
                thlog.info(f"replacing pin_first_vertex with {_v}")
                config.pin_first_vertex = _v
            if _p := loaded_deform_by_csd_cfg.get("local_step_procrustes"):
                if _p is not None:
                    thlog.info(f"replacing local_step_procrustes {_p}")
                    config.local_step_procrustes_lambda_and_normalize = (
                        _p["lamb"],
                        _p["normalize_target_normals"],
                    )
        else:
            thlog.err(
                "unable to parse saved config string in npz file, still using command line/configured values"
            )

        verts = deformations_dARAP.normalize_to_side2_cube_inplace_np_singlemesh(verts)
        if (
            (key := "faces_jacobians") in npz
            or (key := "faces_3x2rotations") in npz
            or (key := "verts_3x2rotations") in npz
            or (key := "faces_normals_offset") in npz
            or (key := "verts_normals") in npz
        ):
            src_deform_qty = (
                (
                    npz[key]
                    + igl.per_face_normals(verts, faces, np.zeros(3, dtype=np.float32))
                )
                if key == "faces_normals_offset"
                else npz[key]
            )
            patient_deform_qty = torch.from_numpy(src_deform_qty).float().to(device)
            patient_v, patient_f = verts, faces
        else:
            raise KeyError("supported deforms not found")
    key = "faces_normals" if key == "faces_normals_offset" else key
    thlog.info(f"deform quantity is: {key}")
    with torch.no_grad():
        thlog.psr.remove_all_structures()
        patient_meshes = pytorch3d.structures.Meshes(
            [torch.from_numpy(patient_v).float()], [torch.from_numpy(patient_f)]
        ).to(device)
        # TODO add or read a key from npz file to say num_verts, num_faces per
        # mesh to  load batched meshes? it's weird if the optim loop supports
        # batched meshes (dataset size > 1) but the loading doesn't...
        deformations_dARAP.normalize_to_side2_cube_inplace(patient_meshes)

        solver = deformations_dARAP.SparseLaplaciansSolvers.from_meshes(
            patient_meshes,
            pin_first_vertex=config.pin_first_vertex,
            compute_poisson_rhs_lefts=config.solve_method == "poisson",
            compute_igl_arap_rhs_lefts=deformations_dARAP.get_igl_arap_energy_type_from_cfg(
                config.arap_energy_type
            )
            if config.solve_method == "arap"
            and (
                config.arap_energy_type == "spokes_and_rims_igl"
                or config.arap_energy_type == "spokes_igl"
            )
            else None,
        )
        quantity_struct = deformations_dARAP.QuantityBeingOptimized(
            tensor=patient_deform_qty,
            this_is=key,
            num_verts_per_mesh=[len(patient_v)],
            num_faces_per_mesh=[len(patient_f)],
            procrustes_struct_if_needed=(
                deformations_dARAP.ProcrustesPrecomputeAndWhetherToNormalize(
                    pp=deformations_dARAP.ProcrustesPrecompute.from_meshes(
                        config.local_step_procrustes_lambda_and_normalize[0],
                        config.arap_energy_type,
                        solver,
                        patient_meshes,
                    ),
                    normalize=config.local_step_procrustes_lambda_and_normalize[1],
                )
                if config.local_step_procrustes_lambda_and_normalize
                and key == "verts_normals"
                else None
            ),
        )
        if config.solve_method is None:
            raise InvalidConfigError(
                "must specify solve_method (was it not loaded from the npz file?)"
            )

        deformed_verts, _ = (
            deformations_dARAP.calc_deformed_verts_solution_according_to_cfg(
                patient_meshes,
                config.solve_method,
                config.arap_energy_type,
                solver,
                deformations_dARAP.calc_inputs_to_solve_for_deformation_according_to_cfg(
                    patient_meshes, quantity_struct
                ),
            )
        )
        deformed_verts = torch.cat(tuple(deformed_verts), dim=0)
    return patient_v, patient_f, quantity_struct, deformed_verts


def main():
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("fname", type=str, help=".npz filename saved from optimization run")
    config, _ = ApplySavedDeformSettings.parse_args(
        "config",
        map_namespace_fields_to_config_fields={"fname": "fname"},
        extend_existing_parser=parser,
    )
    device = torch.device("cuda")

    patient_v, patient_f, quantity_struct, deformed_verts = load_saved_deform_and_apply(
        config, device
    )

    psmesh = thlog.psr.register_surface_mesh("orig", patient_v, patient_f)
    if quantity_struct.this_is == "verts_normals":
        psmesh.add_vector_quantity(
            "trg normals",
            quantity_struct.tensor.cpu().detach().numpy(),
            defined_on="vertices",
            enabled=True,
        )
    deformed_verts_np = deformed_verts.cpu().detach().numpy()
    thlog.psr.register_surface_mesh(
        "deformed", deformed_verts_np, patient_f, material="normal"
    )
    thlog.psr.show()
    if config.save_fname:
        igl.write_triangle_mesh(config.save_fname, deformed_verts_np, patient_f)


if __name__ == "__main__":
    main()
