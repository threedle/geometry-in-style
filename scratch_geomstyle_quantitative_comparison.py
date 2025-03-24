from typing import Sequence, List
import numpy as np
import igl
import sys
import torch


def recenter_to_centroid_and_rescale_new_verts_to_fit_old_bboxes(
    old_verts_list: Sequence[np.ndarray], new_verts_list: Sequence[np.ndarray]
) -> Sequence[np.ndarray]:
    """
    this is to match the method from textdeformer, even though it's probably not what I'd
    immediately think of if asked to preserve old bboxes
        (If I were to do this, the scale factor would be based on max axis-aligned extent,
        rather than length of the diagonal between min coord and max coord points)
    """
    new_verts_recentered: List[np.ndarray] = []
    for old_verts, new_verts in zip(old_verts_list, new_verts_list):
        # scale factor is the ratio between the bounding box diagonal lengths
        new_verts = new_verts - new_verts.mean(axis=0, keepdims=True)
        new_bbox_diag = new_verts.max(axis=0) - new_verts.min(axis=0)
        new_size = np.linalg.norm(new_bbox_diag)

        old_verts = old_verts - old_verts.mean(axis=0, keepdims=True)
        old_bbox_diag = old_verts.max(axis=0) - old_verts.min(axis=0)
        old_size = np.linalg.norm(old_bbox_diag)

        new_verts_recentered.append(new_verts * (old_size / new_size))

    return new_verts_recentered


def calc_mesh_volume_estimate(verts: np.ndarray, faces: np.ndarray) -> float:
    face_verts_coords = verts[faces]
    v1 = face_verts_coords[:, 0]
    v2 = face_verts_coords[:, 1]
    v3 = face_verts_coords[:, 2]
    return float(((v1 * (np.cross(v2, v3))).sum(axis=-1) / 6).sum())


def normalize_to_side2_cube_inplace_np_singlemesh(verts: np.ndarray):
    # actually just AABB
    mincoord = np.min(verts, axis=0)
    maxcoord = np.max(verts, axis=0)
    extent = (maxcoord - mincoord).max()
    scale = 2 / extent
    center = (mincoord + maxcoord) / 2
    return (verts - center) * scale


filelist1fname = sys.argv[1]
filelist2fname = sys.argv[2]


with open(filelist1fname, "r") as f:
    fnames1 = f.readlines()

with open(filelist2fname, "r") as f:
    fnames2 = f.readlines()

fnames1 = [fname.rstrip() for fname in fnames1]
fnames2 = [fname.rstrip() for fname in fnames2]
actually_used_fnames1 = []
actually_used_fnames2 = []

# should be orig verts, mapped verts
face_areas_ratio_arrays = []
volume_ratios = []


for fname1, fname2 in zip(fnames1, fnames2):
    if fname1.startswith("#") or fname2.startswith("#"):
        # rudimentary way to skip bad lines / placeholder filenames...
        continue
    verts1, faces1 = igl.read_triangle_mesh(fname1, dtypef=np.float32)
    verts2, faces2 = igl.read_triangle_mesh(fname2, dtypef=np.float32)
    assert verts1.shape == verts2.shape, f"\nfname 1:{fname1}\nfname 2 {fname2}"
    assert faces1.shape == faces2.shape, f"\nfname 1:{fname1}\nfname 2 {fname2}"

    # normalize src as is done by the deform methods
    verts1 = normalize_to_side2_cube_inplace_np_singlemesh(verts1)

    # perform the same post-global-solve normalizing as done in the methods
    verts2_bboxdiagnormed = recenter_to_centroid_and_rescale_new_verts_to_fit_old_bboxes(
        [verts1], [verts2]
    )[0]

    _absdiff = np.abs(verts2 - verts2_bboxdiagnormed)
    _absdiffmean = np.mean(_absdiff)
    _absdiffmax = np.max(_absdiff)
    _absdiffmin = np.min(_absdiff)
    # the absdiff should be very tiny if fname2 really is from meshup/geomstyle but NOT td because it doesnt use the same normalizing

    verts2 = verts2_bboxdiagnormed

    face_areas1 = igl.doublearea(verts1, faces1) * 0.5
    face_areas2 = igl.doublearea(verts2, faces2) * 0.5

    volume1 = calc_mesh_volume_estimate(verts1, faces1)
    volume2 = calc_mesh_volume_estimate(verts2, faces2)

    verts1_torch = torch.from_numpy(verts1)
    verts2_torch = torch.from_numpy(verts2)
    faces_torch = torch.from_numpy(faces1)

    face_areas_ratio = face_areas2 / face_areas1  # same as jac2x2det

    volume_ratio = volume2 / volume1

    print(f"""
    from: {fname1}
    to: {fname2}
    absdiff between verts2 and verts2_bboxdiagnormed mean {_absdiffmean} min {_absdiffmin} max {_absdiffmax} (should all be very small)
    avg face area ratio:  {face_areas_ratio.mean()}
    stdev face area ratio:{face_areas_ratio.std()}
    vol ratio: {volume_ratio}
    """)

    face_areas_ratio_arrays.append(face_areas_ratio)
    volume_ratios.append(volume_ratio)

    actually_used_fnames1.append(fname1)
    actually_used_fnames2.append(fname2)

# now print summary
face_areas_ratio_grand = np.concatenate(face_areas_ratio_arrays, axis=0)  # (total n_faces,)
face_areas_ratio_grandmean = np.mean(face_areas_ratio_grand)
face_areas_ratio_std = np.std(face_areas_ratio_grand)

volume_ratios = np.array(volume_ratios)  # (n_meshes,)
volume_ratios_mean = np.mean(volume_ratios)
volume_ratios_std = np.std(volume_ratios)

print(f"""
grand average  face area ratio: {face_areas_ratio_grandmean}
grand stddev   face area ratio: {face_areas_ratio_std}
meshes average volume ratio:    {volume_ratios_mean}
meshes stddev  volume ratio:    {volume_ratios_std}
""")


# save "face_areas_ratio_grand" array and array of face numbers, and source and target filenames
ARRAY_SAVE_FNAME = "face_areas_ratio_arrays.npz"
np.savez_compressed(
    ARRAY_SAVE_FNAME,
    face_areas_ratio_grand=face_areas_ratio_grand,
    volume_ratios=volume_ratios,
    num_faces_per_mesh=np.array(tuple(map(len, face_areas_ratio_arrays))),
    fnames1=np.array("\n".join(actually_used_fnames1)),
    fnames2=np.array("\n".join(actually_used_fnames2)),
)
print(f"saved arrays to {ARRAY_SAVE_FNAME}")
