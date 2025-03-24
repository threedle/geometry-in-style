from typing import List, Optional
import torch
import pytorch3d.structures
from iopath.common.file_io import PathManager


def load_objs_as_meshes(files: List[str],
    device: Optional[torch.device] = None,
    load_textures: bool = True,
    create_texture_atlas: bool = False,
    texture_atlas_size: int = 4,
    texture_wrap: Optional[str] = "repeat",
    path_manager: Optional[PathManager] = None,
) -> pytorch3d.structures.Meshes:...

def save_obj(
    f: str,
    verts: torch.Tensor,
    faces: torch.Tensor,
    decimal_places: Optional[int] = None,
    path_manager: Optional[PathManager] = None,
    *,
    normals: Optional[torch.Tensor] = None,
    faces_normals_idx: Optional[torch.Tensor] = None,
    verts_uvs: Optional[torch.Tensor] = None,
    faces_uvs: Optional[torch.Tensor] = None,
    texture_map: Optional[torch.Tensor] = None,
) -> None:...