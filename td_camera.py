from dataclasses import dataclass
from typing import Tuple, Dict
import random
import numpy as np
import glm
import torch
import torch.utils.data
import torchvision.transforms as transforms

import resize_right
from thronf import Thronfig


@dataclass
class CamsAndLights_Settings(Thronfig):
    raster_res: int
    dist_minmax: Tuple[float, float]
    azim_minmax: Tuple[float, float]
    elev_minmax: Tuple[float, float]
    fov_minmax: Tuple[float, float]
    light_power: float
    aug_loc: bool
    """ whether to do augmentation on camera position """
    aug_light: bool
    """ whether to do augmentation on light position """
    look_at: Tuple[float, float, float]  # default in cameras.py was [0,0,0]
    up: Tuple[float, float, float]  # default in cameras.py is [0,-1,0]


blurs = [
    transforms.Compose([transforms.GaussianBlur(11, sigma=(5, 5))]),
    transforms.Compose([transforms.GaussianBlur(11, sigma=(2, 2))]),
    transforms.Compose([transforms.GaussianBlur(5, sigma=(5, 5))]),
    transforms.Compose([transforms.GaussianBlur(5, sigma=(2, 2))]),
]


def get_random_bg(h, w, rand_solid=False):
    """
    from meshfusion/meshup/td repo  utilities/camera.py
    """
    p = torch.rand(1)
    if p > 0.66666:
        if rand_solid:
            background = torch.vstack(
                [
                    torch.full((1, h, w), torch.rand(1).item()),
                    torch.full((1, h, w), torch.rand(1).item()),
                    torch.full((1, h, w), torch.rand(1).item()),
                ]
            ).unsqueeze(0) + torch.rand(1, 3, h, w)
            background = (background - background.amin()) / (
                background.amax() - background.amin()
            )
            background = blurs[random.randint(0, 3)](background).permute(0, 2, 3, 1)
        else:
            background = blurs[random.randint(0, 3)](torch.rand((1, 3, h, w))).permute(
                0, 2, 3, 1
            )
    elif p > 0.333333:
        size = random.randint(5, 10)
        background = torch.vstack(
            [
                torch.full((1, size, size), torch.rand(1).item() / 2),
                torch.full((1, size, size), torch.rand(1).item() / 2),
                torch.full((1, size, size), torch.rand(1).item() / 2),
            ]
        ).unsqueeze(0)

        second = torch.rand(3)

        background[:, 0, ::2, ::2] = second[0]
        background[:, 1, ::2, ::2] = second[1]
        background[:, 2, ::2, ::2] = second[2]

        background[:, 0, 1::2, 1::2] = second[0]
        background[:, 1, 1::2, 1::2] = second[1]
        background[:, 2, 1::2, 1::2] = second[2]

        background = blurs[random.randint(0, 3)](
            resize_right.resize(background, out_shape=(h, w))
        )

        background = background.permute(0, 2, 3, 1)

    else:
        background = (
            torch.vstack(
                [
                    torch.full((1, h, w), torch.rand(1).item()),
                    torch.full((1, h, w), torch.rand(1).item()),
                    torch.full((1, h, w), torch.rand(1).item()),
                ]
            )
            .unsqueeze(0)
            .permute(0, 2, 3, 1)
        )

    return background


def cosine_sample(N: np.ndarray) -> np.ndarray:
    """
    #----------------------------------------------------------------------------
    # Cosine sample around a vector N
    #----------------------------------------------------------------------------

    Copied from nvdiffmodelling

    """
    # construct local frame
    N = N / np.linalg.norm(N)

    dx0 = np.array([0, N[2], -N[1]])
    dx1 = np.array([-N[2], 0, N[0]])

    dx = dx0 if np.dot(dx0, dx0) > np.dot(dx1, dx1) else dx1
    dx = dx / np.linalg.norm(dx)
    dy = np.cross(N, dx)
    dy = dy / np.linalg.norm(dy)

    # cosine sampling in local frame
    phi = 2.0 * np.pi * np.random.uniform()
    s = np.random.uniform()
    costheta = np.sqrt(s)
    sintheta = np.sqrt(1.0 - s)

    # cartesian vector in local space
    x = np.cos(phi) * sintheta
    y = np.sin(phi) * sintheta
    z = costheta

    # local to world
    return dx * x + dy * y + N * z


def persp_proj(fov_x=45.0, ar=1.0, near=1.0, far=50.0):
    """
    From https://github.com/rgl-epfl/large-steps-pytorch by @bathal1 (Baptiste Nicolet)

    Build a perspective projection matrix.
    Parameters
    ----------
    fov_x : float
        Horizontal field of view (in degrees).
    ar : float
        Aspect ratio (w/h).
    near : float
        Depth of the near plane relative to the camera.
    far : float
        Depth of the far plane relative to the camera.
    """
    fov_rad = np.deg2rad(fov_x)

    tanhalffov = np.tan((fov_rad / 2))
    max_y = tanhalffov * near
    min_y = -max_y
    max_x = max_y * ar
    min_x = -max_x

    z_sign = -1.0
    proj_mat = np.array([[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]])

    proj_mat[0, 0] = 2.0 * near / (max_x - min_x)
    proj_mat[1, 1] = 2.0 * near / (max_y - min_y)
    proj_mat[0, 2] = (max_x + min_x) / (max_x - min_x)
    proj_mat[1, 2] = (max_y + min_y) / (max_y - min_y)
    proj_mat[3, 2] = z_sign

    proj_mat[2, 2] = z_sign * far / (far - near)
    proj_mat[2, 3] = -(far * near) / (far - near)

    return proj_mat


def get_batch_of_cameras_and_lights(
    cams_and_lights_cfg: CamsAndLights_Settings,
    view_batch_size: int,
    dist_multiplier: float = 1.0,
) -> Dict[str, torch.Tensor]:
    cfg = cams_and_lights_cfg
    elev_min, elev_max = cfg.elev_minmax
    azim_min, azim_max = cfg.azim_minmax
    dist_min, dist_max = cfg.dist_minmax
    dist_min = dist_multiplier * dist_min
    dist_max = dist_multiplier * dist_max
    fov_min, fov_max = cfg.fov_minmax

    # this function does one item, and then we'll use torch's default collate_fn
    # to combine a list of these results into a batch
    def __get_single_camera_and_light_dict():
        """
        from meshfusion/meshup/td repo utilities/camera.py
        """
        elev = np.radians(np.random.uniform(elev_min, elev_max))
        azim = np.radians(np.random.uniform(azim_min, azim_max + 1.0))
        dist = np.random.uniform(dist_min, dist_max)
        fov = np.random.uniform(fov_min, fov_max)
        proj_mtx = persp_proj(fov)

        # Generate random view
        cam_z = dist * np.cos(elev) * np.sin(azim)
        cam_y = dist * np.sin(elev)
        cam_x = dist * np.cos(elev) * np.cos(azim)

        if cfg.aug_loc:
            # Random offset
            limit = dist_min // 2
            rand_x = np.random.uniform(-limit, limit)
            rand_y = np.random.uniform(-limit, limit)

            modl = glm.translate(glm.mat4(), glm.vec3(rand_x, rand_y, 0))

        else:

            modl = glm.mat4()

        view = glm.lookAt(
            glm.vec3(cam_x, cam_y, cam_z),
            glm.vec3(cfg.look_at[0], cfg.look_at[1], cfg.look_at[2]),
            glm.vec3(cfg.up[0], cfg.up[1], cfg.up[2]),
        )

        r_mv = view * modl
        r_mv = np.array(r_mv.to_list()).T

        mvp = np.matmul(proj_mtx, r_mv).astype(np.float32)
        campos = np.linalg.inv(r_mv)[:3, 3]

        if cfg.aug_light:
            lightpos = cosine_sample(campos) * dist
        else:
            lightpos = campos * dist

        # if cfg.aug_background:
        #     bkgs = get_random_bg(cfg.raster_res, cfg.raster_res, rand_solid=True).squeeze(0)
        # else:
        #     bkgs = torch.ones(cfg.raster_res, cfg.raster_res, 3)

        return {
            "mvp": torch.from_numpy(mvp).float(),
            "lightpos": torch.from_numpy(lightpos).float(),
            "campos": torch.from_numpy(campos).float(),
            # "bkgs": bkgs,
            "azim": torch.tensor(azim).float(),
            "elev": torch.tensor(elev).float(),
            "fov": torch.tensor(fov).float(),
            "dist": torch.tensor(dist).float(),
        }

    batch = [__get_single_camera_and_light_dict() for _ in range(view_batch_size)]
    batch = torch.utils.data.default_collate(batch)
    return batch
