import copy
import random
import numpy as np
import torch
from jaxtyping import Float
from torch import Tensor

from ..types import AnyExample, AnyViews


def reflect_extrinsics(extrinsics: Tensor) -> Tensor:
    """
    Reflect extrinsics along the x axis.

    Supports both:
      - (..., 4, 4) homogeneous transforms
      - (..., 3, 4) [R|t] transforms

    Returns the same shape as input.
    """
    if extrinsics.shape[-2:] not in ((4, 4), (3, 4)):
        raise ValueError(f"Expected extrinsics shape (...,4,4) or (...,3,4), got {tuple(extrinsics.shape)}")

    # keep dtype consistent (important for bf16/fp16/fp32)
    reflect = torch.eye(4, dtype=extrinsics.dtype, device=extrinsics.device)
    reflect[0, 0] = -1

    if extrinsics.shape[-2:] == (4, 4):
        return reflect @ extrinsics @ reflect

    # extrinsics is (..., 3, 4) -> lift to (..., 4, 4), reflect, then drop back to (..., 3, 4)
    # build bottom row [0, 0, 0, 1] with correct batch shape
    bottom = torch.tensor([0, 0, 0, 1], dtype=extrinsics.dtype, device=extrinsics.device)
    bottom = bottom.view(*([1] * (extrinsics.ndim - 2)), 1, 4).expand(*extrinsics.shape[:-2], 1, 4)

    extr4 = torch.cat([extrinsics, bottom], dim=-2)  # (..., 4, 4)
    extr4_ref = reflect @ extr4 @ reflect
    return extr4_ref[..., :3, :]  # (..., 3, 4)


def reflect_views(views: AnyViews) -> AnyViews:
    if "depth" in views.keys():
        return {
            **views,
            "image": views["image"].flip(-1),
            "extrinsics": reflect_extrinsics(views["extrinsics"]),
            "depth": views["depth"].flip(-1),
        }
    else:
        return {
            **views,
            "image": views["image"].flip(-1),
            "extrinsics": reflect_extrinsics(views["extrinsics"]),
        }


def apply_augmentation_shim(
    example: AnyExample,
    generator: torch.Generator | None = None,
) -> AnyExample:
    """Randomly augment the training images."""
    # Do not augment with 50% chance.
    if torch.rand(tuple(), generator=generator) < 0.5:
        return example

    return {
        **example,
        "context": reflect_views(example["context"]),
        "target": reflect_views(example["target"]),
    }


def rotate_90_degrees(
    image: torch.Tensor, depth_map: torch.Tensor | None, extri_opencv: torch.Tensor, intri_opencv: torch.Tensor, clockwise=True
):
    """
    Rotates the input image, depth map, and camera parameters by 90 degrees.
    """
    image_height, image_width = image.shape[-2:]

    rotated_image, rotated_depth_map = rotate_image_and_depth_rot90(image, depth_map, clockwise)
    new_intri_opencv = adjust_intrinsic_matrix_rot90(intri_opencv, image_width, image_height, clockwise)
    new_extri_opencv = adjust_extrinsic_matrix_rot90(extri_opencv, clockwise)

    return (
        rotated_image,
        rotated_depth_map,
        new_extri_opencv,
        new_intri_opencv,
    )


def rotate_image_and_depth_rot90(image: torch.Tensor, depth_map: torch.Tensor | None, clockwise: bool):
    rotated_depth_map = None
    if clockwise:
        rotated_image = torch.rot90(image, k=-1, dims=[-2, -1])
        if depth_map is not None:
            rotated_depth_map = torch.rot90(depth_map, k=-1, dims=[-2, -1])
    else:
        rotated_image = torch.rot90(image, k=1, dims=[-2, -1])
        if depth_map is not None:
            rotated_depth_map = torch.rot90(depth_map, k=1, dims=[-2, -1])
    return rotated_image, rotated_depth_map


def adjust_extrinsic_matrix_rot90(extri_opencv: torch.Tensor, clockwise: bool):
    """
    Adjusts the extrinsic matrix (3x4) for a 90-degree rotation of the image.

    NOTE: Despite docstring saying 3x4, this function actually returns 4x4
    (it appends the bottom row [0,0,0,1]).
    """
    R = extri_opencv[:3, :3]
    t = extri_opencv[:3, 3]

    if clockwise:
        R_rotation = torch.tensor(
            [[0, -1, 0], [1, 0, 0], [0, 0, 1]],
            dtype=extri_opencv.dtype,
            device=extri_opencv.device,
        )
    else:
        R_rotation = torch.tensor(
            [[0, 1, 0], [-1, 0, 0], [0, 0, 1]],
            dtype=extri_opencv.dtype,
            device=extri_opencv.device,
        )

    new_R = torch.matmul(R_rotation, R)
    new_t = torch.matmul(R_rotation, t)
    new_extri_opencv = torch.cat((new_R, new_t.reshape(-1, 1)), dim=1)
    new_extri_opencv = torch.cat(
        (
            new_extri_opencv,
            torch.tensor([[0, 0, 0, 1]], dtype=extri_opencv.dtype, device=extri_opencv.device),
        ),
        dim=0,
    )
    return new_extri_opencv


def adjust_intrinsic_matrix_rot90(intri_opencv: torch.Tensor, image_width: int, image_height: int, clockwise: bool):
    """
    Adjusts the intrinsic matrix (3x3) for a 90-degree rotation of the image in the image plane.
    """
    intri_opencv = copy.deepcopy(intri_opencv)
    intri_opencv[0, :] *= image_width
    intri_opencv[1, :] *= image_height

    fx, fy, cx, cy = (
        intri_opencv[0, 0],
        intri_opencv[1, 1],
        intri_opencv[0, 2],
        intri_opencv[1, 2],
    )

    new_intri_opencv = torch.eye(3, dtype=intri_opencv.dtype, device=intri_opencv.device)
    if clockwise:
        new_intri_opencv[0, 0] = fy
        new_intri_opencv[1, 1] = fx
        new_intri_opencv[0, 2] = image_height - cy
        new_intri_opencv[1, 2] = cx
    else:
        new_intri_opencv[0, 0] = fy
        new_intri_opencv[1, 1] = fx
        new_intri_opencv[0, 2] = cy
        new_intri_opencv[1, 2] = image_width - cx

    new_intri_opencv[0, :] /= image_height
    new_intri_opencv[1, :] /= image_width

    return new_intri_opencv