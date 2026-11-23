from __future__ import annotations

from typing import TYPE_CHECKING, cast

import torch

from mjlab.utils.lab_api.math import (
  matrix_from_quat,
  subtract_frame_transforms,
)

from .commands import MotionCommand

if TYPE_CHECKING:
  from mjlab.envs import ManagerBasedRlEnv


def motion_anchor_pos_b(env: ManagerBasedRlEnv, command_name: str) -> torch.Tensor:
  command = cast(MotionCommand, env.command_manager.get_term(command_name))

  pos, _ = subtract_frame_transforms(
    command.robot_anchor_pos_w,
    command.robot_anchor_quat_w,
    command.anchor_pos_w,
    command.anchor_quat_w,
  )

  return pos.view(env.num_envs, -1)


def motion_anchor_ori_b(env: ManagerBasedRlEnv, command_name: str) -> torch.Tensor:
  """Compute the reference anchor orientation in the robot's body frame.

  Steps:
    1. Use subtract_frame_transforms to get the relative orientation between
       the robot anchor and the reference anchor.
    2. Convert the resulting quaternion to a rotation matrix using matrix_from_quat.
    3. Extract the first two columns of the rotation matrix and flatten.

  Returns:
    Tensor of shape (num_envs, 6) — the first two columns of the relative
    rotation matrix, flattened.
  """
  command = cast(MotionCommand, env.command_manager.get_term(command_name))
  _, ori = subtract_frame_transforms(command.robot_anchor_pos_w,command.robot_anchor_quat_w,command.anchor_pos_w, command.anchor_quat_w,)
  mat = matrix_from_quat(ori)
  return mat[..., :2].reshape(mat.shape[0], -1)


def robot_body_pos_b(env: ManagerBasedRlEnv, command_name: str) -> torch.Tensor:
  """Compute the positions of all tracked bodies in the robot's body (anchor) frame.

  Steps:
    1. Get the number of tracked bodies from command.cfg.body_names.
    2. Expand robot_anchor_pos_w and robot_anchor_quat_w to match the number of bodies
       (use [:, None, :].repeat(1, num_bodies, 1)).
    3. Use subtract_frame_transforms to get body positions relative to the anchor.
    4. Flatten and return shape (num_envs, num_bodies * 3).

  Returns:
    Tensor of shape (num_envs, num_bodies * 3) — body positions in the anchor frame.
  """
  command = cast(MotionCommand, env.command_manager.get_term(command_name))
  num_bodies = len(command.cfg.body_names)
  pos_b, _ = subtract_frame_transforms(command.robot_anchor_pos_w[:, None, :].repeat(1, num_bodies, 1), command.robot_anchor_quat_w[:, None, :].repeat(1, num_bodies, 1), command.robot_body_pos_w, command.robot_body_quat_w, )
  return pos_b.view(env.num_envs, -1)


def robot_body_ori_b(env: ManagerBasedRlEnv, command_name: str) -> torch.Tensor:
  command = cast(MotionCommand, env.command_manager.get_term(command_name))

  num_bodies = len(command.cfg.body_names)
  _, ori_b = subtract_frame_transforms(
    command.robot_anchor_pos_w[:, None, :].repeat(1, num_bodies, 1),
    command.robot_anchor_quat_w[:, None, :].repeat(1, num_bodies, 1),
    command.robot_body_pos_w,
    command.robot_body_quat_w,
  )
  mat = matrix_from_quat(ori_b)
  return mat[..., :2].reshape(mat.shape[0], -1)
