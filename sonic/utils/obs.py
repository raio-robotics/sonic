"""Observation builders for the policy and encoder networks.

Replicates the observation registry from g1_deploy_onnx_ref.cpp:1704-1791. The
layout below is locked to the *default* observation_config.yaml that ships with
the release checkpoint:

Policy obs (994 floats, observation_config.yaml `observations`):
    token_state                                  [0:64]
    his_base_angular_velocity_10frame_step1      [64:94]    (10 × 3)
    his_body_joint_positions_10frame_step1       [94:384]   (10 × 29, IsaacLab, q - default)
    his_body_joint_velocities_10frame_step1      [384:674]  (10 × 29, IsaacLab)
    his_last_actions_10frame_step1               [674:964]  (10 × 29, IsaacLab, raw)
    his_gravity_dir_10frame_step1                [964:994]  (10 × 3, body-frame)

Encoder obs (1762 floats). For g1 mode (mode_id=0) only these are filled —
everything else is zeroed (mode gating from encoder.hpp).
    encoder_mode_4                                [0:4]
    motion_joint_positions_10frame_step5          [4:294]
    motion_joint_velocities_10frame_step5         [294:584]
    motion_anchor_orientation_10frame_step5       [601:661]
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass

import numpy as np
import yaml

from . import math_utils, params
from .motion import MotionSequence


@dataclass
class StateSample:
    base_quat: np.ndarray  # (4,) wxyz
    base_ang_vel: np.ndarray  # (3,) body-frame
    joint_pos_isaaclab: np.ndarray  # (29,) IsaacLab order, q - default
    joint_vel_isaaclab: np.ndarray  # (29,) IsaacLab order
    last_action: np.ndarray  # (29,) IsaacLab order, raw policy output


class StateHistory:
    """Ring buffer of past StateSample. Used to construct the his_* observations."""

    def __init__(self, capacity: int = 64) -> None:
        self.buf: deque[StateSample] = deque(maxlen=capacity)

    def push(self, s: StateSample) -> None:
        self.buf.append(s)

    def get_history(self, num_frames: int, step: int) -> list[StateSample]:
        """Samples `step` ticks apart, oldest first, padded at the old end with zero samples.

        Matches StateLogger::GetLatest (state_logger.cpp:199-233), which pads with zero
        entries until the buffer holds num_frames samples.
        """
        latest = len(self.buf) - 1
        out: list[StateSample] = []
        for f in range(num_frames):
            idx = latest - (num_frames - 1 - f) * step
            out.append(self.buf[idx] if idx >= 0 else ZERO_SAMPLE)
        return out


ZERO_SAMPLE = StateSample(
    base_quat=np.zeros(4),
    base_ang_vel=np.zeros(3),
    joint_pos_isaaclab=np.zeros(params.G1_NUM_MOTOR),
    joint_vel_isaaclab=np.zeros(params.G1_NUM_MOTOR),
    last_action=np.zeros(params.G1_NUM_MOTOR),
)


class PolicyObsBuilder:
    """Builds the 994-d policy observation."""

    POLICY_DIM = 994

    def __init__(self) -> None:
        self.buf = np.zeros(self.POLICY_DIM, dtype=np.float64)

    def build(self, token_state: np.ndarray, history: StateHistory) -> np.ndarray:
        b = self.buf
        b[0:64] = token_state
        hist = history.get_history(10, 1)
        for f, s in enumerate(hist):
            b[64 + 3 * f : 67 + 3 * f] = s.base_ang_vel
            b[94 + 29 * f : 123 + 29 * f] = s.joint_pos_isaaclab
            b[384 + 29 * f : 413 + 29 * f] = s.joint_vel_isaaclab
            b[674 + 29 * f : 703 + 29 * f] = s.last_action
            grav = math_utils.quat_rotate(
                math_utils.quat_conjugate(s.base_quat), np.array([0.0, 0.0, -1.0])
            )
            b[964 + 3 * f : 967 + 3 * f] = grav
        return b


class EncoderObsBuilder:
    """Builds the 1762-d encoder observation for g1 mode (mode_id=0)."""

    ENCODER_DIM = 1762

    OFFSETS = {
        "encoder_mode_4": (0, 4),
        "motion_joint_positions_10frame_step5": (4, 290),
        "motion_joint_velocities_10frame_step5": (294, 290),
        "motion_root_z_position_10frame_step5": (584, 10),
        "motion_root_z_position": (594, 1),
        "motion_anchor_orientation": (595, 6),
        "motion_anchor_orientation_10frame_step5": (601, 60),
        "motion_joint_positions_lowerbody_10frame_step5": (661, 120),
        "motion_joint_velocities_lowerbody_10frame_step5": (781, 120),
        "vr_3point_local_target": (901, 9),
        "vr_3point_local_orn_target": (910, 12),
        "smpl_joints_10frame_step1": (922, 720),
        "smpl_anchor_orientation_10frame_step1": (1642, 60),
        "motion_joint_positions_wrists_10frame_step1": (1702, 60),
    }
    G1_REQUIRED = {
        "encoder_mode_4",
        "motion_joint_positions_10frame_step5",
        "motion_joint_velocities_10frame_step5",
        "motion_anchor_orientation_10frame_step5",
    }

    def __init__(self) -> None:
        self.buf = np.zeros(self.ENCODER_DIM, dtype=np.float64)

    def build(
        self,
        motion: MotionSequence,
        current_frame: int,
        base_quat_wxyz: np.ndarray,
        apply_delta_heading: np.ndarray,
    ) -> np.ndarray:
        """Build the 1762-d encoder observation of `motion` playing at `current_frame`.

        ControlNode always plays the motion it tracks (operator_state.play in the C++:
        the planner trajectory, or a reference motion after T), so the 10 gathered
        frames run 5 frames apart from current_frame, holding the last frame at the
        end, with the motion's joint velocities (g1_deploy_onnx_ref.cpp:758-766, 819-860).

        Args:
            base_quat_wxyz: robot's current base IMU quaternion (wxyz). Used
                as the "left quaternion" in the anchor-orientation diff.
            apply_delta_heading: heading-alignment quaternion that rotates the
                motion data to match the robot's current heading.
        """
        b = self.buf
        b.fill(0.0)
        b[0] = float(motion.encode_mode)

        # Motion CSVs store joint data in IsaacLab order (the C++
        # GatherMotionJointPositions just memcpys JointPositions(frame) into the
        # obs buffer at g1_deploy_onnx_ref.cpp:768-777). No remap needed.
        frames = [min(current_frame + 5 * f, motion.timesteps - 1) for f in range(10)]
        for f, fr in enumerate(frames):
            b[4 + 29 * f : 33 + 29 * f] = motion.jp(fr)
            b[294 + 29 * f : 323 + 29 * f] = motion.jv(fr)

        # motion_anchor_orientation_10frame_step5: motion root orientation
        # expressed in the ROBOT body frame, 6-D row-wise flatten of the first
        # two columns of the rotation matrix of
        #   base_to_ref = inv(base_quat) * apply_delta_heading * motion_root_quat
        # (g1_deploy_onnx_ref.cpp:644-688).
        base_inv = math_utils.quat_conjugate(base_quat_wxyz)
        for f, fr in enumerate(frames):
            new_ref = math_utils.quat_mul(apply_delta_heading, motion.bq(fr, 0))
            base_to_ref = math_utils.quat_mul(base_inv, new_ref)
            b[601 + 6 * f : 607 + 6 * f] = math_utils.quat_to_6d_rowwise(base_to_ref)
        return b


def parse_observation_config(yaml_path: str) -> dict:
    with open(yaml_path) as f:
        return yaml.safe_load(f)
