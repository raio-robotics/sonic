"""Dataclasses exchanged between raio nodes.

raio pickles every value that crosses a connection, so we keep these small —
numpy arrays of fixed shape, plain Python ints/floats/bools. raio stamps each
value with its timestamp, which the receiving port reports as `port.time`.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


@dataclass
class LowStateSnapshot:
    """/rt/lowstate, MuJoCo joint order (hardware index)."""

    motor_q: np.ndarray  # (29,) float64
    motor_dq: np.ndarray  # (29,) float64
    base_quat_wxyz: np.ndarray  # (4,)
    base_ang_vel: np.ndarray  # (3,) body-frame
    base_lin_acc: np.ndarray  # (3,)
    mode_machine: int = 0


@dataclass
class IMUTorsoSnapshot:
    """/rt/secondary_imu — G1 torso IMU."""

    quat_wxyz: np.ndarray
    ang_vel: np.ndarray


@dataclass
class MotorCommand:
    """29-DOF motor command — direct LowCmd payload, MuJoCo joint order."""

    q_target: np.ndarray  # (29,) float32
    dq_target: np.ndarray
    kp: np.ndarray
    kd: np.ndarray
    tau_ff: np.ndarray
    mode_pr: int = 0


@dataclass
class MovementCommand:
    locomotion_mode: int = 0
    movement_direction: np.ndarray = field(
        default_factory=lambda: np.array([0.0, 0.0, 0.0], dtype=np.float64)
    )
    facing_direction: np.ndarray = field(
        default_factory=lambda: np.array([1.0, 0.0, 0.0], dtype=np.float64)
    )
    movement_speed: float = -1.0
    height: float = -1.0


@dataclass
class OperatorCommand:
    start: bool = False
    stop: bool = False
    play: bool = False
