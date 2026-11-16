"""MotionSequence dataclass and CSV motion-folder loader.

Matches motion_data_reader.hpp ReadFromCSV. Each motion clip is a folder with:
  joint_pos.csv      [T, 29]
  joint_vel.csv      [T, 29]
  body_pos.csv       [T, num_bodies*3]   (flattened xyz)
  body_quat.csv      [T, num_bodies*4]   (flattened wxyz)
  body_lin_vel.csv   [T, num_bodies*3]
  body_ang_vel.csv   [T, num_bodies*3]
  smpl_joint.csv     [T, 24*3]           (optional)
  smpl_pose.csv      [T, 21*3]           (optional)
  metadata.txt       contains "Body part indexes: [...]"
"""

from __future__ import annotations

import os
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np


@dataclass
class MotionSequence:
    name: str = ""
    timesteps: int = 0
    num_joints: int = 29
    num_bodies: int = 0
    num_body_quaternions: int = 0
    num_smpl_joints: int = 24
    num_smpl_poses: int = 21
    encode_mode: int = 0  # 0=g1, 1=teleop, 2=smpl
    body_part_indexes: list[int] = field(default_factory=list)
    joint_positions: np.ndarray = field(default_factory=lambda: np.zeros((0, 29), dtype=np.float64))
    joint_velocities: np.ndarray = field(default_factory=lambda: np.zeros((0, 29), dtype=np.float64))
    body_positions: np.ndarray = field(default_factory=lambda: np.zeros((0, 0, 3), dtype=np.float64))
    body_quaternions: np.ndarray = field(default_factory=lambda: np.zeros((0, 0, 4), dtype=np.float64))
    body_lin_velocities: np.ndarray = field(default_factory=lambda: np.zeros((0, 0, 3), dtype=np.float64))
    body_ang_velocities: np.ndarray = field(default_factory=lambda: np.zeros((0, 0, 3), dtype=np.float64))
    smpl_joints: np.ndarray = field(default_factory=lambda: np.zeros((0, 24, 3), dtype=np.float64))
    smpl_poses: np.ndarray = field(default_factory=lambda: np.zeros((0, 21, 3), dtype=np.float64))

    @staticmethod
    def empty_with_default_pose(default_height: float, num_frames: int = 512) -> "MotionSequence":
        ms = MotionSequence(name="<planner_buffer>")
        ms.num_joints = 29
        ms.num_bodies = 1
        ms.num_body_quaternions = 1
        ms.timesteps = 0
        ms.joint_positions = np.zeros((num_frames, 29), dtype=np.float64)
        ms.joint_velocities = np.zeros((num_frames, 29), dtype=np.float64)
        ms.body_positions = np.zeros((num_frames, 1, 3), dtype=np.float64)
        ms.body_positions[:, 0, 2] = default_height
        ms.body_quaternions = np.zeros((num_frames, 1, 4), dtype=np.float64)
        ms.body_quaternions[:, 0, 0] = 1.0
        return ms

    def jp(self, frame: int) -> np.ndarray:
        f = min(max(frame, 0), self.timesteps - 1)
        return self.joint_positions[f]

    def jv(self, frame: int) -> np.ndarray:
        f = min(max(frame, 0), self.timesteps - 1)
        return self.joint_velocities[f]

    def bq(self, frame: int, body: int = 0) -> np.ndarray:
        f = min(max(frame, 0), self.timesteps - 1)
        return self.body_quaternions[f, body]

    def bp(self, frame: int, body: int = 0) -> np.ndarray:
        f = min(max(frame, 0), self.timesteps - 1)
        return self.body_positions[f, body]


def _read_csv(path: Path) -> Optional[np.ndarray]:
    if not path.is_file():
        return None
    return np.loadtxt(path, delimiter=",", skiprows=1, dtype=np.float64, ndmin=2)


def _parse_metadata(path: Path) -> list[int]:
    if not path.is_file():
        return []
    text = path.read_text()
    m = re.search(r"Body part indexes:\s*\[([^\]]*)\]", text)
    if not m:
        return []
    return [int(x) for x in re.findall(r"-?\d+", m.group(1))]


def load_motion_folder(folder: Path, encode_mode: int = 0) -> Optional[MotionSequence]:
    jp = _read_csv(folder / "joint_pos.csv")
    if jp is None:
        return None
    jv = _read_csv(folder / "joint_vel.csv")
    bp = _read_csv(folder / "body_pos.csv")
    bq = _read_csv(folder / "body_quat.csv")
    blv = _read_csv(folder / "body_lin_vel.csv")
    bav = _read_csv(folder / "body_ang_vel.csv")
    smpl_j = _read_csv(folder / "smpl_joint.csv")
    smpl_p = _read_csv(folder / "smpl_pose.csv")

    T, num_joints = jp.shape
    ms = MotionSequence(name=folder.name, timesteps=T, num_joints=num_joints, encode_mode=encode_mode)
    ms.joint_positions = jp
    ms.joint_velocities = jv if jv is not None and jv.shape[0] == T else np.zeros_like(jp)

    if bp is not None and bp.shape[0] == T and bp.shape[1] % 3 == 0:
        nb = bp.shape[1] // 3
        ms.num_bodies = nb
        ms.body_positions = bp.reshape(T, nb, 3)
    if bq is not None and bq.shape[0] == T and bq.shape[1] % 4 == 0:
        nbq = bq.shape[1] // 4
        ms.num_body_quaternions = nbq
        ms.body_quaternions = bq.reshape(T, nbq, 4)
    if blv is not None and blv.shape[0] == T and ms.num_bodies > 0:
        ms.body_lin_velocities = blv.reshape(T, ms.num_bodies, 3)
    if bav is not None and bav.shape[0] == T and ms.num_bodies > 0:
        ms.body_ang_velocities = bav.reshape(T, ms.num_bodies, 3)
    if smpl_j is not None and smpl_j.shape[0] == T and smpl_j.shape[1] % 3 == 0:
        nj = smpl_j.shape[1] // 3
        ms.num_smpl_joints = nj
        ms.smpl_joints = smpl_j.reshape(T, nj, 3)
    if smpl_p is not None and smpl_p.shape[0] == T and smpl_p.shape[1] % 3 == 0:
        npose = smpl_p.shape[1] // 3
        ms.num_smpl_poses = npose
        ms.smpl_poses = smpl_p.reshape(T, npose, 3)

    ms.body_part_indexes = _parse_metadata(folder / "metadata.txt")
    return ms


def load_motion_library(base_dir: Path, encode_mode: int = 0) -> list[MotionSequence]:
    base_dir = Path(base_dir)
    motions: list[MotionSequence] = []
    if not base_dir.is_dir():
        raise FileNotFoundError(f"motion data path not found: {base_dir}")
    for entry in sorted(os.listdir(base_dir)):
        if entry.startswith(".") or entry == "motion_summary.txt":
            continue
        sub = base_dir / entry
        if not sub.is_dir():
            continue
        try:
            m = load_motion_folder(sub, encode_mode=encode_mode)
        except Exception as exc:
            print(f"[motion] failed to load {sub}: {exc}")
            continue
        if m is not None and m.timesteps > 0:
            motions.append(m)
    if not motions:
        raise RuntimeError(f"no usable motions in {base_dir}")
    return motions
