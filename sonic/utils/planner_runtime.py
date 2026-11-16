"""Kinematic-planner runtime — Python port of LocalMotionPlannerTensorRT.

Responsibilities (matches localmotion_kplanner.hpp):
  - initialize() : seed the 4-frame context (root at default height, identity yaw,
                   current joint positions) and run first inference.
  - update_planning() : replan using context derived from `current_motion` at
                        gen_frame + look_ahead_steps.
  - resample_30hz_to_50hz() : lerp pos/joints, slerp base quat, central-diff vel.

blend_trajectory() ports the planner branch of CurrentFrameAdvancement
(g1_deploy_onnx_ref.cpp:3184-3333): how a new plan joins the trajectory being tracked.
"""

from __future__ import annotations

import numpy as np

from . import math_utils, params
from .inference import PlannerEngine
from .motion import MotionSequence


class KinematicPlanner:
    def __init__(self, engine: PlannerEngine) -> None:
        self.engine = engine
        self.gen_frame: int = 0
        self.motion_available: bool = False
        self.initialized: bool = False
        self.enabled: bool = False

        self._context = np.zeros((4, 36), dtype=np.float32)
        self._context[:, 2] = params.DEFAULT_PLANNER_HEIGHT
        self._context[:, 3] = 1.0  # wxyz identity (w)

        self.planner_motion_50hz = MotionSequence.empty_with_default_pose(
            params.DEFAULT_PLANNER_HEIGHT, num_frames=512
        )

        self.current_random_seed = params.DEFAULT_PLANNER_SEED

    def _seed_context_from_robot(self, base_quat_wxyz: np.ndarray, joint_positions_mujoco: np.ndarray) -> None:
        # All 4 context frames identical, zero-yaw frame. Robot joint angles
        # arrive in MuJoCo order — the planner ONNX consumes MuJoCo order in
        # slots 7..35 of context_qpos (matches localmotion_kplanner.hpp:621
        # which writes joint_positions[i] directly without remap).
        for n in range(4):
            self._context[n, 0:3] = [0.0, 0.0, params.DEFAULT_PLANNER_HEIGHT]
            self._context[n, 3:7] = [1.0, 0.0, 0.0, 0.0]
            self._context[n, 7:36] = joint_positions_mujoco

    def _update_context_from_motion(self, motion: MotionSequence) -> None:
        # localmotion_kplanner.hpp:628-678: sample 4 frames at 30 Hz starting at
        # gen_time = gen_frame_ / 50.0 from the current motion (50 Hz).
        # MotionSequence stores joint data in IsaacLab order; the planner
        # context_qpos slots 7..35 want MuJoCo order, so we remap iso → mujoco.
        # Matches `context_qpos_values[7 + mujoco_to_isaaclab[i]] = JointPositions[i]`
        # at localmotion_kplanner.hpp:675.
        if motion.timesteps == 0:
            return
        gen_time = self.gen_frame / 50.0
        for n in range(4):
            t = gen_time + n / 30.0
            f50 = t * 50.0
            f0 = min(int(np.floor(f50)), motion.timesteps - 1)
            f1 = min(f0 + 1, motion.timesteps - 1)
            w0 = 1.0 - (f50 - f0)
            w1 = 1.0 - w0
            p = w0 * motion.bp(f0, 0) + w1 * motion.bp(f1, 0)
            self._context[n, 0:3] = p.astype(np.float32)
            q = math_utils.quat_slerp(motion.bq(f0, 0), motion.bq(f1, 0), f50 - f0)
            self._context[n, 3:7] = q.astype(np.float32)
            jp_iso = w0 * motion.jp(f0) + w1 * motion.jp(f1)
            self._context[n, 7:36] = jp_iso[params.isaaclab_to_mujoco].astype(np.float32)

    def _resample_30hz_to_50hz(self, qpos_30hz: np.ndarray, num_pred_frames: int) -> None:
        if num_pred_frames <= 0 or num_pred_frames > 64:
            self.motion_available = False
            return
        valid = qpos_30hz[:num_pred_frames]
        if np.any(np.isnan(valid)):
            self.motion_available = False
            return
        n50 = int(np.floor(num_pred_frames / 30.0 * 50.0))
        self.planner_motion_50hz.timesteps = n50
        for f50 in range(n50):
            t = f50 / 50.0
            f30 = t * 30.0
            f0 = int(np.floor(f30))
            f1 = min(f0 + 1, num_pred_frames - 1)
            w0 = 1.0 - (f30 - f0)
            w1 = 1.0 - w0
            self.planner_motion_50hz.body_positions[f50, 0] = w0 * valid[f0, 0:3] + w1 * valid[f1, 0:3]
            q0 = valid[f0, 3:7].astype(np.float64)
            q1 = valid[f1, 3:7].astype(np.float64)
            self.planner_motion_50hz.body_quaternions[f50, 0] = math_utils.quat_slerp(q0, q1, f30 - f0)
            # Planner output slots 7..35 are MuJoCo-order qpos; MotionSequence
            # stores IsaacLab order (matches localmotion_kplanner.hpp:494-496
            # which writes mujoco_qpos[7 + mujoco_to_isaaclab[joint]] into
            # JointPositions[joint]).
            jp_mujoco = w0 * valid[f0, 7:36] + w1 * valid[f1, 7:36]
            self.planner_motion_50hz.joint_positions[f50] = jp_mujoco[params.mujoco_to_isaaclab]
        for fr in range(n50 - 1):
            self.planner_motion_50hz.joint_velocities[fr] = (
                self.planner_motion_50hz.joint_positions[fr + 1]
                - self.planner_motion_50hz.joint_positions[fr]
            ) * 50.0
        if n50 >= 2:
            self.planner_motion_50hz.joint_velocities[n50 - 1] = self.planner_motion_50hz.joint_velocities[n50 - 2]
        self.motion_available = True

    def initialize(self, base_quat_wxyz: np.ndarray, joint_positions_mujoco: np.ndarray) -> bool:
        self.initialized = False
        self._seed_context_from_robot(base_quat_wxyz, joint_positions_mujoco)
        self.gen_frame = 0
        qpos, n = self.engine.run(
            context_qpos=self._context,
            mode=params.LOCOMOTION_IDLE,
            target_vel=-1.0,
            height=-1.0,
            movement_direction=np.zeros(3, dtype=np.float32),
            facing_direction=np.array([1.0, 0.0, 0.0], dtype=np.float32),
            random_seed=self.current_random_seed,
        )
        self._resample_30hz_to_50hz(qpos, n)
        if not self.motion_available:
            return False
        self.initialized = True
        return True

    def update_planning(
        self,
        gen_frame: int,
        current_motion: MotionSequence,
        mode: int,
        target_vel: float,
        target_height: float,
        movement_direction: np.ndarray,
        facing_direction: np.ndarray,
        random_seed: int = -1,
    ) -> bool:
        if not self.initialized or not self.enabled:
            return False
        self.gen_frame = gen_frame + params.DEFAULT_PLANNER_LOOK_AHEAD_STEPS
        self._update_context_from_motion(current_motion)
        if random_seed >= 0:
            self.current_random_seed = random_seed
        qpos, n = self.engine.run(
            context_qpos=self._context,
            mode=mode,
            target_vel=target_vel,
            height=target_height,
            movement_direction=movement_direction.astype(np.float32),
            facing_direction=facing_direction.astype(np.float32),
            random_seed=self.current_random_seed,
        )
        self._resample_30hz_to_50hz(qpos, n)
        return self.motion_available

    def latest_plan(self) -> MotionSequence:
        """A copy of the last plan, trimmed to its frames (planner_motion_50hz is a reused buffer)."""
        m = self.planner_motion_50hz
        T = m.timesteps
        return MotionSequence(
            name="<planner>",
            timesteps=T,
            num_bodies=1,
            num_body_quaternions=1,
            joint_positions=m.joint_positions[:T].copy(),
            joint_velocities=m.joint_velocities[:T].copy(),
            body_positions=m.body_positions[:T].copy(),
            body_quaternions=m.body_quaternions[:T].copy(),
        )


def blend_trajectory(
    old: MotionSequence, current_frame: int, new: MotionSequence, gen_frame: int
) -> MotionSequence:
    """Continue `old` from `current_frame` into `new`, a plan whose frame 0 follows old frame `gen_frame`.

    Frame f of the result is old frame current_frame + f, cross-faded over
    PLANNER_BLEND_FRAMES frames into new frame current_frame + f - gen_frame
    (g1_deploy_onnx_ref.cpp:3226-3295). The cross-fade starts at once if the plan
    arrives after its gen_frame, as it does here. The result's frame 0 is the frame
    tracked when it takes over, so the motion carries on without a jump.
    """
    length = gen_frame - current_frame + new.timesteps
    assert length > 0, f"plan for frame {gen_frame} ends before frame {current_frame}"
    f = np.arange(length)
    f_old = np.clip(f + current_frame, 0, old.timesteps - 1)
    f_new = np.clip(f + current_frame - gen_frame, 0, new.timesteps - 1)
    w_new = np.clip((f - max(0, gen_frame - current_frame)) / params.PLANNER_BLEND_FRAMES, 0.0, 1.0)
    w_old = 1.0 - w_new
    return MotionSequence(
        name="<planner>",
        timesteps=length,
        num_bodies=1,
        num_body_quaternions=1,
        joint_positions=w_old[:, None] * old.joint_positions[f_old] + w_new[:, None] * new.joint_positions[f_new],
        joint_velocities=w_old[:, None] * old.joint_velocities[f_old] + w_new[:, None] * new.joint_velocities[f_new],
        body_positions=w_old[:, None, None] * old.body_positions[f_old] + w_new[:, None, None] * new.body_positions[f_new],
        body_quaternions=np.array(
            [[math_utils.quat_slerp(old.bq(a), new.bq(b), w)] for a, b, w in zip(f_old, f_new, w_new)]
        ),
    )
