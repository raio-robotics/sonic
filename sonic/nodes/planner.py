"""PlannerNode — 10 Hz kinematic planner.

Owns a `KinematicPlanner` (planner_sonic.onnx, V2) and the 50 Hz trajectory that
ControlNode tracks. Mirrors the Planner thread (g1_deploy_onnx_ref.cpp:3595-3814)
together with the planner branch of CurrentFrameAdvancement (:3184-3333).

The planner starts with control (OperatorCommand.start), after the INIT ramp has
stood the robot up: it seeds its 4-frame context from the robot's joint positions
and plans an IDLE trajectory.

ControlNode plays a trajectory from the timestamp it arrives, one frame per control
period, holding the last frame at the end. The connection delays every trajectory by
exactly PLANNER_MOTION_DELAY, so this node knows which frame ControlNode tracks at any
time without hearing back from it. On a replan it plans from the tracked frame
(gen_frame) and cross-fades the plan into the trajectory at the frame ControlNode
tracks when the plan arrives, as the C++ control thread does.

Replans follow the C++: on a change of locomotion mode, facing direction or height;
and in a moving mode on a change of speed or direction, or periodically while the
speed is non-zero. IDLE never replans on its own, so a standing robot plays its
initial trajectory and holds its last frame.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import raio
from raio import InputPort, Node, OutputPort, S, Time, task

from ..utils import RealtimeDelay, params
from ..utils.inference import PlannerEngine
from ..utils.messages import LowStateSnapshot, MovementCommand, OperatorCommand
from ..utils.motion import MotionSequence
from ..utils.planner_runtime import KinematicPlanner, blend_trajectory


class PlannerNode(Node):
    low_state = InputPort[LowStateSnapshot]()
    operator_command = InputPort[OperatorCommand]()
    movement_command = InputPort[MovementCommand]()

    planner_motion = OutputPort[MotionSequence]()

    # The trajectory ControlNode tracks, set by send() from the planner's first plan on.
    trajectory: MotionSequence

    def __init__(self, planner_path: str, use_cuda: bool = True) -> None:
        self.planner_path = Path(planner_path)
        self.use_cuda = use_cuda
        self.trajectory_start = Time.ZERO  # the timestamp at which `trajectory` reaches ControlNode
        # The command of the last plan; the C++ starts from the default IDLE command.
        self.last_command = MovementCommand()
        self.replan_counter: float = 0.0
        self.delay = RealtimeDelay("Planner 10Hz", every=10)

    def __setstate__(self, state: dict) -> None:
        """Rebuild the node in its own process and load the planner model there.

        raio unpickles each node in its own process before it wires the app, so the
        model is loaded before any node's clock starts: no tick waits for it, and
        loading is not bound by the 5 s heartbeat deadline of a running node. ONNX
        sessions cannot be pickled, so the launcher never builds one.
        """
        self.__dict__.update(state)
        self.planner = KinematicPlanner(PlannerEngine(self.planner_path, use_cuda=self.use_cuda))
        self.planner.enabled = True  # default-mode equivalent of input setting planner_state_.enabled = True
        print(f"[Planner] loaded {self.planner_path.name}")

    # ---------------------------------------------------------------- loop

    @task
    async def plan_loop(self) -> None:
        while True:
            await raio.wait(params.PLANNER_PERIOD)
            self.step()

    def step(self) -> None:
        self.delay.record()
        ls = self.low_state.value
        op = self.operator_command.value
        if ls is None or op is None or not op.start:
            return  # the planner starts with control

        if not self.planner.initialized:
            if not self.planner.initialize(ls.base_quat_wxyz, ls.motor_q):
                print("[Planner] WARNING: initialize() failed; will retry next tick")
                return
            print("[Planner] initialised")
            self.send(self.planner.latest_plan())
            return

        cmd = self.movement_command.value
        last = self.last_command
        self.replan_counter += params.PLANNER_PERIOD / S
        interval = params.REPLAN_INTERVAL_RUN_SEC if cmd.locomotion_mode == params.LOCOMOTION_RUN else params.REPLAN_INTERVAL_SEC
        time_to_replan = self.replan_counter >= interval
        if time_to_replan:
            self.replan_counter = 0.0
        static_mode = cmd.locomotion_mode == params.LOCOMOTION_IDLE
        if (
            cmd.locomotion_mode != last.locomotion_mode
            or not np.array_equal(cmd.facing_direction, last.facing_direction)
            or cmd.height != last.height
        ):
            need_replan = True
        else:
            need_replan = not static_mode and (
                cmd.movement_speed != last.movement_speed
                or not np.array_equal(cmd.movement_direction, last.movement_direction)
                or (time_to_replan and cmd.movement_speed != 0)
            )
        if not need_replan:
            return
        self.last_command = cmd

        # Plan from the frame ControlNode tracks now, as the C++ passes current_frame_.
        ok = self.planner.update_planning(
            gen_frame=self.frame_at(raio.now()),
            current_motion=self.trajectory,
            mode=cmd.locomotion_mode,
            target_vel=float(cmd.movement_speed),
            target_height=float(cmd.height),
            movement_direction=np.asarray(cmd.movement_direction, dtype=np.float32),
            facing_direction=np.asarray(cmd.facing_direction, dtype=np.float32),
        )
        if not ok:
            print("[Planner] WARNING: update_planning() failed; keeping the current trajectory")
            return
        # Join the plan at the frame ControlNode tracks when the plan arrives.
        arrival_frame = self.frame_at(raio.now() + params.PLANNER_MOTION_DELAY)
        self.send(blend_trajectory(self.trajectory, arrival_frame, self.planner.latest_plan(), self.planner.gen_frame))

    # ---------------------------------------------------------------- helpers

    def frame_at(self, t: Time) -> int:
        """The trajectory frame ControlNode tracks at timestamp `t`."""
        return min((t - self.trajectory_start) // params.CONTROL_PERIOD, self.trajectory.timesteps - 1)

    def send(self, trajectory: MotionSequence) -> None:
        self.trajectory = trajectory
        self.trajectory_start = raio.now() + params.PLANNER_MOTION_DELAY
        self.planner_motion.set(trajectory)
