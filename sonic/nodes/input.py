"""DefaultInputNode — 100 Hz "input" thread replacement.

In the original C++, the Input thread polls a keyboard / gamepad / ZMQ /
ROS2 source for OperatorState and MovementState. We skip all interactive
inputs and emit the default sequence an operator types in the C++ sim2sim
flow once the INIT ramp has stood the robot up:

  start  = True   once INIT_RAMP_DURATION_SEC have passed since the first
                  LowState — pressing `]` (start control) and Enter (enable
                  the planner). ControlNode ramps for that long from its first
                  LowState, so the planner seeds its context from the standing
                  pose the ramp ends in.
  IDLE locomotion mode, no movement, facing forward, default height.

To exercise locomotion, switch `locomotion_mode` to LOCOMOTION_WALK after a
delay via --walk-after.
"""

from __future__ import annotations

import numpy as np
import raio
from raio import InputPort, Node, OutputPort, S, Time, task

from ..utils import RealtimeDelay, params
from ..utils.messages import LowStateSnapshot, MovementCommand, OperatorCommand


class DefaultInputNode(Node):
    low_state = InputPort[LowStateSnapshot]()

    movement_command = OutputPort[MovementCommand]()
    operator_command = OutputPort[OperatorCommand]()

    def __init__(self, walk_after_seconds: float = -1.0) -> None:
        """walk_after_seconds: if >=0, switches to WALK locomotion after this many seconds.
        Default -1 keeps the robot in IDLE mode for the whole run.
        """
        self.walk_after = walk_after_seconds
        self.first_low_state_time: Time | None = None
        self.delay = RealtimeDelay("Input 100Hz", every=100)

    @task
    async def input_loop(self) -> None:
        while True:
            await raio.wait(params.INPUT_PERIOD)
            self.delay.record()

            if self.first_low_state_time is None and self.low_state.value is not None:
                self.first_low_state_time = raio.now()
            start = (
                self.first_low_state_time is not None
                and raio.now() - self.first_low_state_time >= params.INIT_RAMP_DURATION_SEC * S
            )
            self.operator_command.set(OperatorCommand(start=start, stop=False, play=False))

            mode = params.LOCOMOTION_IDLE
            movement_dir = np.array([0.0, 0.0, 0.0])
            speed = -1.0
            if self.walk_after >= 0.0 and raio.now() / S >= self.walk_after:
                mode = params.LOCOMOTION_WALK
                movement_dir = np.array([1.0, 0.0, 0.0])
                speed = 0.6
            mc = MovementCommand(
                locomotion_mode=mode,
                movement_direction=movement_dir,
                facing_direction=np.array([1.0, 0.0, 0.0]),
                movement_speed=speed,
                height=-1.0,
            )
            self.movement_command.set(mc)
