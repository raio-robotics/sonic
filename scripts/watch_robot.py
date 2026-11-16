"""Live sensor + command monitor for sonic.

Subscribes to /rt/lowstate (what the robot is doing) and optionally /rt/lowcmd
(what sonic is asking it to do). Prints a snapshot every ~250 ms with the
robot's tilt (deg), leg joint error, and the current command. Useful for
seeing whether the sim is already in a fallen state BEFORE sonic engages.
"""

import sys
import time
from collections import deque
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np

from unitree_sdk2py.core.channel import ChannelFactoryInitialize, ChannelSubscriber
from unitree_sdk2py.idl.unitree_hg.msg.dds_ import LowCmd_, LowState_

from sonic.utils import params

LEG_INDICES = list(range(12))  # MuJoCo idx 0..11 are the legs
NUM = params.G1_NUM_MOTOR


def main(seconds: float = 10.0) -> None:
    ChannelFactoryInitialize(0, "lo")

    state = {"ls": None, "cmd": None, "ls_count": 0, "cmd_count": 0}

    def on_state(m: LowState_) -> None:
        state["ls"] = m
        state["ls_count"] += 1

    def on_cmd(m: LowCmd_) -> None:
        state["cmd"] = m
        state["cmd_count"] += 1

    sub_state = ChannelSubscriber("rt/lowstate", LowState_)
    sub_state.Init(on_state, 5)
    sub_cmd = ChannelSubscriber("rt/lowcmd", LowCmd_)
    sub_cmd.Init(on_cmd, 5)

    print(f"{'t(s)':>5} {'ls#':>5} {'cmd#':>5} {'z':>6} {'tilt°':>6} {'leg_rms':>8} {'L_knee':>8} {'R_knee':>8} {'waist_p':>8}  {'cmd_L_knee':>10} {'cmd_R_knee':>10} {'cmd_w_p':>8}")
    t0 = time.time()
    last_print = 0.0
    while time.time() - t0 < seconds:
        time.sleep(0.05)
        now = time.time() - t0
        if now - last_print < 0.25:
            continue
        last_print = now
        ls = state["ls"]
        cmd = state["cmd"]
        if ls is None or cmd is None:
            continue

        q_actual = np.array([ls.motor_state[i].q for i in range(NUM)], dtype=np.float64)
        leg_err = q_actual[LEG_INDICES] - params.default_angles[LEG_INDICES]
        leg_rms = float(np.sqrt(np.mean(leg_err ** 2)))

        L_knee = q_actual[3] - params.default_angles[3]      # mj 3 = left_knee
        R_knee = q_actual[9] - params.default_angles[9]      # mj 9 = right_knee
        waist_p = q_actual[14] - params.default_angles[14]   # mj 14 = waist_pitch

        qw, qx, qy, qz = ls.imu_state.quaternion
        # tilt = angle of gravity vector from straight-down in body frame
        # body-frame gravity = R^T * [0,0,-1]; tilt = acos(-grav_z_body)
        grav_z_body = 1.0 - 2.0 * (qw * qw + qz * qz)  # rotmat[2,2] entry
        tilt_deg = float(np.degrees(np.arccos(np.clip(-grav_z_body, -1.0, 1.0)))) if False else float(np.degrees(np.arccos(np.clip(grav_z_body, -1.0, 1.0))))
        # qpos z not available in lowstate, fall back to omitting it (use 0)
        z = float("nan")

        cmd_q = np.array([cmd.motor_cmd[i].q for i in range(NUM)], dtype=np.float64)
        cmd_L_knee = cmd_q[3] - params.default_angles[3]
        cmd_R_knee = cmd_q[9] - params.default_angles[9]
        cmd_waist_p = cmd_q[14] - params.default_angles[14]

        print(
            f"{now:5.2f} {state['ls_count']:5d} {state['cmd_count']:5d} {z:6.2f} {tilt_deg:6.1f}"
            f" {leg_rms:8.4f} {L_knee:+8.3f} {R_knee:+8.3f} {waist_p:+8.3f} "
            f" {cmd_L_knee:+10.3f} {cmd_R_knee:+10.3f} {cmd_waist_p:+8.3f}"
        )


if __name__ == "__main__":
    main(seconds=float(sys.argv[1]) if len(sys.argv) > 1 else 10.0)
