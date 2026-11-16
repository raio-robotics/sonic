"""Subscribe to /rt/lowcmd and dump the first 5 messages so we can verify what's
being published over DDS while sonic main.py is running.

Usage:
    # in one shell:
    python main.py --no-policy
    # in another shell:
    python scripts/inspect_lowcmd.py
"""

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np

from unitree_sdk2py.core.channel import ChannelFactoryInitialize, ChannelSubscriber
from unitree_sdk2py.idl.unitree_hg.msg.dds_ import LowCmd_

from sonic.utils import params


def main() -> None:
    ChannelFactoryInitialize(0, "lo")

    received: list = []

    def cb(msg: LowCmd_) -> None:
        if len(received) >= 5:
            return
        snap = {
            "mode_pr": int(msg.mode_pr),
            "mode_machine": int(msg.mode_machine),
            "q": [float(msg.motor_cmd[i].q) for i in range(params.G1_NUM_MOTOR)],
            "kp": [float(msg.motor_cmd[i].kp) for i in range(params.G1_NUM_MOTOR)],
            "kd": [float(msg.motor_cmd[i].kd) for i in range(params.G1_NUM_MOTOR)],
            "mode": [int(msg.motor_cmd[i].mode) for i in range(params.G1_NUM_MOTOR)],
        }
        received.append(snap)

    sub = ChannelSubscriber("rt/lowcmd", LowCmd_)
    sub.Init(cb, 10)

    t0 = time.time()
    while len(received) < 5 and time.time() - t0 < 5.0:
        time.sleep(0.05)

    print(f"captured {len(received)} LowCmd messages")
    if not received:
        print("nothing arrived — is sonic main.py publishing on lo / domain 0?")
        return

    s = received[-1]
    np.set_printoptions(precision=4, suppress=True, linewidth=200)
    print(f"mode_pr      = {s['mode_pr']}")
    print(f"mode_machine = {s['mode_machine']}")
    print(f"mode[0..4]   = {s['mode'][:5]}  (expect 1 = enabled)")
    print(f"q  = {np.asarray(s['q'])}")
    print(f"kp = {np.asarray(s['kp'])}")
    print(f"kd = {np.asarray(s['kd'])}")
    print()
    print("expected default_angles (MuJoCo order):")
    print(f"q  = {params.default_angles}")
    print(f"kp = {params.kps}")
    print(f"kd = {params.kds}")
    print()
    same_q = np.allclose(s["q"], params.default_angles, atol=1e-3)
    same_kp = np.allclose(s["kp"], params.kps, atol=1e-3)
    same_kd = np.allclose(s["kd"], params.kds, atol=1e-3)
    print(f"match: q={same_q} kp={same_kp} kd={same_kd}")


if __name__ == "__main__":
    main()
