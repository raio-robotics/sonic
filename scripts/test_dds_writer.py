"""Smoke test: DDSWriterNode publishes /rt/lowcmd at 500 Hz.

Wires a tiny CommandSource (50 Hz) that emits a fixed MotorCommand (hold default
pose) into DDSWriterNode (500 Hz loop). A Python subscriber in a background
thread of this process counts /rt/lowcmd messages — pass criterion is ≥ ~50
LowCmd in total (sim only reads at its own rate; we publish at 500 Hz).

Usage:
    python scripts/test_dds_writer.py                  # Linux loopback
    python scripts/test_dds_writer.py --interface lo0  # macOS loopback
"""

import argparse
import sys
import threading
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import raio
from raio import Node, OutputPort, task
from unitree_sdk2py.core.channel import ChannelFactoryInitialize, ChannelSubscriber
from unitree_sdk2py.idl.unitree_hg.msg.dds_ import LowCmd_

from sonic.nodes.dds_writer import DDSWriterNode
from sonic.utils import params
from sonic.utils.messages import MotorCommand


class CommandSource(Node):
    """Emits a constant MotorCommand (hold default pose) at the control rate."""

    motor_command = OutputPort[MotorCommand]()

    def __init__(self, target_emit_count: int = 500) -> None:
        self.target = target_emit_count

    @task
    async def emit(self) -> None:
        for count in range(1, self.target + 1):
            await raio.wait(params.CONTROL_PERIOD)
            cmd = MotorCommand(
                q_target=params.default_angles.astype(np.float32),
                dq_target=np.zeros(params.G1_NUM_MOTOR, dtype=np.float32),
                kp=params.kps.copy(),
                kd=params.kds.copy(),
                tau_ff=np.zeros(params.G1_NUM_MOTOR, dtype=np.float32),
                mode_pr=params.MODE_PR,
            )
            self.motor_command.set(cmd)
            if count in (1, 100, 500):
                print(f"[Source] emitted MotorCommand count={count}")
        print("[Source] reached target — shutting down")
        raio.request_shutdown()


def subscribe_lowcmd(interface: str, counters: dict, stop: threading.Event) -> None:
    """Background-thread subscriber that counts /rt/lowcmd messages."""
    ChannelFactoryInitialize(0, interface)

    def on_lowcmd(msg: LowCmd_) -> None:
        counters["n"] += 1
        if counters["n"] == 1:
            counters["first_q"] = float(msg.motor_cmd[0].q)
            counters["first_kp"] = float(msg.motor_cmd[0].kp)

    sub = ChannelSubscriber(params.HG_CMD_TOPIC, LowCmd_)
    sub.Init(on_lowcmd, 50)
    while not stop.is_set():
        time.sleep(0.05)


def main() -> None:
    p = argparse.ArgumentParser(description="Smoke test: DDSWriterNode publishes /rt/lowcmd at 500 Hz")
    p.add_argument("--interface", default="lo", help="DDS network interface (default: lo)")
    args = p.parse_args()

    # Spin up a subscriber FIRST so we don't miss the early LowCmds.
    counters = {"n": 0, "first_q": None, "first_kp": None}
    stop = threading.Event()
    sub_thread = threading.Thread(target=subscribe_lowcmd, args=(args.interface, counters, stop), daemon=True)
    sub_thread.start()
    time.sleep(0.5)  # let cyclonedds discover the publisher when it appears

    source = CommandSource(target_emit_count=200)
    writer = DDSWriterNode(network_interface=args.interface, domain_id=0)
    raio.connect(source.motor_command, writer.motor_command)
    raio.run(source, writer, fast_forward=True)  # don't gate the test on wall-clock

    stop.set()
    sub_thread.join(timeout=1.0)
    print(
        f"[verify] received {counters['n']} rt/lowcmd messages, "
        f"first_q={counters['first_q']}, first_kp={counters['first_kp']}"
    )
    print(f"RESULT: lowcmd_count={counters['n']}")


if __name__ == "__main__":
    main()
