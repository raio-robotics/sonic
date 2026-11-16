"""Smoke test: DDSReaderNode emits LowStateSnapshot + IMUTorsoSnapshot.

Run with the MuJoCo sim already publishing on /rt/lowstate /rt/secondary_imu.
Verifies that the reader's DDS subscribers fire and that raio routes the
output to a tiny downstream printer node.

Usage:
    python scripts/test_dds_reader.py                  # Linux loopback
    python scripts/test_dds_reader.py --interface lo0  # macOS loopback
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import raio
from raio import InputPort, Node, reaction

from sonic.nodes.dds_reader import DDSReaderNode
from sonic.utils.messages import IMUTorsoSnapshot, LowStateSnapshot


class StatePrinter(Node):
    low_state = InputPort[LowStateSnapshot]()
    imu_torso = InputPort[IMUTorsoSnapshot]()

    def __init__(self, target_count: int = 50) -> None:
        self.target = target_count
        self.low_state_count = 0
        self.imu_count = 0

    @reaction(low_state)
    def on_low_state(self) -> None:
        self.low_state_count += 1
        if self.low_state_count in (1, 10, 50):
            ls = self.low_state.value
            print(
                f"[Printer] low_state #{self.low_state_count} at {self.low_state.time}: "
                f"motor[0].q={ls.motor_q[0]:.3f} "
                f"quat={ls.base_quat_wxyz.round(3).tolist()} "
                f"mode_machine={ls.mode_machine}"
            )
        if self.low_state_count >= self.target and self.imu_count >= 1:
            print(f"[Printer] reached target counts; ls={self.low_state_count} imu={self.imu_count}")
            raio.request_shutdown()

    @reaction(imu_torso)
    def on_imu(self) -> None:
        self.imu_count += 1
        if self.imu_count in (1, 10, 50):
            print(f"[Printer] imu #{self.imu_count}: quat={self.imu_torso.value.quat_wxyz.round(3).tolist()}")


def main() -> None:
    p = argparse.ArgumentParser(description="Smoke test: DDSReaderNode emits LowState + IMU snapshots")
    p.add_argument("--interface", default="lo", help="DDS network interface (default: lo)")
    args = p.parse_args()

    reader = DDSReaderNode(network_interface=args.interface, domain_id=0)
    printer = StatePrinter(target_count=50)
    raio.connect(reader.low_state, printer.low_state)
    raio.connect(reader.imu_torso, printer.imu_torso)
    # The reader's values are physical events, so fast-forward changes nothing for them;
    # it keeps the test from waiting on the wall clock anywhere else.
    raio.run(reader, printer, fast_forward=True)


if __name__ == "__main__":
    main()
