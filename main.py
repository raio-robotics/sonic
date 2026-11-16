"""sonic — raio port of deploy.sh sim2sim loop.

Equivalent to running:
    deploy.sh sim
    → just run g1_deploy_onnx_ref lo policy/release/model_decoder.onnx reference/example/ \\
        --obs-config policy/release/observation_config.yaml \\
        --encoder-file policy/release/model_encoder.onnx \\
        --planner-file planner/target_vel/V2/planner_sonic.onnx \\
        --input-type manager --output-type all --disable-crc-check

CLI:
    python main.py                       # default: lo interface, IDLE locomotion
    python main.py --interface enP8p1s0  # talk to real robot
    python main.py --walk-after 5.0      # switch from IDLE → WALK after 5 s

Node graph (acyclic). Each node runs in its own process; the four loops run
at the C++ RT-thread rates on raio's logical clock:

    DDSReaderNode            ── low_state (physical) ──► DefaultInputNode, PlannerNode, ControlNode
                             ── imu_torso (physical) ──► ControlNode
    Keyboard                 ── key (physical) ──► ControlNode
    DefaultInputNode  100 Hz ── movement_command ──► PlannerNode, ControlNode
                             ── operator_command ──► PlannerNode, ControlNode
    PlannerNode        10 Hz ── planner_motion (100 ms delay) ──► ControlNode
    ControlNode        50 Hz ── motor_command ──► DDSWriterNode
    DDSWriterNode     500 Hz ──► /rt/lowcmd (DDS)

DDSReaderNode and Keyboard receive their values from threads through actions.
The keyboard connection is physical: each key is stamped with ControlNode's
clock when it arrives, and ControlNode never waits for a keypress. Every other
connection is logical. The DDS samples keep the timestamp DDSReaderNode gave
them, so all receivers see a sample at the same timestamp; DDSReaderNode only
waits on its actions, so after its first sample it promises never to hold its
receivers back. Over a logical connection a receiver handles a timestamp
only once its sender has finished it, so the motor command of a control tick
is published at that tick, right after ControlNode computes it.
The planner connection has a delay of one planner period, so ControlNode runs
up to 100 ms ahead of PlannerNode and a planner inference never stalls the
control loop. ControlNode plays a trajectory from the timestamp it arrives,
one frame per control period, so PlannerNode knows the frame ControlNode
tracks at any time and needs no connection back from it.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import raio

from sonic.nodes import (
    ControlNode,
    DDSReaderNode,
    DDSWriterNode,
    DefaultInputNode,
    Keyboard,
    PlannerNode,
)
from sonic.utils import params

PROJECT_ROOT = Path(__file__).resolve().parent


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="raio port of gear_sonic_deploy sim2sim loop")
    p.add_argument("--interface", default="lo", help="DDS network interface (default: lo for MuJoCo sim)")
    p.add_argument("--domain-id", type=int, default=0, help="DDS domain ID (default: 0)")
    p.add_argument(
        "--policy", default=str(PROJECT_ROOT / "checkpoints" / "model_decoder.onnx"),
        help="path to policy decoder ONNX",
    )
    p.add_argument(
        "--encoder", default=str(PROJECT_ROOT / "checkpoints" / "model_encoder.onnx"),
        help="path to encoder ONNX",
    )
    p.add_argument(
        "--planner", default=str(PROJECT_ROOT / "checkpoints" / "planner_sonic.onnx"),
        help="path to planner ONNX",
    )
    p.add_argument(
        "--obs-config", default=str(PROJECT_ROOT / "assets" / "observation_config.yaml"),
        help="path to observation config YAML",
    )
    p.add_argument(
        "--motion-data", default=str(PROJECT_ROOT / "assets" / "reference" / "example"),
        help="path to reference motion library",
    )
    p.add_argument(
        "--walk-after", type=float, default=-1.0,
        help="seconds after start to switch from IDLE to WALK (default: -1, stay IDLE)",
    )
    p.add_argument("--no-cuda", action="store_true", help="force CPU ONNX execution")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    use_cuda = not args.no_cuda

    dds_reader = DDSReaderNode(network_interface=args.interface, domain_id=args.domain_id)
    dds_writer = DDSWriterNode(network_interface=args.interface, domain_id=args.domain_id)
    inp = DefaultInputNode(walk_after_seconds=args.walk_after)
    planner = PlannerNode(planner_path=args.planner, use_cuda=use_cuda)
    control = ControlNode(
        policy_path=args.policy,
        encoder_path=args.encoder,
        obs_config_path=args.obs_config,
        motion_data_path=args.motion_data,
        use_cuda=use_cuda,
    )
    keyboard = Keyboard()

    # ----- DDS reader fan-out (logical: every receiver sees a sample at the same timestamp) -----
    raio.connect(dds_reader.low_state, control.low_state)
    raio.connect(dds_reader.low_state, planner.low_state)
    raio.connect(dds_reader.low_state, inp.low_state)
    raio.connect(dds_reader.imu_torso, control.imu_torso)

    # ----- input fan-out (logical) -----
    raio.connect(inp.movement_command, planner.movement_command)
    raio.connect(inp.movement_command, control.movement_command)
    raio.connect(inp.operator_command, planner.operator_command)
    raio.connect(inp.operator_command, control.operator_command)

    # ----- planner → control (logical, one planner period of delay) -----
    raio.connect(planner.planner_motion, control.planner_motion, delay=params.PLANNER_MOTION_DELAY)

    # ----- control → DDS writer (logical) -----
    raio.connect(control.motor_command, dds_writer.motor_command)

    # ----- keyboard shortcuts (T/N/P/Esc, physical: control never waits for a key) -----
    raio.connect(keyboard.key, control.key, physical=True)

    nodes = (dds_reader, dds_writer, inp, planner, control, keyboard)

    # Always refresh the node graph snapshot before executing.
    diagram_path = PROJECT_ROOT / "diagram.svg"
    raio.diagram(diagram_path, *nodes)
    print(f"diagram written to {diagram_path}")

    raio.run(*nodes)


if __name__ == "__main__":
    main()
