"""ControlNode — 50 Hz control loop.

Mirrors G1Deploy::Control in g1_deploy_onnx_ref.cpp:3834-4079. The 50 Hz
inference runs in a 20 ms loop on raio's logical clock.

Keyboard shortcuts (when a Keyboard node is wired to the `key` input port):
  T  toggle play  — start/stop playing the selected reference motion
  N  next motion  — select the next motion in the library
  P  prev motion  — select the previous motion in the library
  Esc  request shutdown

State machine:
  INIT             linear-interpolate motor q from initial reading to default_angles
                   over 3 s, then transition to WAIT_FOR_CONTROL.
  WAIT_FOR_CONTROL hold default pose until OperatorCommand.start == True and the
                   planner's first trajectory has arrived.
  CONTROL          per tick: gather observation → encoder → policy → motor command.
                   The node plays the planner trajectory, as the C++ does in planner
                   mode; after T it plays the selected reference motion once and then
                   returns to the planner trajectory.

Inputs:
  key                                   physical, from Keyboard: never waited for.
  low_state, imu_torso                  logical, from DDSReaderNode, read as their latest value.
  movement_command, operator_command,   logical, from DefaultInputNode and
  planner_motion                        PlannerNode.

Outputs:
  motor_command      29-DOF LowCmd payload, MuJoCo joint order.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import raio
from raio import InputPort, Node, OutputPort, S, reaction, task

from ..utils import RealtimeDelay, math_utils, params
from ..utils.inference import EncoderEngine, PolicyEngine
from ..utils.messages import (
    IMUTorsoSnapshot,
    LowStateSnapshot,
    MotorCommand,
    MovementCommand,
    OperatorCommand,
)
from ..utils.motion import MotionSequence, load_motion_library
from ..utils.obs import EncoderObsBuilder, PolicyObsBuilder, StateHistory, StateSample, parse_observation_config


class ControlNode(Node):
    low_state = InputPort[LowStateSnapshot]()
    imu_torso = InputPort[IMUTorsoSnapshot]()
    movement_command = InputPort[MovementCommand]()
    operator_command = InputPort[OperatorCommand]()
    planner_motion = InputPort[MotionSequence]()
    key = InputPort[str]()

    motor_command = OutputPort[MotorCommand]()

    def __init__(
        self,
        policy_path: str,
        encoder_path: str,
        obs_config_path: str,
        motion_data_path: str,
        use_cuda: bool = True,
    ) -> None:
        self.policy_path = Path(policy_path)
        self.encoder_path = Path(encoder_path)
        self.obs_config_path = obs_config_path
        self.motion_data_path = Path(motion_data_path)
        self.use_cuda = use_cuda

        self.policy_obs = PolicyObsBuilder()
        self.encoder_obs = EncoderObsBuilder()
        self.history = StateHistory(capacity=64)

        self.program_state: str = "INIT"
        self.init_time: float = 0.0
        self.init_pose_start: np.ndarray | None = None  # initial measured q at first valid LowState

        # Last action stored in IsaacLab order, raw (pre-scale) — fed back to obs.
        self.last_action_raw = np.zeros(params.G1_NUM_MOTOR, dtype=np.float64)
        self.last_token_state = np.zeros(64, dtype=np.float64)
        self.policy_step_count: int = 0

        # Keyboard-driven playback state. Press T to toggle `play`; N/P to
        # select a motion. Mirrors the C++ SimpleKeyboard reference-motion mode
        # (keyboard_handler.hpp:69-77, 380-440).
        self.motion_index: int = 0
        self.play: bool = False  # True while the selected reference motion plays
        self.reference_frame: int = 0

        # Heading alignment between motion data and the robot's current pose, as
        # HeadingState in the C++: init_base_quat is the robot's orientation when
        # the heading was last reinitialised, init_motion_quat the tracked motion's
        # root orientation then. See UpdateHeadingState / ComputeApplyDeltaHeading
        # in g1_deploy_onnx_ref.cpp:559-606.
        self.init_base_quat = np.array([1.0, 0.0, 0.0, 0.0])
        self.init_motion_quat = np.array([1.0, 0.0, 0.0, 0.0])
        self.reinitialize_heading: bool = True  # the first CONTROL tick anchors the heading

        self.delay = RealtimeDelay("Control 50Hz", every=50)

    def __setstate__(self, state: dict) -> None:
        """Rebuild the node in its own process and load the models and motions there.

        raio unpickles each node in its own process before it wires the app, so all
        of this is loaded before any node's clock starts: no tick waits for it, and
        loading is not bound by the 5 s heartbeat deadline of a running node. ONNX
        sessions cannot be pickled, so the launcher never builds one.
        """
        self.__dict__.update(state)
        cfg = parse_observation_config(self.obs_config_path)
        print(f"[Control] loaded obs config: {len(cfg.get('observations', []))} entries")
        self.policy = PolicyEngine(self.policy_path, use_cuda=self.use_cuda)
        assert self.policy.input_dim == 994, f"policy input dim {self.policy.input_dim} != 994"
        assert self.policy.output_dim == params.G1_NUM_MOTOR, f"policy out {self.policy.output_dim}"

        self.encoder = EncoderEngine(self.encoder_path, use_cuda=self.use_cuda)
        assert self.encoder.input_dim == 1762, f"encoder input dim {self.encoder.input_dim} != 1762"
        assert self.encoder.output_dim == 64, f"encoder out {self.encoder.output_dim}"

        self.motions = load_motion_library(self.motion_data_path, encode_mode=0)
        self.current_motion = self.motions[self.motion_index]
        print(
            f"[Control] loaded {len(self.motions)} motions, "
            f"using {self.current_motion.name} ({self.current_motion.timesteps} frames)"
        )

    # ---------------------------------------------------------------- reactions

    @reaction(key)
    def on_key(self) -> None:
        """Handle a single keystroke from the Keyboard node."""
        ch = self.key.value
        if ch in ("t", "T"):
            # Toggle: first T plays the selected motion from frame 0; pressing T
            # again while it plays stops it and returns to the planner trajectory.
            # The end of the motion in step() also clears self.play.
            self.play = not self.play
            self.reference_frame = 0
            self.reinitialize_heading = True
            if self.play:
                # Print the full motion library, marking the one about to play.
                names = "\n".join(
                    f"  {'>' if i == self.motion_index else ' '} [{i}] {m.name}"
                    for i, m in enumerate(self.motions)
                )
                print(
                    f"[Keyboard] T → play motion={self.current_motion.name}\n"
                    f"motions ({len(self.motions)}):\n{names}"
                )
            else:
                print("[Keyboard] T → stop")
        elif ch in ("n", "N", "p", "P"):
            # N/P only select which reference motion T plays next; the robot keeps
            # tracking the planner trajectory meanwhile.
            step = 1 if ch in ("n", "N") else -1
            self.motion_index = (self.motion_index + step) % len(self.motions)
            self.current_motion = self.motions[self.motion_index]
            self.reference_frame = 0
            if self.play:
                self.play = False
                self.reinitialize_heading = True
            print(f"[Keyboard] motion → [{self.motion_index}] {self.current_motion.name} (press T to play)")
        elif ch == "\x1b":  # Esc
            print("[Keyboard] Esc → shutdown")
            raio.request_shutdown()

    # ---------------------------------------------------------------- loop

    @task
    async def control_loop(self) -> None:
        while True:
            await raio.wait(params.CONTROL_PERIOD)
            self.step()

    def step(self) -> None:
        self.delay.record()
        ls = self.low_state.value
        if ls is None:
            # Cannot do anything until we've seen a LowState. Stay silent.
            return

        # ----- INIT: ramp from initial q to default_angles over 3 s.
        if self.program_state == "INIT":
            if self.init_pose_start is None:
                self.init_pose_start = ls.motor_q.copy()
            self.init_time += params.CONTROL_PERIOD / S
            r = min(self.init_time / params.INIT_RAMP_DURATION_SEC, 1.0)
            q_target = (1.0 - r) * self.init_pose_start + r * params.default_angles
            self.send_motor_command(q_target)
            if r >= 1.0:
                self.program_state = "WAIT_FOR_CONTROL"
                print("[Control] INIT ramp done → WAIT_FOR_CONTROL")
            return

        # ----- WAIT_FOR_CONTROL: hold default pose, wait for operator.start
        # AND for the planner's first trajectory, which control starts tracking.
        if self.program_state == "WAIT_FOR_CONTROL":
            op = self.operator_command.value
            self.send_motor_command(params.default_angles)
            if op is None or not op.start:
                return
            if self.planner_motion.value is None:
                return  # planner trajectory not ready — keep holding default.
            self.program_state = "CONTROL"
            print("[Control] operator.start=True and planner ready → CONTROL")
            return

        # ----- CONTROL: gather obs, run policy, emit motor command.
        # 1) Update state history (proprio for policy obs).
        # ls.motor_q / motor_dq are MuJoCo-order (hardware index).
        # Convert to IsaacLab layout — matches C++ body_q[i] =
        # unitree_joint_state[mujoco_to_isaaclab[i]].q() - default_angles[mujoco_to_isaaclab[i]]
        # at g1_deploy_onnx_ref.cpp:2861.
        q_iso = (ls.motor_q - params.default_angles)[params.mujoco_to_isaaclab]
        dq_iso = ls.motor_dq[params.mujoco_to_isaaclab]
        self.history.push(
            StateSample(
                base_quat=ls.base_quat_wxyz,
                base_ang_vel=ls.base_ang_vel,
                joint_pos_isaaclab=q_iso,
                joint_vel_isaaclab=dq_iso,
                last_action=self.last_action_raw,
            )
        )

        # 2) Choose the motion and frame to play.
        # - After T (self.play) the selected reference motion, from frame 0.
        # - Otherwise the planner trajectory, which plays from the timestamp it
        #   arrived, one frame per control period, holding its last frame
        #   (CurrentFrameAdvancement, g1_deploy_onnx_ref.cpp:3335-3404).
        if self.play:
            motion, frame = self.current_motion, self.reference_frame
        else:
            motion = self.planner_motion.value
            elapsed = (raio.now() - self.planner_motion.time) // params.CONTROL_PERIOD
            frame = min(elapsed, motion.timesteps - 1)

        # 3) Heading alignment (UpdateHeadingState, g1_deploy_onnx_ref.cpp:559-585).
        # apply_delta_heading rotates the motion into the robot's heading frame.
        # Without it the motion's "forward" is wherever the recording happened
        # to face, so the robot would move in a direction unrelated to where it
        # is standing. The motion's anchor follows it to every frame 0, so each
        # new planner trajectory is aligned from where it starts.
        if self.reinitialize_heading:
            self.init_base_quat = np.asarray(ls.base_quat_wxyz, dtype=np.float64).copy()
            self.init_motion_quat = np.asarray(motion.bq(frame), dtype=np.float64).copy()
            self.reinitialize_heading = False
        if frame == 0:
            self.init_motion_quat = np.asarray(motion.bq(0), dtype=np.float64).copy()
        apply_delta_heading = math_utils.quat_mul(
            math_utils.calc_heading_quat(self.init_base_quat),
            math_utils.calc_heading_quat_inv(self.init_motion_quat),
        )

        # 4) Encoder forward — token_state (64 floats).
        enc_in = self.encoder_obs.build(
            motion,
            frame,
            base_quat_wxyz=ls.base_quat_wxyz,
            apply_delta_heading=apply_delta_heading,
        )
        token_state = self.encoder.encode(enc_in)
        self.last_token_state = token_state.astype(np.float64)

        # 5) Policy forward — 29-d action in IsaacLab order.
        policy_in = self.policy_obs.build(self.last_token_state, self.history)
        action_iso = self.policy.infer(policy_in).astype(np.float64)
        self.last_action_raw = action_iso

        self.policy_step_count += 1
        # Dump full obs slices on the first tick so we can sanity-check the
        # observation values the policy receives.
        if self.policy_step_count == 1:
            last_ang_vel = policy_in[64 + 27 : 64 + 30]
            last_jp = policy_in[94 + 29 * 9 : 94 + 29 * 10]
            last_jv = policy_in[384 + 29 * 9 : 384 + 29 * 10]
            last_act = policy_in[674 + 29 * 9 : 674 + 29 * 10]
            last_grav = policy_in[964 + 27 : 964 + 30]
            print(f"[Control] first obs token_state[0:6]={policy_in[0:6].round(3)}")
            print(f"[Control] first obs last ang_vel={last_ang_vel.round(3)}")
            print(f"[Control] first obs last gravity_dir={last_grav.round(3)} (expect ~[0,0,-1] upright)")
            print(f"[Control] first obs last jp(iso, q-default)={last_jp.round(3)}")
            print(f"[Control] first obs last jv(iso)={last_jv.round(3)}")
            print(f"[Control] first obs last_action(iso)={last_act.round(3)}  (should be 0 on tick 1)")

        # 6) Convert action (IsaacLab) → motor target (MuJoCo).
        # Matches C++ CreatePolicyCommand at g1_deploy_onnx_ref.cpp:3156:
        #     q_target[mujoco_i] = default_angles[mujoco_i]
        #         + action[isaaclab_to_mujoco[mujoco_i]] * g1_action_scale[mujoco_i]
        # g1_action_scale is MuJoCo-ordered.
        action_mujoco = action_iso[params.isaaclab_to_mujoco]
        q_target = params.default_angles + action_mujoco * params.g1_action_scale
        self.send_motor_command(q_target)

        # 7) Advance the reference motion; the planner trajectory advances with time.
        if self.play:
            self.reference_frame += 1
            if self.reference_frame >= motion.timesteps:
                # Reached the end of the reference motion. Mirror the C++
                # non-planner-branch behaviour at g1_deploy_onnx_ref.cpp:3409-3427:
                # stop, rewind to frame 0 and reinitialise the heading, here to
                # the planner trajectory that the robot tracks again.
                print(f"[Keyboard] motion {self.current_motion.name} completed")
                self.play = False
                self.reference_frame = 0
                self.reinitialize_heading = True

    # ---------------------------------------------------------------- helpers

    def send_motor_command(self, q_target_mujoco: np.ndarray) -> None:
        cmd = MotorCommand(
            q_target=np.asarray(q_target_mujoco, dtype=np.float32),
            dq_target=np.zeros(params.G1_NUM_MOTOR, dtype=np.float32),
            kp=params.kps.copy(),
            kd=params.kds.copy(),
            tau_ff=np.zeros(params.G1_NUM_MOTOR, dtype=np.float32),
            mode_pr=params.MODE_PR,
        )
        self.motor_command.set(cmd)
