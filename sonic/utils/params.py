"""Hardware / policy constants ported verbatim from policy_parameters.hpp and robot_parameters.hpp.

Joint orderings:
  * MuJoCo order: URDF/hardware index — used by simulator, DDS, reference motion CSVs.
  * IsaacLab order: training-time index — used by the policy and encoder.

`isaaclab_to_mujoco[i]` answers: which MuJoCo motor index does IsaacLab joint i map to.
`mujoco_to_isaaclab[i]` answers: which IsaacLab index does MuJoCo motor i map to.
"""

import math

import numpy as np
from raio import MS, NS

# Periods of the four RT threads in g1_deploy_onnx_ref.cpp, one loop per node.
INPUT_PERIOD = 10 * MS  # 100 Hz
PLANNER_PERIOD = 100 * MS  # 10 Hz
CONTROL_PERIOD = 20 * MS  # 50 Hz
COMMAND_PERIOD = 2 * MS  # 500 Hz

# Delay of the planner → control connection. A trajectory the planner sends at t
# reaches ControlNode at exactly t + PLANNER_MOTION_DELAY, which lets PlannerNode know
# which frame ControlNode tracks at any time. One planner period: the planner sends at
# most one trajectory per period, so ControlNode always has the latest one it sent.
PLANNER_MOTION_DELAY = 0 * NS

# Unitree SDK DDS topic names (HG IDL — G1 / H1-2)
HG_CMD_TOPIC = "rt/lowcmd"
HG_IMU_TORSO = "rt/secondary_imu"
HG_STATE_TOPIC = "rt/lowstate"

G1_NUM_MOTOR = 29

# Motor armature constants (used for PID gain computation)
ARMATURE_5020 = 0.003609725
ARMATURE_7520_14 = 0.010177520
ARMATURE_7520_22 = 0.025101925
ARMATURE_4010 = 0.00425

NATURAL_FREQ = 10.0 * 2.0 * math.pi  # 10 Hz
DAMPING_RATIO = 2.0

STIFFNESS_5020 = ARMATURE_5020 * NATURAL_FREQ * NATURAL_FREQ
STIFFNESS_7520_14 = ARMATURE_7520_14 * NATURAL_FREQ * NATURAL_FREQ
STIFFNESS_7520_22 = ARMATURE_7520_22 * NATURAL_FREQ * NATURAL_FREQ
STIFFNESS_4010 = ARMATURE_4010 * NATURAL_FREQ * NATURAL_FREQ

DAMPING_5020 = 2.0 * DAMPING_RATIO * ARMATURE_5020 * NATURAL_FREQ
DAMPING_7520_14 = 2.0 * DAMPING_RATIO * ARMATURE_7520_14 * NATURAL_FREQ
DAMPING_7520_22 = 2.0 * DAMPING_RATIO * ARMATURE_7520_22 * NATURAL_FREQ
DAMPING_4010 = 2.0 * DAMPING_RATIO * ARMATURE_4010 * NATURAL_FREQ

EFFORT_LIMIT_5020 = 25.0
EFFORT_LIMIT_7520_14 = 88.0
EFFORT_LIMIT_7520_22 = 139.0
EFFORT_LIMIT_4010 = 5.0

# Joint mapping arrays — names copied verbatim from policy_parameters.hpp.
#
# WARNING: the C++ variable names are reversed from their semantic role.
#
#   isaaclab_to_mujoco[mujoco_idx] = isaaclab_idx       (the array IS indexed by mujoco)
#   mujoco_to_isaaclab[isaaclab_idx] = mujoco_idx       (the array IS indexed by isaaclab)
#
# In numpy fancy-indexing terms:
#   arr_iso    = arr_mujoco[mujoco_to_isaaclab]    # mujoco-layout → isaaclab-layout
#   arr_mujoco = arr_iso[isaaclab_to_mujoco]       # isaaclab-layout → mujoco-layout
#
# These match the C++ usage at policy_parameters.hpp:100-104 and the action
# remap at g1_deploy_onnx_ref.cpp:3120 (q_target[mujoco_i] uses
# action[isaaclab_to_mujoco[mujoco_i]]).
isaaclab_to_mujoco = np.array(
    [0, 3, 6, 9, 13, 17, 1, 4, 7, 10, 14, 18, 2, 5, 8, 11, 15, 19, 21, 23, 25, 27, 12, 16, 20, 22, 24, 26, 28],
    dtype=np.int32,
)
mujoco_to_isaaclab = np.array(
    [0, 6, 12, 1, 7, 13, 2, 8, 14, 3, 9, 15, 22, 4, 10, 16, 23, 5, 11, 17, 24, 18, 25, 19, 26, 20, 27, 21, 28],
    dtype=np.int32,
)

lower_body_joint_mujoco_order_in_isaaclab_index = np.array(
    [0, 3, 6, 9, 13, 17, 1, 4, 7, 10, 14, 18], dtype=np.int32
)
wrist_joint_isaaclab_order_in_isaaclab_index = np.array([23, 24, 25, 26, 27, 28], dtype=np.int32)

# action_scale[i] = 0.25 * effort_limit[i] / stiffness[i] — MuJoCo joint order
# (per the joint-name comments below and policy_parameters.hpp:109-139).
g1_action_scale = np.array(
    [
        0.25 * EFFORT_LIMIT_7520_22 / STIFFNESS_7520_22,  # 0 left_hip_pitch
        0.25 * EFFORT_LIMIT_7520_22 / STIFFNESS_7520_22,  # 1 left_hip_roll
        0.25 * EFFORT_LIMIT_7520_14 / STIFFNESS_7520_14,  # 2 left_hip_yaw
        0.25 * EFFORT_LIMIT_7520_22 / STIFFNESS_7520_22,  # 3 left_knee
        0.25 * EFFORT_LIMIT_5020 / STIFFNESS_5020,        # 4 left_ankle_pitch
        0.25 * EFFORT_LIMIT_5020 / STIFFNESS_5020,        # 5 left_ankle_roll
        0.25 * EFFORT_LIMIT_7520_22 / STIFFNESS_7520_22,  # 6 right_hip_pitch
        0.25 * EFFORT_LIMIT_7520_22 / STIFFNESS_7520_22,  # 7 right_hip_roll
        0.25 * EFFORT_LIMIT_7520_14 / STIFFNESS_7520_14,  # 8 right_hip_yaw
        0.25 * EFFORT_LIMIT_7520_22 / STIFFNESS_7520_22,  # 9 right_knee
        0.25 * EFFORT_LIMIT_5020 / STIFFNESS_5020,        # 10 right_ankle_pitch
        0.25 * EFFORT_LIMIT_5020 / STIFFNESS_5020,        # 11 right_ankle_roll
        0.25 * EFFORT_LIMIT_7520_14 / STIFFNESS_7520_14,  # 12 waist_yaw
        0.25 * EFFORT_LIMIT_5020 / STIFFNESS_5020,        # 13 waist_roll
        0.25 * EFFORT_LIMIT_5020 / STIFFNESS_5020,        # 14 waist_pitch
        0.25 * EFFORT_LIMIT_5020 / STIFFNESS_5020,        # 15 left_shoulder_pitch
        0.25 * EFFORT_LIMIT_5020 / STIFFNESS_5020,        # 16 left_shoulder_roll
        0.25 * EFFORT_LIMIT_5020 / STIFFNESS_5020,        # 17 left_shoulder_yaw
        0.25 * EFFORT_LIMIT_5020 / STIFFNESS_5020,        # 18 left_elbow
        0.25 * EFFORT_LIMIT_5020 / STIFFNESS_5020,        # 19 left_wrist_roll
        0.25 * EFFORT_LIMIT_4010 / STIFFNESS_4010,        # 20 left_wrist_pitch
        0.25 * EFFORT_LIMIT_4010 / STIFFNESS_4010,        # 21 left_wrist_yaw
        0.25 * EFFORT_LIMIT_5020 / STIFFNESS_5020,        # 22 right_shoulder_pitch
        0.25 * EFFORT_LIMIT_5020 / STIFFNESS_5020,        # 23 right_shoulder_roll
        0.25 * EFFORT_LIMIT_5020 / STIFFNESS_5020,        # 24 right_shoulder_yaw
        0.25 * EFFORT_LIMIT_5020 / STIFFNESS_5020,        # 25 right_elbow
        0.25 * EFFORT_LIMIT_5020 / STIFFNESS_5020,        # 26 right_wrist_roll
        0.25 * EFFORT_LIMIT_4010 / STIFFNESS_4010,        # 27 right_wrist_pitch
        0.25 * EFFORT_LIMIT_4010 / STIFFNESS_4010,        # 28 right_wrist_yaw
    ],
    dtype=np.float64,
)

kps = np.array(
    [
        STIFFNESS_7520_22, STIFFNESS_7520_22, STIFFNESS_7520_14, STIFFNESS_7520_22,
        2.0 * STIFFNESS_5020, 2.0 * STIFFNESS_5020,
        STIFFNESS_7520_22, STIFFNESS_7520_22, STIFFNESS_7520_14, STIFFNESS_7520_22,
        2.0 * STIFFNESS_5020, 2.0 * STIFFNESS_5020,
        STIFFNESS_7520_14, 2.0 * STIFFNESS_5020, 2.0 * STIFFNESS_5020,
        STIFFNESS_5020, STIFFNESS_5020, STIFFNESS_5020, STIFFNESS_5020, STIFFNESS_5020,
        STIFFNESS_4010, STIFFNESS_4010,
        STIFFNESS_5020, STIFFNESS_5020, STIFFNESS_5020, STIFFNESS_5020, STIFFNESS_5020,
        STIFFNESS_4010, STIFFNESS_4010,
    ],
    dtype=np.float32,
)

kds = np.array(
    [
        DAMPING_7520_22, DAMPING_7520_22, DAMPING_7520_14, DAMPING_7520_22,
        2.0 * DAMPING_5020, 2.0 * DAMPING_5020,
        DAMPING_7520_22, DAMPING_7520_22, DAMPING_7520_14, DAMPING_7520_22,
        2.0 * DAMPING_5020, 2.0 * DAMPING_5020,
        DAMPING_7520_14, 2.0 * DAMPING_5020, 2.0 * DAMPING_5020,
        DAMPING_5020, DAMPING_5020, DAMPING_5020, DAMPING_5020, DAMPING_5020,
        DAMPING_4010, DAMPING_4010,
        DAMPING_5020, DAMPING_5020, DAMPING_5020, DAMPING_5020, DAMPING_5020,
        DAMPING_4010, DAMPING_4010,
    ],
    dtype=np.float32,
)

# Standing pose, MuJoCo joint order.
default_angles = np.array(
    [
        -0.312, 0.0, 0.0, 0.669, -0.363, 0.0,
        -0.312, 0.0, 0.0, 0.669, -0.363, 0.0,
        0.0, 0.0, 0.0,
        0.2, 0.2, 0.0, 0.6, 0.0, 0.0, 0.0,
        0.2, -0.2, 0.0, 0.6, 0.0, 0.0, 0.0,
    ],
    dtype=np.float64,
)

# Ankle mode for LowCmd_.mode_pr.
MODE_PR = 0
MODE_AB = 1

INIT_RAMP_DURATION_SEC = 3.0

DEFAULT_PLANNER_HEIGHT = 0.788740
DEFAULT_PLANNER_LOOK_AHEAD_STEPS = 2
DEFAULT_PLANNER_SEED = 1234
PLANNER_VERSION = 2

# Periodic replan intervals (g1_deploy_onnx_ref.cpp:231-234); they apply only to moving
# modes with a non-zero speed.
REPLAN_INTERVAL_SEC = 1.0
REPLAN_INTERVAL_RUN_SEC = 0.1
# Frames over which a new plan is cross-faded into the tracked trajectory (g1_deploy_onnx_ref.cpp:3248).
PLANNER_BLEND_FRAMES = 8

LOCOMOTION_IDLE = 0
LOCOMOTION_SLOW_WALK = 1
LOCOMOTION_WALK = 2
LOCOMOTION_RUN = 3
