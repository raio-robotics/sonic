"""DDSReaderNode — subscribes to G1 DDS state topics, emits raio outputs.

It has NO input ports, so it cannot create a cycle with downstream nodes.

Subscribes:
  /rt/lowstate       (HG LowState_)   →  output port `low_state`
  /rt/secondary_imu  (HG IMUState_)   →  output port `imu_torso`

The SDK's reader threads send each sample to an action, so each one enters the
app as a physical event, stamped with the wall-clock time it arrived. Over
logical connections every receiver sees a sample at that same timestamp. The
node waits only on its actions, so from its first sample on it promises never
to send at a logical timestamp: receivers wait for it only until then. A
reaction converts the sample on the node's own thread, so a failed conversion
fails the node.

CRC validation on inbound is intentionally skipped — sim publishes with no CRC,
mirroring deploy.sh's --disable-crc-check flag for the C++ binary.
"""

from __future__ import annotations

import numpy as np
from raio import Action, Node, OutputPort, reaction, task
from unitree_sdk2py.core.channel import ChannelFactoryInitialize, ChannelSubscriber
from unitree_sdk2py.idl.unitree_hg.msg.dds_ import IMUState_, LowState_

from ..utils import RealtimeDelay, params
from ..utils.messages import IMUTorsoSnapshot, LowStateSnapshot


class DDSReaderNode(Node):
    low_state_sample = Action[LowState_]()
    imu_torso_sample = Action[IMUState_]()

    low_state = OutputPort[LowStateSnapshot]()
    imu_torso = OutputPort[IMUTorsoSnapshot]()

    def __init__(self, network_interface: str = "lo", domain_id: int = 0) -> None:
        self.network_interface = network_interface
        self.domain_id = domain_id
        self.delay = RealtimeDelay("DDSReader low_state", every=200)

    @task
    async def subscribe(self) -> None:
        ChannelFactoryInitialize(self.domain_id, self.network_interface)
        self.low_state_subscriber = ChannelSubscriber(params.HG_STATE_TOPIC, LowState_)
        self.low_state_subscriber.Init(self.low_state_sample.send, 10)
        self.imu_torso_subscriber = ChannelSubscriber(params.HG_IMU_TORSO, IMUState_)
        self.imu_torso_subscriber.Init(self.imu_torso_sample.send, 10)
        print(
            f"[DDSReader] ready (interface={self.network_interface} "
            f"topics={params.HG_STATE_TOPIC}, {params.HG_IMU_TORSO})"
        )

    @reaction(low_state_sample)
    def forward_low_state(self) -> None:
        self.delay.record()
        msg = self.low_state_sample.value
        motors = msg.motor_state[: params.G1_NUM_MOTOR]
        self.low_state.set(
            LowStateSnapshot(
                motor_q=np.array([m.q for m in motors], dtype=np.float64),
                motor_dq=np.array([m.dq for m in motors], dtype=np.float64),
                base_quat_wxyz=np.array(msg.imu_state.quaternion, dtype=np.float64),
                base_ang_vel=np.array(msg.imu_state.gyroscope, dtype=np.float64),
                base_lin_acc=np.array(msg.imu_state.accelerometer, dtype=np.float64),
                mode_machine=msg.mode_machine,
            )
        )

    @reaction(imu_torso_sample)
    def forward_imu_torso(self) -> None:
        msg = self.imu_torso_sample.value
        self.imu_torso.set(
            IMUTorsoSnapshot(
                quat_wxyz=np.array(msg.quaternion, dtype=np.float64),
                ang_vel=np.array(msg.gyroscope, dtype=np.float64),
            )
        )
