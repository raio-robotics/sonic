"""DDSWriterNode — receives raio motor_command, publishes /rt/lowcmd at 500 Hz.

It has NO output ports back into the graph, so it cannot create a cycle with
upstream nodes.

A 2 ms loop publishes the latest MotorCommand from ControlNode. The connection
from ControlNode is logical: at a control tick the writer publishes only after
ControlNode has computed that tick's command, so each new command goes out as
soon as it exists. CRC is left at 0 — the MuJoCo sim bridge ignores CRC
(mirrors deploy.sh's --disable-crc-check sim flag).
"""

from __future__ import annotations

import raio
from raio import InputPort, Node, task
from unitree_sdk2py.core.channel import ChannelFactoryInitialize, ChannelPublisher
from unitree_sdk2py.idl.default import unitree_hg_msg_dds__LowCmd_
from unitree_sdk2py.idl.unitree_hg.msg.dds_ import LowCmd_

from ..utils import RealtimeDelay, params
from ..utils.messages import MotorCommand


class DDSWriterNode(Node):
    motor_command = InputPort[MotorCommand]()

    def __init__(self, network_interface: str = "lo", domain_id: int = 0) -> None:
        self.network_interface = network_interface
        self.domain_id = domain_id
        self.delay = RealtimeDelay("DDSWriter 500Hz", every=500)

    @task
    async def publish_loop(self) -> None:
        ChannelFactoryInitialize(self.domain_id, self.network_interface)
        cmd_dds = unitree_hg_msg_dds__LowCmd_()
        publisher = ChannelPublisher(params.HG_CMD_TOPIC, LowCmd_)
        publisher.Init()
        print(f"[DDSWriter] ready (interface={self.network_interface} topic={params.HG_CMD_TOPIC})")
        while True:
            await raio.wait(params.COMMAND_PERIOD)
            self.delay.record()
            cmd = self.motor_command.value
            if cmd is None:
                continue  # ControlNode has not computed a command yet
            cmd_dds.mode_pr = cmd.mode_pr
            for i in range(params.G1_NUM_MOTOR):
                mc = cmd_dds.motor_cmd[i]
                mc.mode = 1
                mc.q = float(cmd.q_target[i])
                mc.dq = float(cmd.dq_target[i])
                mc.kp = float(cmd.kp[i])
                mc.kd = float(cmd.kd[i])
                mc.tau = float(cmd.tau_ff[i])
            publisher.Write(cmd_dds)
