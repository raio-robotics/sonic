"""Keyboard — sends each keypress on the terminal into the app.

A thread reads file descriptor 0 one byte at a time. raio starts each node's
process with the launcher's stdin, so fd 0 is the terminal that runs main.py.
The node puts the terminal into cbreak mode (no Enter required, no echo) and
restores it when it shuts down. Each keypress enters the app as a physical
event, stamped with the wall-clock time of the press. Connect `key` physically,
so the receiver never waits for a keypress.

POSIX-only (Linux/macOS). When fd 0 is not a TTY (e.g. piped input), the node
reads bytes with no terminal mode change; at end of input its output closes.
"""

from __future__ import annotations

import os
import termios
import threading
import tty

from raio import Action, Node, OutputPort, task


class Keyboard(Node):
    pressed = Action[str]()

    key = OutputPort[str]()

    def __init__(self) -> None:
        self.terminal_settings: list | None = None  # the terminal mode to restore at shutdown

    @task
    async def forward(self) -> None:
        if os.isatty(0):
            self.terminal_settings = termios.tcgetattr(0)
            tty.setcbreak(0)
        threading.Thread(target=self.read_keys, daemon=True).start()
        async for key in self.pressed:
            self.key.set(key)

    def read_keys(self) -> None:
        while data := os.read(0, 1):
            self.pressed.send(data.decode(errors="replace"))
        self.pressed.close()

    def shutdown(self) -> None:
        if self.terminal_settings is not None:
            termios.tcsetattr(0, termios.TCSADRAIN, self.terminal_settings)
