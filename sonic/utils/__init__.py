"""Helpers shared by all nodes — constants, math, motion loader, ONNX wrappers."""

import raio
from raio import MS


class RealtimeDelay:
    """Prints, every `every` calls, how far the node's logical time trails the wall clock.

    A node handles a timestamp no earlier than the wall clock reaches it, so the delay
    is about zero, or slightly positive when the node's work takes time. A delay that
    keeps growing means the node falls behind the wall clock — e.g. policy inference
    exceeds the tick period.
    """

    def __init__(self, tag: str, every: int) -> None:
        self.tag = tag
        self.every = every
        self.count = 0

    def record(self) -> None:
        self.count += 1
        if self.count % self.every == 0:
            delay_ms = (raio.physical_now() - raio.now()) / MS
            print(f"[{self.tag}] count={self.count} delay={delay_ms:+.2f}ms")
