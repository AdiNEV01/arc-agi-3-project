"""Opt-in synchronized wall timings; disabled mode never synchronizes CUDA."""
from collections import defaultdict
from contextlib import contextmanager
import time
import torch

class StageTimer:
    def __init__(self, device, enabled=False):
        self.device = torch.device(device)
        self.enabled = enabled
        self.seconds = defaultdict(float)
        self.calls = defaultdict(int)

    def sync(self):
        if self.device.type == "cuda":
            torch.cuda.synchronize(self.device)

    @contextmanager
    def measure(self, name):
        if not self.enabled:
            yield
            return
        self.sync()
        start = time.perf_counter()
        try:
            yield
        finally:
            self.sync()
            self.seconds[name] += time.perf_counter() - start
            self.calls[name] += 1
