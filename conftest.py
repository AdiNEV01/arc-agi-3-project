"""Portable CPU tests; optional historical binary fixtures stay external."""
import os
from pathlib import Path
import pytest


def pytest_sessionstart(session):
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    os.environ.setdefault("MKL_NUM_THREADS", "1")
    import torch
    torch.set_num_threads(1)


def pytest_collection_modifyitems(config, items):
    missing = [Path(f"DQN/checkpoints/dqn_{variant}_ls20_seed{seed}.pt")
               for variant in ("v0", "v1", "v2") for seed in (42,123,456)
               if not Path(f"DQN/checkpoints/dqn_{variant}_ls20_seed{seed}.pt").exists()]
    if missing:
        for item in items:
            if item.nodeid.endswith("Task1Tests::test_legacy_encoding_and_checkpoint_shapes"):
                item.add_marker(pytest.mark.skip(reason="Historical3K checkpoint fixtures are distributed separately; see docs/ARTIFACT_ACCESS.md"))
