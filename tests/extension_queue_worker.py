"""Real CPU subprocess: continue an existing state, fail once, then finish."""
import argparse
import json
import os
from pathlib import Path
from DQN.extended_train import train
from tests.fake_arc import FakeArcade

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    for name in ("config", "run", "persistent", "variant", "seed", "resume", "sentinel"):
        p.add_argument("--" + name, required=True)
    a = p.parse_args()
    config = json.loads(Path(a.config).read_text())
    first = not Path(a.sentinel).exists()
    train(config, a.variant, int(a.seed), "cpu", Path(a.run), stop_at=36 if first else config["total_actions"],
          resume=Path(a.resume), persistent=Path(a.persistent), arcade_factory=FakeArcade)
    if first:
        Path(a.sentinel).write_text("saved continuation at36")
        os._exit(137)
