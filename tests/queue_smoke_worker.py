"""Subprocess fixture: commit a checkpoint, fail once, then continue exactly."""
import argparse
import json
import os
from pathlib import Path
from DQN.extended_train import train
from tests.fake_arc import FakeArcade

if __name__=="__main__":
    ap=argparse.ArgumentParser()
    for key in ("config","run","persistent","sentinel","variant","seed","resume"):
        ap.add_argument("--"+key)
    a=ap.parse_args()
    c=json.loads(Path(a.config).read_text())
    fail=not Path(a.sentinel).exists()
    train(c,a.variant,int(a.seed),"cpu",Path(a.run),stop_at=12 if fail else c["total_actions"],
          resume=Path(a.resume) if a.resume else None,persistent=Path(a.persistent),arcade_factory=FakeArcade)
    if fail:
        Path(a.sentinel).write_text("one controlled failure")
        os._exit(137)
