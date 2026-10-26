"""Real declared three-game continuation preflight; runs before costly GPU pilots."""
import argparse
import json
from pathlib import Path
import torch
from DQN.extended_train import train, check_resume, json_write


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=False)
    c = json.loads(Path("Configs/dqn_extended_v1.json").read_text())
    train(c,"v2",42,args.device,args.out/"full",stop_at=1200)
    train(c,"v2",42,args.device,args.out/"prefix",stop_at=1100)
    train(c,"v2",42,args.device,args.out/"restored",stop_at=1200,resume=args.out/"prefix/resume/step001100")
    a,b = [check_resume(args.out/x/"resume/step001200") for x in ("full","restored")]
    assert a["journal"]==b["journal"]
    for key in ("online_state_dict","target_state_dict"):
        for name,tensor in a[key].items():
            torch.testing.assert_close(tensor,b[key][name],rtol=0,atol=0)
    json_write(args.out/"reconstruction_validation.json",{"valid":True,"device":args.device,
        "selected_actions":1200,"prefix_actions":1100,"identical_transcript_and_parameters":True,
        "scope":"fixed package versions, source, runtime assets and device; not cross-hardware bitwise equivalence"})
    print("Three-game continuation verified")


if __name__ == "__main__":
    main()
