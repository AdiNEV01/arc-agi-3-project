"""Download ONLY declared exposed training/development games through the toolkit.

Never reads game rules. Exact offline game IDs and opaque runtime asset hashes
are checked before training. No substitution is allowed if a game is unavailable.
"""
import argparse
import json
from pathlib import Path

import arc_agi
from DQN.extended_train import hash_file, json_write


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="Configs/dqn_extended_v1.json")
    ap.add_argument("--include-development", action="store_true")
    ap.add_argument("--download", action="store_true")
    args = ap.parse_args()
    config = json.loads(Path(args.config).read_text())
    games = config["training_games"] + (config["development_games"] if args.include_development else [])
    if args.download:
        arcade = arc_agi.Arcade(operation_mode=arc_agi.OperationMode.NORMAL)
        for game in games:
            env = arcade.make(game, seed=42)
            if env is None:
                raise RuntimeError(f"Required game {game} unavailable; no replacements permitted")
    # Separate OFFLINE construction ensures no accidental remote-only fallback.
    arcade = arc_agi.Arcade(operation_mode=arc_agi.OperationMode.OFFLINE)
    hashes = {}
    for game in games:
        env = arcade.make(game, seed=42)
        if env is None or env.info.game_id != game:
            raise RuntimeError(f"Exact offline game missing: {game}; run with --download")
        short, version = game.split("-", 1)
        folder = Path("environment_files") / short / version
        files = [p for p in folder.rglob("*") if p.is_file() and p.suffix in (".py", ".json")]
        if not files:
            raise RuntimeError(f"Offline runtime assets not found: {folder}")
        hashes.update({p.as_posix(): hash_file(p) for p in files})
    Path("outputs").mkdir(exist_ok=True)
    json_write(Path("outputs/extended_game_assets.json"), {"games": games, "sha256": hashes})
    print("Verified exact offline games:", games)


if __name__ == "__main__":
    main()
