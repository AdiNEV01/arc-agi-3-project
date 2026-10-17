"""Click-coverage probe (Task 3, evidence for the ACTION6 discretisation choice).

From the start of level 1, clicks every pixel (or every `--stride` pixels), records
which clicks change the settled frame, then level-resets. Then reports how many
responsive regions each click lattice (4x4, 8x8, 16x16, 32x32) can reach.

Use TRAIN games only (never held-out games):
  python probe_clicks.py --game lp85
  python probe_clicks.py --game ft09 --stride 2      # faster, coarser sweep
Writes results/click_probe_<game>.json
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import arc_agi
from arcengine import GameAction, GameState


def components(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    """4-connected components of True cells, as lists of (x, y)."""
    seen = np.zeros_like(mask, dtype=bool)
    comps = []
    h, w = mask.shape
    for y0 in range(h):
        for x0 in range(w):
            if not mask[y0, x0] or seen[y0, x0]:
                continue
            stack, comp = [(y0, x0)], []
            seen[y0, x0] = True
            while stack:
                y, x = stack.pop()
                comp.append((x, y))
                for dy, dx in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                    ny, nx = y + dy, x + dx
                    if 0 <= ny < h and 0 <= nx < w and mask[ny, nx] and not seen[ny, nx]:
                        seen[ny, nx] = True
                        stack.append((ny, nx))
            comps.append(comp)
    return comps


def lattice(k: int) -> set[tuple[int, int]]:
    cell = 64 // k
    return {(c * cell + cell // 2, r * cell + cell // 2) for r in range(k) for c in range(k)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--game", required=True)
    ap.add_argument("--stride", type=int, default=1, help="1 = every pixel (4096 clicks)")
    args = ap.parse_args()

    arc = arc_agi.Arcade()
    env = arc.make(args.game, seed=0)
    start = env.reset()
    if GameAction.ACTION6.value not in start.available_actions:
        raise SystemExit(f"{args.game} does not offer ACTION6")
    base = np.array(start.frame[-1])

    responsive = np.zeros((64, 64), dtype=bool)
    game_over_clicks, t0, n = 0, time.perf_counter(), 0
    for y in range(0, 64, args.stride):
        for x in range(0, 64, args.stride):
            f = env.step(GameAction.ACTION6, data={"x": x, "y": y})
            n += 1
            if f is not None and f.frame and not np.array_equal(np.array(f.frame[-1]), base):
                responsive[y, x] = True
            if f is not None and f.state == GameState.GAME_OVER:
                game_over_clicks += 1
            r = env.reset()                         # level reset back to level 1 start
            if not np.array_equal(np.array(r.frame[-1]), base):
                print("warning: reset did not restore the start frame")

    comps = components(responsive)
    report = {"game_id": env.info.game_id, "stride": args.stride, "clicks": n,
              "seconds": round(time.perf_counter() - t0, 1),
              "responsive_pixels": int(responsive.sum()),
              "responsive_fraction": round(float(responsive.sum()) / (64 * 64 / args.stride ** 2), 4),
              "responsive_regions": len(comps),
              "region_sizes": sorted(len(c) for c in comps),
              "game_over_clicks": game_over_clicks, "lattices": {}}
    for k in (4, 8, 16, 32):
        pts = lattice(k)
        hit = sum(any(p in pts for p in c) for c in comps)
        report["lattices"][f"{k}x{k}"] = {
            "targets": k * k, "regions_reachable": hit,
            "coverage": round(hit / len(comps), 3) if comps else None}
    if args.stride > 1:
        report["note"] = "stride > 1: regions smaller than the stride may be missed entirely"

    Path("results").mkdir(exist_ok=True)
    out = Path(f"results/click_probe_{args.game}.json")
    out.write_text(json.dumps(report, indent=2))
    print(json.dumps({k: v for k, v in report.items() if k != "region_sizes"}, indent=2))
    print(f"Full report: {out}")


if __name__ == "__main__":
    main()
