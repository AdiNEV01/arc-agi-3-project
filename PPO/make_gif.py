"""Build an animated GIF from ordered step_*.png frames."""

from __future__ import annotations

import argparse
from pathlib import Path

from PIL import Image


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--frames", required=True, help="Folder containing step_*.png images")
    parser.add_argument("--output", default=None, help="Output .gif path")
    parser.add_argument("--duration", type=int, default=150,
                        help="Milliseconds each frame remains visible")
    parser.add_argument("--loop", type=int, default=0,
                        help="0 loops forever; 1 plays twice; etc.")
    args = parser.parse_args()

    if args.duration < 20:
        parser.error("--duration must be at least 20 milliseconds")

    frames_dir = Path(args.frames)
    paths = sorted(frames_dir.glob("step_*.png"))
    if len(paths) < 2:
        raise FileNotFoundError(
            f"Expected at least two step_*.png frames in {frames_dir.resolve()}"
        )

    output_path = Path(args.output) if args.output else frames_dir / "episode.gif"
    output_path.parent.mkdir(parents=True, exist_ok=True)

    gif_frames = []
    for path in paths:
        with Image.open(path) as frame:
            gif_frames.append(frame.convert("RGB").quantize(colors=256))

    first_frame = gif_frames[0]
    first_frame.save(
        output_path,
        save_all=True,
        append_images=gif_frames[1:],
        duration=args.duration,
        loop=args.loop,
        optimize=False,
        disposal=2,
    )
    print(f"GIF created from {len(paths)} frames: {output_path.resolve()}")


if __name__ == "__main__":
    main()
