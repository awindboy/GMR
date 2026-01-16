#!/usr/bin/env python3
import argparse
import pickle
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Merge per-motion .pkl files into a single dataset .pkl."
    )
    parser.add_argument(
        "--motion_folder",
        required=True,
        help="Folder containing motion .pkl files.",
    )
    parser.add_argument(
        "--output",
        required=True,
        help="Output .pkl path for the merged dataset.",
    )
    parser.add_argument(
        "--pattern",
        default="*.pkl",
        help="Glob pattern to match motion files (default: *.pkl).",
    )
    args = parser.parse_args()

    motion_dir = Path(args.motion_folder)
    if not motion_dir.exists():
        raise FileNotFoundError(f"Motion folder not found: {motion_dir}")

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    motion_files = sorted(motion_dir.glob(args.pattern))
    motions = []
    for path in motion_files:
        if path.resolve() == out_path.resolve():
            continue
        with open(path, "rb") as f:
            data = pickle.load(f)
        entry = {"motion_file": path.name}
        if isinstance(data, dict):
            entry.update(data)
        else:
            entry["motion_data"] = data
        motions.append(entry)

    merged = {
        "motions": motions,
        "motion_files": [m["motion_file"] for m in motions],
    }

    with open(out_path, "wb") as f:
        pickle.dump(merged, f)

    print(f"Saved {len(motions)} motions to {out_path}")


if __name__ == "__main__":
    main()
