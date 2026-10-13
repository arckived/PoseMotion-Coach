import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from .pose_utils import extract_pose_sequence


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--seq-len", type=int, default=48)
    args = parser.parse_args()

    manifest = pd.read_csv(args.manifest)
    out_dir = Path(args.out_dir)
    seq_dir = out_dir / "sequences"
    seq_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    skipped = 0
    for i, row in manifest.iterrows():
        video_path = row["video_path"]
        label = row["label"]
        seq = extract_pose_sequence(video_path, seq_len=args.seq_len)
        if seq is None:
            skipped += 1
            print(f"[skip] could not read pose: {video_path}")
            continue

        save_path = seq_dir / f"sample_{i:06d}.npy"
        np.save(save_path, seq)
        rows.append({"sequence_path": str(save_path), "label": label, "source_video": video_path})
        print(f"[{i+1}/{len(manifest)}] {label} -> {save_path.name}")

    processed = pd.DataFrame(rows)
    out_csv = out_dir / "processed_manifest.csv"
    processed.to_csv(out_csv, index=False)
    print(f"Saved {len(processed)} processed samples to {out_csv}. Skipped {skipped} videos.")


if __name__ == "__main__":
    main()
