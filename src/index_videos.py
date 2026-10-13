import argparse
from pathlib import Path
import pandas as pd

VIDEO_EXTS = {".mp4", ".mov", ".avi", ".mkv", ".webm"}


def build_manifest(data_dir: Path) -> pd.DataFrame:
    rows = []
    for class_dir in sorted([p for p in data_dir.iterdir() if p.is_dir()]):
        label = class_dir.name
        for video_path in sorted(class_dir.rglob("*")):
            if video_path.suffix.lower() in VIDEO_EXTS:
                rows.append({"video_path": str(video_path), "label": label})
    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", required=True, help="Folder containing class subfolders")
    parser.add_argument("--out", required=True, help="Output manifest CSV")
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)

    df = build_manifest(data_dir)
    if df.empty:
        raise SystemExit(f"No videos found in {data_dir}. Expected data/raw/<class_name>/*.mp4")

    df.to_csv(out, index=False)
    print(f"Saved {len(df)} videos to {out}")
    print(df["label"].value_counts())


if __name__ == "__main__":
    main()
