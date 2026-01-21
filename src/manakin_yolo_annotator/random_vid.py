import argparse
import random
from pathlib import Path
from typing import Dict, List, Optional

from manakin_yolo_annotator.box_downloader import BoxNavigator


DEFAULT_BOX_FOLDER_ID = "162855223530"


def filter_box_entries(entries: List[Dict[str, str]]) -> List[Dict[str, str]]:
    filtered: List[Dict[str, str]] = []
    for item in entries:
        name = item.get("name", "")
        lower = name.lower()
        if "x" in lower:
            continue
        if not lower.endswith(".mp4"):
            continue
        filtered.append(item)
    return filtered


def confirm_cross_split(video_name: str, other_root: Path) -> bool:
    other_data = other_root / "data" / Path(video_name).stem
    other_video = other_root / "videos" / Path(video_name).name
    if not (other_data.exists() or other_video.exists()):
        return True
    print(f"[warn] '{video_name}' already exists in the other split.")
    if other_data.exists():
        print(f"  - data: {other_data}")
    if other_video.exists():
        print(f"  - video: {other_video}")
    response = input("Annotations/test should not share videos. Proceed anyway? [y/N]: ").strip().lower()
    return response in ("y", "yes")


def select_random_box_video_names(
    navigator: BoxNavigator,
    *,
    count: int = 1,
    folder_id: str = DEFAULT_BOX_FOLDER_ID,
    rebuild_index: bool = False,
    seed: Optional[int] = None,
) -> List[str]:
    if seed is not None:
        random.seed(seed)
    entries = None if rebuild_index else navigator.load_box_index(folder_id)
    if entries is None:
        print("[box] Building local Box index (one-time, may take a while)...")
        entries = navigator.build_box_index(folder_id)
    else:
        print("[box] Using cached Box index.")

    candidates = filter_box_entries(entries)
    if not candidates:
        return []

    existing = {p.name for p in Path(navigator.download_dir).glob("*.mp4")}
    candidates = [item for item in candidates if item.get("name") not in existing]
    if not candidates:
        return []

    pick = random.sample(candidates, min(count, len(candidates)))
    return [item.get("name", "") for item in pick if item.get("name")]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Download random videos from a Box folder.")
    parser.add_argument("--count", type=int, default=5, help="Number of random videos to download.")
    parser.add_argument("--base-dir", type=Path, default=Path("."), help="Project root (controls annotations/, test/, and system_files/).")
    parser.add_argument("--folder-id", type=str, default=DEFAULT_BOX_FOLDER_ID, help="Box folder ID to sample directly from Box.")
    parser.add_argument("--rebuild-index", action="store_true", help="Rebuild local Box index cache for the folder ID.")
    parser.add_argument("--seed", type=int, default=None, help="Optional random seed for reproducibility.")
    split_group = parser.add_mutually_exclusive_group()
    split_group.add_argument(
        "--annotate",
        dest="split",
        action="store_const",
        const="annotations",
        help="Download into annotations/videos (default).",
    )
    split_group.add_argument(
        "--test",
        dest="split",
        action="store_const",
        const="test",
        help="Download into test/videos.",
    )
    parser.set_defaults(split="annotations")
    return parser.parse_args()


def main():
    args = parse_args()
    if args.seed is not None:
        random.seed(args.seed)

    project_root = args.base_dir
    split_root = project_root / args.split
    other_root = project_root / ("test" if args.split == "annotations" else "annotations")
    download_dir = split_root / "videos"
    system_files_dir = project_root / "system_files"
    navigator = BoxNavigator(
        base_dir=str(split_root),
        system_files_dir=str(system_files_dir),
        download_dir=str(download_dir),
    )
    existing = {p.name.lower() for p in Path(navigator.download_dir).glob("*.mp4")}
    names = select_random_box_video_names(
        navigator,
        count=args.count,
        folder_id=args.folder_id,
        rebuild_index=args.rebuild_index,
        seed=args.seed,
    )
    if not names:
        raise SystemExit("No videos found after filtering (.mp4 only, no 'X' in name).")

    print(f"Selected {len(names)} videos to download.")
    for name in names:
        if name.lower() in existing:
            print(f"[skip] Already downloaded in {download_dir}: {name}")
            continue
        if not confirm_cross_split(name, other_root):
            print(f"[skip] Skipped {name} due to annotations/test conflict.")
            continue
        path = navigator.download_vid(name)
        if path is None:
            print(f"[miss] '{name}' not found or failed to download.")
        else:
            print(f"[ok] Saved to {path}")


if __name__ == "__main__":
    main()
