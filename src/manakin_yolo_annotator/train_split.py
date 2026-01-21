import argparse
import json
import random
import shutil
from datetime import datetime, timezone
from pathlib import Path


IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp"}


def find_root(path: Path) -> Path:
    """Return the provided path if it exists, otherwise try common legacy names."""
    if path.exists():
        return path
    candidates = [
        Path(str(path).replace("annotations", "train")),
        Path(str(path).replace("train", "annotations")),
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return path


def collect_video_dirs(root: Path) -> list[Path]:
    return [p for p in sorted(root.iterdir()) if p.is_dir()]


def parse_label_counts(label_path: Path) -> tuple[int, dict]:
    """Return (box_count, class_counts) for a YOLO label file."""
    box_count = 0
    class_counts: dict[str, int] = {}
    with open(label_path, "r", encoding="utf-8") as handle:
        for line in handle:
            parts = line.strip().split()
            if not parts:
                continue
            if len(parts) >= 6:
                class_id = parts[1]
            elif len(parts) >= 5:
                class_id = parts[0]
            else:
                continue
            box_count += 1
            key = str(class_id)
            class_counts[key] = class_counts.get(key, 0) + 1
    return box_count, class_counts


def copy_video(video_dir: Path, dest_images: Path, dest_labels: Path) -> dict:
    """Copy image/label pairs for one video and return stats."""
    stats = {
        "images": 0,
        "labels": 0,
        "boxes": 0,
        "missing_labels": 0,
        "present_frames": 0,
        "absent_frames": 0,
        "classes": {},
    }
    images_dir = video_dir / "images"
    labels_dir = video_dir / "labels"
    if not images_dir.is_dir() or not labels_dir.is_dir():
        print(f"[warn] Skipping {video_dir}: missing images/ or labels/ folder")
        return stats

    for img_path in sorted(images_dir.iterdir()):
        if img_path.suffix.lower() not in IMG_EXTS or not img_path.is_file():
            continue
        label_path = labels_dir / (img_path.stem + ".txt")
        if not label_path.exists():
            print(f"[warn] No label for {img_path}, skipping")
            stats["missing_labels"] += 1
            continue
        shutil.copy2(img_path, dest_images / img_path.name)
        shutil.copy2(label_path, dest_labels / label_path.name)
        stats["images"] += 1
        stats["labels"] += 1
        box_count, class_counts = parse_label_counts(label_path)
        stats["boxes"] += box_count
        if box_count > 0:
            stats["present_frames"] += 1
        else:
            stats["absent_frames"] += 1
        for class_id, count in class_counts.items():
            stats["classes"][class_id] = stats["classes"].get(class_id, 0) + count
    return stats


def ensure_dirs(base: Path):
    for split in ("train", "val"):
        (base / "images" / split).mkdir(parents=True, exist_ok=True)
        (base / "labels" / split).mkdir(parents=True, exist_ok=True)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Split annotations/Error_Detector frames into YOLO train/val sets by video.")
    p.add_argument(
        "--root",
        type=Path,
        default=Path("annotations/data"),
        help="Root of annotation frames (per-video subfolders with images/ and labels/).",
    )
    p.add_argument(
        "--split-name",
        type=str,
        default="yolo-split",
        help="Name for this split (stored under splits/<split-name>).",
    )
    p.add_argument("--val-fraction", type=float, default=0.2, help="Validation fraction (video-level) for annotation frames.")
    p.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility.")
    return p.parse_args()


def main():
    args = parse_args()
    curated_root = find_root(args.root)
    rng = random.Random(args.seed)
    split_name = args.split_name.strip()
    if not split_name:
        raise SystemExit("split-name cannot be empty.")
    if Path(split_name).is_absolute():
        raise SystemExit("split-name must be a relative name.")
    output_dir = Path("splits") / split_name

    curated_videos = collect_video_dirs(curated_root)
    if not curated_videos:
        raise SystemExit(f"No videos found under {curated_root}")

    rng.shuffle(curated_videos)
    split_idx = max(1, int(len(curated_videos) * (1 - args.val_fraction)))
    train_vids = curated_videos[:split_idx]
    val_vids = curated_videos[split_idx:] or curated_videos[-1:]

    if output_dir.exists():
        response = input(f"Output dir exists: {output_dir}. Overwrite? [y/N]: ").strip().lower()
        if response not in ("y", "yes"):
            raise SystemExit("Aborted.")
        shutil.rmtree(output_dir)
    ensure_dirs(output_dir)
    train_img_dir = output_dir / "images" / "train"
    train_lbl_dir = output_dir / "labels" / "train"
    val_img_dir = output_dir / "images" / "val"
    val_lbl_dir = output_dir / "labels" / "val"

    copied_train = 0
    copied_val = 0
    train_stats = {
        "videos": 0,
        "images": 0,
        "labels": 0,
        "boxes": 0,
        "missing_labels": 0,
        "present_frames": 0,
        "absent_frames": 0,
        "classes": {},
    }
    val_stats = {
        "videos": 0,
        "images": 0,
        "labels": 0,
        "boxes": 0,
        "missing_labels": 0,
        "present_frames": 0,
        "absent_frames": 0,
        "classes": {},
    }
    per_video = {"train": {}, "val": {}}

    for vid in train_vids:
        vid_stats = copy_video(vid, train_img_dir, train_lbl_dir)
        per_video["train"][vid.name] = vid_stats
        copied_train += vid_stats["images"]
        train_stats["videos"] += 1
        train_stats["images"] += vid_stats["images"]
        train_stats["labels"] += vid_stats["labels"]
        train_stats["boxes"] += vid_stats["boxes"]
        train_stats["missing_labels"] += vid_stats["missing_labels"]
        train_stats["present_frames"] += vid_stats["present_frames"]
        train_stats["absent_frames"] += vid_stats["absent_frames"]
        for class_id, count in vid_stats["classes"].items():
            train_stats["classes"][class_id] = train_stats["classes"].get(class_id, 0) + count
    for vid in val_vids:
        vid_stats = copy_video(vid, val_img_dir, val_lbl_dir)
        per_video["val"][vid.name] = vid_stats
        copied_val += vid_stats["images"]
        val_stats["videos"] += 1
        val_stats["images"] += vid_stats["images"]
        val_stats["labels"] += vid_stats["labels"]
        val_stats["boxes"] += vid_stats["boxes"]
        val_stats["missing_labels"] += vid_stats["missing_labels"]
        val_stats["present_frames"] += vid_stats["present_frames"]
        val_stats["absent_frames"] += vid_stats["absent_frames"]
        for class_id, count in vid_stats["classes"].items():
            val_stats["classes"][class_id] = val_stats["classes"].get(class_id, 0) + count

    created_at = datetime.now(timezone.utc).isoformat()
    manifest = {
        "created_at": created_at,
        "split_name": split_name,
        "root": str(curated_root),
        "output_dir": str(output_dir),
        "seed": args.seed,
        "val_fraction": args.val_fraction,
        "splits": {
            "train": {"videos": [v.name for v in train_vids]},
            "val": {"videos": [v.name for v in val_vids]},
        },
    }
    stats = {
        "created_at": created_at,
        "split_name": split_name,
        "counts": {
            "train": train_stats,
            "val": val_stats,
            "total": {
                "videos": train_stats["videos"] + val_stats["videos"],
                "images": train_stats["images"] + val_stats["images"],
                "labels": train_stats["labels"] + val_stats["labels"],
                "boxes": train_stats["boxes"] + val_stats["boxes"],
                "missing_labels": train_stats["missing_labels"] + val_stats["missing_labels"],
                "present_frames": train_stats["present_frames"] + val_stats["present_frames"],
                "absent_frames": train_stats["absent_frames"] + val_stats["absent_frames"],
                "classes": {
                    **train_stats["classes"],
                },
            },
        },
        "per_video": per_video,
    }
    for class_id, count in val_stats["classes"].items():
        stats["counts"]["total"]["classes"][class_id] = stats["counts"]["total"]["classes"].get(class_id, 0) + count

    with open(output_dir / "manifest.json", "w", encoding="utf-8") as handle:
        json.dump(manifest, handle, indent=2, sort_keys=True)
    with open(output_dir / "stats.json", "w", encoding="utf-8") as handle:
        json.dump(stats, handle, indent=2, sort_keys=True)

    print(f"Curated videos: {len(curated_videos)} (train {len(train_vids)}, val {len(val_vids)})")
    print(f"Images copied -> train: {copied_train}, val: {copied_val}")
    print(f"Wrote metadata: {output_dir / 'manifest.json'}")
    print(f"Wrote stats: {output_dir / 'stats.json'}")


if __name__ == "__main__":
    main()
