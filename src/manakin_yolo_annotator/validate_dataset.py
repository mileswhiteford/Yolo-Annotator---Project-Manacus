import argparse
import json
import math
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

try:
    import cv2
except ImportError:  # pragma: no cover - optional for image integrity checks
    cv2 = None


IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp"}


def parse_label_line(line: str) -> dict:
    parts = line.strip().split()
    if not parts:
        return {"skip": True}
    if len(parts) not in (5, 6, 7):
        return {"error": "invalid_field_count"}

    if len(parts) == 5:
        class_idx = 0
        coord_start = 1
    elif len(parts) == 6:
        class_idx = 1
        coord_start = 2
    else:
        class_idx = 1
        coord_start = 3

    class_raw = parts[class_idx]
    coord_parts = parts[coord_start:coord_start + 4]
    if len(coord_parts) != 4:
        return {"error": "invalid_field_count"}

    try:
        class_float = float(class_raw)
    except ValueError:
        return {"error": "invalid_class"}

    class_is_int = class_float.is_integer()
    class_id = int(class_float) if class_is_int else None
    if not class_is_int or class_id < 0:
        return {"error": "invalid_class"}

    try:
        x_center, y_center, width, height = [float(v) for v in coord_parts]
    except ValueError:
        return {"error": "invalid_bbox"}

    if not all(math.isfinite(v) for v in (x_center, y_center, width, height)):
        return {"error": "invalid_bbox"}

    bbox_valid = (
        0.0 <= x_center <= 1.0
        and 0.0 <= y_center <= 1.0
        and 0.0 < width <= 1.0
        and 0.0 < height <= 1.0
    )
    if not bbox_valid:
        return {"error": "invalid_bbox"}

    return {"class_id": class_id}


def iter_video_dirs(root: Path) -> list[Path]:
    return [p for p in sorted(root.iterdir()) if p.is_dir()]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate YOLO dataset structure and labels.")
    parser.add_argument(
        "--root",
        type=Path,
        default=Path("annotations/data"),
        help="Root of per-video dataset folders (each with images/ and labels/).",
    )
    parser.add_argument(
        "--other-root",
        type=Path,
        default=None,
        help="Optional root of the other split to check for overlapping videos.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("validation.json"),
        help="Output path for the validation report JSON.",
    )
    parser.add_argument(
        "--skip-image-check",
        action="store_true",
        help="Skip loading images to validate readability.",
    )
    return parser.parse_args()


def guess_other_root(root: Path) -> Optional[Path]:
    parts = root.resolve().parts
    if len(parts) < 2 or parts[-1] != "data":
        return None
    if parts[-2] == "annotations":
        return root.parent.parent / "test" / "data"
    if parts[-2] == "test":
        return root.parent.parent / "annotations" / "data"
    return None


def init_split_stats() -> dict:
    return {
        "videos": 0,
        "images": 0,
        "labels": 0,
        "missing_labels": 0,
        "extra_labels": 0,
        "present_frames": 0,
        "absent_frames": 0,
        "invalid_lines": 0,
        "invalid_classes": 0,
        "invalid_bboxes": 0,
        "corrupt_images": 0,
        "missing_images_dir": 0,
        "missing_labels_dir": 0,
        "classes": {},
    }


def add_class_counts(dst: dict, src: dict) -> None:
    for class_id, count in src.items():
        dst[class_id] = dst.get(class_id, 0) + count


def validate_root(root: Path, other_root: Optional[Path], check_images: bool) -> dict:
    report = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "root": str(root),
        "other_root": str(other_root) if other_root else None,
        "image_check_enabled": check_images,
        "image_check_supported": cv2 is not None,
        "summary": init_split_stats(),
        "per_video": {},
        "leakage": {
            "overlap_videos": [],
            "overlap_count": 0,
        },
    }

    if not root.exists():
        raise SystemExit(f"Root does not exist: {root}")

    video_dirs = iter_video_dirs(root)
    summary = report["summary"]
    summary["videos"] = len(video_dirs)

    for video_dir in video_dirs:
        stats = init_split_stats()
        images_dir = video_dir / "images"
        labels_dir = video_dir / "labels"
        if not images_dir.is_dir():
            stats["missing_images_dir"] = 1
        if not labels_dir.is_dir():
            stats["missing_labels_dir"] = 1

        if not images_dir.is_dir() or not labels_dir.is_dir():
            report["per_video"][video_dir.name] = stats
            summary["missing_images_dir"] += stats["missing_images_dir"]
            summary["missing_labels_dir"] += stats["missing_labels_dir"]
            continue

        images = {p.stem: p for p in images_dir.iterdir() if p.is_file() and p.suffix.lower() in IMG_EXTS}
        labels = {p.stem: p for p in labels_dir.iterdir() if p.is_file() and p.suffix.lower() == ".txt"}

        missing_labels = sorted(set(images) - set(labels))
        extra_labels = sorted(set(labels) - set(images))
        stats["images"] = len(images)
        stats["labels"] = len(labels)
        stats["missing_labels"] = len(missing_labels)
        stats["extra_labels"] = len(extra_labels)

        if check_images and cv2 is not None:
            for image_path in images.values():
                img = cv2.imread(str(image_path))
                if img is None:
                    stats["corrupt_images"] += 1

        for stem, label_path in labels.items():
            if stem not in images:
                continue
            with open(label_path, "r", encoding="utf-8") as handle:
                lines = handle.readlines()
            valid_boxes = 0
            class_counts = {}
            for line in lines:
                result = parse_label_line(line)
                if result.get("skip"):
                    continue
                if "error" in result:
                    stats["invalid_lines"] += 1
                    if result["error"] == "invalid_class":
                        stats["invalid_classes"] += 1
                    if result["error"] == "invalid_bbox":
                        stats["invalid_bboxes"] += 1
                    continue
                class_id = str(result["class_id"])
                class_counts[class_id] = class_counts.get(class_id, 0) + 1
                valid_boxes += 1

            if valid_boxes > 0:
                stats["present_frames"] += 1
            else:
                stats["absent_frames"] += 1
            add_class_counts(stats["classes"], class_counts)

        report["per_video"][video_dir.name] = stats
        summary["images"] += stats["images"]
        summary["labels"] += stats["labels"]
        summary["missing_labels"] += stats["missing_labels"]
        summary["extra_labels"] += stats["extra_labels"]
        summary["present_frames"] += stats["present_frames"]
        summary["absent_frames"] += stats["absent_frames"]
        summary["invalid_lines"] += stats["invalid_lines"]
        summary["invalid_classes"] += stats["invalid_classes"]
        summary["invalid_bboxes"] += stats["invalid_bboxes"]
        summary["corrupt_images"] += stats["corrupt_images"]
        summary["missing_images_dir"] += stats["missing_images_dir"]
        summary["missing_labels_dir"] += stats["missing_labels_dir"]
        add_class_counts(summary["classes"], stats["classes"])

    if other_root:
        if other_root.exists():
            overlap = sorted({p.name for p in iter_video_dirs(root)} & {p.name for p in iter_video_dirs(other_root)})
            report["leakage"]["overlap_videos"] = overlap
            report["leakage"]["overlap_count"] = len(overlap)
        else:
            report["leakage"]["other_root_missing"] = True

    return report


def main() -> None:
    args = parse_args()
    root = args.root
    other_root = args.other_root or guess_other_root(root)
    check_images = not args.skip_image_check

    report = validate_root(root, other_root, check_images)
    output_path = args.output
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2, sort_keys=True)

    summary = report["summary"]
    print(f"Videos: {summary['videos']}")
    print(f"Images: {summary['images']}, Labels: {summary['labels']}")
    print(f"Missing labels: {summary['missing_labels']}, Extra labels: {summary['extra_labels']}")
    print(f"Present frames: {summary['present_frames']}, Absent frames: {summary['absent_frames']}")
    print(f"Invalid lines: {summary['invalid_lines']}, Invalid boxes: {summary['invalid_bboxes']}")
    if summary["corrupt_images"]:
        print(f"Corrupt images: {summary['corrupt_images']}")
    if report["leakage"]["overlap_count"]:
        print(f"Overlap videos: {report['leakage']['overlap_count']}")
        for name in report["leakage"]["overlap_videos"]:
            print(f"- {name}")
    print(f"Wrote report: {output_path}")

    error_count = (
        summary["missing_labels"]
        + summary["extra_labels"]
        + summary["invalid_lines"]
        + summary["invalid_classes"]
        + summary["invalid_bboxes"]
        + summary["corrupt_images"]
        + summary["missing_images_dir"]
        + summary["missing_labels_dir"]
        + report["leakage"]["overlap_count"]
    )
    raise SystemExit(1 if error_count else 0)


if __name__ == "__main__":
    main()
