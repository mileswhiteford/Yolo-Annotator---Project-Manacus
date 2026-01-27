import argparse
import csv
import hashlib
import json
import math
import random
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
from ultralytics import YOLO

from .box_downloader import BoxNavigator

IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate a YOLO model on test videos with frame-level labels.")
    parser.add_argument("--weights", type=Path, required=True, help="Path to YOLO weights (.pt).")
    parser.add_argument("--base-dir", type=Path, default=Path("."), help="Project root (controls test/).")
    parser.add_argument("--data-root", type=Path, default=None, help="Override data root (default: test/data).")
    parser.add_argument("--video-root", type=Path, default=None, help="Override video root (default: test/videos).")
    parser.add_argument("--match-iou", type=float, default=0.5, help="IoU threshold to count a match (higher = stricter).")
    parser.add_argument("--track-iou", type=float, default=0.1, help="IoU for YOLO tracking.")
    parser.add_argument("--track-conf", type=float, default=0.01, help="Confidence threshold for YOLO tracking.")
    parser.add_argument(
        "--conf-thresholds",
        type=str,
        default="0.1,0.25,0.5",
        help="Comma-separated confidence thresholds for metrics (e.g., 0.1,0.25,0.5).",
    )
    parser.add_argument("--output-dir", type=Path, default=None, help="Output directory for eval artifacts.")
    parser.add_argument("--run-name", type=str, default=None, help="Optional tag appended to the output dir name.")
    parser.add_argument("--write-predictions", action="store_true", help="Write predictions.json for inspection.")
    parser.add_argument("--write-csv", action="store_true", help="Write per-video metrics CSV.")
    parser.add_argument("--bootstrap-samples", type=int, default=0, help="Bootstrap samples for confidence intervals.")
    parser.add_argument("--bootstrap-alpha", type=float, default=0.05, help="Alpha for CI (0.05 => 95%% CI).")
    parser.add_argument("--verbose", action="store_true", help="Print progress while evaluating videos.")
    parser.add_argument(
        "--download-missing",
        type=int,
        choices=[0, 1],
        default=1,
        help="Download missing test videos from Box (1 = yes, 0 = no).",
    )
    return parser.parse_args()


def build_run_id(weights: Path, run_name: Optional[str]) -> str:
    now = datetime.now(timezone.utc)
    timestamp = f"{now.day}.{now.month}.{now.year}_{now:%H.%M}"
    run_id = f"{timestamp}_{weights.name}"
    if run_name:
        run_id = f"{run_id}_{run_name}"
    return run_id


def hash_file(path: Path) -> str:
    hasher = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


def parse_conf_thresholds(raw: str) -> List[float]:
    thresholds = []
    for part in raw.split(","):
        part = part.strip()
        if not part:
            continue
        thresholds.append(float(part))
    thresholds = sorted(set(thresholds))
    if not thresholds:
        raise SystemExit("No confidence thresholds provided.")
    return thresholds


def format_summary_text(eval_report: Dict, output_dir: Path) -> str:
    summary = eval_report["summary"]
    label_errors = summary["label_errors"]
    downloads = eval_report.get("downloads", {})
    metrics = summary["metrics"]

    lines = [
        "Evaluation summary",
        f"Run: {output_dir.name}",
        f"Created at: {eval_report['created_at']}",
        f"Weights: {eval_report['weights']}",
        f"Weights sha256: {eval_report['weights_sha256']}",
        f"Data root: {eval_report['data_root']}",
        f"Video root: {eval_report['video_root']}",
        "",
        "Coverage",
        f"Videos evaluated: {summary['videos_evaluated']}",
        f"Frames evaluated: {summary['frames_evaluated']}",
        f"Present frames: {summary['present_frames']}",
        f"Absent frames: {summary['absent_frames']}",
        f"Missing labels: {summary['missing_labels']}",
        f"Out of range labels: {summary['out_of_range_labels']}",
        (
            "Label errors: "
            f"invalid_lines={label_errors['invalid_lines']} "
            f"invalid_bboxes={label_errors['invalid_bboxes']} "
            f"invalid_classes={label_errors['invalid_classes']} "
            f"invalid_filenames={label_errors['invalid_filenames']}"
        ),
        "",
        "Missing data",
        f"Missing data videos: {len(eval_report['missing_data_videos'])}",
        f"Missing video data: {len(eval_report['missing_video_data'])}",
        (
            "Downloads: "
            f"attempted={len(downloads.get('attempted', []))} "
            f"succeeded={len(downloads.get('succeeded', []))} "
            f"failed={len(downloads.get('failed', []))}"
        ),
        "",
        "Metrics by confidence threshold",
    ]

    for thr in eval_report["confidence_thresholds"]:
        key = str(thr)
        metric = metrics[key]
        lines.append(
            f"thr={thr} precision={metric['precision']:.4f} recall={metric['recall']:.4f} "
            f"f1={metric['f1']:.4f} tp={metric['tp']} fp={metric['fp']} fn={metric['fn']}"
        )

    if "bootstrap" in eval_report:
        bootstrap = eval_report["bootstrap"]
        lines.extend(
            [
                "",
                "Bootstrap confidence intervals",
                f"samples={bootstrap['samples']} alpha={bootstrap['alpha']}",
                "See eval.json for detailed bootstrap metrics.",
            ]
        )

    return "\n".join(lines) + "\n"


def yolo_to_xyxy(line: str, frame_w: int, frame_h: int) -> Tuple[Optional[List[float]], Dict[str, int]]:
    parts = line.strip().split()
    stats = {"invalid_lines": 0, "invalid_bbox": 0, "invalid_class": 0}
    if not parts:
        return None, stats
    if len(parts) not in (5, 6, 7):
        stats["invalid_lines"] = 1
        return None, stats

    if len(parts) == 5:
        class_idx = 0
        coord_start = 1
    elif len(parts) == 6:
        class_idx = 1
        coord_start = 2
    else:
        class_idx = 1
        coord_start = 3

    try:
        class_float = float(parts[class_idx])
    except ValueError:
        stats["invalid_class"] = 1
        return None, stats
    if not class_float.is_integer() or int(class_float) < 0:
        stats["invalid_class"] = 1
        return None, stats

    coords = parts[coord_start:coord_start + 4]
    if len(coords) != 4:
        stats["invalid_lines"] = 1
        return None, stats
    try:
        x_center, y_center, width, height = [float(v) for v in coords]
    except ValueError:
        stats["invalid_bbox"] = 1
        return None, stats
    if not all(math.isfinite(v) for v in (x_center, y_center, width, height)):
        stats["invalid_bbox"] = 1
        return None, stats
    if not (0.0 <= x_center <= 1.0 and 0.0 <= y_center <= 1.0 and 0.0 < width <= 1.0 and 0.0 < height <= 1.0):
        stats["invalid_bbox"] = 1
        return None, stats

    x1 = (x_center - width / 2.0) * frame_w
    y1 = (y_center - height / 2.0) * frame_h
    x2 = (x_center + width / 2.0) * frame_w
    y2 = (y_center + height / 2.0) * frame_h
    return [x1, y1, x2, y2], stats


def box_iou(box1: List[float], box2: List[float]) -> float:
    xA = max(box1[0], box2[0])
    yA = max(box1[1], box2[1])
    xB = min(box1[2], box2[2])
    yB = min(box1[3], box2[3])
    inter_w = max(0.0, xB - xA)
    inter_h = max(0.0, yB - yA)
    inter = inter_w * inter_h
    if inter <= 0:
        return 0.0
    area1 = max(0.0, box1[2] - box1[0]) * max(0.0, box1[3] - box1[1])
    area2 = max(0.0, box2[2] - box2[0]) * max(0.0, box2[3] - box2[1])
    union = area1 + area2 - inter
    return float(inter / union) if union > 0 else 0.0


def _is_zero(value: float, eps: float = 1e-9) -> bool:
    return abs(value) <= eps


def hungarian(cost_matrix: List[List[float]]) -> List[Tuple[int, int]]:
    n = len(cost_matrix)
    if n == 0:
        return []
    cost = [row[:] for row in cost_matrix]

    # Step 1: subtract row minima
    for i in range(n):
        row_min = min(cost[i])
        for j in range(n):
            cost[i][j] -= row_min

    # Step 2: subtract column minima
    for j in range(n):
        col_min = min(cost[i][j] for i in range(n))
        for i in range(n):
            cost[i][j] -= col_min

    starred = [[False] * n for _ in range(n)]
    primed = [[False] * n for _ in range(n)]
    row_covered = [False] * n
    col_covered = [False] * n

    # Step 3: star zeros
    for i in range(n):
        for j in range(n):
            if _is_zero(cost[i][j]) and not row_covered[i] and not col_covered[j]:
                starred[i][j] = True
                row_covered[i] = True
                col_covered[j] = True
    row_covered = [False] * n
    col_covered = [False] * n

    def cover_columns_with_starred() -> int:
        for j in range(n):
            col_covered[j] = any(starred[i][j] for i in range(n))
        return sum(col_covered)

    def find_uncovered_zero() -> Optional[Tuple[int, int]]:
        for i in range(n):
            if row_covered[i]:
                continue
            for j in range(n):
                if not col_covered[j] and _is_zero(cost[i][j]):
                    return i, j
        return None

    def find_star_in_row(row: int) -> Optional[int]:
        for j in range(n):
            if starred[row][j]:
                return j
        return None

    def find_star_in_col(col: int) -> Optional[int]:
        for i in range(n):
            if starred[i][col]:
                return i
        return None

    def find_prime_in_row(row: int) -> Optional[int]:
        for j in range(n):
            if primed[row][j]:
                return j
        return None

    while True:
        if cover_columns_with_starred() == n:
            break

        while True:
            zero = find_uncovered_zero()
            if zero is None:
                # Step 6: adjust matrix
                min_uncovered = None
                for i in range(n):
                    if row_covered[i]:
                        continue
                    for j in range(n):
                        if col_covered[j]:
                            continue
                        val = cost[i][j]
                        if min_uncovered is None or val < min_uncovered:
                            min_uncovered = val
                if min_uncovered is None:
                    min_uncovered = 0.0
                for i in range(n):
                    for j in range(n):
                        if row_covered[i]:
                            cost[i][j] += min_uncovered
                        if not col_covered[j]:
                            cost[i][j] -= min_uncovered
                continue

            row, col = zero
            primed[row][col] = True
            star_col = find_star_in_row(row)
            if star_col is None:
                # Step 5: augmenting path
                path = [(row, col)]
                while True:
                    star_row = find_star_in_col(path[-1][1])
                    if star_row is None:
                        break
                    path.append((star_row, path[-1][1]))
                    prime_col = find_prime_in_row(path[-1][0])
                    if prime_col is None:
                        break
                    path.append((path[-1][0], prime_col))

                for r, c in path:
                    starred[r][c] = not starred[r][c]
                primed = [[False] * n for _ in range(n)]
                row_covered = [False] * n
                col_covered = [False] * n
                break
            row_covered[row] = True
            col_covered[star_col] = False

    assignments = []
    for i in range(n):
        for j in range(n):
            if starred[i][j]:
                assignments.append((i, j))
                break
    return assignments


def match_boxes(
    gt_boxes: List[List[float]],
    pred_boxes: List[List[float]],
    match_iou: float,
) -> Tuple[int, int, int, List[Tuple[int, int, float]]]:
    if not gt_boxes and not pred_boxes:
        return 0, 0, 0, []
    if not gt_boxes:
        return 0, len(pred_boxes), 0, []
    if not pred_boxes:
        return 0, 0, len(gt_boxes), []

    iou_matrix = [[box_iou(gt, pred[:4]) for pred in pred_boxes] for gt in gt_boxes]
    max_size = max(len(gt_boxes), len(pred_boxes))
    padded = []
    for row in iou_matrix:
        padded.append(row + [0.0] * (max_size - len(pred_boxes)))
    for _ in range(max_size - len(gt_boxes)):
        padded.append([0.0] * max_size)

    cost = [[1.0 - val for val in row] for row in padded]
    assignments = hungarian(cost)
    matches: List[Tuple[int, int, float]] = []
    tp = 0
    for row, col in assignments:
        if row >= len(gt_boxes) or col >= len(pred_boxes):
            continue
        iou_val = iou_matrix[row][col]
        if iou_val >= match_iou:
            tp += 1
            matches.append((row, col, iou_val))

    fp = len(pred_boxes) - tp
    fn = len(gt_boxes) - tp
    return tp, fp, fn, matches


def precision_recall_f1(tp: int, fp: int, fn: int) -> Dict[str, float]:
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0
    return {"precision": precision, "recall": recall, "f1": f1}


def run_tracking(model: YOLO, video_path: Path, conf: float, iou: float) -> List[List[List[float]]]:
    predictions: List[List[List[float]]] = []
    for result in model.track(str(video_path), stream=True, persist=True, verbose=False, conf=conf, iou=iou):
        frame_preds: List[List[float]] = []
        boxes = result.boxes
        if boxes is not None and len(boxes) > 0:
            xyxy = boxes.xyxy.cpu().numpy()
            confs = boxes.conf.cpu().numpy()
            for i in range(len(xyxy)):
                x1, y1, x2, y2 = xyxy[i]
                conf_val = float(confs[i]) if confs is not None else 0.0
                frame_preds.append([float(x1), float(y1), float(x2), float(y2), conf_val])
        predictions.append(frame_preds)
    return predictions


def parse_labels(labels_dir: Path, frame_w: int, frame_h: int) -> Tuple[Dict[int, List[List[float]]], Dict[str, int]]:
    gt_by_frame: Dict[int, List[List[float]]] = {}
    stats = {
        "invalid_lines": 0,
        "invalid_bboxes": 0,
        "invalid_classes": 0,
        "invalid_filenames": 0,
    }
    for label_path in sorted(labels_dir.iterdir()):
        if not label_path.is_file() or label_path.suffix.lower() != ".txt":
            continue
        try:
            frame_idx = int(label_path.stem)
        except ValueError:
            stats["invalid_filenames"] += 1
            continue
        boxes: List[List[float]] = []
        with open(label_path, "r", encoding="utf-8") as handle:
            for line in handle:
                box, line_stats = yolo_to_xyxy(line, frame_w, frame_h)
                stats["invalid_lines"] += line_stats["invalid_lines"]
                stats["invalid_bboxes"] += line_stats["invalid_bbox"]
                stats["invalid_classes"] += line_stats["invalid_class"]
                if box:
                    boxes.append(box)
        gt_by_frame[frame_idx] = boxes
    return gt_by_frame, stats


def evaluate_video(
    model: YOLO,
    video_path: Path,
    data_dir: Path,
    thresholds: List[float],
    match_iou: float,
    track_conf: float,
    track_iou: float,
) -> Tuple[Dict, Optional[Dict]]:
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise SystemExit(f"Could not open video: {video_path}")
    frame_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
    frame_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    cap.release()
    if frame_w == 0 or frame_h == 0:
        raise SystemExit(f"Could not read video dimensions: {video_path}")

    labels_dir = data_dir / "labels"
    if not labels_dir.is_dir():
        raise SystemExit(f"Missing labels dir: {labels_dir}")
    gt_by_frame, label_stats = parse_labels(labels_dir, frame_w, frame_h)

    predictions = run_tracking(model, video_path, conf=track_conf, iou=track_iou)
    pred_frame_count = len(predictions)

    expected_frames = max(frame_count, pred_frame_count)
    label_frames = {idx for idx in gt_by_frame.keys() if idx < expected_frames}
    missing_labels = max(0, expected_frames - len(label_frames))

    metrics = {
        str(thr): {"tp": 0, "fp": 0, "fn": 0} for thr in thresholds
    }
    present_frames = 0
    absent_frames = 0
    frames_evaluated = 0
    out_of_range_labels = 0

    for frame_idx in sorted(label_frames):
        gt_boxes = gt_by_frame.get(frame_idx, [])
        if frame_idx >= pred_frame_count:
            out_of_range_labels += 1
            continue
        frames_evaluated += 1
        if gt_boxes:
            present_frames += 1
        else:
            absent_frames += 1
        frame_preds_all = predictions[frame_idx]
        for thr in thresholds:
            frame_preds = [p for p in frame_preds_all if p[4] >= thr]
            tp, fp, fn, _ = match_boxes(gt_boxes, frame_preds, match_iou)
            metrics[str(thr)]["tp"] += tp
            metrics[str(thr)]["fp"] += fp
            metrics[str(thr)]["fn"] += fn

    for thr, values in metrics.items():
        values.update(precision_recall_f1(values["tp"], values["fp"], values["fn"]))

    summary = {
        "video": video_path.name,
        "frames_total": expected_frames,
        "frames_evaluated": frames_evaluated,
        "present_frames": present_frames,
        "absent_frames": absent_frames,
        "missing_labels": missing_labels,
        "out_of_range_labels": out_of_range_labels,
        "label_stats": label_stats,
        "metrics": metrics,
    }
    predictions_payload = None
    if predictions:
        predictions_payload = {
            "video": video_path.name,
            "frames": [
                {"frame": idx, "boxes": preds}
                for idx, preds in enumerate(predictions)
                if preds
            ],
        }
    return summary, predictions_payload


def bootstrap_confidence_intervals(
    per_video: Dict[str, Dict],
    thresholds: List[float],
    samples: int,
    alpha: float,
) -> Dict[str, Dict]:
    if samples <= 0:
        return {}
    names = list(per_video.keys())
    if not names:
        return {}
    rng = random.Random(0)
    results: Dict[str, Dict] = {str(thr): {"precision": [], "recall": [], "f1": []} for thr in thresholds}
    for _ in range(samples):
        picked = [rng.choice(names) for _ in names]
        agg = {str(thr): {"tp": 0, "fp": 0, "fn": 0} for thr in thresholds}
        for name in picked:
            video_metrics = per_video[name]["metrics"]
            for thr in thresholds:
                key = str(thr)
                agg[key]["tp"] += video_metrics[key]["tp"]
                agg[key]["fp"] += video_metrics[key]["fp"]
                agg[key]["fn"] += video_metrics[key]["fn"]
        for thr in thresholds:
            key = str(thr)
            pr = precision_recall_f1(agg[key]["tp"], agg[key]["fp"], agg[key]["fn"])
            results[key]["precision"].append(pr["precision"])
            results[key]["recall"].append(pr["recall"])
            results[key]["f1"].append(pr["f1"])

    def quantiles(values: List[float]) -> Tuple[float, float]:
        values = sorted(values)
        lo_idx = int((alpha / 2) * (len(values) - 1))
        hi_idx = int((1 - alpha / 2) * (len(values) - 1))
        return values[lo_idx], values[hi_idx]

    ci = {}
    for thr in thresholds:
        key = str(thr)
        ci[key] = {
            "precision": quantiles(results[key]["precision"]),
            "recall": quantiles(results[key]["recall"]),
            "f1": quantiles(results[key]["f1"]),
        }
    return ci


def main() -> None:
    args = parse_args()
    thresholds = parse_conf_thresholds(args.conf_thresholds)

    base_dir = args.base_dir
    data_root = args.data_root or (base_dir / "test" / "data")
    video_root = args.video_root or (base_dir / "test" / "videos")
    if not data_root.exists():
        raise SystemExit(f"Data root not found: {data_root}")
    if not video_root.exists():
        video_root.mkdir(parents=True, exist_ok=True)

    model = YOLO(str(args.weights))
    weights_hash = hash_file(args.weights)

    data_dirs = {p.name: p for p in data_root.iterdir() if p.is_dir()}
    video_files = {p.stem: p for p in video_root.iterdir() if p.is_file() and p.suffix.lower() == ".mp4"}
    missing_video = sorted(set(data_dirs.keys()) - set(video_files.keys()))
    downloads = {"attempted": [], "succeeded": [], "failed": []}

    if args.download_missing and missing_video:
        system_files_dir = base_dir / "system_files"
        navigator = BoxNavigator(
            base_dir=str(base_dir / "test"),
            system_files_dir=str(system_files_dir),
            download_dir=str(video_root),
        )
        if args.verbose:
            print(f"[eval] Downloading {len(missing_video)} missing video(s) from Box...")
        for name in missing_video:
            candidate_names = [f"{name}.MP4", f"{name}.mp4"]
            downloads["attempted"].append(name)
            downloaded = None
            for candidate in candidate_names:
                downloaded = navigator.download_vid(candidate)
                if downloaded:
                    break
            if downloaded:
                downloads["succeeded"].append(name)
            else:
                downloads["failed"].append(name)

        video_files = {p.stem: p for p in video_root.iterdir() if p.is_file() and p.suffix.lower() == ".mp4"}
        missing_video = sorted(set(data_dirs.keys()) - set(video_files.keys()))

    common = sorted(set(data_dirs.keys()) & set(video_files.keys()))
    missing_data = sorted(set(video_files.keys()) - set(data_dirs.keys()))

    run_id = build_run_id(args.weights, args.run_name)
    output_dir = args.output_dir or (base_dir / "reports" / "eval" / run_id)
    output_dir.mkdir(parents=True, exist_ok=True)

    if args.verbose:
        print("[eval] Starting evaluation")
        print(f"[eval] Weights: {args.weights}")
        print(f"[eval] Data root: {data_root}")
        print(f"[eval] Video root: {video_root}")
        print(f"[eval] Match IoU: {args.match_iou}")
        print(f"[eval] Track IoU: {args.track_iou}")
        print(f"[eval] Track conf: {args.track_conf}")
        print(f"[eval] Confidence thresholds: {', '.join(str(t) for t in thresholds)}")
        print(f"[eval] Output dir: {output_dir}")
        print(f"[eval] Videos with data: {len(data_dirs)}")
        print(f"[eval] Videos with mp4: {len(video_files)}")
        print(f"[eval] Videos to evaluate: {len(common)}")
        if missing_data:
            print(f"[eval] Missing data for {len(missing_data)} video(s).")
        if missing_video:
            print(f"[eval] Missing video for {len(missing_video)} data folder(s).")
        if downloads["succeeded"] or downloads["failed"]:
            print(f"[eval] Downloads succeeded: {len(downloads['succeeded'])}, failed: {len(downloads['failed'])}.")

    per_video: Dict[str, Dict] = {}
    predictions_payload: Dict[str, Dict] = {}

    start_time = time.time()
    for idx, name in enumerate(common, start=1):
        if args.verbose:
            print(f"[eval] ({idx}/{len(common)}) Evaluating {name} ...")
        summary, preds = evaluate_video(
            model=model,
            video_path=video_files[name],
            data_dir=data_dirs[name],
            thresholds=thresholds,
            match_iou=args.match_iou,
            track_conf=args.track_conf,
            track_iou=args.track_iou,
        )
        per_video[name] = summary
        if args.write_predictions and preds:
            predictions_payload[name] = preds
        if args.verbose:
            print(
                f"[eval] {name}: frames={summary['frames_evaluated']} "
                f"present={summary['present_frames']} absent={summary['absent_frames']} "
                f"missing_labels={summary['missing_labels']}"
            )

    totals = {str(thr): {"tp": 0, "fp": 0, "fn": 0} for thr in thresholds}
    present_frames = 0
    absent_frames = 0
    missing_labels = 0
    frames_evaluated = 0
    label_errors = {"invalid_lines": 0, "invalid_bboxes": 0, "invalid_classes": 0, "invalid_filenames": 0}
    out_of_range_labels = 0

    for summary in per_video.values():
        present_frames += summary["present_frames"]
        absent_frames += summary["absent_frames"]
        missing_labels += summary["missing_labels"]
        frames_evaluated += summary["frames_evaluated"]
        out_of_range_labels += summary["out_of_range_labels"]
        for key in label_errors:
            label_errors[key] += summary["label_stats"][key]
        for thr in thresholds:
            key = str(thr)
            totals[key]["tp"] += summary["metrics"][key]["tp"]
            totals[key]["fp"] += summary["metrics"][key]["fp"]
            totals[key]["fn"] += summary["metrics"][key]["fn"]

    summary_metrics = {}
    for thr in thresholds:
        key = str(thr)
        summary_metrics[key] = {**totals[key], **precision_recall_f1(totals[key]["tp"], totals[key]["fp"], totals[key]["fn"])}

    eval_report = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "weights": str(args.weights),
        "weights_sha256": weights_hash,
        "data_root": str(data_root),
        "video_root": str(video_root),
        "match_iou": args.match_iou,
        "track_iou": args.track_iou,
        "track_conf": args.track_conf,
        "confidence_thresholds": thresholds,
        "download_missing_enabled": bool(args.download_missing),
        "missing_data_videos": missing_data,
        "missing_video_data": missing_video,
        "downloads": downloads,
        "summary": {
            "videos_evaluated": len(common),
            "frames_evaluated": frames_evaluated,
            "present_frames": present_frames,
            "absent_frames": absent_frames,
            "missing_labels": missing_labels,
            "out_of_range_labels": out_of_range_labels,
            "label_errors": label_errors,
            "metrics": summary_metrics,
        },
    }

    if args.bootstrap_samples > 0:
        eval_report["bootstrap"] = {
            "samples": args.bootstrap_samples,
            "alpha": args.bootstrap_alpha,
            "metrics": bootstrap_confidence_intervals(per_video, thresholds, args.bootstrap_samples, args.bootstrap_alpha),
        }

    with open(output_dir / "eval.json", "w", encoding="utf-8") as handle:
        json.dump(eval_report, handle, indent=2, sort_keys=True)

    with open(output_dir / "per_video.json", "w", encoding="utf-8") as handle:
        json.dump(per_video, handle, indent=2, sort_keys=True)

    if args.write_predictions:
        with open(output_dir / "predictions.json", "w", encoding="utf-8") as handle:
            json.dump(predictions_payload, handle, indent=2, sort_keys=True)

    if args.write_csv:
        csv_path = output_dir / "per_video.csv"
        with open(csv_path, "w", newline="", encoding="utf-8") as handle:
            writer = csv.writer(handle)
            writer.writerow(
                [
                    "video",
                    "confidence_threshold",
                    "tp",
                    "fp",
                    "fn",
                    "precision",
                    "recall",
                    "f1",
                    "frames_evaluated",
                    "present_frames",
                    "absent_frames",
                    "missing_labels",
                ]
            )
            for name, summary in per_video.items():
                for thr in thresholds:
                    key = str(thr)
                    metrics = summary["metrics"][key]
                    writer.writerow(
                        [
                            name,
                            thr,
                            metrics["tp"],
                            metrics["fp"],
                            metrics["fn"],
                            f"{metrics['precision']:.6f}",
                            f"{metrics['recall']:.6f}",
                            f"{metrics['f1']:.6f}",
                            summary["frames_evaluated"],
                            summary["present_frames"],
                            summary["absent_frames"],
                            summary["missing_labels"],
                        ]
                    )

    summary_path = output_dir / "summary.txt"
    with open(summary_path, "w", encoding="utf-8") as handle:
        handle.write(format_summary_text(eval_report, output_dir))

    elapsed = time.time() - start_time
    print(f"Videos evaluated: {len(common)}")
    print(f"Frames evaluated: {frames_evaluated}")
    print(f"Wrote eval.json: {output_dir / 'eval.json'}")
    print(f"Wrote per_video.json: {output_dir / 'per_video.json'}")
    print(f"Wrote summary.txt: {summary_path}")
    if args.write_predictions:
        print(f"Wrote predictions.json: {output_dir / 'predictions.json'}")
    if args.write_csv:
        print(f"Wrote per_video.csv: {output_dir / 'per_video.csv'}")
    print(f"Elapsed: {elapsed:.1f}s")
    if args.verbose:
        print("[eval] Evaluation finished.")


if __name__ == "__main__":
    main()
