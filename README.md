# YOLO Video Annotator

A PyQt5 tool for running YOLO inference on a video, reviewing/editing bounding boxes, and exporting YOLO-formatted datasets. It can read a local video or download one from Box on launch.

## Project layout
- `src/` — application code (`annotator_app.py`, `annotator_view.py`, `bounding_box.py`, `inference.py`, `box_downloader.py`)
- `models/` — YOLO weights (e.g., `best.pt`; ignored by Git if you store large checkpoints)
- `data/sample_videos/` — optional sample clips (ignored by Git)
- `outputs/` — generated files (downloads, system files/credentials, temp labels, exported train data)
- `requirements.txt` — runtime dependencies

## Setup
```bash
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\\Scripts\\activate
pip install -r requirements.txt
```

## Usage
Run the app as a module so relative imports resolve correctly:
```bash
python -m src.annotator_app --video_path data/sample_videos/your_video.mp4 --yolo_weights models/best.pt
```

Download from Box instead:
```bash
python -m src.annotator_app --box_video ExactFileName.mp4 --yolo_weights models/best.pt
```
Downloads, YOLO temp labels, and exports land under `outputs/`.

## Notes
- Box credentials are stored locally at `outputs/system_files/box_credentials.txt` (not committed).
- Exported datasets are written to `outputs/train_data/<video_name>/`.
- YOLO temp label files live in `outputs/yolo_temp_labels/`.
