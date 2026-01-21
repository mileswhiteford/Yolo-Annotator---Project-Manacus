# Manakin YOLO Annotator

A set of PyQt5 tools for running YOLO inference on videos, reviewing/editing boxes, and exporting YOLO-formatted datasets for training and testing.

Note: Requires an already pre-trained YOLO model. This pipeline is designed for improving model performance via targeted curation, error review, and clean train/test splits (not training from scratch). 

## Project layout
- `src/manakin_yolo_annotator/` — core application code
- `scripts/` — CLI wrappers for the tools
- `annotations/` — curated training annotations + videos (data, videos, audio annotations)
- `test/` — test outputs (data, videos, audio annotations)
- `splits/` — derived train/val splits + metadata
- `system_files/` — shared Box credentials + cached Box index (gitignored)
- `models/` — YOLO weights (gitignored)
- `requirements.txt` — runtime dependencies
- `pyproject.toml` — package + console entrypoints

## Setup
```bash
cd Yolo-Annotator---Project-Manacus
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install -r requirements.txt
pip install -e .
```

## Quickstart
1. Add pre-trained yolo model to "models/" folder
1. Build the Box index cache:
   ```bash
   random-vid --annotate --count 1
   ```
2. Curate training frames:
   ```bash
   frame-curator --weights models/best.pt
   ```
3. Flag failure cases for review:
   ```bash
   error-detector --weights models/best.pt
   ```
4. Create a test set:
   ```bash
   annotator --weights models/best.pt --box-video ExactFileName.mp4
   ```
5. Split and validate:
   ```bash
   train-split --root annotations/data --split-name yolo-split
   validate-dataset --root annotations/data --output validation.json
   ```

## Tools
### Annotator (test by default)
Annotate full videos (or snippets), edit boxes, and export YOLO datasets.

```bash
annotator --video-path path/to/video.mp4 --weights models/best.pt
# or download from Box
annotator --box-video ExactFileName.mp4 --weights models/best.pt
```

Defaults to saving in `test/data/<video_name>/images|labels`.
Use `--train_set` to save into `annotations/data/<video_name>/...` instead.
Tip: If you run from outside the repo, pass `--base-dir /path/to/manakin_YOLO_annotator` so split checks use the right folders.

### Frame Curator (annotations)
Curate frames with UI editing and save selected frames.

```bash
frame-curator --weights models/best.pt
# optional
# --video-path path/to/video.mp4
# --box-video ExactFileName.mp4
# --frame-stride 30
# --folder-id 162855223530
# --rebuild-index
# --seed 123
# --run-new
```

If no video is specified, a random video is selected from the Box folder index.
Saves to `annotations/data/<video_name>/images|labels` and downloads into `annotations/videos/`.
Note: `random-vid` builds the Box index cache on first run; run it once to create the JSON index used for random selection.

### Error_Detector (annotations)
Detect likely failure frames, edit boxes, and save corrected frames.

```bash
error-detector --weights models/best.pt
# optional
# --video-path path/to/video.mp4
# --box-video ExactFileName.mp4
# --random-count 5
# --folder-id 162855223530
# --rebuild-index
# --seed 123
# --run-new
# --iou-thresh 0.1
# --flicker-len 3
```

If no video is specified, a random video is selected from the Box folder index.
Saves to `annotations/data/<video_name>/images|labels` and downloads into `annotations/videos/`.

### Random video downloader
Download random videos from Box.

```bash
random-vid --annotate --count 5
random-vid --test --count 5
```
Note: `random-vid` builds the Box index cache on first run; run it once to create the JSON index used for random selection.

### Train/val split
Create a YOLO train/val split across videos (optional). Writes `manifest.json` and `stats.json`.

```bash
train-split --root annotations/data --split-name yolo-split
# or
# python scripts/train_split.py --root annotations/data --split-name yolo-split
```

### Validation
Validate dataset structure + labels and write `validation.json`.

```bash
validate-dataset --root annotations/data --output validation.json
# or
# python scripts/validate_dataset.py --root annotations/data --output validation.json
```

## Output structure
```
annotations/
├─ data/
│  └─ <video_name>/
│     ├─ images/
│     └─ labels/
├─ jump_audio_data/
│  └─ <video_name>/
└─ videos/
   └─ <video_name>.mp4

test/
├─ data/
│  └─ <video_name>/
│     ├─ images/
│     └─ labels/
├─ jump_audio_data/
│  └─ <video_name>/
└─ videos/
   └─ <video_name>.mp4

splits/
└─ yolo-split/
   ├─ images/
   │  ├─ train/
   │  └─ val/
   ├─ labels/
   │  ├─ train/
   │  └─ val/
   ├─ manifest.json
   └─ stats.json

system_files/
└─ box_credentials.txt
```

## Notes
- Box credentials are stored at `system_files/box_credentials.txt` and are not committed.
- Labels are written as single-class YOLO with class id `0`.
- If you want to run wrappers without installing, use `PYTHONPATH=src`:
  ```bash
  PYTHONPATH=src python scripts/annotator.py --video-path ... --weights ...
  ```
