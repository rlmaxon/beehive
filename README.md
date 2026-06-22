# Beehive Monitor — MVP

Detects and tracks bees at a hive entrance from an RTSP feed, counts directional
traffic (in/out) across a virtual line, and logs aggregated metrics you can use
as a colony-strength proxy.

Built for a Ubuntu box. No GPU required to run this — inference (the live
pipeline) is light enough for CPU. Training a model is the one step that
benefits from a GPU; see TRAINING_PLAN.md for a free-cloud-GPU path if your
box doesn't have one (most don't).

## What this gives you

- Live RTSP ingestion (no need to save raw video)
- Bee detection + tracking (Ultralytics YOLO + built-in ByteTrack/BoT-SORT)
- Directional counting across a configurable line (in vs. out per minute)
- Per-class counts (worker / drone / queen / pollen-bearer)
- CSV time-series log you can graph or feed into a dashboard
- Optional annotated video output for debugging/calibration

## What this does NOT give you (yet)

- True population size inside the hive (traffic is a proxy, not a census)
- Health/disease diagnosis

## Setup

```bash
cd beehive_monitor
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

No GPU: `pip install -r requirements.txt` pulls the CPU build of torch
automatically — nothing extra to configure. `config.yaml`'s `device: auto`
will detect there's no CUDA available and run on CPU.

## Configure

Edit `config.yaml`:

- `rtsp_url`: your camera's RTSP stream
- `model_path`: path to a YOLO model (`yolov8n.pt` to start — see below)
- `line`: two points defining the counting line across the entrance, in
  pixel coordinates of your stream's resolution
- `classes_of_interest`: which class IDs to count (depends on your model)

### Finding line coordinates

Run:

```bash
python -m src.calibrate --config config.yaml
```

This grabs one frame from the stream and opens it in a window — click two
points to define the counting line, coordinates get printed to paste into
config.yaml. Press `q` to quit without saving.

## Run

```bash
python -m src.pipeline --config config.yaml
```

Counts get appended to `logs/traffic_log.csv` every `log_interval_seconds`
(default 60s): timestamp, count_in, count_out, net, and per-class breakdowns
if available.

Add `--display` to pop up an annotated window while it runs (skip this once
you're running headless on a server). Add `--save-video out.mp4` to save the
annotated stream for later review — useful while calibrating, not something
you want running 24/7 (fills disk fast).

## Starting model

`yolov8n.pt` (stock, COCO-trained) does NOT know what a bee is. It's only
useful to sanity-check that your pipeline runs end-to-end (it'll detect
nothing bee-related, but you'll confirm RTSP decode, tracking, and counting
logic work). You need a bee-trained model before the counts mean anything —
see TRAINING_PLAN.md, which covers merging public worker/drone/queen/
pollen-bearer datasets (via `train/merge_datasets.py`) and training on CPU
or a free cloud GPU.
