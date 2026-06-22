# Model Training Plan

Goal: a YOLOv8 model that detects worker bees, drones, queens, and
pollen-bearing bees, trained entirely from public datasets — no footage from
your own camera required for this version.

## 0. No local GPU — pick a training path

Your Ubuntu box has no GPU, so training runs on CPU by default
(`train/train.py` defaults to `--device cpu`). Two honest options:

**Option A: Train locally on CPU.** Works, just slow. For ~1,500-3,000 images
(roughly what you get merging the 3 datasets below) at `yolov8n`, imgsz 512,
batch 8, expect somewhere in the ballpark of 6-15 hours for 60 epochs on a
typical multi-core desktop CPU — it varies a lot with core count, so treat
that as a rough planning number, not a promise. It's a background job: kick
it off, walk away, check back later. Lower `--epochs` (e.g. 30) for a faster
first pass to confirm the pipeline works before committing to a long run.

**Option B (recommended): train on a free cloud GPU, deploy on your CPU box.**
Training is the only step that benefits from a GPU — running the live
pipeline afterward is cheap (single image at a time, not a batch) and works
fine on CPU. Google Colab's free tier gives you a T4 GPU with no payment
info required:

1. Open a new Colab notebook (colab.research.google.com), set
   Runtime → Change runtime type → T4 GPU.
2. Upload your merged dataset folder (zip it first) or download the public
   datasets directly into Colab.
3. ```python
   !pip install ultralytics
   from ultralytics import YOLO
   model = YOLO("yolov8n.pt")
   model.train(data="data.yaml", epochs=60, imgsz=640, batch=16, device=0)
   ```
4. This takes roughly 10-20 minutes on a free T4 instead of hours on CPU.
5. Download `runs/detect/train/weights/best.pt`, copy it to your Ubuntu box's
   `beehive_monitor/models/` folder, point `config.yaml`'s `model_path` at it.

Either way the *inference* side (the actual live pipeline watching your
RTSP feed) runs on CPU on your Ubuntu box — that part was never GPU-dependent.
`config.yaml` already has `vid_stride: 2` set to ease CPU load by skipping
every other frame; raise it to 3-4 if the live pipeline can't keep up with
your camera's frame rate (tradeoff: a fast-moving bee could cross the line
between sampled frames and go uncounted — start at 2 and only raise it if you
see the pipeline visibly falling behind the stream).

## 1. Pull and merge public datasets (day 1)

Public datasets that include pollen-bearer alongside worker/drone/queen:

- Roboflow Universe "Honey Bee Detection Model" (Matt Nudi) — 909 images,
  classes worker/drone/queen/pollen-bearer. This is your main source for the
  pollen-bearer class.
  https://universe.roboflow.com/matt-nudi/honey-bee-detection-model-zgjnb
- Roboflow Universe "Bee Detection" — bee/drone/pollenbee/queen, YOLOv8 format.
  https://universe.roboflow.com/workspace-1-tu7e5/bee-detection-h59jf
- Roboflow Universe "Honey Bee Project" — actively maintained, YOLOv8/v11
  export formats. Check its class list on the page before downloading; it may
  not include pollen-bearer in every version.
  https://universe.roboflow.com/honey-bee-project/honey-bee-project
- Browse more at https://universe.roboflow.com/search?q=class:honeybee — search
  also for "pollen" specifically if you want more pollen-bearer-labeled data,
  since it's the sparsest class (fewer images naturally catch a bee mid-pollen-load
  vs. a plain worker).

For each, download in "YOLOv8" export format (a Roboflow account is free) and
unzip — you get a folder with `data.yaml` plus `train/`, `valid/`, `test/`
subfolders, each containing `images/` and `labels/`.

Each dataset numbers its classes differently (one might have drone=0, another
drone=1) and spells class names differently ("Worker Bee" vs "worker",
"pollenbee" vs "pollen_bearer"), so don't just dump the label files together.
Use the merge script, which reads each dataset's own `data.yaml` and remaps
every class to a single canonical worker=0/drone=1/queen=2/pollen_bearer=3
scheme by name:

```bash
python train/merge_datasets.py \
  --sources path/to/roboflow_export_1 path/to/roboflow_export_2 path/to/roboflow_export_3 \
  --out train/data
```

It prints the per-class mapping it inferred for each source — read that
output before trusting the merge. If a dataset uses a spelling the script
doesn't recognize, it'll say "DROPPED (no canonical match)" for that class;
add the spelling to the `SYNONYMS` dict at the top of
`train/merge_datasets.py` and rerun rather than losing that data silently.

Two things to expect with pollen_bearer specifically: it'll be the smallest
class by image count (fewer frames happen to catch a bee mid-pollen-load),
so the model will likely have lower recall on it than on worker/drone — don't
be surprised if it needs the most attention later. Also, since
"pollen-bearer" isn't a distinct bee caste, the same bee gets labeled
"worker" in one dataset's convention and "pollen_bearer" in another's for a
visually similar shot — that's an inherent inconsistency in combining
independently-labeled datasets, not a bug in the merge.

## 2. Train (day 1-2, depending on path chosen above)

```bash
python train/train.py --data train/data/data.yaml --base yolov8n.pt --epochs 60
```
(or run the Colab snippet from step 0 if going the cloud-GPU route, then copy
`best.pt` back).

## 3. Evaluate against what actually matters

mAP is a useful training signal but the metric you actually care about is
traffic-count accuracy. After training:

1. Pick a 10-15 minute clip you have NOT used in training.
2. Manually count bees crossing the entrance in that clip (yes, by eye — slow
   it down, it's tedious but it's your ground truth).
3. Run the pipeline on that same clip and compare counts.
4. Track mean absolute error between manual and model counts. The Be-Hive
   Project research benchmark (Univ. Twente) reports ~5.7 MAE as a reference
   point for what "good" looks like in this problem space — use that as a
   rough target, not a hard bar.

If counts are systematically off in one direction, it's usually one of:
missed detections in low light (public datasets are mostly daylight shots —
this is the most likely gap), double counting from a flickering/lost track ID
(tune tracker confidence/IOU thresholds in `bytetrack.yaml`), or bees clumping
at the entrance causing occlusion.

## 4. When you outgrow the public-only model

A model trained purely on public datasets will likely underperform on your
specific camera's angle, distance, and lighting compared to one fine-tuned on
your own footage — that's the expected domain-gap tradeoff for skipping data
collection in this version. If accuracy isn't good enough once you check it
against a real clip from your hive (step 3 above), the next step is
collecting and labeling a few hundred frames from your own RTSP feed and
fine-tuning further:

```bash
python train/extract_frames.py --source rtsp://your-camera-url --out train/raw_frames --every 2 --max 500
```

Label those in Roboflow (model-assisted labeling using your current model
speeds this up a lot — you're correcting boxes, not drawing from scratch),
export in YOLOv8 format, and fine-tune again starting from your current
weights rather than from scratch:

```bash
python train/train.py --data <your_export>/data.yaml --base train/runs/bee_detector/weights/best.pt --epochs 40
```

300-500 of your own labeled frames on top of the public-dataset base usually
closes most of the domain gap.

## Reference reading

- Survey: "Machine Learning and Computer Vision Techniques in Continuous
  Beehive Monitoring Applications" — https://arxiv.org/pdf/2208.00085
- "Accuracy vs. Energy" comparison of YOLOv3/v4-tiny/v7-tiny on bee video —
  useful if you later need to shrink the model for lower-power hardware:
  https://www.ncbi.nlm.nih.gov/pmc/articles/PMC10422429/
- Be-Hive Project (pose-estimation based counting, useful design reference
  for v2): https://research.utwente.nl/en/publications/the-be-hive-projectcounting-bee-traffic-based-on-deep-learning-an/
