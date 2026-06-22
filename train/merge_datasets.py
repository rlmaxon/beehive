"""Merge several public Roboflow (YOLOv8-format) bee datasets into one
dataset with a unified worker/drone/queen/pollen_bearer class scheme.

Why this is needed: each public dataset defines its own class IDs in its own
data.yaml (e.g. one dataset might have worker=0,drone=1,queen=2,pollen=3,
another might order them differently or use different label spellings). You
can't just concatenate label files across datasets without remapping class
IDs first, or you'll silently train on garbage (a "drone" box in one dataset
becomes a "queen" box in the merged set).

This script:
  1. Reads each source dataset's data.yaml to get its class names.
  2. Maps each source class name to one of the canonical classes
     (worker / drone / queen / pollen_bearer) via a synonym table --
     unrecognized classes (e.g. "varroa") are DROPPED (their boxes are
     removed from the label files, the image itself is kept).
  3. Copies images + rewritten label files into train/<out>/{train,val,test}.
  4. Writes a merged data.yaml pointing at the canonical 4-class scheme.

Note on pollen_bearer specifically: this isn't a separate bee caste, it's a
worker bee that happens to be carrying a pollen load when the photo was
taken. A handful of source images may have the SAME bee labeled "worker" in
one dataset and "pollen_bearer" in another depending on each dataset's
labeling convention -- there's no way to reconcile that automatically, it's
an inherent inconsistency in combining independently-labeled public datasets.
Worth a spot check of a sample of merged images once this runs.

Usage:
  python train/merge_datasets.py \
      --sources path/to/roboflow_export_1 path/to/roboflow_export_2 \
      --out train/data

Each --sources path should be the root of a Roboflow YOLOv8 export, i.e. the
folder containing data.yaml and train/, valid/ (or val/), test/ subfolders.

IMPORTANT: print the per-source class mapping this script reports and sanity
check it before trusting the merge -- label spellings vary across datasets
and a wrong synonym match silently corrupts your labels.
"""
from __future__ import annotations

import argparse
import shutil
from pathlib import Path

import yaml

CANONICAL = {"worker": 0, "drone": 1, "queen": 2, "pollen_bearer": 3}

# Synonyms seen across public bee datasets. Extend this if a dataset you pull
# uses a spelling not covered here -- the script will tell you which class
# names it could not map.
SYNONYMS = {
    "worker": {"worker", "worker_bee", "workerbee", "bee", "honeybee", "honey_bee"},
    "drone": {"drone", "drone_bee", "dronebee"},
    "queen": {"queen", "queen_bee", "queenbee"},
    "pollen_bearer": {
        "pollen_bearer", "pollenbearer", "pollen_bee", "pollenbee",
        "bee_with_pollen", "pollen", "forager_pollen",
    },
}

SPLIT_ALIASES = {"train": "train", "valid": "val", "val": "val", "test": "test"}


def normalize(name: str) -> str:
    return name.strip().lower().replace(" ", "_").replace("-", "_")


def map_class_name(name: str) -> str | None:
    n = normalize(name)
    for canon, syns in SYNONYMS.items():
        if n in syns:
            return canon
    return None


def load_class_names(data_yaml: Path) -> dict[int, str]:
    with open(data_yaml) as f:
        cfg = yaml.safe_load(f)
    names = cfg["names"]
    if isinstance(names, dict):
        return {int(k): v for k, v in names.items()}
    return dict(enumerate(names))  # list form


def find_split_dir(source: Path, split_alias: str) -> Path | None:
    candidate = source / split_alias
    if (candidate / "images").is_dir():
        return candidate
    return None


def merge(sources: list[Path], out_dir: Path):
    out_dir = Path(out_dir)
    for split in ("train", "val", "test"):
        (out_dir / "images" / split).mkdir(parents=True, exist_ok=True)
        (out_dir / "labels" / split).mkdir(parents=True, exist_ok=True)

    totals = {"kept_boxes": 0, "dropped_boxes": 0, "images": 0}

    for src_idx, source in enumerate(sources):
        data_yaml = source / "data.yaml"
        if not data_yaml.exists():
            print(f"SKIP {source}: no data.yaml found")
            continue

        src_names = load_class_names(data_yaml)
        id_map: dict[int, int | None] = {}
        print(f"\nSource: {source}")
        for cid, name in src_names.items():
            canon = map_class_name(name)
            id_map[cid] = CANONICAL[canon] if canon else None
            status = f"-> {canon} ({CANONICAL[canon]})" if canon else "-> DROPPED (no canonical match)"
            print(f"  class {cid} '{name}' {status}")

        for split_alias, out_split in SPLIT_ALIASES.items():
            split_dir = find_split_dir(source, split_alias)
            if split_dir is None:
                continue
            img_dir = split_dir / "images"
            lbl_dir = split_dir / "labels"

            for img_path in img_dir.iterdir():
                if not img_path.is_file():
                    continue
                stem = img_path.stem
                label_path = lbl_dir / f"{stem}.txt"

                new_lines = []
                if label_path.exists():
                    for line in label_path.read_text().splitlines():
                        if not line.strip():
                            continue
                        parts = line.split()
                        src_cid = int(parts[0])
                        new_cid = id_map.get(src_cid)
                        if new_cid is None:
                            totals["dropped_boxes"] += 1
                            continue
                        parts[0] = str(new_cid)
                        new_lines.append(" ".join(parts))
                        totals["kept_boxes"] += 1

                # prefix filenames with source index to avoid collisions across datasets
                out_stem = f"src{src_idx}_{stem}"
                out_img = out_dir / "images" / out_split / f"{out_stem}{img_path.suffix}"
                out_lbl = out_dir / "labels" / out_split / f"{out_stem}.txt"

                shutil.copy2(img_path, out_img)
                out_lbl.write_text("\n".join(new_lines) + ("\n" if new_lines else ""))
                totals["images"] += 1

    data_yaml_out = out_dir / "data.yaml"
    data_yaml_out.write_text(
        yaml.dump(
            {
                "path": str(out_dir.resolve()),
                "train": "images/train",
                "val": "images/val",
                "test": "images/test",
                "names": {v: k for k, v in CANONICAL.items()},
            },
            sort_keys=False,
        )
    )

    print(f"\nMerged {totals['images']} images, kept {totals['kept_boxes']} boxes, "
          f"dropped {totals['dropped_boxes']} boxes (unmapped classes).")
    print(f"Wrote {data_yaml_out}")
    print("Train with: python train/train.py --data", data_yaml_out)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--sources", nargs="+", required=True, type=Path)
    parser.add_argument("--out", default="train/data", type=Path)
    args = parser.parse_args()
    merge(args.sources, args.out)


if __name__ == "__main__":
    main()
