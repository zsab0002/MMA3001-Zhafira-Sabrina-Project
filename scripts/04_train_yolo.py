"""Step 4 - Fine-tune YOLOv8 detectors.

Trains every model listed in ``configs/config.yaml -> yolo.models`` on the
chosen split. Runs are saved to ``results/runs/<split>_<model>/``; Ultralytics
writes training curves (results.png), the confusion matrix and best.pt there.

Usage:
    python scripts/04_train_yolo.py --split grouped                 # main experiment
    python scripts/04_train_yolo.py --split original                # leakage comparison
    python scripts/04_train_yolo.py --split grouped --quick         # 3-epoch smoke test
    python scripts/04_train_yolo.py --split grouped --models yolov8n.pt --epochs 30
    python scripts/04_train_yolo.py --split grouped --models yolov8n.pt --fraction 0.25
"""
import _setup  # noqa: F401

import argparse
import json
import time

from porkqc.config import load_config
from porkqc.data import write_data_yaml
from porkqc.detector import train


def main() -> None:
    cfg = load_config()
    y = cfg["yolo"]
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--split", choices=["grouped", "original"], default="grouped")
    ap.add_argument("--models", nargs="+", default=y["models"])
    ap.add_argument("--epochs", type=int, default=y["epochs"])
    ap.add_argument("--imgsz", type=int, default=y["imgsz"])
    ap.add_argument("--batch", type=int, default=y["batch"])
    ap.add_argument("--device", default=y["device"])
    ap.add_argument("--fraction", type=float, default=1.0, help="fraction of training images used")
    ap.add_argument("--quick", action="store_true", help="3 epochs on 20%% of data to test the pipeline")
    a = ap.parse_args()
    if a.quick:
        a.epochs, a.fraction = 3, 0.2

    root = cfg["paths"]["grouped_data" if a.split == "grouped" else "raw_data"]
    if not (root / "train").exists():
        raise SystemExit(f"{root} not found - run steps 1-2 first.")
    data_yaml = write_data_yaml(root)
    runs = cfg["paths"]["results"] / "runs"

    for weights in a.models:
        name = f"{a.split}_{weights.replace('.pt', '')}"
        if a.fraction < 1:
            name += f"_frac{int(a.fraction * 100)}"
        if a.imgsz != y["imgsz"]:
            name += f"_img{a.imgsz}"
        print(f"\n=== Training {weights} on '{a.split}' -> {runs / name} ===")
        t0 = time.perf_counter()
        best = train(weights, data_yaml, runs, name, epochs=a.epochs, imgsz=a.imgsz,
                     batch=a.batch, patience=y["patience"], device=a.device,
                     workers=y["workers"], seed=cfg["seed"], fraction=a.fraction)
        minutes = (time.perf_counter() - t0) / 60
        meta = {"weights": weights, "split": a.split, "epochs_max": a.epochs, "imgsz": a.imgsz,
                "fraction": a.fraction, "train_minutes": minutes, "best": str(best)}
        (runs / name / "train_meta.json").write_text(json.dumps(meta, indent=2))
        print(f"done in {minutes:.1f} min, best weights: {best}")


if __name__ == "__main__":
    main()
