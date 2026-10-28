"""Step 6 - Sensitivity, optimisation and robustness studies.

A. Input resolution: evaluate the trained detector at several inference sizes
   -> accuracy (mAP50, tray recall) vs latency vs GFLOPs trade-off.
B. Robustness: apply realistic disturbances (lighting, motion blur, sensor
   noise, film glare) to the test trays and measure how the PASS/FAIL
   decision degrades, for both the detector and the classical baseline.

Outputs: results/sensitivity/<run>/{imgsz.json, imgsz.png, robustness.json, robustness.png}

Usage:
    python scripts/06_sensitivity.py --run grouped_yolov8n
"""
import _setup  # noqa: F401

import argparse
import json

import cv2
import matplotlib.pyplot as plt
import numpy as np

from porkqc.baseline import build_baseline, extract_features, load_features
from porkqc.config import load_config
from porkqc.data import label_path_for, list_images, read_label_file, tray_label, write_data_yaml
from porkqc.detector import resolve_device, validate
from porkqc.metrics import binary_metrics
from porkqc.profiling import scale_flops, time_function
from porkqc.robustness import PERTURBATIONS
from porkqc.reporting import save_json


def yolo_scores(model, imgs, imgsz, dev):
    """Max defect confidence per image for a list of BGR arrays."""
    scores = []
    for i in range(0, len(imgs), 16):
        for r in model.predict(imgs[i:i + 16], imgsz=imgsz, conf=0.01, device=dev, verbose=False):
            scores.append(float(r.boxes.conf.max()) if len(r.boxes) else 0.0)
    return np.array(scores)


def main() -> None:
    cfg = load_config()
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run", default="grouped_yolov8n")
    ap.add_argument("--skip-imgsz", action="store_true")
    a = ap.parse_args()

    from ultralytics import YOLO

    run_dir = cfg["paths"]["results"] / "runs" / a.run
    ev = json.loads((cfg["paths"]["results"] / "eval" / a.run / "evaluation.json").read_text())
    thr = ev["threshold_from_val"]
    meta = json.loads((run_dir / "train_meta.json").read_text())
    root = cfg["paths"]["grouped_data" if meta["split"] == "grouped" else "raw_data"]
    weights = run_dir / "weights" / "best.pt"
    dev = resolve_device(cfg["yolo"]["device"])
    model = YOLO(str(weights))
    out = cfg["paths"]["results"] / "sensitivity" / a.run
    out.mkdir(parents=True, exist_ok=True)

    test_imgs = list_images(root / "test" / "images")
    y = np.array([tray_label(read_label_file(label_path_for(p))) for p in test_imgs])
    arrays = [cv2.imread(str(p)) for p in test_imgs]
    gflops640 = ev["complexity"]["gflops"]
    gflops640 = gflops640 / (meta["imgsz"] / 640) ** 2 if gflops640 else None

    # ---- A. input resolution -------------------------------------------------
    if not a.skip_imgsz:
        rows = []
        for sz in cfg["sensitivity"]["imgsz_values"]:
            det = validate(weights, write_data_yaml(root), "test", sz, dev, out, f"val_img{sz}")
            s = yolo_scores(model, arrays, sz, dev)
            tray = binary_metrics(y, (s >= thr).astype(int))
            lat = time_function(lambda: model.predict(arrays[0], imgsz=sz, device=dev, verbose=False), 30)
            rows.append({"imgsz": sz, "mAP50": det["mAP50"], "mAP50_95": det["mAP50_95"],
                         "tray_recall": tray["recall"], "tray_specificity": tray["specificity"],
                         "median_ms": lat["median_ms"], "fps": lat["fps"],
                         "gflops": scale_flops(gflops640, sz) if gflops640 else None})
            print(rows[-1])
        save_json(rows, out / "imgsz.json")
        fig, ax1 = plt.subplots(figsize=(6, 3.5))
        sz = [r["imgsz"] for r in rows]
        ax1.plot(sz, [r["mAP50"] for r in rows], "o-", label="mAP50")
        ax1.plot(sz, [r["tray_recall"] for r in rows], "s-", label="tray recall")
        ax1.set_xlabel("Inference image size (px)")
        ax1.set_ylabel("Accuracy")
        ax2 = ax1.twinx()
        ax2.plot(sz, [r["median_ms"] for r in rows], "k^--", label="latency")
        ax2.set_ylabel(f"Median latency (ms) on {dev}")
        fig.legend(loc="lower right", fontsize=8)
        ax1.set_title("Accuracy vs speed trade-off")
        fig.tight_layout()
        fig.savefig(out / "imgsz.png", dpi=150)
        plt.close(fig)

    # ---- B. robustness -------------------------------------------------------
    # Baseline re-trained with its validation-selected C, so both models are
    # stressed with identical images.
    bfile = cfg["paths"]["results"] / "baseline" / f"baseline_{meta['split']}.json"
    baseline = None
    if bfile.exists():
        C = json.loads(bfile.read_text())["chosen_C"]
        tr = list_images(root / "train" / "images")
        ytr = np.array([tray_label(read_label_file(label_path_for(p))) for p in tr])
        baseline = build_baseline(C=C, seed=cfg["seed"]).fit(load_features(tr), ytr)

    results = {}
    for pname, (fn, levels) in PERTURBATIONS.items():
        results[pname] = []
        for lvl in levels:
            pert = [fn(im, lvl) for im in arrays]
            s = yolo_scores(model, pert, meta["imgsz"], dev)
            row = {"level": lvl, "yolo": binary_metrics(y, (s >= thr).astype(int))}
            if baseline is not None:
                bp = baseline.predict(np.vstack([extract_features(im) for im in pert]))
                row["baseline"] = binary_metrics(y, bp)
            results[pname].append(row)
            print(pname, lvl, "yolo bal.acc=%.3f" % row["yolo"]["balanced_accuracy"],
                  "baseline bal.acc=%.3f" % row["baseline"]["balanced_accuracy"] if baseline else "")
        cv2.imwrite(str(out / f"example_{pname}.jpg"), np.hstack([fn(arrays[0], levels[0]), fn(arrays[0], levels[-1])]))
    save_json(results, out / "robustness.json")

    fig, axes = plt.subplots(1, len(results), figsize=(14, 3.3), sharey=True)
    for ax, (pname, rows) in zip(axes, results.items()):
        lv = [str(r["level"]) for r in rows]
        ax.plot(lv, [r["yolo"]["recall"] for r in rows], "o-", label="YOLO recall")
        ax.plot(lv, [r["yolo"]["specificity"] for r in rows], "o--", label="YOLO specificity")
        if baseline is not None:
            ax.plot(lv, [r["baseline"]["balanced_accuracy"] for r in rows], "s:", label="baseline bal.acc")
        ax.set_title(pname)
        ax.set_xlabel("severity")
    axes[0].set_ylabel("Test metric")
    axes[0].legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(out / "robustness.png", dpi=150)
    print(f"Saved to {out}")


if __name__ == "__main__":
    main()
