"""Step 5 - Evaluate trained detectors and compare against the baseline.

For every run in results/runs/ (or those given with --runs):

1. Detection metrics on the run's TEST split (precision, recall, mAP50,
   mAP50-95, per-class AP50) via Ultralytics.
2. Tray-level PASS/FAIL metrics. The decision threshold is tuned on the
   VALIDATION set only (highest threshold that keeps FAIL-recall >= target),
   then applied once to the test set - the test set never influences a choice.
3. Bootstrap 95 % confidence intervals, a threshold sweep plot, and images of
   the worst errors (missed defects and false rejections) for failure analysis.
4. A comparison table (Markdown) including the classical baseline.

Usage:
    python scripts/05_evaluate.py
    python scripts/05_evaluate.py --runs grouped_yolov8n
"""
import _setup  # noqa: F401

import argparse
import json
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np

from porkqc.config import load_config
from porkqc.data import label_path_for, list_images, read_label_file, tray_label, write_data_yaml
from porkqc.detector import model_complexity, predict, validate
from porkqc.inspection import tray_scores
from porkqc.metrics import binary_metrics, bootstrap_ci, threshold_sweep
from porkqc.profiling import time_function
from porkqc.reporting import draw_boxes, markdown_table, save_json


def tune_threshold(y_val, s_val, grid, target_recall):
    """Highest threshold on the grid whose validation recall >= target (else best F1)."""
    sweep = threshold_sweep(y_val, s_val, grid)
    ok = [r for r in sweep if r["recall"] >= target_recall]
    return (max(ok, key=lambda r: r["threshold"]) if ok else max(sweep, key=lambda r: r["f1"]))["threshold"]


def save_error_examples(images, y, scores, preds, thr, out_dir, n=6):
    """Save the most confident mistakes: missed defects (FN) and false rejects (FP)."""
    out_dir.mkdir(parents=True, exist_ok=True)
    fn = [i for i in np.argsort(scores) if y[i] == 1 and scores[i] < thr][:n]          # lowest score FAIL trays
    fp = [i for i in np.argsort(-scores) if y[i] == 0 and scores[i] >= thr][:n]        # highest score PASS trays
    for tag, idxs in (("missed_defect", fn), ("false_reject", fp)):
        for i in idxs:
            p = images[i]
            img = cv2.imread(str(p))
            h, w = img.shape[:2]
            gt = [(b.cls, None, b.to_xyxy(w, h)) for b in read_label_file(label_path_for(p))]
            det = [(d.cls, d.conf, d.xyxy) for d in preds.get(p.name, []) if d.conf >= 0.05]
            img = draw_boxes(img, gt + det, f"{tag}  score={scores[i]:.2f}  thr={thr:.2f}  "
                                             f"(thin=truth, thick=predicted)", (0, 0, 255))
            cv2.imwrite(str(out_dir / f"{tag}_{p.name}"), img)


def evaluate_run(run_dir: Path, cfg: dict) -> dict:
    meta = json.loads((run_dir / "train_meta.json").read_text())
    root = cfg["paths"]["grouped_data" if meta["split"] == "grouped" else "raw_data"]
    data_yaml = write_data_yaml(root)
    weights = run_dir / "weights" / "best.pt"
    imgsz = meta["imgsz"]
    out = cfg["paths"]["results"] / "eval" / run_dir.name
    out.mkdir(parents=True, exist_ok=True)
    print(f"\n=== Evaluating {run_dir.name} ===")

    det = validate(weights, data_yaml, "test", imgsz, cfg["yolo"]["device"], out, "test_metrics")

    tray = {}
    preds_all = {}
    for split in ("valid", "test"):
        images = list_images(root / split / "images")
        preds = predict(weights, images, imgsz, conf=0.01, device=cfg["yolo"]["device"])
        y = np.array([tray_label(read_label_file(label_path_for(p))) for p in images])
        s = tray_scores(preds, [p.name for p in images])
        tray[split] = (images, y, s)
        preds_all[split] = preds

    grid = cfg["sensitivity"]["conf_values"]
    thr = tune_threshold(tray["valid"][1], tray["valid"][2], grid, cfg["inspection"]["target_recall"])
    images, y, s = tray["test"]
    pred = (s >= thr).astype(int)
    test = binary_metrics(y, pred)
    test["recall_95ci"] = bootstrap_ci(y, pred, "recall")
    test["specificity_95ci"] = bootstrap_ci(y, pred, "specificity")
    default_thr = binary_metrics(y, (s >= cfg["inspection"]["conf_threshold"]).astype(int))

    # Threshold sensitivity on the test set (reported, NOT used for selection).
    sweep = threshold_sweep(y, s, np.linspace(0.01, 0.95, 60))
    fig, ax = plt.subplots(figsize=(6, 3.5))
    t = [r["threshold"] for r in sweep]
    for k in ("recall", "specificity", "f1"):
        ax.plot(t, [r[k] for r in sweep], label=k)
    ax.axvline(thr, color="k", ls="--", lw=1, label=f"chosen (val) = {thr:.2f}")
    ax.set_xlabel("Confidence threshold for FAIL")
    ax.set_ylabel("Test metric")
    ax.set_title(f"Tray decision vs threshold - {run_dir.name}")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out / "threshold_sweep.png", dpi=150)
    plt.close(fig)

    save_error_examples(images, y, s, preds_all["test"], thr, out / "errors")

    # Latency of the full predict call for one image (includes pre/post-processing).
    from ultralytics import YOLO
    from porkqc.detector import resolve_device
    model = YOLO(str(weights))
    dev = resolve_device(cfg["yolo"]["device"])
    img = cv2.imread(str(images[0]))
    latency = time_function(lambda: model.predict(img, imgsz=imgsz, device=dev, verbose=False), n_repeat=30)

    result = {
        "run": run_dir.name, "split": meta["split"], "model": meta["weights"], "imgsz": imgsz,
        "train_minutes": meta.get("train_minutes"), "detection_test": det,
        "threshold_from_val": thr, "tray_test": test, "tray_test_default_threshold": default_thr,
        "latency_per_image": latency, "device": dev,
        "complexity": model_complexity(weights, imgsz),
    }
    save_json(result, out / "evaluation.json")
    save_json({k: [d.__dict__ for d in v] for k, v in preds_all["test"].items()}, out / "test_predictions.json")
    print(f"mAP50={det['mAP50']:.3f} mAP50-95={det['mAP50_95']:.3f} | tray recall={test['recall']:.3f} "
          f"specificity={test['specificity']:.3f} @thr={thr:.2f} | {latency['median_ms']:.1f} ms/img on {dev}")
    return result


def main() -> None:
    cfg = load_config()
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--runs", nargs="*", help="run folder names (default: all in results/runs)")
    a = ap.parse_args()
    runs_dir = cfg["paths"]["results"] / "runs"
    runs = [runs_dir / r for r in a.runs] if a.runs else sorted(
        p for p in runs_dir.glob("*") if (p / "weights" / "best.pt").exists())
    if not runs:
        raise SystemExit("No trained runs found - run scripts/04_train_yolo.py first.")

    rows = []
    for r in runs:
        res = evaluate_run(r, cfg)
        rows.append({
            "method": res["run"], "split": res["split"],
            "mAP50": res["detection_test"]["mAP50"], "mAP50-95": res["detection_test"]["mAP50_95"],
            "tray recall": res["tray_test"]["recall"], "tray specificity": res["tray_test"]["specificity"],
            "bal. acc": res["tray_test"]["balanced_accuracy"],
            "ms/img": res["latency_per_image"]["median_ms"],
            "params (M)": res["complexity"]["params_M"],
        })
    for name in ("original", "grouped"):
        f = cfg["paths"]["results"] / "baseline" / f"baseline_{name}.json"
        if f.exists():
            b = json.loads(f.read_text())
            rows.append({"method": "HOG+colour+LogReg baseline", "split": name, "mAP50": "n/a",
                         "mAP50-95": "n/a", "tray recall": b["test"]["recall"],
                         "tray specificity": b["test"]["specificity"],
                         "bal. acc": b["test"]["balanced_accuracy"],
                         "ms/img": b["latency_per_image"]["median_ms"], "params (M)": b["n_features"] / 1e6})
    cols = ["method", "split", "mAP50", "mAP50-95", "tray recall", "tray specificity", "bal. acc",
            "ms/img", "params (M)"]
    table = markdown_table(rows, cols)
    (cfg["paths"]["results"] / "comparison_table.md").write_text(table + "\n")
    save_json(rows, cfg["paths"]["results"] / "comparison_table.json")
    print("\n" + table)


if __name__ == "__main__":
    main()
