"""Step 7 - Inspect new tray images: the deployable end product.

Input : one image or a folder of images (JPG/PNG, ideally 720x540 RGB).
Output: for each image an annotated copy, plus reports.csv and reports.json
        with decision (PASS/FAIL/INVALID), fail score, severity, defect
        counts and box coordinates.

Invalid inputs (missing, corrupt, unsupported format) are reported as INVALID
instead of stopping the batch.

Usage:
    python scripts/07_inspect.py --source data/grouped/test/images --run grouped_yolov8n
    python scripts/07_inspect.py --source my_tray.jpg --weights path/to/best.pt --conf 0.3
"""
import _setup  # noqa: F401

import argparse
import csv
import json
from pathlib import Path

import cv2

from porkqc.config import IMAGE_EXTENSIONS, IMAGE_SIZE, load_config
from porkqc.detector import predict
from porkqc.inspection import TrayReport, inspect_tray, validate_image
from porkqc.reporting import draw_boxes, save_json

COLOURS = {"PASS": (0, 200, 0), "FAIL": (0, 0, 255)}


def main() -> None:
    cfg = load_config()
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--source", required=True, help="image file or folder")
    ap.add_argument("--run", default="grouped_yolov8n", help="run name in results/runs")
    ap.add_argument("--weights", help="explicit checkpoint (overrides --run)")
    ap.add_argument("--conf", type=float, help="FAIL threshold (default: value tuned in step 5)")
    ap.add_argument("--out", default=None, help="output folder (default results/inspection)")
    a = ap.parse_args()

    weights = Path(a.weights) if a.weights else cfg["paths"]["results"] / "runs" / a.run / "weights" / "best.pt"
    if not weights.exists():
        raise SystemExit(f"weights not found: {weights}")
    thr = a.conf
    if thr is None:
        ev = cfg["paths"]["results"] / "eval" / a.run / "evaluation.json"
        thr = json.loads(ev.read_text())["threshold_from_val"] if ev.exists() else cfg["inspection"]["conf_threshold"]

    src = Path(a.source)
    files = sorted(p for p in src.iterdir() if p.is_file()) if src.is_dir() else [src]
    out = Path(a.out) if a.out else cfg["paths"]["results"] / "inspection"
    out.mkdir(parents=True, exist_ok=True)

    reports, valid = [], []
    for f in files:
        ok, msg = validate_image(f, IMAGE_SIZE)
        if ok:
            valid.append(f)
        elif f.suffix.lower() in IMAGE_EXTENSIONS or src.is_file():
            reports.append(TrayReport(image=f.name, decision="INVALID", message=msg))

    preds = predict(weights, valid, conf=0.01, device=cfg["yolo"]["device"]) if valid else {}
    for f in valid:
        img = cv2.imread(str(f))
        h, w = img.shape[:2]
        rep = inspect_tray(f.name, preds.get(f.name, []), (w, h), thr, cfg["inspection"]["severity_bands"])
        reports.append(rep)
        boxes = [(d.cls, d.conf, d.xyxy) for d in preds.get(f.name, []) if d.conf >= thr]
        header = f"{rep.decision}  score={rep.fail_score:.2f}  severity={rep.severity}"
        cv2.imwrite(str(out / f.name), draw_boxes(img, boxes, header, COLOURS[rep.decision]))

    save_json([r.to_dict() for r in reports], out / "reports.json")
    with open(out / "reports.csv", "w", newline="") as fh:
        wr = csv.writer(fh)
        wr.writerow(["image", "decision", "fail_score", "severity", "defect_area_fraction", "defects", "message"])
        for r in reports:
            wr.writerow([r.image, r.decision, r.fail_score, r.severity, r.defect_area_fraction,
                         json.dumps(r.defects), r.message])
    n = {d: sum(r.decision == d for r in reports) for d in ("PASS", "FAIL", "INVALID")}
    print(f"Inspected {len(reports)} images at threshold {thr:.2f}: {n}. Results in {out}")


if __name__ == "__main__":
    main()
