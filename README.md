# porkqc — Automated Packaging-Defect Inspection of Pork Rasher Trays

MMA3001 Numerical Methods and Machine Learning — individual project (Dataset 3).
Author: Zhafira Ayu Sabrina

A computer-vision pipeline that takes RGB images of packaged pork-rasher trays on a
production line, **detects and localises packaging defects** with a fine-tuned YOLOv8
object detector and turns them into a **tray-level PASS / FAIL decision** with a
severity grade and an inspection report.

The project's key engineering finding is about *validation*: the official Roboflow
test split is built from video frames that sit 1–3 frames away from training frames
(12 val/test frames are exact duplicates). A leakage-free split grouped by video
sequence is built and used for every headline result, and the gap between the two
splits is reported as evidence of how much the official split overstates accuracy.

---

## 1. Problem, inputs and outputs (summary)

| | Description |
|---|---|
| **Input** | RGB image, JPG/PNG, 720 × 540 px, 8-bit (0–255) per channel, top-down view of a tray on the conveyor. Other sizes are accepted (letterboxed) but flagged as outside the validated domain. |
| **Output 1 – detections** | List of boxes: class ∈ {loose-meat, packaging-error, twisted-meat, unsealed, wrinkle}, confidence ∈ [0, 1], pixel corners (x1, y1, x2, y2). |
| **Output 2 – decision** | `PASS` / `FAIL` / `INVALID`. FAIL if any defect has confidence ≥ threshold (threshold tuned on the validation set for ≥ 95 % defect recall). |
| **Output 3 – severity** | `none` / `minor` (< 1 % of image) / `major` (1–5 %) / `critical` (> 5 %), from total defect box area. |
| **Invalid input** | Missing, corrupt or unsupported files → `INVALID` row in the report; the batch continues. Malformed label files raise `ValueError` during training/evaluation. |

Full details: `report/REPORT.md` §2.

## 2. Repository structure

```
pork-qc/
├── configs/config.yaml        all settings (paths, epochs, thresholds) — one place to change things
├── src/porkqc/                the Python package (documented with Google-style docstrings)
│   ├── config.py              class names, paths, config loader
│   ├── data.py                label reading/validation, statistics, leakage check, grouped split
│   ├── baseline.py            classical baseline: colour histogram + HOG + logistic regression
│   ├── detector.py            YOLOv8 train / validate / predict wrappers
│   ├── inspection.py          detections -> PASS/FAIL decision, severity, input validation
│   ├── metrics.py             tray-level metrics, IoU, threshold sweep, bootstrap CI
│   ├── robustness.py          lighting / blur / noise / glare perturbations
│   ├── profiling.py           timing, memory, FLOP scaling
│   └── reporting.py           JSON, figures, annotated images, Markdown tables
├── scripts/                   the workflow, run in numerical order
│   ├── 01_explore_data.py     dataset statistics, figures, leakage measurement
│   ├── 02_make_grouped_split.py  leakage-free split by video sequence
│   ├── 03_baseline.py         train/evaluate the classical baseline
│   ├── 04_train_yolo.py       fine-tune YOLOv8n (main) and YOLOv8s (alternative)
│   ├── 05_evaluate.py         test metrics, threshold tuning on val, error images, comparison table
│   ├── 06_sensitivity.py      input-size trade-off and robustness tests
│   ├── 07_inspect.py          the end product: inspect new images -> annotated images + CSV/JSON
│   └── run_all.py             run everything in order
├── tests/                     pytest unit tests (synthetic data, run in seconds)
├── docs/                      generated HTML API docs (pdoc) and pytest HTML report
├── report/REPORT.md           project documentation / report draft
├── results/                   generated JSON, figures and tables (weights are git-ignored)
└── data/                      dataset goes here (not committed, see data/README.md)
```

## 3. Setup (VS Code, Windows / macOS / Linux)

```bash
git clone <your-repo-url> pork-qc && cd pork-qc
python -m venv .venv
# Windows:  .venv\Scripts\activate        macOS/Linux:  source .venv/bin/activate
pip install -r requirements.txt
pip install -e .                 # makes `import porkqc` work everywhere
```

In VS Code: *Ctrl+Shift+P → Python: Select Interpreter → .venv*.

**GPU (strongly recommended for training):** with an NVIDIA GPU install the CUDA build of
PyTorch first, e.g. `pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121`,
then `pip install -r requirements.txt`. `device: auto` in the config picks CUDA → Apple MPS → CPU.

**Data:** unzip `Pork_Rasher_Error-Packaging-_v4i_yolov8.zip` into `data/raw/` so that
`data/raw/data.yaml` exists.

## 4. Running the workflow

```bash
python scripts/01_explore_data.py
python scripts/02_make_grouped_split.py
python scripts/03_baseline.py
python scripts/04_train_yolo.py --split grouped          # YOLOv8n (main)
python scripts/04_train_yolo.py --split grouped --models yolov8s.pt   # optional larger alternative
python scripts/04_train_yolo.py --split original --models yolov8n.pt   # leakage comparison
python scripts/05_evaluate.py
python scripts/06_sensitivity.py --run grouped_yolov8n
python scripts/07_inspect.py --source data/grouped/test/images --run grouped_yolov8n
```

Or all at once: `python scripts/run_all.py` (add `--quick` for a 3-epoch smoke test first).

Approximate run time: steps 1–3 ≈ 1 min; each YOLOv8n training ≈ 20–40 min on a laptop GPU,
several hours on CPU (use `--epochs 20` or `--quick` if needed); YOLOv8s ≈ 2–3× longer.

| Step | Main outputs |
|---|---|
| 01 | `results/data/summary_original.json`, `leakage_original.json`, `class_distribution.png`, `box_sizes.png`, `samples.png` |
| 02 | `data/grouped/`, `results/data/summary_grouped.json` |
| 03 | `results/baseline/baseline_{original,grouped}.json` |
| 04 | `results/runs/<split>_<model>/` (Ultralytics curves, confusion matrix, `weights/best.pt`) |
| 05 | `results/eval/<run>/evaluation.json`, `threshold_sweep.png`, `errors/*.jpg`; `results/comparison_table.md` |
| 06 | `results/sensitivity/<run>/imgsz.png`, `robustness.png` (+ JSON) |
| 07 | `results/inspection/*.jpg`, `reports.csv`, `reports.json` |

## 5. Tests and documentation

```bash
pytest                                                    # unit tests
pytest --html=docs/test_report.html --self-contained-html # HTML test report
pdoc src/porkqc -o docs/api --docformat google            # HTML API documentation
```

Open `docs/api/index.html` in a browser. Re-run pdoc whenever docstrings change so the
docs stay in sync with the code.

## 6. Reproducibility

* All randomness is seeded (`seed: 42` in the config; Ultralytics `deterministic=True`).
* The grouped split is a deterministic function of the file names and the seed.
* GPU training is not bit-for-bit deterministic across hardware; expect ± ~0.01 mAP.
* Validation is never used for training; the test set is used once per model after all
  choices (C for the baseline, the decision threshold, early stopping) were made on validation.

## 7. Licence and acknowledgements

Code: MIT (see `LICENSE`). Dataset: *Pork Rasher Error(Packaging)* v4, Roboflow Universe,
CC BY 4.0 (not redistributed). YOLOv8: Ultralytics (AGPL-3.0). AI-tool use is disclosed in
`report/REPORT.md` §8.
