# Automated Packaging-Defect Inspection of Pork Rasher Trays with YOLOv8

**MMA3001 Numerical Methods and Machine Learning — Individual Project Documentation**
Zhafira Ayu Sabrina · GitHub: `<repo link>`

> **How to use this draft.** Sections marked **[FILL]** need numbers from your own runs of
> steps 4–6 (copy them from `results/comparison_table.md`, `results/eval/*/evaluation.json`
> and `results/sensitivity/*/`). Numbers already given (dataset, leakage, baseline) were
> produced by steps 1–3 and are reproducible. Rewrite the text in your own words; you will
> be questioned on it. Target 8–10 pages with figures.

---

## 1. Engineering problem

**Context.** Sliced pork (rashers) is packed into clear PET trays and sealed with film on
high-speed lines. Defects — film not sealed, the film or tray damaged/mis-placed, film
wrinkles across the seal, rashers twisted or lying loose outside their lane — cause leaks,
shortened shelf life, contamination risk, product recalls and customer rejection. Today
these are found by human inspectors watching the conveyor, which is tiring, inconsistent
and limited to a sample of trays.

**Problem addressed.** Given a top-down RGB camera image of a tray on the conveyor,
(i) localise and classify visible packaging/placement defects and (ii) decide whether the
tray should be rejected.

**Why it matters.** A missed defect (false PASS) can reach a consumer; a false reject only
costs one re-pack. The system is therefore designed and tuned around **high defect recall**,
with false rejects as the secondary cost.

**Intended use.** Decision support for an in-line reject gate / operator alarm, with the
annotated image explaining each rejection. It is *not* a food-safety certification tool.

**Scope and limitations of scope.**
* Packaging and placement defects visible from above only; meat *quality* (colour, fat
  ratio, spoilage) is out of scope because the dataset has no quality labels.
* Decisions are per *image*. On the real line the same tray appears in several frames;
  aggregation over frames (tracking) is proposed as future work (§7).
* One camera, one lighting set-up, one product type — the model is only validated for that
  domain.
* `wrinkle` has only 18 training boxes and none in the original val/test, so performance on
  it cannot be established.

## 2. Inputs, outputs and data

### 2.1 Top-level inputs

| Input | Format | Units / domain | Notes |
|---|---|---|---|
| Tray image | JPG/PNG, 3-channel RGB, 8-bit | 720 × 540 px; pixel values 0–255 | validated domain; other sizes letterboxed and flagged |
| Trained weights | `best.pt` (PyTorch) | — | from step 4 |
| Decision threshold | float | [0, 1] (dimensionless confidence) | tuned on validation (§3.4) |
| Severity bands | floats | fraction of image area, 0.01 and 0.05 | engineering choice, configurable |

### 2.2 Top-level outputs

| Output | Format | Meaning / intended use |
|---|---|---|
| Decision | `PASS`/`FAIL`/`INVALID` | drive reject gate; INVALID = image could not be read → treat as reject and alarm |
| Fail score | float in [0, 1] | highest defect confidence; allows the line to change the threshold without re-running the model |
| Detections | class, confidence, box `(x1,y1,x2,y2)` in pixels | tells operator *what* and *where* |
| Severity | none/minor/major/critical | prioritise operator attention; total defect-box area / image area (upper bound, boxes may overlap) |
| Reports | `reports.csv`, `reports.json`, annotated JPGs | traceability / QA records |

### 2.3 Invalid, missing or unsupported inputs
* Missing file, unsupported extension, truncated/corrupt image → `INVALID` with a message;
  the batch continues (`inspection.validate_image`).
* Different image size → processed but flagged “outside validated domain”.
* Malformed label (wrong number of values, class id out of range, coordinate outside [0, 1])
  → `ValueError` with file and line number (`data.read_label_file`), so corrupt annotations
  cannot silently enter training or validation. All 3,549 label files passed this check.

### 2.4 Dataset
*Pork Rasher Error(Packaging)* v4, Roboflow Universe, CC BY 4.0: 3,549 images
(1,272 distinct video frames), YOLO-format boxes, 5 classes. Training images were tripled
by Roboflow with ±10 % exposure augmentation.

**Table 1 – Original split (from `results/data/summary_original.json`).**

| Split | Images | Distinct frames | FAIL trays | PASS trays | loose | pack.-err | twisted | unsealed | wrinkle |
|---|---|---|---|---|---|---|---|---|---|
| train | 3,349 | 1,085 | 2,665 | 684 | 66 | 1,712 | 72 | 1,055 | 18 |
| valid | 120 | 120 | 106 | 14 | 3 | 34 | 3 | 81 | 0 |
| test | 80 | 80 | 68 | 12 | 3 | 23 | 4 | 49 | 0 |

Observations (Figures `class_distribution.png`, `box_sizes.png`, `samples.png`):
* **Severe class imbalance** — packaging-error and unsealed make up 95 % of boxes.
* **~80 % of images are FAIL**, so accuracy is misleading: “always FAIL” scores 85 % accuracy
  on the original test set. Balanced accuracy, recall and specificity are reported instead.
* **Packaging-error boxes are tiny** (median 0.36 % of the image, ≈ 40 × 30 px) → input
  resolution is expected to matter (tested in §5.2).
* Many images show a tray only partly in view as it enters the frame.

### 2.5 Data leakage in the official split (key validation finding)
File names encode the video frame number. For every val/test image the gap to the nearest
training frame was measured (`data.leakage_report`):

| Split | Exact duplicate frames | Median gap to nearest train frame | Within 5 frames |
|---|---|---|---|
| valid | 8 | 3 frames | 67.5 % |
| test | 4 | 2.5 frames | 67.5 % |

Consecutive frames are near-identical, so the official test set mostly measures
*memorisation*, not generalisation to new trays. A **grouped split** was therefore built
(`data.make_grouped_split`): frames are grouped into video sequences (a new sequence starts
after a gap > 20 frames; 117 sequences), whole sequences are randomly assigned 70/15/15 to
train/val/test, and only one copy per frame is kept in val/test.

| Grouped split | Images | Sequences | FAIL | PASS | Median gap to train | Duplicates |
|---|---|---|---|---|---|---|
| train | 2,492 | 94 | 2,002 | 490 | — | — |
| valid | 183 | 10 | 147 | 36 | 116 frames | 0 |
| test | 188 | 13 | 153 | 35 | 77 frames | 0 |

All headline results use the grouped split. Limitation: the grouped test set contains no
loose-meat and only 2 twisted-meat boxes, so per-class results for those are not meaningful.

## 3. Computational solution

### 3.1 Pipeline
`image → validation → YOLOv8 detector → boxes (class, conf) → decision layer (threshold,
severity) → report`

### 3.2 Method: YOLOv8 object detection (how it works)
YOLOv8 is a single-stage convolutional detector. A CSP-Darknet **backbone** extracts
features, a **PAN-FPN neck** fuses them at three scales (strides 8, 16, 32 → 80×80, 40×40,
20×20 grids at 640 px), and an **anchor-free decoupled head** predicts, for every grid cell,
class probabilities and the distances from the cell to the four box edges. Training
minimises a weighted sum of a classification loss (binary cross-entropy), a box-regression
loss (CIoU) and distribution focal loss, using stochastic gradient descent with momentum
(numerical optimisation of ~3 M parameters). Overlapping predictions are removed by
non-maximum suppression (NMS, IoU 0.7).

**Why selected.** (i) One forward pass per image → real-time capable; (ii) it *localises*
defects, which an image classifier cannot, so decisions are explainable to operators;
(iii) transfer learning from COCO-pretrained weights suits the small dataset (~1,000
distinct frames); (iv) the data is already in YOLOv8 format; (v) mature, documented tooling.

**Key assumptions.** Defects are visible from above; training labels are correct and
complete; test-time images come from the same camera/lighting distribution.

**Settings** (`configs/config.yaml`): pretrained `yolov8n.pt`; image size 640; batch 16;
max 50 epochs with early stopping (patience 15 on validation mAP50-95); default Ultralytics
augmentation (mosaic, flips, HSV jitter, scaling); seed 42, deterministic mode.

### 3.3 Decision layer
`FAIL if max(defect confidence) ≥ threshold`. Simple and transparent: the operator sees
exactly which box caused the rejection. Severity from total box area (§2.2).

### 3.4 Threshold selection (validation only)
Predictions are stored at confidence ≥ 0.01, then thresholds 0.05–0.70 are swept on the
**validation** set and the *highest* threshold that still gives ≥ 95 % FAIL-recall is
chosen (prioritising missed defects over false rejects). The test set is touched once, after
this choice. Chosen threshold: **[FILL]**.

### 3.5 Implementation
Python package `porkqc` (9 modules, Google-style docstrings), 8 numbered scripts, central
YAML config, pytest unit tests and pdoc HTML documentation. Ultralytics provides training
and mAP computation; dataset handling, leakage analysis, splitting, the decision layer,
tray-level metrics, bootstrap intervals, robustness tests and profiling were written for
this project.

### 3.6 Known limitations of the method
Per-frame decisions; small objects (packaging-error) are hard at low resolution; rare
classes have too few examples; sensitive to domain shift (new lighting, tray type).

## 4. Validation

**What was validated.** (a) Correctness of my own code — unit tests; (b) detection accuracy
of boxes; (c) correctness of the tray PASS/FAIL decision; (d) robustness and speed.

**Verification of code (unit tests, `tests/`).** 41 tests, including known-answer tests:
IoU of half-overlapping boxes = 1/3; YOLO→pixel box conversion; hand-calculated confusion
matrix; “always FAIL” gives balanced accuracy exactly 0.5; recall is monotonic in threshold;
grouped split never splits a sequence and removes all duplicates on a synthetic dataset with
planted leakage; corrupt/malformed inputs raise or are flagged. Test report:
`docs/test_report.html`.

**Metrics and why.**
* *Detection:* precision, recall, mAP50, mAP50-95 (standard COCO metrics; mAP50-95 also
  rewards tight boxes, relevant for severity).
* *Tray decision:* recall (defect detection rate — primary, acceptance target ≥ 0.95),
  specificity (good trays passed), balanced accuracy (imbalance-robust), with **bootstrap
  95 % confidence intervals** because the test sets are small (188 trays, only 35 PASS).
* *Engineering:* latency (median of 30 warm runs), throughput, parameters, GFLOPs.

**Acceptance criteria (proposed for this application).** tray recall ≥ 0.95; specificity
≥ 0.80; latency well below the conveyor period (e.g. < 100 ms for ~10 trays/s with margin).

**Did validation influence model selection?** Yes, and only validation did: baseline C,
YOLO early stopping, the decision threshold and the choice between YOLOv8n and YOLOv8s.

**What validation cannot establish.** Performance on other lines/lighting/products; on
loose-meat, twisted-meat and wrinkle (too few test examples); on real per-tray decisions
(per-frame here); long-term drift. Labels themselves are assumed correct — some
annotations looked inconsistent in the error images **[FILL: check `results/eval/*/errors/`]**.

### 4.1 Results — **[FILL from `results/comparison_table.md`]**

**Table 2 – Test-set comparison (grouped split unless stated).**

| Method | Split | mAP50 | mAP50-95 | Tray recall [95 % CI] | Specificity [95 % CI] | Bal. acc | ms/img | Params |
|---|---|---|---|---|---|---|---|---|
| Always FAIL (trivial) | grouped | — | — | 1.000 | 0.000 | 0.500 | 0 | 0 |
| HOG + colour + LogReg | grouped | — | — | 0.830 [0.77, 0.89] | 0.514 [0.36, 0.68] | 0.672 | 2.8 (CPU) | 2,072 features |
| HOG + colour + LogReg | original | — | — | 0.809 [0.72, 0.90] | 0.667 [0.36, 0.92] | 0.738 | 2.8 (CPU) | 2,072 features |
| YOLOv8n | grouped | [FILL] | [FILL] | [FILL] | [FILL] | [FILL] | [FILL] | 3.0 M |
| YOLOv8s | grouped | [FILL] | [FILL] | [FILL] | [FILL] | [FILL] | [FILL] | 11.1 M |
| YOLOv8n | original (leaky) | [FILL] | [FILL] | [FILL] | [FILL] | [FILL] | [FILL] | 3.0 M |

Discuss: (1) YOLO vs baseline — how much does localisation-based detection gain?
(2) n vs s — accuracy gain vs 3–4× compute; (3) original vs grouped — size of the leakage
optimism (the baseline already drops 0.74 → 0.67 balanced accuracy); (4) per-class AP50 from
`evaluation.json` — which defects are reliable; (5) error images — what causes misses
(partial trays at frame edge, reflections, tiny boxes, label noise?).

Figures to include: Ultralytics `results.png` (training curves — check val loss for
over-fitting), `confusion_matrix_normalized.png`, `threshold_sweep.png`, 2–4 error images.

## 5. Comparison, optimisation and performance

### 5.1 Alternatives considered
| Option | Accuracy | Speed | Interpretability | Localises? | Decision |
|---|---|---|---|---|---|
| Always FAIL | none | — | — | no | lower bound |
| HOG + colour + logistic regression | low | ~3 ms CPU, 1 MB | high (linear) | no | baseline |
| YOLOv8n (3 M params, 8.7 GFLOPs @640) | [FILL] | [FILL] | medium (boxes) | yes | **selected if [FILL]** |
| YOLOv8s (11 M params, 28.6 GFLOPs @640) | [FILL] | [FILL] | medium | yes | alternative |
| Two-stage detector (Faster R-CNN) | often higher on small objects | ~5–10× slower | medium | yes | not tested — not real-time on CPU |

### 5.2 Input resolution study (`06_sensitivity.py`, Figure `imgsz.png`)
Convolution cost scales with pixel count, so GFLOPs ∝ imgsz²: 320 px needs ¼ of the
operations of 640 px. Because packaging-error boxes are ~40×30 px at full resolution
(only ~18×13 px at 320), a resolution reduction is expected to hurt that class most.
**[FILL table: imgsz, mAP50, tray recall, ms/img, GFLOPs]** — choose the smallest size that
still meets the acceptance criteria.

**Arithmetic intensity / FLOPs estimate.** YOLOv8n ≈ 8.7 GFLOPs per 640² image. At
measured latency *t* the achieved throughput is 8.7 / *t* GFLOP/s **[FILL]**; compare with
your CPU/GPU peak to judge whether the run is compute- or memory/overhead-bound (small
models on GPUs are usually overhead-bound, which is why latency does not fall 4× at 320 px).

### 5.3 Threshold sensitivity (`threshold_sweep.png`)
Raising the threshold trades recall for specificity. Report the operating point chosen on
validation and how test recall/specificity move ±0.1 around it **[FILL]**.

### 5.4 Robustness (`robustness.png`)
Perturbations representing line conditions: brightness ×0.4–1.6 (lighting drift), horizontal
motion blur 5–15 px (conveyor speed), Gaussian noise σ = 10–35 (low-light sensor), synthetic
specular glare (film reflection). **[FILL: which disturbance breaks the model first and at
what level; compare with baseline]** → defines operating limits (e.g. “lighting must stay
within ±20 %”).

### 5.5 Data requirement (optional)
`04_train_yolo.py --models yolov8n.pt --fraction 0.25 / 0.5` → mAP vs training-set size
shows whether collecting more data would help **[FILL if run]**.

### 5.6 Final design justification
**[FILL]** e.g. “YOLOv8n at 640 px, threshold 0.xx: meets recall target, x ms on CPU,
YOLOv8s gave only +0.0x mAP for 3.3× compute.”

## 6. Limitations
* Official split leaks; grouped test set is small → wide confidence intervals.
* Rare classes (wrinkle, loose, twisted) not validated.
* Per-frame, not per-tray, decisions; partially visible trays at frame edges are ambiguous.
* Single site/camera; no meat-quality assessment.
* Labels assumed correct; no inter-annotator check.
* Severity is box-area based (upper bound, overlap not removed), not a physical measure of
  seal integrity.

## 7. Lessons learned and improvements
* **Validation design matters more than model choice** — the leakage check changed what
  “accuracy” meant for this project. Lesson: always check how a split was made.
* Accuracy is misleading on imbalanced data; choose metrics from the cost of each error.
* Improvements, justified by results: aggregate decisions over the frames of one tray
  (object tracking + majority/any vote); higher input resolution or tiling for small
  packaging errors; collect more rare-class examples or merge rare classes; class-weighted
  loss; polarising filter to remove film glare if robustness tests confirm it as a failure
  mode; export to ONNX/TensorRT for faster inference.

## 8. AI use and reflection  — **write this in your own words**
Points to cover (brief requirements): tools used (e.g. Claude, Anthropic, for code
scaffolding, docstrings, README/report structure); what for; approximate contribution
(be honest — e.g. most boilerplate code generated, all experiments run, results
interpreted and decisions made by me); why; **how you checked it** (read every function,
unit tests with hand-calculated answers, re-ran scripts, checked figures and label overlays
visually, compared against Ultralytics docs); decisions that were yours (problem scope,
pass/fail cost priority, acceptance criteria, final model); **errors/limitations of AI
output** you found (note any bug or wrong assumption you had to fix, e.g. API mismatches
with your Ultralytics version, paths on Windows); how it affected your learning.

## References
1. Roboflow user hello-2aqe0, *Pork Rasher Error(Packaging) Dataset v4*, Roboflow Universe, 2025. CC BY 4.0.
2. G. Jocher, A. Chaurasia, J. Qiu, *Ultralytics YOLOv8*, v8.x, 2023. https://github.com/ultralytics/ultralytics
3. J. Redmon et al., “You Only Look Once: Unified, Real-Time Object Detection,” CVPR 2016.
4. N. Dalal, B. Triggs, “Histograms of Oriented Gradients for Human Detection,” CVPR 2005.
5. T.-Y. Lin et al., “Microsoft COCO: Common Objects in Context,” ECCV 2014 (mAP definition).
6. S. Kaufman et al., “Leakage in Data Mining: Formulation, Detection, and Avoidance,” ACM TKDD, 2012.
7. B. Efron, R. Tibshirani, *An Introduction to the Bootstrap*, Chapman & Hall, 1993.
8. F. Pedregosa et al., “Scikit-learn: Machine Learning in Python,” JMLR 12, 2011.
