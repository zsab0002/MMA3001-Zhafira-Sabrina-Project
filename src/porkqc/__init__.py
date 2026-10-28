"""porkqc - automated quality inspection of packaged pork rasher trays.

The package turns RGB images of pork-rasher trays into

1. localised defect detections (YOLOv8 object detector), and
2. a tray-level PASS / FAIL decision with a short inspection report.

Modules
-------
- `porkqc.config`      : class names, paths and the project configuration file.
- `porkqc.data`        : reading/validating YOLO labels, dataset statistics,
                         video-frame leakage analysis and the leakage-free split.
- `porkqc.metrics`     : image-level classification metrics and box IoU.
- `porkqc.inspection`  : turning detections into a PASS/FAIL decision + report.
- `porkqc.baseline`    : classical (non deep-learning) baseline classifier.
- `porkqc.robustness`  : image perturbations used for robustness testing.
- `porkqc.profiling`   : timing, memory and FLOP helpers.
- `porkqc.detector`    : thin wrappers around Ultralytics YOLOv8.
"""

__version__ = "1.0.0"
