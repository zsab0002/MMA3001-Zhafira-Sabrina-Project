"""Step 3 - Classical baseline: colour histogram + HOG + logistic regression.

Predicts PASS/FAIL per tray. The regularisation strength C is chosen on the
VALIDATION set; the test set is used once at the end. Run on both the original
and the grouped split to show how much leakage inflates accuracy.

Outputs: results/baseline/baseline_<split>.json

Usage:
    python scripts/03_baseline.py
"""
import _setup  # noqa: F401

import time

import numpy as np

from porkqc.baseline import build_baseline, load_features
from porkqc.config import load_config
from porkqc.data import label_path_for, list_images, read_label_file, tray_label
from porkqc.metrics import binary_metrics, bootstrap_ci
from porkqc.profiling import peak_python_memory, time_function
from porkqc.reporting import save_json


def load_split(root, split):
    images = list_images(root / split / "images")
    y = np.array([tray_label(read_label_file(label_path_for(p))) for p in images])
    return images, y


def run(root, name, cfg):
    print(f"\n=== Baseline on '{name}' split ===")
    data = {s: load_split(root, s) for s in ("train", "valid", "test")}
    t0 = time.perf_counter()
    X = {s: load_features(imgs) for s, (imgs, _) in data.items()}
    feat_time = time.perf_counter() - t0
    y = {s: lab for s, (_, lab) in data.items()}
    print(f"features: {X['train'].shape[1]} per image, extracted in {feat_time:.1f}s")

    # Model selection on validation balanced accuracy (robust to class imbalance).
    best = None
    for C in (0.001, 0.01, 0.1, 1.0):
        clf = build_baseline(C=C, seed=cfg["seed"]).fit(X["train"], y["train"])
        score = binary_metrics(y["valid"], clf.predict(X["valid"]))["balanced_accuracy"]
        print(f"  C={C:<6} val balanced acc={score:.3f}")
        if best is None or score > best[0]:
            best = (score, C, clf)
    _, C, clf = best

    pred = clf.predict(X["test"])
    test = binary_metrics(y["test"], pred)
    test["recall_95ci"] = bootstrap_ci(y["test"], pred, "recall")
    test["specificity_95ci"] = bootstrap_ci(y["test"], pred, "specificity")
    always_fail = binary_metrics(y["test"], np.ones_like(y["test"]))

    import cv2
    img = cv2.imread(str(data["test"][0][0]))
    from porkqc.baseline import extract_features
    latency = time_function(lambda: clf.predict(extract_features(img)[None]), n_repeat=50)
    mem = peak_python_memory(lambda: clf.predict(extract_features(img)[None]))
    result = {
        "split": name, "chosen_C": C, "test": test, "always_fail_test": always_fail,
        "latency_per_image": latency, "peak_memory_MB": mem,
        "n_features": int(X["train"].shape[1]),
    }
    print(f"  chosen C={C}; TEST recall={test['recall']:.3f} specificity={test['specificity']:.3f} "
          f"bal.acc={test['balanced_accuracy']:.3f}  (always-FAIL bal.acc=0.500)")
    print(f"  latency {latency['median_ms']:.1f} ms/image, peak mem {mem:.1f} MB")
    return result


def main() -> None:
    cfg = load_config()
    out = cfg["paths"]["results"] / "baseline"
    for name, key in (("original", "raw_data"), ("grouped", "grouped_data")):
        root = cfg["paths"][key]
        if not (root / "train").exists():
            print(f"skip {name}: {root} missing (run step 2 first)")
            continue
        save_json(run(root, name, cfg), out / f"baseline_{name}.json")


if __name__ == "__main__":
    main()
