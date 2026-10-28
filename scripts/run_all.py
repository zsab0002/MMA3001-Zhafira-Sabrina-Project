"""Run the whole workflow in order (steps 1-7) and the tests.

Usage:
    python scripts/run_all.py            # full experiment (hours on CPU, ~30-60 min on a GPU)
    python scripts/run_all.py --quick    # 3-epoch smoke test of every step (~10-20 min on CPU)
"""
import argparse
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def run(*args: str) -> None:
    print("\n>>>", " ".join(args), flush=True)
    subprocess.run([sys.executable, *args], cwd=ROOT, check=True)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    a = ap.parse_args()
    q = ["--quick"] if a.quick else []
    run("scripts/01_explore_data.py")
    run("scripts/02_make_grouped_split.py")
    run("scripts/03_baseline.py")
    run("scripts/04_train_yolo.py", "--split", "grouped", *q)
    run("scripts/04_train_yolo.py", "--split", "original", "--models", "yolov8n.pt", *q)
    run("scripts/05_evaluate.py")
    main_run = "grouped_yolov8n_frac20" if a.quick else "grouped_yolov8n"
    run("scripts/06_sensitivity.py", "--run", main_run)
    run("scripts/07_inspect.py", "--source", "data/grouped/test/images", "--run", main_run)
    run("-m", "pytest", "--html=docs/test_report.html", "--self-contained-html")


if __name__ == "__main__":
    main()
