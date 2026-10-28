"""Step 2 - Build a leakage-free train/val/test split grouped by video sequence.

The original split places near-identical neighbouring video frames in train
and test. Here whole sequences (runs of frames with gaps <= sequence_gap)
are assigned to one split only, so test performance reflects genuinely
unseen production footage.

Usage:
    python scripts/02_make_grouped_split.py
"""
import _setup  # noqa: F401

from porkqc.config import load_config
from porkqc.data import dataset_summary, leakage_report, make_grouped_split, write_data_yaml
from porkqc.reporting import save_json


def main() -> None:
    cfg = load_config()
    raw, dst = cfg["paths"]["raw_data"], cfg["paths"]["grouped_data"]
    gs = cfg["grouped_split"]
    write_data_yaml(raw)  # absolute-path yaml for the ORIGINAL split as well
    info = make_grouped_split(raw, dst, gap=gs["sequence_gap"],
                              fractions=tuple(gs["fractions"]), seed=cfg["seed"])
    summary = dataset_summary(dst)
    leak = leakage_report(dst)
    out = cfg["paths"]["results"] / "data"
    save_json({"split": info, "summary": summary, "leakage": leak}, out / "summary_grouped.json")
    print("Grouped split:", info)
    for split, s in summary.items():
        print(f"{split:6s} images={s['images']:5d} FAIL={s['fail_trays']:5d} "
              f"PASS={s['pass_trays']:5d} boxes={s['boxes_per_class']}")
    print("Leakage after grouping:", leak)
    print(f"Written to {dst}")


if __name__ == "__main__":
    main()
