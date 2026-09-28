"""Run the engine on an injected split and write eval/reports/metrics.json.

Owner: M1

Usage:  python eval/run_eval.py --split test
"""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd  # noqa: E402

from skyguard.config import INJECTED_DIR, REPORTS_DIR  # noqa: E402


def run(labelled: pd.DataFrame) -> dict:
    """TODO: build Climatology on the train split, load the IFModel,
    run Engine over `labelled`, then return evaluate(results, labelled)."""
    raise NotImplementedError


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--split", default="test", choices=["val", "test"])
    args = parser.parse_args()

    labelled = pd.read_parquet(INJECTED_DIR / f"{args.split}.parquet")
    metrics = run(labelled)

    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    (REPORTS_DIR / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
