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

from skyguard.config import INJECTED_DIR, REPORTS_DIR, load_config  # noqa: E402
from skyguard.data.climatology import Climatology  # noqa: E402
from skyguard.data.io import load_data, load_stations, select_split  # noqa: E402
from skyguard.engine import Engine  # noqa: E402
from skyguard.evaluation import evaluate  # noqa: E402
from skyguard.models.iforest import IFModel  # noqa: E402


def run(labelled: pd.DataFrame) -> dict:
    cfg = load_config()
    clim = Climatology().fit(select_split(load_data(), "train", cfg))
    engine = Engine(df=labelled, stations=load_stations(), clim=clim, model=IFModel.load(), cfg=cfg)
    results = engine.run(labelled["ts"].min(), labelled["ts"].max())
    return evaluate(results, labelled)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--split", default="test", choices=["val", "test"])
    args = parser.parse_args()

    labelled = pd.read_parquet(INJECTED_DIR / f"{args.split}.parquet")
    metrics = run(labelled)

    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    out = REPORTS_DIR / "metrics.json"
    out.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    print(f"Saved {out}")


if __name__ == "__main__":
    main()
