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
    from skyguard.config import load_config
    from skyguard.data.io import load_data, load_stations
    from skyguard.data.climatology import Climatology
    from skyguard.models.iforest import IFModel
    from skyguard.engine import Engine
    from skyguard.evaluation import evaluate

    cfg = load_config()
    df_all = load_data()
    stations = load_stations()

    train_start, train_end = cfg["splits"]["train"]
    # train_end is "2023-12-31" which parses to 00:00:00, use < "2024-01-01" to include the whole day
    train_end_exclusive = str(pd.to_datetime(train_end) + pd.Timedelta(days=1)).split()[0]
    train_df = df_all[(df_all["ts"] >= train_start) & (df_all["ts"] < train_end_exclusive)]

    clim = Climatology().fit(train_df)
    
    model = IFModel.load()

    engine = Engine(df=labelled, stations=stations, clim=clim, model=model, cfg=cfg)
    results_df = engine.run(labelled["ts"].min(), labelled["ts"].max())

    return evaluate(results_df, labelled)


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
