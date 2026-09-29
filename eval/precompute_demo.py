"""Precompute a demo result window using the real M2+M3 engine."""
from __future__ import annotations

import sys
from pathlib import Path

# Allow ``python eval/precompute_demo.py`` from the project root.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd

from skyguard.config import HOURLY_FILE, INJECTED_DIR, MODELS_DIR, load_config
from skyguard.data.climatology import Climatology
from skyguard.data.io import load_data, load_stations, select_split
from skyguard.models.iforest import IFModel
from skyguard.engine import Engine

OUT = INJECTED_DIR.parent / "demo_results.parquet"


def main():
    cfg = load_config()
    df = load_data(HOURLY_FILE)
    train = select_split(df, "train", cfg)
    clim = Climatology().fit(train)
    model = IFModel.load(MODELS_DIR / "iforest.joblib")

    test = select_split(df, "test", cfg)
    start = test["ts"].min() + pd.Timedelta(days=1)
    end = min(test["ts"].max(), start + pd.Timedelta(days=7))
    engine = Engine(test, load_stations(), clim, model, cfg)
    results = engine.run(start, end)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    results["shap_top"] = results["shap_top"].apply(
        lambda x: str(x) if x is not None else ""
    )
    results.to_parquet(OUT, index=False)
    print(f"wrote {len(results):,} rows to {OUT}")


if __name__ == "__main__":
    main()
