"""Project paths and settings loaded from config.yaml."""

from functools import lru_cache
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]

DATA_DIR = ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"
INJECTED_DIR = DATA_DIR / "injected"
MOCK_DIR = DATA_DIR / "mock"
STATIONS_FILE = DATA_DIR / "stations.json"
HOURLY_FILE = PROCESSED_DIR / "ncr_hourly.parquet"

MODELS_DIR = ROOT / "models"
REPORTS_DIR = ROOT / "eval" / "reports"

CONFIG_FILE = ROOT / "config.yaml"


@lru_cache(maxsize=None)
def load_config(path: str | None = None) -> dict:
    """Read config.yaml once and cache it."""
    with open(path or CONFIG_FILE, encoding="utf-8") as f:
        return yaml.safe_load(f)
