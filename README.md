# SkyGuard AI

Real-time anomaly detection for Automatic Weather Stations (AWS) using temperature, pressure and relative humidity.

SkyGuard checks every reading and decides whether it is **normal**, a **genuine weather event** (for example a thunderstorm) or a **sensor fault** (spike, frozen value, drift, dropout, noise, out-of-range). Each decision comes with a confidence score, a severity, a plain-English explanation and a corrected value. Every sensor also gets a trust score.

The core idea: a reading is accepted as real weather only if it is **physically plausible** and **neighbouring stations agree**. Otherwise it is flagged as a sensor fault.

> Smart India Hackathon 2026 · Problem Statement 26073 · Ministry of Earth Sciences / India Meteorological Department
>
> Status: early prototype, under active development.

## How it works

For every station and every hour:

1. **Physics checks**: valid ranges, rate of change, dewpoint plausibility, flat-lined values, one variable moving on its own
2. **Neighbour check**: compares the station with nearby stations after removing each station's normal daily/seasonal pattern
3. **ML score**: Isolation Forest on time-window features
4. **Decision**: normal / genuine event / suspect / sensor fault, with confidence and severity
5. **Fault type, explanation and corrected value**
6. **Trust score** update for the sensor

## Data

Hourly data for 8 IMD station locations in Delhi-NCR, 2022–2024, from the [Open-Meteo](https://open-meteo.com/) historical weather API. This is ECMWF reanalysis at the station coordinates, not raw station observations. Faults and storms are injected on top of it (`skyguard/inject/injector.py`) so detection can be measured against known labels.

## Project layout

```
skyguard/
  config.py, schemas.py, geo.py   shared settings, data formats, distance helpers
  data/        download, clean, climatology
  inject/      synthetic fault + storm injection (for testing and the demo)
  physics/     dewpoint and physics rules
  spatial/     neighbour consistency
  models/      features, Isolation Forest
  fusion/      decision, fault type, explanations
  heal/        corrected values
  health/      trust score
  engine.py    runs the full pipeline
  evaluation.py
dashboard/     Streamlit app
eval/          evaluation script and reports
data/          processed data (raw downloads are not committed)
models/        trained models
tests/
config.yaml    all thresholds and settings
```

## Setup (Windows)

Requires Python 3.11 or newer.

```bat
git clone <repo-url>
cd skyguard-ai
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

## Usage

```bat
:: run tests
pytest

:: download and prepare data (already in the repo; only needed to rebuild it)
python -m skyguard.data.fetch
python -m skyguard.data.clean

:: build the validation and test sets with injected faults
python -m skyguard.inject.injector

:: start the dashboard (from PowerShell: .\run_demo.bat)
run_demo.bat
```

The dashboard replays the test period (July to December 2024) hour by hour. Use the sidebar to add a temperature spike, a frozen sensor or a thunderstorm at any station and watch how each one is classified. Until `skyguard/engine.py` is implemented it runs in preview mode: faults come from the injector and are confirmed with simple physics and neighbour checks. The header says so, and the real detector is used automatically once the engine works.

## Team

| Name | Role |
|---|---|
| | |
