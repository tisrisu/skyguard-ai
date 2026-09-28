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

:: download and prepare data
python -m skyguard.data.fetch
python -m skyguard.data.clean

:: start the dashboard
run_demo.bat
```

## Team

| Name | Role |
|---|---|
| | |
