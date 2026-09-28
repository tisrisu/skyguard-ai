# SkyGuard AI — Initial Demo Plan (for the idea submission)

**Deadline: 30 Sept 2026** · Today: 27 Sept · Team: M1, M2, M3, M4 + A1 (auxiliary)

This plan covers **only** the demo that goes in with the PPT. The bigger build (MQTT streaming, FastAPI, LSTM autoencoder, ESP32 TinyML) is in [TEAM_PLAN.md](TEAM_PLAN.md) and comes later. The demo code goes in the same folders, so none of it is wasted.

---

## 1. What we submit

| # | Deliverable | Owner | Where it goes |
|---|---|---|---|
| 1 | **Idea PPT** (6 slides, PDF), with real numbers and one prototype screenshot | A1 (+ everyone reviews) | SIH portal |
| 2 | **Demo video**, 2–3 min, uploaded to YouTube as *Unlisted* | A1 + M4 | link in portal / on slide 6 |
| 3 | **Public GitHub repo** with working code + README | M3 (repo), A1 (README) | link on slide 6 |

> ✅ **Do this today:** one person opens the SIH portal's idea-submission form and checks exactly which fields and links it asks for (video link? length limit? GitHub?). Tell the group. If there's no video field, put the YouTube and GitHub links on slide 6.

---

## 2. What the demo must prove (and nothing more)

Judges watching a 2-minute video should come away with **one message**:

> **"The same-looking jump is flagged as a *sensor fault* at one station and accepted as *real weather* when the neighbouring stations agree, and SkyGuard explains why in plain English."**

So the demo shows exactly **five moments**:

1. Live replay of real weather data from ~6 Delhi-NCR stations on a map, all green.
2. **Spike** injected (the PS's own 55 °C example) → red alert card: SENSOR_FAULT · SPIKE · confidence · 3 reasons · corrected value · trust score drops.
3. **Storm** injected across several neighbours → shown in **blue as GENUINE_EVENT, no alarm**.
4. **Frozen sensor** → FROZEN detected.
5. A **scorecard**: detection F1 per fault type + the false-alarm rate on storms.

---

## 3. What we cut for the demo (on purpose)

| In the demo ✅ | Cut from the demo ❌ → comes in the full build |
|---|---|
| Real Meteostat data for 6 NCR stations | MQTT broker, FastAPI, database |
| Anomaly injector: SPIKE, FROZEN, DRIFT, DROPOUT, NOISE, OUT_OF_RANGE, STORM | LSTM autoencoder |
| Physics rules (incl. dewpoint checks) | RandomForest fault classifier (rules name the fault type instead) |
| Neighbour (spatial) consistency check | ESP32 TinyML model |
| Isolation Forest + SHAP | WebSockets, multi-page app |
| Decision rules → status, confidence, severity, reasons | Maintenance-prediction page (stretch only) |
| Corrected value + trust score | |
| **One Streamlit app** that replays the data *in-process* | |

**Key simplification:** no servers. The Streamlit app replays the data hour by hour and calls the detection engine directly. This means fewer things that can break and no install problems, and one command runs everything.

In the video, say it honestly: *"This prototype runs the detection engine and dashboard. The edge (ESP32) and streaming layers shown in our architecture are the next build phase."*

**ESP32:** only if someone **already owns** an ESP32 + BME280 (or DHT22/BMP180). If so, add a 15-second clip of it reading live values with the range/flatline rules running. If nobody has one, skip it. There's no time to order one.

---

## 4. How the code fits together

```
data/processed/ncr_hourly.parquet  (M1)
          │
          ▼
┌──────────────────── skyguard/engine.py  (M3 glues it) ────────────────────┐
│  for each hour t, for each station, for each variable:                     │
│    physics  = check_physics(...)                     (M3)                  │
│    spatial  = spatial_check(...)                     (M3)                  │
│    ml       = iforest_score(...)                     (M2)                  │
│    status, confidence, severity = decide(...)        (M2)                  │
│    fault_type = fingerprint(...)                     (M2)                  │
│    reasons, shap_top = explain(...)                  (M2)                  │
│    corrected = impute(...)   if fault                (M1)                  │
│    trust.update(...)                                 (M3)                  │
└───────────────────────────────────────────────────────────────────────────┘
          │  list of result dicts
          ▼
dashboard/app.py  (M4) ── Inject buttons ──► injector (M1) modifies upcoming data
```

### 4.1 Shared data format (agree tonight, don't change after)

**Data table** `data/processed/ncr_hourly.parquet` (long format, UTC):

| station_id | ts | temp_c | pressure_hpa | rh_pct |
|---|---|---|---|---|

**Injected table** = same, plus `label_temp_c`, `label_pressure_hpa`, `label_rh_pct` (values: `NONE, SPIKE, FROZEN, DRIFT, DROPOUT, NOISE, OUT_OF_RANGE`) and `event_id` (storms: `STORM_…`, label stays `NONE`).

**Stations** `data/stations.json`: `[{station_id, name, lat, lon, elevation_m}]`

**Result dict** (one per station × hour × variable). This is exactly what the dashboard reads:
```python
{
  "station_id": "DEL-03", "ts": "2024-08-14T09:00:00Z", "variable": "temp_c",
  "value": 55.0,
  "status": "SENSOR_FAULT",        # NORMAL | GENUINE_EVENT | SUSPECT | SENSOR_FAULT
  "fault_type": "SPIKE",           # None | SPIKE | FROZEN | DRIFT | DROPOUT | NOISE | OUT_OF_RANGE
  "confidence": 0.96,              # 0..1
  "severity": "CRITICAL",          # LOW | MEDIUM | HIGH | CRITICAL
  "reasons": ["Implied dewpoint 50.8 °C exceeds the physical maximum (~35 °C)", "..."],
  "shap_top": [["sudden 1-hour change", 0.41], ["...", 0.2]],
  "corrected_value": 31.6,
  "trust": 42.0,
  "scores": {"physics": 1.0, "iforest": 0.998, "spatial_z": 9.4}
}
```

### 4.2 Function signatures (each owner delivers exactly these)

```python
# M1 — skyguard/data/ , skyguard/inject/ , skyguard/heal/ , eval/
load_data() -> pd.DataFrame                                   # the parquet above
load_stations() -> list[dict]
climatology(df) -> Clim            # Clim.expected(station_id, ts, var) -> (mean, std)
inject_fault(df, station_id, var, fault_type, t0, duration_h, magnitude, seed=42) -> df
inject_storm(df, stations, center_id, t0, radius_km=80, strength=1.0, seed=42) -> df
build_test_set(df, seed=42) -> df  # many random faults + storms, with labels
impute(history, neighbours_now, clim, station_id, ts, var) -> float
evaluate(results_df, labels_df) -> dict   # writes eval/reports/metrics.json

# M3 — skyguard/physics/ , skyguard/spatial/ , skyguard/health/ , skyguard/engine.py
check_physics(station_hist: pd.DataFrame, var: str) -> {"hard": bool, "soft": bool, "rule_ids": [...], "reasons": [...]}
spatial_check(df_window, stations, clim, station_id, ts, var) -> {"z": float, "n": int, "consistent": bool | None, "reason": str}
class TrustTracker: update(station_id, var, status, severity) -> float
class Engine:
    def __init__(self, df, stations, clim, model): ...
    def step(self, ts) -> list[dict]        # results for all stations/variables at ts
    def run(self, start, end) -> pd.DataFrame

# M2 — skyguard/models/ , skyguard/fusion/
make_features(station_hist: pd.DataFrame, clim, var) -> np.ndarray   # features for the latest row
class IFModel: fit(clean_df, clim); score(features, var) -> float (0..1 percentile); save(); load()
decide(physics, spatial, ml_score, dev_sigma) -> (status, confidence, severity)
fingerprint(station_hist, var, physics) -> fault_type | None
explain(physics, spatial, ml_score, features, var) -> (reasons, shap_top)
```

**Unblock rule:** M1 shares a small sample file (`data/sample_2stations.parquet`, one month) **within 2 hours of kickoff**. Everyone develops against it. M4 builds the UI on a hand-written `data/mock_results.json` in the result-dict format above.

---

## 5. Tasks per person

Each person has roughly the same ~14–16 focused hours spread over 28–29 Sept.

### M1 — Data, Injector, Correction & Scorecard
| ID | Task | Done when | Due |
|---|---|---|---|
| D1 | Fetch Meteostat hourly for 6–8 stations within ~150 km of Delhi (`Stations().nearby(28.61, 77.21)`), 2022–2024; keep stations with ≥ 80 % data for T, P, RH | parquet + `stations.json` committed | **28 Sep 11:00** (sample file within 2 h of kickoff) |
| D2 | Clean (hourly index, interpolate gaps ≤ 2 h) + climatology (mean/std per station × month × hour) | `climatology()` works | 28 Sep 14:00 |
| D3 | Injector: all 6 fault types + `inject_storm` (formulas and magnitudes in TEAM_PLAN §6, M1-03) | unit test: each fault changes the data and sets the label | 28 Sep 20:00 |
| D4 | `build_test_set`: 2024 Jul–Dec, ~40 faults of each type + 10 storms, seed 42 | `data/injected/test.parquet` | 28 Sep 22:00 |
| D5 | `impute()`: expected(station) + median neighbour anomaly, blended with linear interpolation | MAE vs the original clean value printed per variable | 29 Sep 12:00 |
| D6 | `evaluate()` + `eval/run_eval.py`: per-fault precision/recall/F1, false-alarm rate on normal rows and on **storm rows**, avg latency per step | `metrics.json` + a bar-chart PNG | 29 Sep 17:00 |

### M2 — ML Scoring, Decisions & Explanations
| ID | Task | Done when | Due |
|---|---|---|---|
| L1 | `make_features`: value, Δ1h, Δ3h, rolling std 6h/24h, deviation from climatology (σ units), same-value run length, dewpoint, Δdewpoint, hour sin/cos | works on the sample file | 28 Sep 13:00 |
| L2 | `IFModel`: one IsolationForest per variable, `n_estimators=200, random_state=42`, trained on 2022–2023 clean data; score → percentile against 2024 Jan–Jun | model saved to `models/`, loads in < 2 s | 28 Sep 19:00 |
| L3 | `decide()`: the fusion rules in TEAM_PLAN §7, M2-04 (physics hard → fault; ML high + neighbours disagree → fault; ML high + neighbours agree → GENUINE_EVENT; severity from σ deviation) | unit tests for the 55 °C case and the storm case | 28 Sep 22:00 |
| L4 | `fingerprint()`: rule-based fault type (flatline → FROZEN, missing/−9999 → DROPOUT, rail → OUT_OF_RANGE, jump-and-return → SPIKE, high 6-h std → NOISE, persistent neighbour offset → DRIFT) | correct type on injected examples | 29 Sep 11:00 |
| L5 | `explain()`: physics reasons + spatial reason + SHAP top 3 (`shap.TreeExplainer` on the IsolationForest, feature names turned into plain English), max 4 reasons | every non-NORMAL result has ≥ 1 reason | 29 Sep 14:00 |
| L6 | Tune thresholds on **2024 Jan–Jun** (never on the test set) with M1 | final thresholds in `config.yaml` | 29 Sep 17:00 |

### M3 — Physics, Neighbour Check, Trust & Engine (integrator)
| ID | Task | Done when | Due |
|---|---|---|---|
| E0 | Tonight: GitHub repo, folders, `requirements.txt` (Python 3.11: pandas, numpy, scikit-learn, shap, meteostat, streamlit, plotly, pyyaml, pyarrow, pytest), `config.yaml`, invite everyone | everyone has cloned it and runs `pip install -r requirements.txt` | **27 Sep night** |
| E1 | `check_physics`: RANGE, MISSING, rate-of-change, DEWPOINT_MAX, DEWPOINT_JUMP, FLATLINE, DECOUPLED (thresholds in TEAM_PLAN §6, M1-04) | unit tests pass | 28 Sep 14:00 |
| E2 | `spatial_check`: anomaly vs climatology, robust z against the neighbour median, ±2 h window, `consistent = abs(z) <= 3` | a storm gives consistent=True; a single-station spike gives False | 28 Sep 19:00 |
| E3 | `TrustTracker`: start 100; penalties LOW 2 / MED 5 / HIGH 10 / CRIT 20; +0.5 per clean hour | trust drops and recovers | 28 Sep 21:00 |
| E4 | `Engine.step/run`: calls everything in §4 order, handles missing modules gracefully, records `latency_ms` | runs the whole test set end-to-end | **29 Sep 12:00** (integration deadline) |
| E5 | Precompute `data/demo_results.parquet` for the demo window (fallback for the video, see §8) | file exists | 29 Sep 15:00 |
| E6 | `run_demo.bat` → `streamlit run dashboard/app.py`; test on a 2nd laptop | works from a fresh clone | 29 Sep 18:00 |

### M4 — Dashboard
| ID | Task | Done when | Due |
|---|---|---|---|
| U1 | Streamlit skeleton reading `mock_results.json`; layout: map (left), alert feed (right), station chart (bottom), demo sidebar | runs on mock data | 28 Sep 13:00 |
| U2 | Station map (Plotly, `open-street-map` tiles): green = normal, **blue = genuine event**, red = fault, dot size = severity | colours change with the data | 28 Sep 18:00 |
| U3 | **Alert card**: station · variable · status · fault type · severity · confidence %, "reported → corrected", reasons as bullets, small SHAP bar, trust before → after | matches the example in TEAM_PLAN §9 | 28 Sep 22:00 |
| U4 | Station chart: T/P/RH last 48 h, raw (grey), corrected (colour), red ✕ on faults, blue band on genuine events, dotted neighbour median | readable on a 1080p screen recording | 29 Sep 12:00 |
| U5 | Replay controls: play/pause, speed (1 h per 0.5–2 s), jump-to-date; **Demo buttons**: "Inject 55 °C spike", "Freeze sensor", "Trigger storm" (call M1's injector on the upcoming data, then continue the replay) | all 3 buttons work on the real engine | 29 Sep 16:00 |
| U6 | Scorecard tab: reads `metrics.json` (F1 bars, storm false-alarm rate as a big number); colours matching the PPT (navy `0B2447`, sky `2CA6E0`, amber `F5A623`, red `E5484D`, green `2FA37C`) | looks like the PPT | 29 Sep 18:00 |

### A1 (auxiliary) — Video, PPT, README, QA
| ID | Task | Done when | Due |
|---|---|---|---|
| A1 | Check the SIH portal form requirements (§1) and tell the team | message sent | **27 Sep night** |
| A2 | Write the video script + shot list (§7), time it to ≤ 2:45 | script in `docs/VIDEO_SCRIPT.md` | 28 Sep 18:00 |
| A3 | README: what it is, screenshot, setup on Windows (Python 3.11 venv), `run_demo.bat`, how to run the evaluation, team | README complete | 29 Sep 17:00 |
| A4 | QA run (§9 checklist) on a 2nd laptop with M3 | all boxes ticked or bugs reported | 29 Sep 19:00 |
| A5 | Record + edit the video (OBS Studio, 1080p, voice-over), upload **Unlisted** | YouTube link works in an incognito window | 29 Sep 23:00 |
| A6 | Update PPT: real numbers from `metrics.json` in slide 5, one dashboard screenshot in slide 3 or 4, repo + video links on slide 6, team details on slide 1; export PDF, check it's still 6 slides | final PDF | 30 Sep 11:00 |
| A7 | (Only if hardware already exists) 15-s ESP32 clip: live BME280 readings + range/flatline flags on Serial | clip in video | 29 Sep |

---

## 6. Timeline and checkpoints

| When | What | Checkpoint (all must be true) |
|---|---|---|
| **27 Sep, tonight** | 45-min kickoff call: read this doc, agree §4 formats, assign names, M3 creates repo, A1 checks portal | everyone has the repo and a working Python 3.11 env |
| **28 Sep, 11:00** | M1's sample data out | everyone is developing on real data |
| **28 Sep, 14:00** | Stand-up (15 min) | each module runs on the sample file |
| **28 Sep, 22:00** | Stand-up | physics, spatial, IForest, decide, injector, test set, alert card all merged to `main` |
| **29 Sep, 12:00** | 🔴 **Integration deadline** | `Engine.run()` processes the whole test set; dashboard shows engine output |
| **29 Sep, 17:00** | Numbers frozen | `metrics.json` final; screenshots taken |
| **29 Sep, 19:00** | QA done | §9 checklist passed on a 2nd laptop |
| **29 Sep, 23:00** | Video uploaded | link works in incognito |
| **30 Sep, 11:00** | Final PPT PDF | reviewed by all 5 |
| **30 Sep, by 14:00** | **SUBMIT** | never leave it for the last hours, portals slow down near deadlines |

---

## 7. Demo video script (≈ 2:30)

| Time | Screen | Voice-over (short version) |
|---|---|---|
| 0:00–0:20 | Title slide → problem | "Weather-station sensors fail silently: spikes, frozen values, drift. Bad data looks real and corrupts forecasts and disaster warnings." |
| 0:20–0:40 | Dashboard, replay running, map all green | "SkyGuard AI replaying real hourly data from 6 Delhi-NCR stations. Every reading is checked against physics, neighbouring stations and a machine-learning model." |
| 0:40–1:15 | Click **Inject 55 °C spike** → red dot, alert card | "The PS's own example: 55 °C. Flagged as a sensor fault, 96 % confidence. Why? The implied dewpoint is physically impossible, only temperature moved, and all neighbours are normal. Suggested corrected value: 31.6 °C. The sensor's trust score drops." |
| 1:15–1:45 | Click **Trigger storm** → several blue dots | "Now a real storm: temperature drops 8 °C at several stations together. Same size change, but neighbours agree and physics holds, so it's a **genuine weather event, no false alarm**." |
| 1:45–2:00 | Click **Freeze sensor** → FROZEN alert | "A stuck sensor: identified as FROZEN from its signature." |
| 2:00–2:20 | Scorecard tab | "On our anomaly-injected test set: F1 of X for spikes, Y for frozen sensors, and only Z % false alarms during storms." |
| 2:20–2:30 | Architecture slide | "Next: edge detection on a ₹400 ESP32 and real-time streaming to scale across IMD's network." |

Record at 1080p, zoom the browser to 110–125 % so text is readable, no personal tabs/notifications visible.

---

## 8. Fallbacks (decide at the checkpoints, not at midnight)

| If at the checkpoint… | Then |
|---|---|
| Meteostat stations too gappy (28 Sep 11:00) | switch to another dense region (Mumbai / Bengaluru); if still bad, use fewer stations (minimum 4) |
| IsolationForest not ready (28 Sep 22:00) | engine runs physics + spatial only (already demonstrates the core idea); ML added if ready by 29 Sep 12:00 |
| Integration not working (29 Sep 12:00) | M3 runs the engine offline → `demo_results.parquet`; dashboard **replays precomputed results** with the fault/storm scenarios pre-injected at known times. The video looks identical. |
| SHAP slow or failing | show physics + spatial reasons only; SHAP mentioned as "in progress" |
| Numbers look weak for a fault type | report them honestly and name it as the improvement area (judges trust honest numbers more than perfect ones). Never tune on test. |
| Video recording goes wrong (29 Sep night) | record in 3 short segments and join them; voice-over can be added separately |

---

## 9. QA checklist (29 Sep, before recording)

- [ ] Fresh clone → `pip install -r requirements.txt` → `run_demo.bat` works on a 2nd laptop
- [ ] Replay runs for 5 minutes without crashing
- [ ] 55 °C spike → SENSOR_FAULT · SPIKE · CRITICAL, ≥ 2 reasons, corrected value within ~3 °C of neighbours
- [ ] Storm → affected stations show GENUINE_EVENT (blue), **no** SENSOR_FAULT
- [ ] Freeze → FROZEN within 6 simulated hours
- [ ] Trust drops after a fault, then recovers
- [ ] Scorecard shows `metrics.json` numbers, and they match the PPT
- [ ] README steps work exactly as written
- [ ] Repo is **public**, no API keys or personal files committed
- [ ] Video link opens in an incognito window

---

## 10. Working rules for these 3 days

1. **Format first:** §4 is frozen after tonight's call.
2. **Small commits, push often.** Branch per task (`m2/decide`), and a teammate merges after a quick look. On 29 Sep after 12:00 only M3 merges to `main`.
3. **Stand-ups at 14:00 and 22:00 on the 28th; 12:00 and 17:00 on the 29th.** 15 min max: done / next / blocked.
4. **Stuck > 45 min → ask in the group.** Pairs: M1↔M2 (data & thresholds), M3↔M4 (engine ↔ dashboard), A1↔M4 (video).
5. **Sleep on the 28th.** The 29th is integration + recording and needs clear heads.
