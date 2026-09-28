# Contributing

## Branches

- Never commit directly to `main`.
- One branch per task: `<member>/<task>`, e.g. `m1/injector`, `m2/iforest`, `m3/physics-rules`, `m4/alert-card`.
- Keep branches short-lived. Merge within a day where possible.

```bat
git checkout main
git pull
git checkout -b m1/injector
:: ...work...
git add -A
git commit -m "Add spike and frozen faults to injector"
git push -u origin m1/injector
```

Then open a Pull Request on GitHub. One other member reviews and merges it.

Before starting new work, update your branch:

```bat
git checkout main
git pull
git checkout m1/injector
git merge main
```

## Before opening a PR

- `pytest` passes
- No thresholds hard-coded in the code; put them in `config.yaml`
- No raw data, virtual environments or personal files committed
- Functions keep the signatures documented in their docstrings; if a shared format has to change, tell the team first

## Data conventions

- Long format, one row per station per hour: `station_id, ts, temp_c, pressure_hpa, rh_pct`
- `ts` is a timezone-aware UTC timestamp; convert to IST only in the dashboard
- Missing values are `NaN`. Sentinel values like `-9999` appear only in injected faulty data
- Injected data adds `label_temp_c`, `label_pressure_hpa`, `label_rh_pct` and `event_id`
- Use `seed` from `config.yaml` for anything random
- Tune thresholds on the validation split, report results on the test split
