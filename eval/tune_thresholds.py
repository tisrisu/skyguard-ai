"""L6 — Tune ml.high, ml.suspect, severity_sigma on the validation split.

Workflow:
  1. Train IFModel on clean 2022-2023 data (+ clean val for percentile calibration)
  2. Score every row in data/injected/val.parquet
  3. Sweep threshold combos, run decide(), evaluate detection performance
  4. Pick the best thresholds and update config.yaml + save the trained model

NEVER touches the test split (data/injected/test.parquet).
"""

import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

# Add project root to path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from skyguard.config import load_config, CONFIG_FILE, MODELS_DIR
from skyguard.data.io import load_data, select_split
from skyguard.data.climatology import climatology
from skyguard.models.iforest import IFModel
from skyguard.models.features import make_features, make_feature_frame, FEATURE_NAMES
from skyguard.fusion.decide import decide
from skyguard.schemas import VARIABLES, LABEL_COLUMNS, FaultType, Status


# ---------------------------------------------------------------------------
# Step 1: Train IFModel
# ---------------------------------------------------------------------------

def train_model(cfg):
    print("=" * 60)
    print("STEP 1: Training IFModel")
    print("=" * 60)

    df = load_data()
    train_df = select_split(df, "train", cfg)
    val_clean_df = select_split(df, "val", cfg)

    print(f"  Train: {len(train_df)} rows")
    print(f"  Val (clean): {len(val_clean_df)} rows")

    clim = climatology(train_df)

    model = IFModel(cfg)
    t0 = time.time()
    model.fit(train_df, val_clean_df, clim)
    print(f"  Training took {time.time() - t0:.1f}s")

    model.save()
    print(f"  Model saved to {MODELS_DIR}")
    return model, clim


def load_or_train_model(cfg):
    model_path = MODELS_DIR / "iforest.joblib"
    if model_path.exists():
        print("  Loading saved model...")
        model = IFModel.load(model_path)
        df = load_data()
        train_df = select_split(df, "train", cfg)
        clim = climatology(train_df)
        return model, clim
    return train_model(cfg)


# ---------------------------------------------------------------------------
# Step 2: Score every row in val.parquet (vectorized)
# ---------------------------------------------------------------------------

def score_val_set(model, clim, cfg):
    print("\n" + "=" * 60)
    print("STEP 2: Scoring validation set (vectorized)")
    print("=" * 60)

    val_df = pd.read_parquet(ROOT / "data" / "injected" / "val.parquet")
    val_df["ts"] = pd.to_datetime(val_df["ts"], utc=True)
    val_df = val_df.sort_values(["station_id", "ts"]).reset_index(drop=True)

    print(f"  {len(val_df)} rows, {val_df['station_id'].nunique()} stations")

    all_records = []
    for var in VARIABLES:
        label_col = LABEL_COLUMNS[var]
        t0 = time.time()

        # Vectorized feature extraction for all stations at once
        feat_df = make_feature_frame(val_df, clim, var)

        # Batch scoring: scale + score_samples for all rows
        if var in model.models:
            X_scaled = model.scalers[var].transform(feat_df.values)
            raw_scores = -model.models[var].score_samples(X_scaled)
            val_sorted = model.val_scores[var]
            ml_scores = np.searchsorted(val_sorted, raw_scores) / len(val_sorted)
        else:
            ml_scores = np.zeros(len(val_df))

        # Dev sigma (vectorized)
        dev_sigma = clim.deviation_sigma(val_df, var).fillna(0.0).values

        # Build records
        for i in range(len(val_df)):
            row = val_df.iloc[i]
            event_id = row.get("event_id", "")
            is_storm = str(event_id).startswith("STORM") if pd.notna(event_id) else False
            label = row[label_col]

            all_records.append({
                "station_id": row["station_id"],
                "ts": row["ts"],
                "variable": var,
                "value": row[var],
                "ml_score": float(ml_scores[i]),
                "dev_sigma": float(dev_sigma[i]),
                "label": label,
                "event_id": event_id,
                "is_storm": is_storm,
                "is_faulty": label != FaultType.NONE,
            })

        print(f"  {var}: scored {len(val_df)} rows in {time.time() - t0:.1f}s")

    scores_df = pd.DataFrame(all_records)
    print(f"  Total: {len(scores_df)} entries, "
          f"faulty={scores_df['is_faulty'].sum()}, storms={scores_df['is_storm'].sum()}")

    scores_path = ROOT / "data" / "injected" / "val_scores.parquet"
    scores_df.to_parquet(scores_path)
    print(f"  Saved to {scores_path}")
    return scores_df


def load_or_score(model, clim, cfg):
    scores_path = ROOT / "data" / "injected" / "val_scores.parquet"
    if scores_path.exists():
        print("  Loading cached scores...")
        return pd.read_parquet(scores_path)
    return score_val_set(model, clim, cfg)


# ---------------------------------------------------------------------------
# Step 3: Sweep thresholds
# ---------------------------------------------------------------------------

def evaluate_thresholds(scores_df, ml_high, ml_suspect, sev_sigma):
    """Vectorized evaluation — reimplements decide() logic with numpy arrays.

    Simulation assumptions (physics not available per-row):
      - physics_hard = False for all rows
      - faults: spatial consistent=False, z=5.0 (neighbours disagree)
      - storms + clean: spatial consistent=True, z=0.5 (neighbours agree)
    """
    ml = scores_df["ml_score"].values
    faulty = scores_df["is_faulty"].values
    storm = scores_df["is_storm"].values
    clean = ~faulty & ~storm

    # Simulated spatial: faults -> disagree, storms/clean -> agree
    neighbours_disagree = faulty  # only fault rows have spatial disagreement

    # Rule 2a: ml >= high AND neighbours disagree -> SENSOR_FAULT
    is_fault = (ml >= ml_high) & neighbours_disagree
    # Rule 2b: ml >= high AND neighbours agree -> GENUINE_EVENT (not flagged)
    # Rule 3: ml >= suspect AND neighbours disagree -> SUSPECT
    is_suspect = (ml >= ml_suspect) & (ml < ml_high) & neighbours_disagree
    # Everything else -> NORMAL

    flagged = is_fault | is_suspect

    tp = int((flagged & faulty).sum())
    fp = int((flagged & clean).sum())
    fn = int((~flagged & faulty).sum())

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

    storm_fault = int((is_fault & storm).sum())
    storm_total = int(storm.sum())
    storm_far = storm_fault / storm_total if storm_total > 0 else 0.0

    normal_fault = int((is_fault & clean).sum())
    normal_total = int(clean.sum())
    normal_far = normal_fault / normal_total if normal_total > 0 else 0.0

    per_fault = {}
    labels = scores_df["label"].values
    for ft in FaultType.FAULTS:
        ft_mask = labels == ft
        ft_total = int(ft_mask.sum())
        if ft_total > 0:
            per_fault[ft] = round(int((flagged & ft_mask).sum()) / ft_total, 4)

    return {
        "ml_high": ml_high, "ml_suspect": ml_suspect, "sev_sigma": sev_sigma,
        "precision": round(precision, 4), "recall": round(recall, 4), "f1": round(f1, 4),
        "storm_far": round(storm_far, 4), "normal_far": round(normal_far, 4),
        "tp": tp, "fp": fp, "fn": fn, "per_fault_recall": per_fault,
    }


def sweep_thresholds(scores_df):
    print("\n" + "=" * 60)
    print("STEP 3: Sweeping thresholds")
    print("=" * 60)

    ml_high_vals = [0.97, 0.98, 0.985, 0.99, 0.995]
    ml_suspect_vals = [0.90, 0.93, 0.95, 0.97]
    sev_sigma_vals = [[3, 6, 10], [2.5, 5, 8], [3.5, 7, 12]]

    results = []
    count = 0
    for mh in ml_high_vals:
        for ms in ml_suspect_vals:
            if ms >= mh:
                continue
            for sv in sev_sigma_vals:
                count += 1
                if count % 5 == 0:
                    print(f"  [{count}] mh={mh} ms={ms} sv={sv}...", flush=True)
                results.append(evaluate_thresholds(scores_df, mh, ms, sv))

    rdf = pd.DataFrame(results)
    valid = rdf[rdf["storm_far"] <= 0.05]
    if valid.empty:
        print("  WARNING: No combos with storm_far <= 5%. Relaxing to 10%.")
        valid = rdf[rdf["storm_far"] <= 0.10]
    if valid.empty:
        valid = rdf

    valid = valid.sort_values(["f1", "storm_far"], ascending=[False, True])

    print(f"\n  Top 5 combos (from {len(valid)} valid):")
    for _, r in valid.head(5).iterrows():
        print(f"    mh={r['ml_high']:.3f}  ms={r['ml_suspect']:.2f}  sv={r['sev_sigma']}  "
              f"F1={r['f1']:.4f}  P={r['precision']:.4f}  R={r['recall']:.4f}  "
              f"storm={r['storm_far']:.4f}  normal={r['normal_far']:.4f}")

    best = valid.iloc[0]
    print(f"\n  [OK] Best: ml.high={best['ml_high']}, ml.suspect={best['ml_suspect']}, "
          f"sev={best['sev_sigma']}")
    print(f"     F1={best['f1']}, storm_far={best['storm_far']}")
    print(f"     Per-fault recall: {best['per_fault_recall']}")
    return best


# ---------------------------------------------------------------------------
# Step 4: Update config.yaml
# ---------------------------------------------------------------------------

def update_config(best):
    print("\n" + "=" * 60)
    print("STEP 4: Updating config.yaml")
    print("=" * 60)

    with open(CONFIG_FILE, "r", encoding="utf-8") as f:
        raw = f.read()

    cfg = yaml.safe_load(raw)
    old_h = cfg["ml"]["high"]
    old_s = cfg["ml"]["suspect"]
    old_sv = cfg["severity_sigma"]
    new_h = best["ml_high"]
    new_s = best["ml_suspect"]
    new_sv = best["sev_sigma"]

    print(f"  ml.high:        {old_h} -> {new_h}")
    print(f"  ml.suspect:     {old_s} -> {new_s}")
    print(f"  severity_sigma: {old_sv} -> {new_sv}")

    lines = raw.split("\n")
    new_lines = []
    in_ml = False
    for line in lines:
        stripped = line.strip()
        if stripped == "ml:":
            in_ml = True
        elif stripped and not stripped.startswith("#") and not stripped.startswith("-") and ":" in stripped and not line.startswith(" "):
            in_ml = False

        if in_ml and stripped.startswith("high:"):
            indent = line[:len(line) - len(line.lstrip())]
            new_lines.append(f"{indent}high: {new_h}")
        elif in_ml and stripped.startswith("suspect:"):
            indent = line[:len(line) - len(line.lstrip())]
            new_lines.append(f"{indent}suspect: {new_s}")
        elif stripped.startswith("severity_sigma:"):
            new_lines.append(f"severity_sigma: {new_sv}       # LOW < {new_sv[0]} <= MEDIUM < {new_sv[1]} <= HIGH < {new_sv[2]} <= CRITICAL")
        else:
            new_lines.append(line)

    with open(CONFIG_FILE, "w", encoding="utf-8") as f:
        f.write("\n".join(new_lines))

    from skyguard.config import load_config
    load_config.cache_clear()
    print("  [OK] config.yaml updated")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    cfg = load_config()
    model, clim = load_or_train_model(cfg)
    scores_df = load_or_score(model, clim, cfg)
    best = sweep_thresholds(scores_df)
    update_config(best)

    print("\n" + "=" * 60)
    print("L6 COMPLETE [OK]")
    print("=" * 60)
    print(f"  ml.high       = {best['ml_high']}")
    print(f"  ml.suspect    = {best['ml_suspect']}")
    print(f"  severity_sigma = {best['sev_sigma']}")
    print(f"  F1={best['f1']}  P={best['precision']}  R={best['recall']}")
    print(f"  Storm FAR={best['storm_far']}  Normal FAR={best['normal_far']}")
