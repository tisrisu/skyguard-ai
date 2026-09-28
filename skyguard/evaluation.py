"""Scorecard: how well the engine finds the injected faults.

Owner: M1

A fault event is detected if any of its hours is flagged SENSOR_FAULT or SUSPECT.
A false positive is a run of consecutive flagged hours on a variable with no
injected fault. Runs are counted once, like events, so precision compares like
with like. False-alarm rates are per reading and count SENSOR_FAULT only.
"""

import numpy as np
import pandas as pd

from skyguard.schemas import LABEL_COLUMNS, ORIG_COLUMNS, VARIABLES, FaultType, Status

FLAGGED = [Status.SENSOR_FAULT, Status.SUSPECT]


def _ratio(a: int, b: int) -> float:
    return round(a / b, 4) if b else 0.0


def _align(results: pd.DataFrame, labelled: pd.DataFrame) -> pd.DataFrame:
    """Engine results with the true label, event and clean value of the same variable."""
    results = results.copy()
    results["ts"] = pd.to_datetime(results["ts"], utc=True)
    labelled = labelled.copy()
    labelled["ts"] = pd.to_datetime(labelled["ts"], utc=True)

    truth = []
    for var in VARIABLES:
        cols = {"station_id": "station_id", "ts": "ts", "event_id": "event_id", LABEL_COLUMNS[var]: "label"}
        if ORIG_COLUMNS[var] in labelled.columns:
            cols[ORIG_COLUMNS[var]] = "true_value"
        part = labelled[list(cols)].rename(columns=cols)
        part["variable"] = var
        truth.append(part)
    return results.merge(pd.concat(truth, ignore_index=True), on=["station_id", "variable", "ts"])


def _count_runs(rows: pd.DataFrame) -> int:
    """Number of runs of consecutive hours per station and variable."""
    if rows.empty:
        return 0
    rows = rows.sort_values(["station_id", "variable", "ts"])
    new_run = (
        (rows["station_id"] != rows["station_id"].shift())
        | (rows["variable"] != rows["variable"].shift())
        | (rows["ts"].diff() != pd.Timedelta(hours=1))
    )
    return int(new_run.sum())


def evaluate(results: pd.DataFrame, labelled: pd.DataFrame) -> dict:
    """Compare engine results with the injected labels.

    results    engine output, one row per station, ts and variable (see schemas.Result)
    labelled   injected data with label_*, orig_* and event_id columns
    """
    if results.empty:
        return {}
    df = _align(results, labelled)
    flagged = df["status"].isin(FLAGGED)
    clean = df["label"] == FaultType.NONE
    storm = df["event_id"].str.startswith("STORM", na=False)

    per_fault = {}
    false_flags = df[flagged & clean]
    correct_type = caught_rows = 0
    for fault_type in FaultType.FAULTS:
        rows = df[df["label"] == fault_type]
        hit = rows.groupby(["variable", "event_id"])["status"].agg(lambda s: s.isin(FLAGGED).any())
        tp, fn = int(hit.sum()), int((~hit).sum())
        fp = _count_runs(false_flags[false_flags["fault_type"] == fault_type])
        precision, recall = _ratio(tp, tp + fp), _ratio(tp, tp + fn)
        per_fault[fault_type] = {
            "events": tp + fn, "tp": tp, "fp": fp, "fn": fn,
            "precision": precision,
            "recall": recall,
            "f1": round(2 * precision * recall / (precision + recall), 4) if precision + recall else 0.0,
        }
        caught = rows[rows["status"].isin(FLAGGED)]
        correct_type += int((caught["fault_type"] == fault_type).sum())
        caught_rows += len(caught)

    faults = (df["status"] == Status.SENSOR_FAULT)
    normal = clean & ~storm

    impute_mae = dict.fromkeys(VARIABLES)
    if "true_value" in df.columns and "corrected_value" in df.columns:
        fixed = df[flagged & ~clean & df["corrected_value"].notna() & df["true_value"].notna()]
        for var, part in fixed.groupby("variable"):
            impute_mae[var] = round(float((part["corrected_value"] - part["true_value"]).abs().mean()), 4)

    if "scores" in df.columns:
        latency = df["scores"].map(lambda s: s.get("latency_ms") if isinstance(s, dict) else np.nan)
    else:
        latency = df.get("latency_ms", pd.Series(dtype=float))
    latency = pd.to_numeric(latency, errors="coerce").dropna()

    return {
        "per_fault": per_fault,
        "false_alarm_rate_normal": _ratio(int((faults & normal).sum()), int(normal.sum())),
        "false_alarm_rate_storm": _ratio(int((faults & clean & storm).sum()), int((clean & storm).sum())),
        "fault_type_accuracy": _ratio(correct_type, caught_rows) if caught_rows else None,
        "impute_mae": impute_mae,
        "latency_ms_mean": round(float(latency.mean()), 2) if len(latency) else None,
    }
