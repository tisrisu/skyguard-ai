"""Scorecard: how well the engine finds the injected faults.

Owner: M1
"""

import pandas as pd
import numpy as np

from skyguard.schemas import Status, FaultType, VARIABLES


def evaluate(results: pd.DataFrame, labelled: pd.DataFrame) -> dict:
    """Compare engine results with the injected labels.

    results    engine output (one row per station, ts, variable; see schemas.Result)
    labelled   injected data with label_* and event_id columns
    """
    if results.empty:
        return {}

    # Normalize timestamps before merging (engine Result.ts is ISO string, labelled.ts is datetime)
    results = results.copy()
    labelled = labelled.copy()
    results["ts"] = pd.to_datetime(results["ts"], utc=True)
    labelled["ts"] = pd.to_datetime(labelled["ts"], utc=True)

    merged = pd.merge(results, labelled, on=["station_id", "ts"], how="left")
    
    metrics = {
        "per_fault": {},
        "false_alarm_rate_normal": 0.0,
        "false_alarm_rate_storm": 0.0,
        "fault_type_accuracy": 0.0,
        "impute_mae": {"temp_c": 0.0, "pressure_hpa": 0.0, "rh_pct": 0.0},
        "latency_ms_mean": 0.0
    }
    
    # 1. False alarm rates (only count SENSOR_FAULT)
    normal_mask = (merged["label_temp_c"] == FaultType.NONE) & \
                  (merged["label_pressure_hpa"] == FaultType.NONE) & \
                  (merged["label_rh_pct"] == FaultType.NONE) & \
                  (~merged["event_id"].str.startswith("STORM", na=False))
                  
    normal_rows = merged[normal_mask]
    if len(normal_rows) > 0:
        fp_normal = len(normal_rows[normal_rows["status"] == Status.SENSOR_FAULT])
        metrics["false_alarm_rate_normal"] = round(fp_normal / len(normal_rows), 4)
        
    storm_mask = merged["event_id"].str.startswith("STORM", na=False)
    storm_rows = merged[storm_mask]
    if len(storm_rows) > 0:
        fp_storm = len(storm_rows[storm_rows["status"] == Status.SENSOR_FAULT])
        metrics["false_alarm_rate_storm"] = round(fp_storm / len(storm_rows), 4)

    # 2. Per-fault detection (event-based)
    correct_type_guesses = 0
    total_fault_flags = 0
    
    for ftype in FaultType.FAULTS:
        metrics["per_fault"][ftype] = {"tp": 0, "fp": 0, "fn": 0, "precision": 0.0, "recall": 0.0, "f1": 0.0}

    for ftype in FaultType.FAULTS:
        tp_events = 0
        fn_events = 0
        
        for var in VARIABLES:
            label_col = f"label_{var}"
            if label_col not in merged.columns:
                continue
                
            var_data = merged[merged["variable"] == var]
            actual_fault_events = var_data[var_data[label_col] == ftype]["event_id"].dropna().unique()
            
            for eid in actual_fault_events:
                event_rows = var_data[var_data["event_id"] == eid]
                # Detected if ANY hour is SENSOR_FAULT or SUSPECT
                if event_rows["status"].isin([Status.SENSOR_FAULT, Status.SUSPECT]).any():
                    tp_events += 1
                    
                    # Accuracy check: did we guess the right type?
                    flagged = event_rows[event_rows["status"].isin([Status.SENSOR_FAULT, Status.SUSPECT])]
                    correct_type_guesses += len(flagged[flagged["fault_type"] == ftype])
                    total_fault_flags += len(flagged)
                else:
                    fn_events += 1
            
            # False positives for this type:
            # Row was NONE, but we flagged it as SENSOR_FAULT/SUSPECT *and* gave it this fault_type
            fps = len(var_data[(var_data[label_col] == FaultType.NONE) & 
                               (var_data["status"].isin([Status.SENSOR_FAULT, Status.SUSPECT])) & 
                               (var_data["fault_type"] == ftype)])
            metrics["per_fault"][ftype]["fp"] += fps
            
        metrics["per_fault"][ftype]["tp"] += tp_events
        metrics["per_fault"][ftype]["fn"] += fn_events
        
    for ftype in FaultType.FAULTS:
        d = metrics["per_fault"][ftype]
        tp, fp, fn = d["tp"], d["fp"], d["fn"]
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
        
        d["precision"] = round(precision, 4)
        d["recall"] = round(recall, 4)
        d["f1"] = round(f1, 4)

    if total_fault_flags > 0:
        metrics["fault_type_accuracy"] = round(correct_type_guesses / total_fault_flags, 4)

    # 3. Impute MAE
    faulty_rows = merged[merged["status"].isin([Status.SENSOR_FAULT, Status.SUSPECT])]
    if not faulty_rows.empty and "corrected_value" in faulty_rows.columns and "true_value" in faulty_rows.columns:
        for var in VARIABLES:
            var_faults = faulty_rows[faulty_rows["variable"] == var]
            if not var_faults.empty:
                mae = np.abs(var_faults["true_value"] - var_faults["corrected_value"]).mean()
                metrics["impute_mae"][var] = round(float(mae), 4)

    # 4. Latency — stored inside the scores dict, not as a top-level column
    if "scores" in merged.columns:
        latencies = merged["scores"].apply(
            lambda s: s.get("latency_ms") if isinstance(s, dict) else None
        ).dropna()
        if not latencies.empty:
            metrics["latency_ms_mean"] = round(float(latencies.mean()), 2)
    elif "latency_ms" in merged.columns:
        metrics["latency_ms_mean"] = round(merged["latency_ms"].mean(), 2)

    return metrics
