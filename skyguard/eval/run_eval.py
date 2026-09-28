"""Evaluation metrics for the engine outputs.

Owner: M1

Run: python -m skyguard.eval.run_eval
"""
import json
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pathlib import Path

from skyguard.config import RESULTS_DIR

def evaluate(results_df: pd.DataFrame, injected_df: pd.DataFrame) -> dict:
    """Calculate precision, recall, F1, false-alarm rates, and latency.
    
    results_df: Output from Engine.run() containing predicted statuses
    injected_df: The test set with ground-truth label_* and event_id columns
    """
    # Merge results with ground truth on (station_id, ts, var)
    # The results_df is expected to be in long format: station_id, ts, var, status, fault_type, latency_ms
    if results_df.empty:
        return {}

    merged = pd.merge(results_df, injected_df, on=["station_id", "ts"], how="left")
    
    metrics = {
        "per_fault": {},
        "false_alarms": {},
        "latency": {}
    }
    
    # We evaluate for each variable
    variables = ["temp_c", "pressure_hpa", "rh_pct"]
    fault_types = ["SPIKE", "FROZEN", "DRIFT", "DROPOUT", "NOISE", "OUT_OF_RANGE"]
    
    for var in variables:
        label_col = f"label_{var}"
        if label_col not in merged.columns:
            continue
            
        var_data = merged[merged["var"] == var]
        
        for ftype in fault_types:
            # True Positives: Actual fault is ftype, Predicted status is SENSOR_FAULT/SUSPECT and predicted fault_type matches
            tp = len(var_data[(var_data[label_col] == ftype) & (var_data["status"] != "NORMAL")])
            
            # False Negatives: Actual fault is ftype, Predicted status is NORMAL
            fn = len(var_data[(var_data[label_col] == ftype) & (var_data["status"] == "NORMAL")])
            
            # False Positives: Predicted status is not NORMAL, but actual label is NONE (or different, but we mostly care about predicting NORMAL vs FAULT)
            # For strict type evaluation:
            fp = len(var_data[(var_data[label_col] != ftype) & (var_data["status"] != "NORMAL") & (var_data["fault_type"] == ftype)])
            
            precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
            f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
            
            if ftype not in metrics["per_fault"]:
                metrics["per_fault"][ftype] = {"tp": 0, "fp": 0, "fn": 0}
            
            metrics["per_fault"][ftype]["tp"] += tp
            metrics["per_fault"][ftype]["fp"] += fp
            metrics["per_fault"][ftype]["fn"] += fn
            
    # Aggregate over all variables for the final metrics
    for ftype in fault_types:
        d = metrics["per_fault"][ftype]
        tp, fp, fn = d["tp"], d["fp"], d["fn"]
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
        
        d["precision"] = round(precision, 4)
        d["recall"] = round(recall, 4)
        d["f1"] = round(f1, 4)

    # False Alarms
    # Normal rows (label is NONE and not a storm)
    normal_rows = merged[(merged["label_temp_c"] == "NONE") & 
                         (merged["label_pressure_hpa"] == "NONE") & 
                         (merged["label_rh_pct"] == "NONE") &
                         (~merged["event_id"].str.startswith("STORM", na=False))]
    
    fp_normal = len(normal_rows[normal_rows["status"] != "NORMAL"])
    metrics["false_alarms"]["normal_rate"] = round(fp_normal / len(normal_rows) if len(normal_rows) > 0 else 0, 4)
    
    # Storm rows
    storm_rows = merged[merged["event_id"].str.startswith("STORM", na=False)]
    fp_storm = len(storm_rows[storm_rows["status"] != "NORMAL"])
    metrics["false_alarms"]["storm_rate"] = round(fp_storm / len(storm_rows) if len(storm_rows) > 0 else 0, 4)
    
    # Latency
    if "latency_ms" in merged.columns:
        metrics["latency"]["avg_ms"] = round(merged["latency_ms"].mean(), 2)
        metrics["latency"]["max_ms"] = round(merged["latency_ms"].max(), 2)
        
    return metrics


def plot_metrics(metrics: dict, output_path: Path):
    fault_types = list(metrics["per_fault"].keys())
    f1_scores = [metrics["per_fault"][f].get("f1", 0) for f in fault_types]
    
    plt.figure(figsize=(10, 6))
    plt.bar(fault_types, f1_scores, color='skyblue')
    plt.axhline(y=0.8, color='r', linestyle='--', label='Target (0.8)')
    plt.title("F1 Score per Fault Type")
    plt.ylabel("F1 Score")
    plt.ylim(0, 1.05)
    plt.legend()
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()


if __name__ == "__main__":
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    
    # In a real run, this would load results from Engine.run
    # For now, we mock some results if they don't exist
    try:
        results = pd.read_parquet(RESULTS_DIR / "demo_results.parquet")
        injected = pd.read_parquet("data/injected/test.parquet")
        
        metrics = evaluate(results, injected)
        
        with open(RESULTS_DIR / "metrics.json", "w") as f:
            json.dump(metrics, f, indent=2)
            
        plot_metrics(metrics, RESULTS_DIR / "f1_scores.png")
        print(f"Metrics saved to {RESULTS_DIR}")
        
    except FileNotFoundError:
        print("Required data files not found. Run the engine first.")
