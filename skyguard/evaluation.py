"""Scorecard: how well the engine finds the injected faults.

Owner: M1
"""

import pandas as pd


def evaluate(results: pd.DataFrame, labelled: pd.DataFrame) -> dict:
    """Compare engine results with the injected labels.

    results    engine output (one row per station, ts, variable; see schemas.Result)
    labelled   injected data with label_* and event_id columns

    TODO, return a dict like:
      {
        "per_fault": {"SPIKE": {"precision": .., "recall": .., "f1": ..}, ...},
        "false_alarm_rate_normal": ..,    # clean rows flagged SENSOR_FAULT
        "false_alarm_rate_storm": ..,     # storm rows flagged SENSOR_FAULT (key number)
        "fault_type_accuracy": ..,
        "impute_mae": {"temp_c": .., "pressure_hpa": .., "rh_pct": ..},
        "latency_ms_mean": ..
      }
    A fault counts as detected if any hour of its event_id is flagged SENSOR_FAULT or SUSPECT.
    """
    raise NotImplementedError
