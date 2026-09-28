import numpy as np
import pandas as pd

from skyguard.data.climatology import Climatology
from skyguard.evaluation import evaluate
from skyguard.heal.impute import impute
from skyguard.inject.injector import inject_fault, inject_storm
from skyguard.schemas import LABEL_COLUMNS, ORIG_COLUMNS, VARIABLES

T0 = pd.Timestamp("2024-08-10 12:00", tz="UTC")


def engine_output(labelled, flag_storms=False):
    """Fake engine: flags exactly the injected faults, corrects them to the clean value + 1."""
    out = []
    for var in VARIABLES:
        for row in labelled.itertuples(index=False):
            label = getattr(row, LABEL_COLUMNS[var])
            is_fault = label != "NONE"
            is_storm = row.event_id.startswith("STORM")
            status = "SENSOR_FAULT" if is_fault or (flag_storms and is_storm) else "NORMAL"
            if is_storm and not flag_storms:
                status = "GENUINE_EVENT"
            out.append({
                "station_id": row.station_id, "ts": row.ts.isoformat(), "variable": var,
                "status": status, "fault_type": label if is_fault else None,
                "corrected_value": getattr(row, ORIG_COLUMNS[var]) + 1 if is_fault else None,
                "scores": {"latency_ms": 4.0},
            })
    return pd.DataFrame(out)


def labelled_data(hourly, stations):
    df = inject_storm(hourly, stations, "A", T0 + pd.Timedelta(days=3), seed=0)
    df = inject_fault(df, "A", "temp_c", "SPIKE", T0, 1, magnitude=20)
    return inject_fault(df, "B", "rh_pct", "FROZEN", T0, 8)


def test_perfect_detector(hourly, stations):
    labelled = labelled_data(hourly, stations)
    m = evaluate(engine_output(labelled), labelled)

    assert m["per_fault"]["SPIKE"] == {"events": 1, "tp": 1, "fp": 0, "fn": 0,
                                       "precision": 1.0, "recall": 1.0, "f1": 1.0}
    assert m["per_fault"]["FROZEN"]["recall"] == 1.0
    assert m["false_alarm_rate_normal"] == 0.0
    assert m["false_alarm_rate_storm"] == 0.0        # GENUINE_EVENT is the right answer
    assert m["fault_type_accuracy"] == 1.0
    assert m["impute_mae"]["temp_c"] == 1.0
    assert m["latency_ms_mean"] == 4.0


def test_flagging_storms_counts_as_false_alarms(hourly, stations):
    labelled = labelled_data(hourly, stations)
    m = evaluate(engine_output(labelled, flag_storms=True), labelled)
    assert m["false_alarm_rate_storm"] == 1.0


def test_impute_ignores_bad_neighbours(hourly):
    clim = Climatology().fit(hourly)
    ts = T0
    history = hourly[(hourly["station_id"] == "A") & (hourly["ts"] < ts)]
    neighbours = hourly[(hourly["station_id"] != "A") & (hourly["ts"] == ts)].copy()
    good = impute(history, neighbours, clim, "A", ts, "temp_c")

    neighbours.loc[neighbours.index[0], "temp_c"] = -9999.0
    with_sentinel = impute(history, neighbours, clim, "A", ts, "temp_c")

    true_value = hourly.loc[(hourly["station_id"] == "A") & (hourly["ts"] == ts), "temp_c"].item()
    assert abs(good - true_value) < 0.5
    assert np.isclose(good, with_sentinel)
