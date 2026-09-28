import math

import numpy as np

from skyguard.physics.dewpoint import dewpoint_c


def test_saturated_air_dewpoint_equals_temperature():
    assert math.isclose(dewpoint_c(20.0, 100.0), 20.0, abs_tol=0.01)


def test_typical_value():
    assert math.isclose(dewpoint_c(30.0, 50.0), 18.4, abs_tol=0.1)


def test_spike_example_is_physically_impossible():
    # 55 °C at 80 % RH implies a dewpoint far above anything ever observed (~35 °C)
    assert dewpoint_c(55.0, 80.0) > 50.0


def test_arrays_and_nan():
    out = dewpoint_c(np.array([20.0, np.nan]), np.array([100.0, 50.0]))
    assert math.isclose(out[0], 20.0, abs_tol=0.01)
    assert np.isnan(out[1])
