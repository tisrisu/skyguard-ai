"""Dewpoint from temperature and relative humidity (Magnus formula).

Dewpoint <= temperature always holds when RH <= 100 %, so that alone is not a
useful check. What is useful: dewpoints above ~35 °C have never been observed,
and dewpoint (the amount of moisture in the air) cannot jump by many degrees
within an hour. A temperature spike with unchanged RH implies exactly such a jump.
"""

import numpy as np

B = 17.62
C = 243.12  # °C


def dewpoint_c(temp_c, rh_pct):
    """Dewpoint in °C. Works on scalars and numpy/pandas arrays; NaN in -> NaN out."""
    t = np.asarray(temp_c, dtype=float)
    rh = np.clip(np.asarray(rh_pct, dtype=float), 1e-3, None)
    g = np.log(rh / 100.0) + B * t / (C + t)
    td = C * g / (B - g)
    return td.item() if td.ndim == 0 else td
