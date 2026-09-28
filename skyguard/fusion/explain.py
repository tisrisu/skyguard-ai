"""Plain-English reasons for a decision.

Owner: M2

Order: physics reasons first (most decisive), then the neighbour reason,
then the top SHAP features translated with models.features.FEATURE_LABELS.
At most 4 reasons. Only called for non-NORMAL results (SHAP is slower than detection).
"""

import numpy as np


def explain(physics: dict, spatial: dict, ml_score: float, features: np.ndarray, var: str,
            model=None) -> tuple[list[str], list[tuple[str, float]]]:
    """Return (reasons, shap_top) where shap_top is [(plain-English feature, contribution), ...] top 3.

    TODO: build one shap.TreeExplainer per variable once (cache it), not per call.
    """
    raise NotImplementedError
