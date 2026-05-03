"""Prediction-oriented utilities for the staged modeling core."""

from __future__ import annotations

import numpy as np


def probability_margin(probabilities: dict[str, float], primary_signal_name: str) -> float:
    """Return the edge between the primary class and its strongest runner-up."""

    normalized_probabilities = {
        str(signal_name): float(probability)
        for signal_name, probability in probabilities.items()
    }
    primary_probability = normalized_probabilities.get(primary_signal_name, 0.0)
    runner_up_probability = max(
        (
            probability
            for signal_name, probability in normalized_probabilities.items()
            if signal_name != primary_signal_name
        ),
        default=0.0,
    )
    return float(np.clip(primary_probability - runner_up_probability, 0.0, 1.0))
