"""Application-layer facade for long-running workflows and CLI use cases.

The concrete implementations still live in ``crypto_signal_ml.app``. This
package provides a clearer import surface for new code without breaking the
existing module layout.
"""

from ..app import (
    BacktestApp,
    BaseSignalApp,
    MarketDataRefreshApp,
    MarketEventsRefreshApp,
    MarketUniverseRefreshApp,
    ModelComparisonApp,
    ProductionCycleApp,
    RegimeTrainingApp,
    RegimeWalkForwardValidationApp,
    SignalGenerationApp,
    SignalParameterTuningApp,
    TrainingApp,
    WalkForwardValidationApp,
)

__all__ = [
    "BacktestApp",
    "BaseSignalApp",
    "MarketDataRefreshApp",
    "MarketEventsRefreshApp",
    "MarketUniverseRefreshApp",
    "ModelComparisonApp",
    "ProductionCycleApp",
    "RegimeTrainingApp",
    "RegimeWalkForwardValidationApp",
    "SignalGenerationApp",
    "SignalParameterTuningApp",
    "TrainingApp",
    "WalkForwardValidationApp",
]
