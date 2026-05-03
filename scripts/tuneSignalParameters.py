"""Tune label and confidence settings with walk-forward validation."""

from __future__ import annotations

from scriptSupport import bootstrap_src_path

bootstrap_src_path()

from crypto_signal_ml.application import SignalParameterTuningApp  # noqa: E402


def main() -> None:
    """Run the reusable signal-parameter tuning workflow."""

    results = SignalParameterTuningApp().run()

    print("Signal parameter tuning complete.")
    print(f"Model type: {results['modelType']}")
    print(f"Label tuning results saved to: {results['labelResultsPath']}")
    print(f"Confidence tuning results saved to: {results['confidenceResultsPath']}")
    print(f"Summary saved to: {results['summaryPath']}")
    print(f"Best prediction horizon: {results['bestPredictionHorizon']}")
    print(f"Best buy threshold: {results['bestBuyThreshold']:.4f}")
    print(f"Best sell threshold: {results['bestSellThreshold']:.4f}")
    print(f"Best backtest min confidence: {results['bestBacktestMinConfidence']:.2f}")
    print(f"Balanced accuracy: {results['balancedAccuracy']:.4f}")
    print(f"Average fold balanced accuracy: {results['averageFoldBalancedAccuracy']:.4f}")
    print(f"Trade count: {results['tradeCount']}")
    print(f"Strategy total return: {results['strategyTotalReturn']:.4f}")
    print(f"Max drawdown: {results['maxDrawdown']:.4f}")


if __name__ == "__main__":
    main()
