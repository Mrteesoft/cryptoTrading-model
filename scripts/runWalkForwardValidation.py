"""Run walk-forward validation for the configured signal model."""

from __future__ import annotations

from scriptSupport import bootstrap_src_path

bootstrap_src_path()

from crypto_signal_ml.application import WalkForwardValidationApp  # noqa: E402


def main() -> None:
    """Execute the reusable walk-forward validation workflow."""

    results = WalkForwardValidationApp().run()

    print("Walk-forward validation complete.")
    print(f"Model type: {results['modelType']}")
    print(f"Run label: {results['runLabel']}")
    print(f"Run directory: {results['runDirectory']}")
    print(f"Prepared dataset saved to: {results['datasetPath']}")
    print(f"Fold metrics saved to: {results['walkForwardFoldMetricsPath']}")
    print(f"Predictions saved to: {results['walkForwardPredictionsPath']}")
    print(f"Feature importance saved to: {results['walkForwardFeatureImportancePath']}")
    print(f"Summary saved to: {results['walkForwardSummaryPath']}")
    print(f"Backtest trades saved to: {results['walkForwardBacktestTradesPath']}")
    print(f"Backtest periods saved to: {results['walkForwardBacktestPeriodsPath']}")
    print(f"Backtest summary saved to: {results['walkForwardBacktestSummaryPath']}")
    print(f"Folds: {results['foldCount']}")
    print(f"Out-of-sample rows: {results['outOfSampleRows']}")
    print(f"Accuracy: {results['accuracy']:.4f}")
    print(f"Balanced accuracy: {results['balancedAccuracy']:.4f}")
    print(f"Average fold balanced accuracy: {results['averageFoldBalancedAccuracy']:.4f}")
    print(f"Trade count: {results['tradeCount']}")
    print(f"Strategy total return: {results['strategyTotalReturn']:.4f}")
    print(f"Benchmark total return: {results['benchmarkTotalReturn']:.4f}")
    print(f"Max drawdown: {results['maxDrawdown']:.4f}")


if __name__ == "__main__":
    main()
