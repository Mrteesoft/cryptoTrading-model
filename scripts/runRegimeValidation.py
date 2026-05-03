"""Run walk-forward validation for the standalone market-regime model."""

from __future__ import annotations

from scriptSupport import bootstrap_src_path

bootstrap_src_path()

from crypto_signal_ml.application import RegimeWalkForwardValidationApp  # noqa: E402


def main() -> None:
    """Execute the reusable regime walk-forward validation workflow."""

    results = RegimeWalkForwardValidationApp().run()

    print("Regime walk-forward validation complete.")
    print(f"Model type: {results['modelType']}")
    print(f"Estimator type: {results['estimatorType']}")
    print(f"Run label: {results['runLabel']}")
    print(f"Run directory: {results['runDirectory']}")
    print(f"Prepared dataset saved to: {results['datasetPath']}")
    print(f"Fold metrics saved to: {results['walkForwardFoldMetricsPath']}")
    print(f"Predictions saved to: {results['walkForwardPredictionsPath']}")
    print(f"Feature importance saved to: {results['walkForwardFeatureImportancePath']}")
    print(f"Summary saved to: {results['walkForwardSummaryPath']}")
    print(f"Folds: {results['foldCount']}")
    print(f"Out-of-sample rows: {results['outOfSampleRows']}")
    print(f"Accuracy: {results['accuracy']:.4f}")
    print(f"Balanced accuracy: {results['balancedAccuracy']:.4f}")
    print(f"Average fold balanced accuracy: {results['averageFoldBalancedAccuracy']:.4f}")


if __name__ == "__main__":
    main()
