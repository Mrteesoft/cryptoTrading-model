"""Train the crypto signal model and save artifacts to disk."""

from __future__ import annotations

from scriptSupport import bootstrap_src_path

bootstrap_src_path()

from crypto_signal_ml.application import TrainingApp  # noqa: E402


def main() -> None:
    """Run the training workflow through the reusable app class."""

    results = TrainingApp().run()

    print("Training complete.")
    print(f"Model type: {results['modelType']}")
    print(f"Prepared dataset saved to: {results['datasetPath']}")
    print(f"Model saved to: {results['modelPath']}")
    print(f"Metrics saved to: {results['metricsPath']}")
    print(f"Test predictions saved to: {results['predictionsPath']}")
    print(f"Feature importance saved to: {results['featureImportancePath']}")
    print(f"Train rows: {results['trainRows']}")
    print(f"Test rows: {results['testRows']}")
    print(f"Accuracy: {results['accuracy']:.4f}")
    print(f"Balanced accuracy: {results['balancedAccuracy']:.4f}")


if __name__ == "__main__":
    main()
