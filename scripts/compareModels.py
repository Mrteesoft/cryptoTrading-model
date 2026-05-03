"""Compare multiple signal model subclasses on the same dataset split."""

from __future__ import annotations

from scriptSupport import bootstrap_src_path

bootstrap_src_path()

from crypto_signal_ml.application import ModelComparisonApp  # noqa: E402


def main() -> None:
    """Run the model-comparison workflow through the reusable app class."""

    results = ModelComparisonApp().run()

    print("Model comparison complete.")
    print(f"Prepared dataset saved to: {results['datasetPath']}")
    print(f"Comparison CSV saved to: {results['comparisonCsvPath']}")
    print(f"Comparison JSON saved to: {results['comparisonJsonPath']}")
    print(f"Compared models: {results['comparedModels']}")
    print(f"Best model type: {results['bestModelType']}")


if __name__ == "__main__":
    main()
