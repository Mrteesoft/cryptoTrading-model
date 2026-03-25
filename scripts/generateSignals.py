"""Load a trained model and generate signal files from the latest data."""

from __future__ import annotations

from scriptSupport import bootstrap_src_path

bootstrap_src_path()

from crypto_signal_ml.app import SignalGenerationApp  # noqa: E402


def main() -> None:
    """Generate signals through the reusable app class."""

    results = SignalGenerationApp().run()

    print("Signal generation complete.")
    print(f"Model type: {results['modelType']}")
    print(f"Historical signals saved to: {results['historicalSignalsPath']}")
    print(f"Latest signal saved to: {results['latestSignalPath']}")
    print(f"Latest signals saved to: {results['latestSignalsPath']}")
    print(f"Actionable signals saved to: {results['actionableSignalsPath']}")
    print(f"Frontend snapshot saved to: {results['frontendSignalSnapshotPath']}")
    print(f"Signals generated: {results['signalsGenerated']}")
    print(f"Actionable signals generated: {results['actionableSignalsGenerated']}")
    print(
        "Latest model output: "
        f"{results['signalName']} "
        f"(confidence={results['confidence']:.4f})"
    )
    print(f"Signal explanation: {results['signalChat']}")


if __name__ == "__main__":
    main()
