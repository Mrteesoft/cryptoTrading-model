"""Load a trained model and generate signal files from the latest data."""

from __future__ import annotations

import logging
import os

from scriptSupport import bootstrap_src_path

bootstrap_src_path()

from crypto_signal_ml.application import SignalGenerationApp  # noqa: E402


def main() -> None:
    """Generate signals through the reusable app class."""

    root_logger = logging.getLogger()
    if not root_logger.handlers:
        logging.basicConfig(
            level=os.getenv("LOG_LEVEL", "INFO").upper(),
            format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        )

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
    print(f"Signal source: {results['signalSource']}")
    if results.get("signalStore"):
        print(
            "Current signal store: "
            f"{results['signalStore']['storageBackend']} -> "
            f"{results['signalStore']['databaseTarget']}"
        )
        print(
            "Current signals persisted: "
            f"{results['signalStore']['signalCount']} "
            f"(primary={results['signalStore'].get('primaryProductId') or 'none'})"
        )
    if results.get("marketDataRefresh"):
        print(
            "Market data refreshed: "
            f"{results['marketDataRefresh']['rowsDownloaded']} rows across "
            f"{results['marketDataRefresh']['uniqueProducts']} products"
        )
        print(
            "Market data window: "
            f"{results['marketDataRefresh']['firstTimestamp']} -> "
            f"{results['marketDataRefresh']['lastTimestamp']}"
        )
    if results.get("signalName") is None:
        print("Latest model output: no public signal is currently published.")
        print(
            "Signal explanation: candidates remain on the internal watchlist until a BUY appears "
            "or an open trade needs HOLD, TAKE_PROFIT, or LOSS management."
        )
    else:
        print(
            "Latest model output: "
            f"{results['signalName']} "
            f"(confidence={results['confidence']:.4f})"
        )
        print(f"Signal explanation: {results['signalChat']}")


if __name__ == "__main__":
    main()
