"""Refresh data, retrain the model, and publish a fresh frontend snapshot."""

from __future__ import annotations

from scriptSupport import bootstrap_src_path

bootstrap_src_path()

from crypto_signal_ml.application import ProductionCycleApp  # noqa: E402


def main() -> None:
    """Execute the production update cycle through the reusable app class."""

    results = ProductionCycleApp().run()
    market_refresh = results["marketRefresh"]
    training = results["training"]
    signal_generation = results["signalGeneration"]

    print("Production cycle complete.")
    print(f"Market data refreshed: {market_refresh['rowsDownloaded']} rows")
    print(f"Products refreshed: {market_refresh['uniqueProducts']}")
    print(f"CoinMarketCal events: {market_refresh['coinMarketCalEventsStatus']} ({market_refresh['coinMarketCalEventsRows']} rows)")
    print(f"Model type: {training['modelType']}")
    print(f"Model published to: {training['modelPath']}")
    print(f"Artifact metadata saved to: {training['metadataPath']}")
    print(f"Balanced accuracy: {training['balancedAccuracy']:.4f}")
    print(f"Frontend snapshot saved to: {signal_generation['frontendSignalSnapshotPath']}")
    print(f"Primary signal: {signal_generation['signalName']}")
    print(f"Primary confidence: {signal_generation['confidence']:.4f}")


if __name__ == "__main__":
    main()
