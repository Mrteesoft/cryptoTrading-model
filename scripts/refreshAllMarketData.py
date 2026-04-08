"""Refresh the full configured market universe across all remaining batches."""

from __future__ import annotations

from scriptSupport import bootstrap_src_path

bootstrap_src_path()

from crypto_signal_ml.application import MarketUniverseRefreshApp  # noqa: E402


def main() -> None:
    """Run the multi-batch market refresh workflow."""

    results = MarketUniverseRefreshApp().run()

    print("Full market refresh complete.")
    print(f"Source: {results['marketDataSource']}")
    print(f"Start batch: {results['startBatch']}")
    print(f"End batch: {results['endBatch']}")
    print(f"Batches run: {results['batchesRun']}")
    print(f"Successful batches: {results['successfulBatches']}")
    print(f"Failed batches: {results['failedBatches']}")
    print(f"Total products available: {results['totalProductsAvailable']}")
    print(f"Final rows saved: {results['finalRowsSaved']}")
    print(f"Final unique products: {results['finalUniqueProducts']}")
    print(f"Saved file: {results['savedPath']}")
    print(f"CoinMarketCap context rows: {results['coinMarketCapContextRows']}")
    print(f"CoinMarketCap context products: {results['coinMarketCapContextUniqueProducts']}")
    print(f"CoinMarketCap context path: {results['coinMarketCapContextPath']}")


if __name__ == "__main__":
    main()
