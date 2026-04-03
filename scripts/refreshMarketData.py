"""Download fresh real-market candles and save them into the raw-data CSV."""

from __future__ import annotations

from scriptSupport import bootstrap_src_path

bootstrap_src_path()

from crypto_signal_ml.app import MarketDataRefreshApp  # noqa: E402


def main() -> None:
    """Run the real-market-data refresh workflow."""

    results = MarketDataRefreshApp().run()

    print("Market data refresh complete.")
    print(f"Source: {results['marketDataSource']}")
    print(f"Product mode: {results['productMode']}")
    print(f"Granularity seconds: {results['granularitySeconds']}")
    print(f"Saved file: {results['savedPath']}")
    print(f"Rows downloaded: {results['rowsDownloaded']}")
    print(f"Unique products: {results['uniqueProducts']}")
    print(f"First timestamp: {results['firstTimestamp']}")
    print(f"Last timestamp: {results['lastTimestamp']}")
    print(f"CoinMarketCap context status: {results['coinMarketCapContextStatus']}")
    print(f"CoinMarketCap context rows: {results['coinMarketCapContextRows']}")
    print(f"CoinMarketCap context path: {results['coinMarketCapContextPath']}")
    print(f"CoinMarketCal events status: {results['coinMarketCalEventsStatus']}")
    print(f"CoinMarketCal events rows: {results['coinMarketCalEventsRows']}")
    print(f"CoinMarketCal events path: {results['coinMarketCalEventsPath']}")
    if results["downloadSummary"]:
        print(f"Download summary: {results['downloadSummary']}")


if __name__ == "__main__":
    main()
