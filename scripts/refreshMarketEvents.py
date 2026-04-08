"""Refresh cached CoinMarketCal events for the current tracked market universe."""

from __future__ import annotations

from scriptSupport import bootstrap_src_path

bootstrap_src_path()

from crypto_signal_ml.application import MarketEventsRefreshApp  # noqa: E402


def main() -> None:
    """Run the standalone CoinMarketCal event refresh workflow."""

    results = MarketEventsRefreshApp().run()

    print("Market event refresh complete.")
    print(f"Status: {results['status']}")
    print(f"Tracked products: {results['trackedProducts']}")
    print(f"Tracked base currencies: {results['trackedBaseCurrencies']}")
    print(f"Events rows: {results['eventsRows']}")
    print(f"Market data path: {results['marketDataPath']}")
    print(f"Events path: {results['eventsPath']}")
    if results["refreshSummary"]:
        print(f"Refresh summary: {results['refreshSummary']}")


if __name__ == "__main__":
    main()
