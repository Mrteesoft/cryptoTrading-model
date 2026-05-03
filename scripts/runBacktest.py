"""Train the configured model and backtest the resulting signals."""

from __future__ import annotations

from scriptSupport import bootstrap_src_path

bootstrap_src_path()

from crypto_signal_ml.application import BacktestApp  # noqa: E402


def main() -> None:
    """Run the reusable backtest workflow."""

    results = BacktestApp().run()

    print("Backtest complete.")
    print(f"Model type: {results['modelType']}")
    print(f"Prepared dataset saved to: {results['datasetPath']}")
    print(f"Model saved to: {results['modelPath']}")
    print(f"Training metrics saved to: {results['metricsPath']}")
    print(f"Backtest trades saved to: {results['backtestTradesPath']}")
    print(f"Backtest periods saved to: {results['backtestPeriodsPath']}")
    print(f"Backtest summary saved to: {results['backtestSummaryPath']}")
    print(f"Trade count: {results['tradeCount']}")
    print(f"Strategy total return: {results['strategyTotalReturn']:.4f}")
    print(f"Benchmark total return: {results['benchmarkTotalReturn']:.4f}")
    print(f"Max drawdown: {results['maxDrawdown']:.4f}")


if __name__ == "__main__":
    main()
