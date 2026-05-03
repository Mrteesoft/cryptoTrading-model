"""Run the production cycle repeatedly for hosted continuous learning."""

from __future__ import annotations

import os
import time
from datetime import datetime, timezone

from scriptSupport import bootstrap_src_path


def _load_env_value(env_path: str, key: str) -> str:
    """Read one key from a simple .env file before app config imports run."""

    if not os.path.exists(env_path):
        return ""

    with open(env_path, "r", encoding="utf-8") as env_file:
        for raw_line in env_file:
            line = raw_line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            env_key, env_value = line.split("=", 1)
            if env_key.strip() == key:
                return env_value.strip().strip("\"'")
    return ""


PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CONTINUOUS_SOURCE = os.getenv("CONTINUOUS_MARKET_DATA_SOURCE") or _load_env_value(
    os.path.join(PROJECT_ROOT, ".env"),
    "CONTINUOUS_MARKET_DATA_SOURCE",
)
if CONTINUOUS_SOURCE:
    os.environ["MARKET_DATA_SOURCE"] = CONTINUOUS_SOURCE

bootstrap_src_path()

from crypto_signal_ml.application import BacktestApp, ProductionCycleApp  # noqa: E402


def _env_int(name: str, default_value: int) -> int:
    raw_value = str(os.getenv(name, "")).strip()
    if not raw_value:
        return default_value
    try:
        return int(raw_value)
    except ValueError:
        return default_value


def _env_bool(name: str, default_value: bool) -> bool:
    raw_value = str(os.getenv(name, "")).strip().lower()
    if not raw_value:
        return default_value
    if raw_value in {"1", "true", "yes", "on"}:
        return True
    if raw_value in {"0", "false", "no", "off"}:
        return False
    return default_value


def main() -> None:
    """Refresh data, retrain, publish signals, and repeat."""

    interval_seconds = max(_env_int("CONTINUOUS_LEARNING_INTERVAL_SECONDS", 3600), 60)
    error_sleep_seconds = max(_env_int("CONTINUOUS_LEARNING_ERROR_SLEEP_SECONDS", 300), 60)
    evaluation_interval_cycles = max(_env_int("CONTINUOUS_LEARNING_EVALUATION_INTERVAL_CYCLES", 6), 0)
    run_backtest = _env_bool("CONTINUOUS_LEARNING_RUN_BACKTEST", True)
    max_cycles = _env_int("CONTINUOUS_LEARNING_MAX_CYCLES", 0)
    cycle_number = 0

    print(
        "Continuous learning started. "
        f"source={os.getenv('MARKET_DATA_SOURCE', 'default')} "
        f"interval={interval_seconds}s"
    )

    while True:
        cycle_number += 1
        started_at = datetime.now(timezone.utc).isoformat()
        print(f"==== {started_at} continuous learning cycle {cycle_number} start ====")

        try:
            results = ProductionCycleApp().run()
            market_refresh = results["marketRefresh"]
            training = results["training"]
            signal_generation = results["signalGeneration"]
            print(
                "Cycle complete. "
                f"rows={market_refresh.get('rowsDownloaded')} "
                f"products={market_refresh.get('uniqueProducts')} "
                f"model={training.get('modelType')} "
                f"balancedAccuracy={float(training.get('balancedAccuracy', 0.0)):.4f} "
                f"primarySignal={signal_generation.get('signalName')}"
            )

            should_evaluate = (
                run_backtest
                and evaluation_interval_cycles > 0
                and cycle_number % evaluation_interval_cycles == 0
            )
            if should_evaluate:
                backtest = BacktestApp().run()
                print(
                    "Knowledge test complete. "
                    f"trades={backtest.get('tradeCount')} "
                    f"strategyReturn={float(backtest.get('strategyTotalReturn', 0.0)):.4f} "
                    f"benchmarkReturn={float(backtest.get('benchmarkTotalReturn', 0.0)):.4f} "
                    f"maxDrawdown={float(backtest.get('maxDrawdown', 0.0)):.4f}"
                )
        except Exception as error:
            print(f"Cycle failed: {error}")
            if max_cycles and cycle_number >= max_cycles:
                raise
            time.sleep(error_sleep_seconds)
            continue

        if max_cycles and cycle_number >= max_cycles:
            print("Continuous learning stopped after configured max cycles.")
            return

        time.sleep(interval_seconds)


if __name__ == "__main__":
    main()
