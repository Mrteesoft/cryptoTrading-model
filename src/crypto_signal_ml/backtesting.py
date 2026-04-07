"""Class-based backtesting helpers for evaluating signal quality as trades."""

from abc import ABC, abstractmethod
from typing import Any, Dict

import pandas as pd

from .config import TrainingConfig
from .trading.policy import evaluate_trading_decision


class BaseSignalBacktester(ABC):
    """
    Base class for turning model predictions into trading-style metrics.

    The base class owns the repeated accounting logic:
    - filter unusable predictions
    - compute net trade returns
    - aggregate equity curves
    - build a readable summary

    A subclass only changes the trade-selection rule.
    """

    def __init__(self, config: TrainingConfig) -> None:
        self.config = config

    def run(self, prediction_df: pd.DataFrame) -> Dict[str, Any]:
        """
        Execute the backtest workflow on prediction rows.

        The input should come from model evaluation on labeled data, because
        the backtest needs the realized `future_return` column to score trades.
        """

        trade_candidates = self._select_trade_candidates(prediction_df)
        selected_trades = self._select_trades(trade_candidates)
        trade_df = self._build_trade_frame(selected_trades)
        period_df = self._build_period_frame(
            selected_trade_df=trade_df,
            reference_df=prediction_df,
        )
        summary = self._build_summary(
            trade_df=trade_df,
            period_df=period_df,
        )

        return {
            "trade_df": trade_df,
            "period_df": period_df,
            "summary": summary,
        }

    def _select_trade_candidates(self, prediction_df: pd.DataFrame) -> pd.DataFrame:
        """
        Keep only actionable predictions.

        Spot mode only opens new positions on `BUY` signals.
        `HOLD` means wait and `TAKE_PROFIT` means reduce risk or exit an
        existing spot position, so neither of them opens a new trade here.
        A minimum confidence filter can also be applied from the config.
        """

        candidates = prediction_df.copy()
        candidates = candidates.dropna(subset=["future_return"])
        if candidates.empty:
            return candidates

        policy_rows = [
            evaluate_trading_decision(
                signal_row=signal_row,
                minimum_action_confidence=self.config.backtest_min_confidence,
                config=self.config,
            )
            for _, signal_row in candidates.iterrows()
        ]
        policy_df = pd.DataFrame(
            [
                {
                    "policy_signal_name": policy_row["signalName"],
                    "policy_spot_action": policy_row["spotAction"],
                    "policy_trade_readiness": policy_row["tradeReadiness"],
                    "policy_score": policy_row["policyScore"],
                    "policy_probability_margin": policy_row["probabilityMargin"],
                    "policy_required_action_confidence": policy_row["requiredActionConfidence"],
                    "policy_confidence_gate_applied": policy_row["confidenceGateApplied"],
                    "policy_risk_gate_applied": policy_row["riskGateApplied"],
                }
                for policy_row in policy_rows
            ],
            index=candidates.index,
        )
        candidates = pd.concat([candidates, policy_df], axis=1)
        candidates = candidates[candidates["policy_spot_action"] == "buy"].copy()

        if "policy_score" in candidates.columns:
            sort_columns = ["timestamp", "policy_score", "confidence"]
            ascending = [True, False, False]
        else:
            sort_columns = ["timestamp", "confidence"]
            ascending = [True, False]
        if "policy_probability_margin" in candidates.columns:
            sort_columns.append("policy_probability_margin")
            ascending.append(False)
        if "product_id" in candidates.columns:
            sort_columns.append("product_id")
            ascending.append(True)

        return candidates.sort_values(sort_columns, ascending=ascending).reset_index(drop=True)

    @abstractmethod
    def _select_trades(self, trade_candidates: pd.DataFrame) -> pd.DataFrame:
        """Apply the subclass trade-selection strategy."""

    def _build_trade_frame(self, selected_trades: pd.DataFrame) -> pd.DataFrame:
        """
        Convert selected signals into per-trade returns.

        This starter backtest assumes:
        - equal notional size per selected trade
        - the trade is held for the same horizon used by labeling
        - BUY means a spot entry
        - TAKE_PROFIT does not open a fresh trade in this simplified backtest
        - we pay a round-trip cost when entering and exiting the trade
        """

        trade_df = selected_trades.copy()
        trade_df["direction"] = 1
        trade_df["gross_trade_return"] = trade_df["direction"] * trade_df["future_return"]
        trade_df["round_trip_cost"] = 2 * (
            self.config.backtest_trading_fee_rate + self.config.backtest_slippage_rate
        )
        trade_df["net_trade_return"] = trade_df["gross_trade_return"] - trade_df["round_trip_cost"]
        trade_df["is_winner"] = trade_df["net_trade_return"] > 0

        return trade_df.reset_index(drop=True)

    def _build_period_frame(
        self,
        selected_trade_df: pd.DataFrame,
        reference_df: pd.DataFrame,
    ) -> pd.DataFrame:
        """
        Aggregate returns into timestamp-level strategy and benchmark curves.

        Strategy return:
        - equal-weight mean of selected trade returns at a timestamp

        Benchmark return:
        - equal-weight mean future return of the coin universe at that time
        """

        benchmark_df = (
            reference_df.groupby("timestamp", as_index=False)
            .agg(benchmark_return=("future_return", "mean"))
            .sort_values("timestamp")
            .reset_index(drop=True)
        )

        if selected_trade_df.empty:
            period_df = benchmark_df.copy()
            period_df["strategy_return"] = 0.0
            period_df["trade_count"] = 0
        else:
            strategy_df = (
                selected_trade_df.groupby("timestamp", as_index=False)
                .agg(
                    strategy_return=("net_trade_return", "mean"),
                    trade_count=("net_trade_return", "size"),
                )
                .sort_values("timestamp")
                .reset_index(drop=True)
            )

            period_df = benchmark_df.merge(strategy_df, on="timestamp", how="left")
            period_df["strategy_return"] = period_df["strategy_return"].fillna(0.0)
            period_df["trade_count"] = period_df["trade_count"].fillna(0).astype(int)

        period_df["strategy_equity"] = self.config.backtest_initial_capital * (
            1 + period_df["strategy_return"]
        ).cumprod()
        period_df["benchmark_equity"] = self.config.backtest_initial_capital * (
            1 + period_df["benchmark_return"].fillna(0.0)
        ).cumprod()
        period_df["strategy_drawdown"] = self._calculate_drawdown(period_df["strategy_equity"])
        period_df["benchmark_drawdown"] = self._calculate_drawdown(period_df["benchmark_equity"])

        return period_df

    def _build_summary(
        self,
        trade_df: pd.DataFrame,
        period_df: pd.DataFrame,
    ) -> Dict[str, Any]:
        """Build a compact JSON-friendly summary of backtest performance."""

        starting_capital = float(self.config.backtest_initial_capital)
        if period_df.empty:
            ending_capital = starting_capital
            ending_benchmark_capital = starting_capital
        else:
            ending_capital = float(period_df.iloc[-1]["strategy_equity"])
            ending_benchmark_capital = float(period_df.iloc[-1]["benchmark_equity"])

        return {
            "initialCapital": starting_capital,
            "endingCapital": ending_capital,
            "endingBenchmarkCapital": ending_benchmark_capital,
            "strategyTotalReturn": (ending_capital / starting_capital) - 1,
            "benchmarkTotalReturn": (ending_benchmark_capital / starting_capital) - 1,
            "tradeCount": int(len(trade_df)),
            "activePeriods": int((period_df["trade_count"] > 0).sum()),
            "winRate": float(trade_df["is_winner"].mean()) if not trade_df.empty else 0.0,
            "averageTradeReturn": float(trade_df["net_trade_return"].mean()) if not trade_df.empty else 0.0,
            "maxDrawdown": float(period_df["strategy_drawdown"].min()) if not period_df.empty else 0.0,
            "maxBenchmarkDrawdown": float(period_df["benchmark_drawdown"].min()) if not period_df.empty else 0.0,
            "minConfidence": float(self.config.backtest_min_confidence),
            "maxPositionsPerTimestamp": int(self.config.backtest_max_positions_per_timestamp),
            "tradingFeeRate": float(self.config.backtest_trading_fee_rate),
            "slippageRate": float(self.config.backtest_slippage_rate),
        }

    def _calculate_drawdown(self, equity_series: pd.Series) -> pd.Series:
        """Calculate running drawdown from an equity curve."""

        running_peak = equity_series.cummax()
        return (equity_series / running_peak) - 1


class EqualWeightSignalBacktester(BaseSignalBacktester):
    """
    Keep the highest-confidence actionable signals at each timestamp.

    This is a simple starter strategy for multi-coin modeling:
    - do not open every possible position
    - cap the number of positions
    - spread weight equally across the selected trades
    """

    def _select_trades(self, trade_candidates: pd.DataFrame) -> pd.DataFrame:
        """Select the top-confidence trades at each timestamp."""

        if trade_candidates.empty:
            return trade_candidates.copy()

        return (
            trade_candidates.groupby("timestamp", group_keys=False)
            .head(self.config.backtest_max_positions_per_timestamp)
            .reset_index(drop=True)
        )
