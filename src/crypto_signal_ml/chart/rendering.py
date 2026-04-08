"""Chart rendering utilities for debug snapshots."""

from __future__ import annotations

from pathlib import Path


def _require_mplfinance():
    try:
        import mplfinance as mpf  # type: ignore
    except ImportError as exc:
        raise ImportError(
            "mplfinance is required for chart rendering. "
            "Install it with `pip install mplfinance`."
        ) from exc
    return mpf


def render_chart_snapshot(ohlcv_df, output_path: Path, title: str | None = None) -> None:
    """Render a candlestick snapshot to disk."""

    if ohlcv_df is None or ohlcv_df.empty:
        return
    mpf = _require_mplfinance()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df = ohlcv_df.copy()
    if "timestamp" in df.columns:
        df = df.set_index("timestamp")
    if df.index.name != "timestamp":
        df.index.name = "timestamp"
    mpf.plot(
        df,
        type="candle",
        volume="volume" in df.columns,
        style="yahoo",
        title=title,
        savefig=str(output_path),
    )
