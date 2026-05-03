"""Unified model-service runner for signal refresh and live monitoring."""

from __future__ import annotations

import logging
import os
from threading import Event, Thread
from time import perf_counter
from typing import Any, Callable

import uvicorn

from .app import SignalGenerationApp
from .config import TrainingConfig
from .engine_api import create_app
from .logging_utils import CompactConsoleFormatter, format_bool_for_log, format_path_for_log


LOGGER = logging.getLogger(__name__)


class SignalMonitorService:
    """Run signal generation on a schedule while serving the live engine API."""

    def __init__(
        self,
        config: TrainingConfig | None = None,
        signal_generation_app_factory: Callable[..., SignalGenerationApp] = SignalGenerationApp,
        app_factory: Callable[..., Any] = create_app,
    ) -> None:
        self.config = config or TrainingConfig()
        self.signal_generation_app_factory = signal_generation_app_factory
        self.app_factory = app_factory
        self._stop_event = Event()
        self._refresh_thread: Thread | None = None

    def _configure_logging(self) -> None:
        """Initialize basic logging when the process has no handlers yet."""

        root_logger = logging.getLogger()
        if root_logger.handlers:
            return

        handler = logging.StreamHandler()
        handler.setFormatter(CompactConsoleFormatter())
        root_logger.setLevel(os.getenv("LOG_LEVEL", "INFO").upper())
        root_logger.addHandler(handler)

    def _run_signal_generation(self, cycle_label: str) -> None:
        """Generate and publish the latest watchlist and signal snapshots."""

        cycle_started_at = perf_counter()
        LOGGER.info("Refresh start | cycle=%s", cycle_label)
        results = self.signal_generation_app_factory(config=self.config).run()
        LOGGER.info(
            "Refresh done | cycle=%s | %.2fs | signals=%s | actionable=%s | top=%s | source=%s",
            cycle_label,
            perf_counter() - cycle_started_at,
            int(results.get("signalsGenerated", 0)),
            int(results.get("actionableSignalsGenerated", 0)),
            results.get("signalName") or "none",
            results.get("signalSource") or "unknown",
        )

    def _refresh_loop(self, interval_seconds: int) -> None:
        """Keep refreshing the published signal snapshot on a fixed interval."""

        while not self._stop_event.wait(interval_seconds):
            try:
                self._run_signal_generation(cycle_label="scheduled")
            except Exception:
                LOGGER.exception("Scheduled signal refresh failed.")

    def _start_background_refresh(self) -> None:
        """Launch the periodic signal-refresh worker when enabled."""

        interval_seconds = int(getattr(self.config, "signal_monitor_refresh_interval_seconds", 0) or 0)
        if interval_seconds <= 0:
            LOGGER.info("Refresh loop | disabled")
            return

        self._refresh_thread = Thread(
            target=self._refresh_loop,
            args=(interval_seconds,),
            name="signal-monitor-refresh",
            daemon=True,
        )
        self._refresh_thread.start()
        LOGGER.info(
            "Refresh loop | running every %ss",
            interval_seconds,
        )

    def serve(self) -> None:
        """Run the full model-side service lifecycle in one process."""

        self._configure_logging()
        LOGGER.info(
            "Service start | initial=%s | refresh=%ss | data=%s",
            format_bool_for_log(bool(getattr(self.config, "signal_monitor_run_initial_generation", True))),
            int(getattr(self.config, "signal_monitor_refresh_interval_seconds", 0) or 0),
            format_path_for_log(self.config.data_file),
        )

        if bool(getattr(self.config, "signal_monitor_run_initial_generation", True)):
            try:
                self._run_signal_generation(cycle_label="initial")
            except Exception:
                LOGGER.exception("Initial signal refresh failed; continuing with the live API.")

        self._start_background_refresh()

        host = os.getenv("AI_ENGINE_HOST", "127.0.0.1")
        port = int(os.getenv("AI_ENGINE_PORT", "8100"))
        internal_api_key = str(os.getenv("AI_ENGINE_INTERNAL_API_KEY", "")).strip()
        app = self.app_factory(
            config=self.config,
            require_internal_api_key=bool(internal_api_key),
            internal_api_key=internal_api_key or None,
        )
        LOGGER.info(
            "API start | http://%s:%s | internal_key=%s",
            host,
            port,
            format_bool_for_log(bool(internal_api_key)),
        )

        try:
            uvicorn.run(
                app,
                host=host,
                port=port,
            )
        finally:
            self._stop_event.set()
            LOGGER.info("Service stop")
