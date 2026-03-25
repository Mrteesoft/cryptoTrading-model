"""FastAPI app for serving cached crypto signal snapshots to a frontend."""

from __future__ import annotations

from pathlib import Path
from typing import Optional

from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware

from .config import OUTPUTS_DIR
from .frontend import SignalSnapshotStore


def create_signal_api(snapshot_path: Path = OUTPUTS_DIR / "frontendSignalSnapshot.json") -> FastAPI:
    """
    Build a lightweight API that serves cached signal JSON.

    Important performance idea:
    - this API does not run the ML model per request
    - it reads one already-generated snapshot into memory
    - requests become fast JSON reads instead of expensive inference jobs
    """

    snapshot_store = SignalSnapshotStore(snapshot_path=snapshot_path)
    app = FastAPI(
        title="Crypto Signal ML API",
        version="0.1.0",
        description="Frontend-facing API for cached crypto signal snapshots.",
    )

    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=False,
        allow_methods=["GET"],
        allow_headers=["*"],
    )

    @app.get("/api/health")
    def health() -> dict:
        """Return a tiny health payload so the frontend can detect readiness."""

        try:
            overview = snapshot_store.get_overview()
        except FileNotFoundError:
            return {
                "status": "waiting_for_snapshot",
                "snapshotPath": str(snapshot_path),
            }

        return {
            "status": "ok",
            "generatedAt": overview["generatedAt"],
            "modelType": overview["modelType"],
            "snapshotPath": str(snapshot_path),
        }

    @app.get("/api/overview")
    def overview() -> dict:
        """Return the dashboard summary and primary signal."""

        return snapshot_store.get_overview()

    @app.get("/api/signals")
    def list_signals(
        action: str = Query(default="all"),
        limit: Optional[int] = Query(default=50, ge=1, le=500),
    ) -> dict:
        """Return cached signal rows for the requested action filter."""

        return {
            "action": action,
            "count": len(snapshot_store.list_signals(action=action, limit=limit)),
            "signals": snapshot_store.list_signals(action=action, limit=limit),
        }

    @app.get("/api/signals/{product_id}")
    def signal_detail(product_id: str) -> dict:
        """Return one product's latest signal and explanation."""

        signal_summary = snapshot_store.get_signal_by_product(product_id)
        if signal_summary is None:
            raise HTTPException(status_code=404, detail=f"No signal found for {product_id}.")

        return signal_summary

    return app


app = create_signal_api()
