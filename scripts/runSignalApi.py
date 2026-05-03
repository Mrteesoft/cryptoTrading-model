"""Run the cached signal API for frontend use."""

from __future__ import annotations

from scriptSupport import bootstrap_src_path

bootstrap_src_path()

import uvicorn  # noqa: E402

from crypto_signal_ml.service import signal_api_app  # noqa: E402


def main() -> None:
    """Start the local FastAPI server for the frontend."""

    uvicorn.run(
        signal_api_app,
        host="0.0.0.0",
        port=8000,
    )


if __name__ == "__main__":
    main()
