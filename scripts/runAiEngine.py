"""Run the Python model and AI engine behind the TypeScript backend."""

from __future__ import annotations

import os

import uvicorn

from scriptSupport import bootstrap_src_path

bootstrap_src_path()

from crypto_signal_ml.engine_api import create_app  # noqa: E402


def main() -> None:
    """Start the internal Python AI engine."""

    host = os.getenv("AI_ENGINE_HOST", "127.0.0.1")
    port = int(os.getenv("AI_ENGINE_PORT", "8100"))
    app = create_app()

    uvicorn.run(
        app,
        host=host,
        port=port,
    )


if __name__ == "__main__":
    main()
