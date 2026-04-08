"""Single-command model-service entrypoint."""

from __future__ import annotations

import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from crypto_signal_ml.service import SignalMonitorService  # noqa: E402


def main() -> None:
    """Start the model-side signal monitor and live engine in one process."""

    SignalMonitorService().serve()


if __name__ == "__main__":
    main()
