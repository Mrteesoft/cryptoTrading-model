"""Run the RabbitMQ/Kafka broker worker for heavy async jobs."""

from __future__ import annotations

import logging
import os

from scriptSupport import bootstrap_src_path

bootstrap_src_path()

from crypto_signal_ml.broker import BrokerWorkerService  # noqa: E402


def main() -> None:
    """Start the broker worker process."""

    root_logger = logging.getLogger()
    if not root_logger.handlers:
        logging.basicConfig(
            level=os.getenv("LOG_LEVEL", "INFO").upper(),
            format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        )

    BrokerWorkerService().serve()


if __name__ == "__main__":
    main()
