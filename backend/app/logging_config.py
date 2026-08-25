"""Application-wide logging setup.

Always logs to stdout (so `docker logs` / local console works out of the
box). When CLOUDWATCH_ENABLED=true and AWS credentials are available, an
additional CloudWatch Logs handler is attached so operational logs and
errors are centrally monitorable, satisfying the "Monitoring" requirement
without making CloudWatch a hard dependency for local development.
"""
from __future__ import annotations

import logging
import sys

from app.config import Settings

_CONFIGURED = False


def configure_logging(settings: Settings) -> logging.Logger:
    global _CONFIGURED

    logger = logging.getLogger("app")

    if _CONFIGURED:
        return logger

    logger.setLevel(settings.LOG_LEVEL.upper())

    formatter = logging.Formatter(
        fmt="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
        datefmt="%Y-%m-%dT%H:%M:%S%z",
    )

    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    if settings.CLOUDWATCH_ENABLED:
        try:
            import watchtower  # imported lazily so it's an optional dependency at runtime

            cloudwatch_handler = watchtower.CloudWatchLogHandler(
                log_group_name=settings.CLOUDWATCH_LOG_GROUP,
                log_stream_name=settings.CLOUDWATCH_LOG_STREAM,
                create_log_group=True,
            )
            cloudwatch_handler.setFormatter(formatter)
            logger.addHandler(cloudwatch_handler)
            logger.info("CloudWatch logging enabled (log group=%s)", settings.CLOUDWATCH_LOG_GROUP)
        except Exception as exc:  # noqa: BLE001 - monitoring must never crash the app
            logger.warning("Could not enable CloudWatch logging, continuing without it: %s", exc)

    logger.propagate = False
    _CONFIGURED = True
    return logger
