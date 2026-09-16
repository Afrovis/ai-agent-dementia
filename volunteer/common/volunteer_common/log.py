"""One JSON line per event on stdout, matching `dashboard/main.py::_log`."""

from __future__ import annotations

import json
import logging


def log(service: str, message: str, **fields: object) -> None:
    logger = logging.getLogger(service)
    logger.info(json.dumps({"service": service, "message": message, **fields}))
