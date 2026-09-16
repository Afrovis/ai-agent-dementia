"""Entry point for the `web` service."""

from __future__ import annotations

import logging

import uvicorn
from volunteer_common.log import log

from volunteer_web.app import create_app
from volunteer_web.config import WebConfig

SERVICE_NAME = "volunteer-web"

logging.basicConfig(level=logging.INFO)


def main() -> None:
    config = WebConfig.from_env()
    log(SERVICE_NAME, "starting", port=config.port, public_hostname=config.public_hostname)
    app = create_app(config)
    uvicorn.run(app, host="0.0.0.0", port=config.port, log_level="info")


if __name__ == "__main__":
    main()
