"""Entry point for the `worker` service.

V1 scaffolds startup logging only. The poll loop that claims `finalized`
submissions and runs the pipeline (HANDOFF.md section 5.8) is V7 -- it needs
`media.py` (decrypt/remux) and the result encryption path, neither of which
exist yet, so wiring a fake loop here now would be the half-finished
implementation the project avoids.
"""

from __future__ import annotations

import logging
import os
import time

from volunteer_common.log import log

SERVICE_NAME = "volunteer-worker"

logging.basicConfig(level=logging.INFO)


def main() -> None:
    data_dir = os.environ.get("VOLUNTEER_DATA_DIR_MOUNT", "/data")
    log(SERVICE_NAME, "starting", data_dir=data_dir)
    while True:
        time.sleep(3600)


if __name__ == "__main__":
    main()
