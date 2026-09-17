"""`python -m volunteer_worker.delete <id>` -- the email-request deletion
path (HANDOFF.md section 6), for a volunteer who asks by email rather than
using the status page's delete button."""

from __future__ import annotations

import argparse
import os
import sys
from datetime import UTC, datetime
from pathlib import Path

from volunteer_common import db, paths

from volunteer_worker.purge import purge_submission


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("submission_id")
    parser.add_argument("--data-dir", default=os.environ.get("VOLUNTEER_DATA_DIR_MOUNT", "/data"))
    args = parser.parse_args(argv)

    data_root = Path(args.data_dir)
    conn = db.connect(paths.db_path(data_root))
    row = db.get_submission(conn, args.submission_id)
    if row is None:
        print(f"no submission {args.submission_id}", file=sys.stderr)
        return 1

    now = datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ")
    db.transition(conn, args.submission_id, from_status=row["status"], to_status="deleted", now=now)
    purge_submission(data_root, args.submission_id)
    print(f"deleted {args.submission_id}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
