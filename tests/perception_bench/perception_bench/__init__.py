"""The perception bench (issue #11): measures `perceive`'s pose classification
pipeline against three tiers of fixtures, from weakest to strongest evidence.

See `tests/perception_bench/README.md` for the full picture; the short
version:

- **Tier 1** (`perception_bench.synthetic`): scripted, in-process fixtures.
  Always available, no downloads, no camera, no model weights. The only
  tier that can measure latency, because it is the only one where the exact
  frame a transition happens on is known by construction.
- **Tier 2** (`perception_bench.daylight`): real daylight RGB footage from
  the IndoorActionDataset, opt-in via `fetch_daylight.sh`. Measures
  per-state accuracy for four of six states, never latency, never
  `in_bed` (the dataset has no bed). Lit RGB, not infrared.
- **Tier 3** (`perception_bench.infrared`): real infrared clips from the
  actual room, following the manifest format documented in
  `perception_bench.infrared`. Empty until someone records
  `RECORDING.md`'s protocol.

This package intentionally imports only `perceive.backends`,
`perceive.classify`, and `perceive.zones` -- the pure pipeline pieces --
never `perceive.main`, which pulls in `redis` and `nc_shared` for the bus
glue this bench has no use for. That keeps tier 1 (and this package's own
unit tests) runnable with nothing installed beyond Pillow, numpy, and
PyYAML (see `pyproject.toml`), matching HANDOFF.md section 4 and this
issue's hard constraint that tier 1 needs no downloads, camera, model
weights, Redis, or network.

`perceive` is not an installed dependency of this package; it is the
sibling service this bench measures, so instead of vendoring or requiring
a separate `pip install -e services/perceive` step, this module adds
`services/perceive` to `sys.path` at import time, the same way a test
`conftest.py` would. See `README.md` for how to run the bench.
"""

from __future__ import annotations

import sys
from pathlib import Path

_THIS_FILE = Path(__file__).resolve()
_REPO_ROOT = _THIS_FILE.parents[3]
_PERCEIVE_SRC = _REPO_ROOT / "services" / "perceive"

if str(_PERCEIVE_SRC) not in sys.path and _PERCEIVE_SRC.is_dir():
    sys.path.insert(0, str(_PERCEIVE_SRC))

__version__ = "0.1.0"
