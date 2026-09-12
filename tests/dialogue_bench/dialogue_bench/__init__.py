"""Offline dialogue regression bench for the Night Companion agent."""

from __future__ import annotations

import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[3]
_AGENT_SRC = _REPO_ROOT / "services" / "agent"

if str(_AGENT_SRC) not in sys.path and _AGENT_SRC.is_dir():
    sys.path.insert(0, str(_AGENT_SRC))

__version__ = "0.1.0"
