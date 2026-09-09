"""Night Companion `agent` service package.

The session and strategy core: owns phases, goals, and the LLM conversation (HANDOFF.md section
4). The real session state machine is M2 (issues 12 to 17) and not built yet.

For now (issue #4), this package is a dev-only "fake agent": it cycles
through all `Show` face states and a representative strategy per goal in
the goal tree, publishing on the real bus, so the embodiment page and
dashboard can be built and demoed without perception or an LLM. See
`agent/main.py` and HANDOFF.md section 8 (M0 done criteria).
"""
