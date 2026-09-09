"""Night Companion `capture` service package (issue #7).

Receives camera frames from a pluggable source (`capture.sources`), motion-
gates and rate-limits them (`capture.gate`), and publishes the admitted
frames as `Frame` events on the bus (HANDOFF.md section 4/5). Part of M1.
See `capture/main.py` for the entry point.
"""
