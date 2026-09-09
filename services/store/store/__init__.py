"""Night Companion `store` service: persists bus events to SQLite.

Reads every stream except the capped `frames` and `audio_in` streams (see
HANDOFF.md section 5) via a Redis consumer group and writes each event as
one row in a generic `events` table at `data/night.db`.
"""
