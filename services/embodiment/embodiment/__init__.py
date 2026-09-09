"""Night Companion `embodiment` service package.

Fullscreen embodiment web page: animated face, big text, TTS playback
(HANDOFF.md section 4). Served over HTTPS on the LAN with a mkcert
certificate (HANDOFF.md section 9), driven by `Show`/`Say` events on the
bus. See `embodiment/app.py` for the FastAPI app and `embodiment/main.py`
for the entry point that runs it.
"""
