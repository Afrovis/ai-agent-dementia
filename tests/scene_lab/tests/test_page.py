import asyncio
import base64
import io
import json
import threading
import wave
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import numpy as np
import pytest
import websockets

from scene_lab.page import VirtualPage
from scene_lab.voice import AudioStream, RoomNoise


def wav_bytes(seconds=0.12):
    output = io.BytesIO()
    with wave.open(output, "wb") as wav:
        wav.setnchannels(1)
        wav.setsampwidth(2)
        wav.setframerate(16000)
        wav.writeframes(b"\0\0" * round(seconds * 16000))
    return output.getvalue()


class AudioHandler(BaseHTTPRequestHandler):
    def do_GET(self):
        data = wav_bytes()
        self.send_response(200)
        self.end_headers()
        self.wfile.write(data)

    def log_message(self, *args):
        pass


async def exercise(interruptible=False, barge=False):
    http = ThreadingHTTPServer(("127.0.0.1", 0), AudioHandler)
    thread = threading.Thread(target=http.serve_forever, daemon=True)
    thread.start()
    messages = []
    audio_messages = []
    connected = asyncio.Event()
    finished = asyncio.Event()
    page = VirtualPage(f"http://127.0.0.1:{http.server_port}", AudioStream(RoomNoise(seed=2)))

    async def server(socket):
        if socket.request.path == "/media":
            async for raw in socket:
                audio_messages.append(json.loads(raw))
            return
        messages.append(json.loads(await socket.recv()))
        connected.set()
        await socket.send(
            json.dumps(
                {
                    "type": "say",
                    "text": "Please rest now",
                    "strategy": "soft_greeting",
                    "session_id": "s1",
                    "interruptible": interruptible,
                    "audio_url": f"http://127.0.0.1:{http.server_port}/speech/abc.wav",
                }
            )
        )
        async for raw in socket:
            item = json.loads(raw)
            messages.append(item)
            if item.get("phase") == "playing" and barge:
                await socket.send(json.dumps({"type": "speech_started", "session_id": "s1"}))
            if item.get("phase") in {"ended", "interrupted"}:
                finished.set()
                break

    try:
        async with websockets.serve(server, "127.0.0.1", 0) as ws:
            port = ws.sockets[0].getsockname()[1]
            page.base_url = f"ws://127.0.0.1:{port}"
            task = asyncio.create_task(page.run())
            await asyncio.wait_for(connected.wait(), 2)
            await asyncio.wait_for(finished.wait(), 3)
            task.cancel()
            try:
                await task
            except (asyncio.CancelledError, ExceptionGroup):
                pass
    finally:
        http.shutdown()
        http.server_close()
    return messages, audio_messages, page


def test_playback_and_audio_protocol():
    try:
        messages, audio, page = asyncio.run(exercise())
    except PermissionError:
        pytest.skip("sandbox blocks loopback sockets")
    assert messages[0]["type"] == "hello"
    assert [item["phase"] for item in messages if item["type"] == "playback"] == [
        "received",
        "requested",
        "playing",
        "ended",
    ]
    assert audio[0]["type"] == "hello"
    chunk = next(item for item in audio if item["type"] == "audio")
    assert chunk["sample_rate"] == 16000
    assert np.frombuffer(base64.b64decode(chunk["pcm16_b64"]), dtype="<i2").size == 1365
    assert page.heard[0].heard_fraction == 1


def test_barge_in_only_when_interruptible():
    try:
        yes, _, page = asyncio.run(exercise(interruptible=True, barge=True))
    except PermissionError:
        pytest.skip("sandbox blocks loopback sockets")
    assert yes[-1]["phase"] == "interrupted"
    assert yes[-1]["detail"] == "barge-in"
    assert page.heard[0].interrupted
    no, _, _ = asyncio.run(exercise(interruptible=False, barge=True))
    assert no[-1]["phase"] == "ended"


class FakeSocket:
    def __init__(self):
        self.sent = []

    async def send(self, raw):
        self.sent.append(json.loads(raw))


async def protocol_without_sockets(interruptible, barge):
    page = VirtualPage("http://localhost:1", AudioStream(RoomNoise(seed=2)))
    socket = FakeSocket()
    page._playback_socket = socket
    page._fetch_wav = lambda url: wav_bytes(0.08)
    await page.handle_message(
        {
            "type": "say",
            "text": "Please rest now",
            "strategy": "soft_greeting",
            "session_id": "s1",
            "interruptible": interruptible,
            "audio_url": "/speech/abc.wav",
        }
    )
    await asyncio.sleep(0.01)
    if barge:
        await page.handle_message({"type": "speech_started", "session_id": "s1"})
    await asyncio.sleep(0.1)
    return socket.sent, page


def test_protocol_logic_without_sockets():
    reports, page = asyncio.run(protocol_without_sockets(True, False))
    playback = [item for item in reports if item["type"] == "playback"]
    assert [item["phase"] for item in playback] == ["received", "requested", "playing", "ended"]
    assert playback[0]["audio_id"] == "abc"
    assert isinstance(playback[0]["latency_ms"], int)
    assert page.heard[0].heard_text == "Please rest now"
    reports, page = asyncio.run(protocol_without_sockets(True, True))
    assert reports[-1]["phase"] == "interrupted"
    assert reports[-1]["detail"] == "barge-in"
    assert page.heard[0].heard_fraction < 1
    reports, _ = asyncio.run(protocol_without_sockets(False, True))
    assert reports[-1]["phase"] == "ended"
    audio = page.audio_message(page.audio.next_chunk())
    assert audio["sample_rate"] == 16000
    assert len(base64.b64decode(audio["pcm16_b64"])) == 1365 * 2
