"""Virtual bedside page implementing script.js's two websocket channels."""

from __future__ import annotations

import asyncio
import base64
import io
import json
import time
import uuid
import wave
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from datetime import UTC, datetime
from urllib.parse import urljoin, urlparse
from urllib.request import urlopen

import websockets

from scene_lab.voice import SAMPLE_RATE, AudioStream

Hook = Callable[..., Awaitable[None]]


@dataclass
class Heard:
    text: str
    start: float
    end: float
    interrupted: bool
    heard_fraction: float

    @property
    def heard_text(self) -> str:
        words = self.text.split()
        return " ".join(words[: round(len(words) * self.heard_fraction)])


class VirtualPage:
    def __init__(
        self,
        base_url: str,
        audio: AudioStream,
        *,
        page_id: str | None = None,
        device_id: str | None = None,
        on_playback_started: Hook | None = None,
        on_playback_ended: Hook | None = None,
        on_playback_interrupted: Hook | None = None,
        on_show: Hook | None = None,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.audio = audio
        self.page_id = page_id or uuid.uuid4().hex
        self.device_id = device_id or uuid.uuid4().hex
        self.page_load_time = datetime.now(UTC).isoformat()
        self.on_playback_started = on_playback_started
        self.on_playback_ended = on_playback_ended
        self.on_playback_interrupted = on_playback_interrupted
        self.on_show = on_show
        self.heard: list[Heard] = []
        self.headline = ""
        self.latest_show: dict | None = None
        self._playback_socket = None
        self._media_socket = None
        self._current: dict | None = None
        self._play_task: asyncio.Task | None = None
        self._audio_unlocked = False

    def identity(self, kind: str) -> dict:
        return {
            "type": kind,
            "page_id": self.page_id,
            "device_id": self.device_id,
            "user_agent": "scene_lab virtual page",
            "platform": "Python",
            "screen": {"width": 1280, "height": 800},
            "visibility": "visible",
            "audio_unlocked": self._audio_unlocked,
            "page_load_time": self.page_load_time,
        }

    @staticmethod
    def audio_message(samples) -> dict:
        return {
            "type": "audio",
            "pcm16_b64": base64.b64encode(samples.astype("<i2", copy=False).tobytes()).decode(
                "ascii"
            ),
            "sample_rate": SAMPLE_RATE,
        }

    @staticmethod
    def _audio_id(url: str) -> str | None:
        parsed = urlparse(url)
        parts = parsed.path.split("/")
        if len(parts) >= 3 and parts[-2] in {"speech", "voice"} and parts[-1].endswith(".wav"):
            return parts[-1][:-4][:24]
        return None

    def _report(
        self, record: dict, phase: str, detail: str = "", error: Exception | None = None
    ) -> dict:
        return {
            "type": "playback",
            "phase": phase,
            "strategy": record["strategy"],
            "session_id": record["session_id"],
            "audio_id": record["audio_id"],
            "latency_ms": round((time.monotonic() - record["received_at"]) * 1000),
            "detail": detail,
            "error_name": type(error).__name__[:64] if error else None,
            "error_message": str(error)[:160] if error else None,
        }

    async def _send_report(self, record: dict, phase: str, detail: str = "", error=None) -> None:
        if self._playback_socket is not None:
            await self._playback_socket.send(json.dumps(self._report(record, phase, detail, error)))

    async def _hook(self, hook: Hook | None, *args) -> None:
        if hook is not None:
            await hook(*args)

    async def _interrupt(self, reason: str) -> None:
        record = self._current
        if record is None:
            return
        self._current = None
        if self._play_task is not None:
            self._play_task.cancel()
            self._play_task = None
        now = time.monotonic()
        if record.get("started_at") is not None:
            fraction = min(1.0, max(0.0, (now - record["started_at"]) / record["duration"]))
            heard = Heard(record["text"], record["started_at"], now, True, fraction)
            self.heard.append(heard)
            await self._hook(self.on_playback_interrupted, heard)
        await self._send_report(record, "interrupted", reason)

    @staticmethod
    def _wav_duration(data: bytes) -> float:
        with wave.open(io.BytesIO(data), "rb") as wav:
            return wav.getnframes() / wav.getframerate()

    def _fetch_wav(self, url: str) -> bytes:
        origin = self.base_url.replace("ws://", "http://", 1).replace("wss://", "https://", 1)
        with urlopen(urljoin(origin + "/", url), timeout=10) as response:
            return response.read()

    async def handle_message(self, message: dict) -> None:
        kind = message.get("type")
        if kind == "show":
            self.latest_show = message
            if isinstance(message.get("headline"), str):
                self.headline = message["headline"]
            await self._hook(self.on_show, message)
        elif kind == "speech_started":
            record = self._current
            if (
                record
                and record["interruptible"]
                and (not message.get("session_id") or message["session_id"] == record["session_id"])
            ):
                await self._interrupt("barge-in")
        elif kind == "say":
            await self._interrupt("replaced by newer Say")
            url = message.get("audio_url")
            record = {
                "text": message.get("text", ""),
                "strategy": message.get("strategy", None),
                "session_id": message.get("session_id", None),
                "audio_id": self._audio_id(url) if isinstance(url, str) else None,
                "interruptible": message.get("interruptible") is True,
                "received_at": time.monotonic(),
                "started_at": None,
            }
            await self._send_report(record, "received")
            if not isinstance(url, str) or not url:
                await self._send_report(record, "no_audio")
                return
            self._current = record
            await self._send_report(record, "requested")
            self._play_task = asyncio.create_task(self._play(record, url))

    async def _play(self, record: dict, url: str) -> None:
        try:
            data = await asyncio.to_thread(self._fetch_wav, url)
            record["duration"] = self._wav_duration(data)
            if self._current is not record:
                return
            record["started_at"] = time.monotonic()
            self._audio_unlocked = True
            for socket in (self._playback_socket, self._media_socket):
                if socket is not None:
                    await socket.send(json.dumps(self.identity("heartbeat")))
            await self._send_report(record, "playing")
            await self._hook(self.on_playback_started, record)
            await asyncio.sleep(record["duration"])
            if self._current is record:
                end = time.monotonic()
                heard = Heard(record["text"], record["started_at"], end, False, 1.0)
                self.heard.append(heard)
                await self._send_report(record, "ended")
                await self._hook(self.on_playback_ended, heard)
                self._current = None
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            if self._current is record:
                await self._send_report(record, "failed", "", exc)
                self._current = None

    async def _receive(self, socket) -> None:
        async for raw in socket:
            await self.handle_message(json.loads(raw))

    async def _microphone(self, socket) -> None:
        period = self.audio.chunk_samples / SAMPLE_RATE
        while True:
            await socket.send(json.dumps(self.audio_message(self.audio.next_chunk())))
            await asyncio.sleep(period)

    async def _heartbeats(self, sockets) -> None:
        while True:
            await asyncio.sleep(10)
            for socket in sockets:
                await socket.send(json.dumps(self.identity("heartbeat")))

    async def run(self) -> None:
        """Connect until cancelled; the runner owns restart policy."""
        ws_base = self.base_url.replace("http://", "ws://", 1).replace("https://", "wss://", 1)
        async with websockets.connect(ws_base + "/ws") as playback:
            async with websockets.connect(ws_base + "/media") as media:
                self._playback_socket = playback
                self._media_socket = media
                try:
                    await playback.send(json.dumps(self.identity("hello")))
                    await media.send(json.dumps(self.identity("hello")))
                    async with asyncio.TaskGroup() as group:
                        group.create_task(self._receive(playback))
                        group.create_task(self._microphone(media))
                        group.create_task(self._heartbeats((playback, media)))
                finally:
                    if self._play_task is not None:
                        self._play_task.cancel()
                    self._playback_socket = None
                    self._media_socket = None
