"""Voice activity detection and bounded in-memory utterance segmentation."""

from __future__ import annotations

from collections import deque
from collections.abc import Callable

SAMPLE_RATE = 16_000
FRAME_MS = 30
BYTES_PER_SAMPLE = 2


class WebRtcVoiceDetector:
    """Small adapter around WebRTC VAD, configured for 16 kHz PCM16 frames."""

    def __init__(self, aggressiveness: int = 2) -> None:
        import webrtcvad  # Imported here so unit tests need no native VAD package.

        self._vad = webrtcvad.Vad(aggressiveness)

    def __call__(self, pcm16: bytes, sample_rate: int) -> bool:
        return bool(self._vad.is_speech(pcm16, sample_rate))


class SpeechSegmenter:
    """Turn arbitrary PCM chunks into utterances using fixed 30 ms VAD frames.

    Audio exists only in these in-memory buffers and is discarded by ``reset``;
    this module never opens a file or logs sample data.
    """

    def __init__(
        self,
        detector: Callable[[bytes, int], bool],
        *,
        sample_rate: int = SAMPLE_RATE,
        end_silence_ms: int = 600,
        min_speech_ms: int = 180,
        max_utterance_seconds: float = 15.0,
        pre_roll_ms: int = 300,
    ) -> None:
        self.detector = detector
        self.sample_rate = sample_rate
        self.end_silence_ms = end_silence_ms
        self.min_speech_ms = min_speech_ms
        self.max_utterance_ms = int(max_utterance_seconds * 1000)
        self.frame_bytes = sample_rate * FRAME_MS // 1000 * BYTES_PER_SAMPLE
        self._pre_roll: deque[bytes] = deque(maxlen=max(1, pre_roll_ms // FRAME_MS))
        self._pending = bytearray()
        self._utterance = bytearray()
        self._speech_ms = 0
        self._silence_ms = 0
        self._speech_starts = 0

    def accept(self, pcm16: bytes) -> list[bytes]:
        """Accept an arbitrary chunk and return zero or more complete utterances."""
        self._pending.extend(pcm16)
        complete: list[bytes] = []
        while len(self._pending) >= self.frame_bytes:
            frame = bytes(self._pending[: self.frame_bytes])
            del self._pending[: self.frame_bytes]
            is_speech = self.detector(frame, self.sample_rate)

            if not self._utterance:
                if not is_speech:
                    self._pre_roll.append(frame)
                    continue
                self._speech_starts += 1
                self._utterance.extend(b"".join(self._pre_roll))
                self._pre_roll.clear()

            self._utterance.extend(frame)
            if is_speech:
                self._speech_ms += FRAME_MS
                self._silence_ms = 0
            else:
                self._silence_ms += FRAME_MS

            duration_ms = len(self._utterance) * 1000 // (self.sample_rate * BYTES_PER_SAMPLE)
            if self._silence_ms >= self.end_silence_ms or duration_ms >= self.max_utterance_ms:
                utterance = self._finish()
                if utterance is not None:
                    complete.append(utterance)
        return complete

    def take_speech_starts(self) -> int:
        """Return and clear the number of VAD onsets seen since the last call."""
        starts = self._speech_starts
        self._speech_starts = 0
        return starts

    def reset(self) -> None:
        """Discard all buffered audio, including any partial utterance."""
        self._pending.clear()
        self._pre_roll.clear()
        self._utterance.clear()
        self._speech_ms = 0
        self._silence_ms = 0
        self._speech_starts = 0

    def _finish(self) -> bytes | None:
        utterance = bytes(self._utterance) if self._speech_ms >= self.min_speech_ms else None
        self._utterance.clear()
        self._speech_ms = 0
        self._silence_ms = 0
        return utterance
