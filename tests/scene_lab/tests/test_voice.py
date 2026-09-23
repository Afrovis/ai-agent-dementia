from pathlib import Path

import numpy as np
import pytest

from scene_lab.voice import AudioStream, RoomNoise, Voice


def fake_synth(text, *, length_scale):
    return np.full(round(8000 * length_scale), 10000, dtype=np.float32), 8000


def test_styles_and_stream():
    voice = Voice("en_US-amy-medium", synthesizer=fake_synth)
    normal = voice.synthesize("hello")
    mumble = voice.synthesize("hello", "mumble")
    trailing = voice.synthesize("hello", "trailing")
    assert len(normal) == 16000
    assert np.max(np.abs(mumble)) < 3000
    assert len(trailing) > len(normal)
    assert abs(int(trailing[-1])) < 300
    stream = AudioStream(RoomNoise(seed=1))
    stream.queue(normal)
    assert stream.speaking
    assert len(stream.next_chunk()) == stream.chunk_samples
    stream.cancel()
    assert not stream.speaking


@pytest.mark.skipif(
    not (
        Path("/Users/mathiasserver/Documents/data-ai-agent-dementia/models/piper-sim")
        / "en_US-amy-medium.onnx"
    ).is_file(),
    reason="Piper simulation voice is not installed",
)
def test_real_voice():
    samples = Voice("en_US-amy-medium").synthesize("hello")
    assert samples.dtype == np.int16
    assert len(samples) > 1600
