"""Configuration tests for listen."""

import pytest

from listen.config import ListenConfig


def test_defaults_are_local_small_en_cpu():
    config = ListenConfig.from_env({})

    assert config.model == "small.en"
    assert config.device == "cpu"
    assert config.compute_type == "int8"
    assert config.model_cache == "/app/data/models/faster-whisper"


def test_invalid_vad_aggressiveness_is_rejected():
    with pytest.raises(ValueError, match="between 0 and 3"):
        ListenConfig.from_env({"LISTEN_VAD_AGGRESSIVENESS": "4"})
