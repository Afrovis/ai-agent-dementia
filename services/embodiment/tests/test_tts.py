"""Piper TTS tests with a fake voice: no model weights or audio hardware."""

import wave
from datetime import datetime

import pytest
from agent.strategies import spoken_time_words

from embodiment.tts import PiperSpeech, load_prerender_phrases


class FakeVoice:
    def __init__(self) -> None:
        self.calls: list[tuple[str, object]] = []

    def synthesize_wav(self, text, wav_file, *, syn_config) -> None:
        self.calls.append((text, syn_config))
        wav_file.setnchannels(1)
        wav_file.setsampwidth(2)
        wav_file.setframerate(16000)
        wav_file.writeframes(b"\x00\x00" * 160)


def test_synthesize_writes_valid_wav_and_reuses_cache(tmp_path):
    voice = FakeVoice()
    config = object()
    speech = PiperSpeech(voice, config, tmp_path, cache_namespace="test-voice:0.85")

    first_id = speech.synthesize("Hello Jean.")
    second_id = speech.synthesize("Hello Jean.")

    assert first_id == second_id
    assert len(voice.calls) == 1
    assert voice.calls[0] == ("Hello Jean.", config)
    path = speech.resolve(first_id)
    assert path is not None
    with wave.open(str(path), "rb") as wav_file:
        assert wav_file.getframerate() == 16000
        assert wav_file.getnchannels() == 1


def test_from_model_converts_point_85_speed_to_piper_length_scale(tmp_path, monkeypatch):
    import piper

    model = tmp_path / "voice.onnx"
    model.write_bytes(b"fake model")
    voice = FakeVoice()
    monkeypatch.setattr(piper.PiperVoice, "load", lambda _path: voice)

    speech = PiperSpeech.from_model(model, tmp_path / "cache", speed=0.85)
    speech.synthesize("Rest now.")

    assert voice.calls[0][1].length_scale == pytest.approx(1 / 0.85)


def test_resolve_rejects_non_digest_ids(tmp_path):
    speech = PiperSpeech(FakeVoice(), object(), tmp_path, cache_namespace="test")

    assert speech.resolve("../secret") is None
    assert speech.resolve("not-a-digest") is None


def test_pre_render_deduplicates_phrases(tmp_path):
    voice = FakeVoice()
    speech = PiperSpeech(voice, object(), tmp_path, cache_namespace="test")

    count = speech.pre_render(("Hello.", "Hello.", "  Rest now.  ", ""))

    assert count == 2
    assert [call[0] for call in voice.calls] == ["Hello.", "Rest now."]


def test_load_prerender_phrases_expands_spoken_time_variants(tmp_path):
    strategies = tmp_path / "strategies.yaml"
    strategies.write_text(
        "strategies:\n"
        "  - id: ambient_orient\n"
        "    say: null\n"
        "  - id: soft_greeting\n"
        '    say: "Hello {name}, it is {time_words}."\n'
        "  - id: disabled\n"
        "    enabled: false\n"
        '    say: "Do not warm this."\n'
    )
    person = tmp_path / "person.yaml"
    person.write_text("name: Jean\ncaregiver:\n  name: Tom\n")

    phrases = load_prerender_phrases(strategies, person)

    assert len(phrases) == 7
    assert "Hello Jean, it is late in the evening." in phrases
    assert "Hello Jean, it is the middle of the night." in phrases
    assert "Hello Jean, it is very early in the morning." in phrases
    assert all("o'clock" not in phrase for phrase in phrases)
    expected = {
        spoken_time_words(datetime(2026, 1, 1, hour), variant)
        for hour in range(24)
        for variant in range(3)
    }
    assert {
        phrase.removeprefix("Hello Jean, it is ").removesuffix(".") for phrase in phrases
    } == expected
    assert all("Do not warm this" not in phrase for phrase in phrases)


def test_load_prerender_phrases_uses_safe_defaults_when_files_are_missing(tmp_path):
    phrases = load_prerender_phrases(
        tmp_path / "missing-strategies.yaml",
        tmp_path / "missing-person.yaml",
    )

    assert "Hello there, it's night-time." in phrases
    assert "You are home in your bedroom, and it is night-time." in phrases
    assert "Someone is coming to help." in phrases
