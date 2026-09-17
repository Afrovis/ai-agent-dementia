from __future__ import annotations

import io
import wave

import pytest
import yaml
from PIL import Image

from dashboard.settings_store import (
    load_profile_document,
    load_strategy_document,
    save_photo,
    save_profile_document,
    save_strategy_document,
    save_voice_clip,
)


def _strategy() -> dict:
    return {
        "id": "soft_greeting",
        "order": 2,
        "enabled": True,
        "dwell_seconds": 20,
        "cooldown_seconds": 120,
        "intrusiveness": 2,
        "face": "awake",
        "brightness": 0.5,
        "headline": "Hello, {name}",
        "body": "It is {time_words}.",
        "say": "Hello {name}, it's {time_words}.",
        "photo_id": None,
    }


def _strategy_form(**changes: object) -> dict[str, object]:
    values = _strategy()
    values.update(changes)
    prefix = "soft_greeting__"
    form = {
        prefix + key: str(value)
        for key, value in values.items()
        if key != "id" and value is not None
    }
    if values["enabled"]:
        form[prefix + "enabled"] = "on"
    return form


def _wav_bytes(seconds: float = 0.1) -> bytes:
    output = io.BytesIO()
    with wave.open(output, "wb") as audio:
        audio.setnchannels(1)
        audio.setsampwidth(2)
        audio.setframerate(16000)
        audio.writeframes(b"\0\0" * int(16000 * seconds))
    return output.getvalue()


def test_profile_round_trip_uses_agent_compatible_shape(tmp_path):
    path = tmp_path / "person.yaml"
    save_profile_document(
        path,
        {
            "name": "Jean",
            "preferred_address": "Jeannie",
            "caregiver_name": "Tom",
            "caregiver_relationship": "son",
            "night_themes": "Looks for work\nWorries about children",
            "calming_things": "Garden photo",
            "things_to_avoid": "Urgent language",
            "physical_notes": "Uses a walker",
            "restroom_location": "Outside the door on the left",
            "enable_cloud_fallback": "on",
        },
    )

    assert load_profile_document(path)["caregiver"] == {"name": "Tom", "relationship": "son"}
    assert load_profile_document(path)["night_themes"] == [
        "Looks for work",
        "Worries about children",
    ]
    assert load_profile_document(path)["enable_cloud_fallback"] is True
    assert not list(tmp_path.glob(".*.tmp"))


def test_profile_requires_a_name(tmp_path):
    with pytest.raises(ValueError, match="name is required"):
        save_profile_document(tmp_path / "person.yaml", {"name": " "})


def test_strategy_round_trip_and_reorder(tmp_path):
    path = tmp_path / "strategies.yaml"
    save_strategy_document(path, [_strategy()], _strategy_form(order=1, headline="Good evening"))

    saved = load_strategy_document(path)[0]
    assert saved["order"] == 1
    assert saved["headline"] == "Good evening"


def test_familiar_voice_clip_id_round_trips_and_is_validated(tmp_path):
    strategy = _strategy()
    strategy.update({"id": "familiar_voice", "order": 6, "clip_id": "family-message"})
    form = {
        key.replace("soft_greeting__", "familiar_voice__"): value
        for key, value in _strategy_form(order=6).items()
    }
    form["familiar_voice__clip_id"] = "family-message"
    path = tmp_path / "strategies.yaml"

    save_strategy_document(path, [strategy], form)

    assert load_strategy_document(path)[0]["clip_id"] == "family-message"
    form["familiar_voice__clip_id"] = "../secret"
    with pytest.raises(ValueError, match="clip id"):
        save_strategy_document(path, [strategy], form)


def test_unknown_strategy_id_is_not_rendered_or_rewritten(tmp_path):
    path = tmp_path / "strategies.yaml"
    path.write_text("strategies:\n  - id: '<script>alert(1)</script>'\n")

    assert load_strategy_document(path) == []


def test_terminal_strategy_may_keep_infinite_until_acknowledged_dwell(tmp_path):
    strategy = _strategy()
    strategy["id"] = "escalate_phone"
    form = {
        key.replace("soft_greeting__", "escalate_phone__"): value
        for key, value in _strategy_form(dwell_seconds="inf").items()
    }

    save_strategy_document(tmp_path / "strategies.yaml", [strategy], form)

    assert load_strategy_document(tmp_path / "strategies.yaml")[0]["dwell_seconds"] == float("inf")


def test_non_terminal_strategy_rejects_non_finite_numbers(tmp_path):
    with pytest.raises(ValueError, match="dwell"):
        save_strategy_document(
            tmp_path / "strategies.yaml",
            [_strategy()],
            _strategy_form(dwell_seconds="nan"),
        )


@pytest.mark.parametrize(
    "say",
    [
        "Do you remember your room?",
        "No, go back to bed.",
        "Rest now. Everyone is sleeping.",
        "This has an {unclosed placeholder.",
    ],
)
def test_strategy_save_rejects_unsafe_or_malformed_speech(tmp_path, say):
    with pytest.raises(ValueError):
        save_strategy_document(tmp_path / "strategies.yaml", [_strategy()], _strategy_form(say=say))


def test_photo_upload_validates_content_and_assigns_safe_unique_ids(tmp_path):
    image = io.BytesIO()
    Image.new("RGB", (4, 4), "red").save(image, format="PNG")

    first_id, first = save_photo(tmp_path, "Family photo.png", image.getvalue())
    second_id, second = save_photo(tmp_path, "Family photo.png", image.getvalue())

    assert first_id == "family-photo"
    assert second_id == "family-photo-2"
    assert first.suffix == second.suffix == ".png"
    with pytest.raises(ValueError, match="valid"):
        save_photo(tmp_path, "fake.jpg", b"not an image")


def test_voice_upload_accepts_pcm_wav_and_rejects_arbitrary_bytes(tmp_path):
    media_id, path = save_voice_clip(tmp_path, "Tom's message.wav", _wav_bytes())

    assert media_id == "tom-s-message"
    assert path.read_bytes().startswith(b"RIFF")
    with pytest.raises(ValueError, match="WAV"):
        save_voice_clip(tmp_path, "not-audio.wav", b"not audio")


def test_written_yaml_contains_no_python_specific_tags(tmp_path):
    path = tmp_path / "strategies.yaml"
    save_strategy_document(path, [_strategy()], _strategy_form())

    assert yaml.safe_load(path.read_text())["strategies"][0]["id"] == "soft_greeting"
