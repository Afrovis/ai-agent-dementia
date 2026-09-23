import pytest
from pydantic import ValidationError

from scene_lab.scene import Scene, load_scene


def card():
    return {
        "id": "test-scene",
        "category": "conversation",
        "start": "02:40",
        "duration_s": 60,
        "profile": "profiles/anna.yaml",
        "persona": {"summary": "Retired teacher"},
        "opening": [
            {"at": 0, "move": {"state": "sitting_up", "zone": "bed"}},
            {"at": 5, "say": "Is it morning?", "style": "mumble"},
            {"at": 10, "wait": 2},
            {"at": 12, "end": True},
        ],
    }


def test_scene_card_and_loader(tmp_path):
    import yaml

    path = tmp_path / "scene.yaml"
    path.write_text(yaml.safe_dump(card()))
    scene = load_scene(path)
    assert scene.id == "test-scene"
    assert len(scene.opening) == 4
    assert scene.mind == "script"


def test_invalid_beat():
    data = card()
    data["opening"][0]["say"] = "too much"
    with pytest.raises(ValidationError):
        Scene.model_validate(data)
