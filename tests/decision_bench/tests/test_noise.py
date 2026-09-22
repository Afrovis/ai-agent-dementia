import pytest

from decision_bench.noise import _typo, perturb, variant_text
from decision_bench.schema import SCENARIOS_DIR, load_scenario

TIMELINE = [
    {"t": 0, "person": {"state": "sitting_up", "zone": "bed"}},
    {"t": 10, "utterance": {"text": "I need the toilet."}},
    {"t": 20, "person": {"state": "walking", "zone": "bathroom_path"}},
    {"t": 60, "utterance": {"text": "Where is it?"}},
    {"t": 80, "person": {"state": "absent", "zone": "other"}},
]


def test_typo_is_small_and_deterministic():
    assert _typo("I need the toilet.") == "i need the toliet"
    assert _typo("I need the toilet.") == _typo("I need the toilet.")


def test_flicker_inserts_a_brief_neighbour_state_and_returns():
    events = perturb(TIMELINE, "flicker")
    states = [(e["t"], e["person"]["state"]) for e in events if "person" in e]
    assert (22.0, "standing") in states and (24.0, "walking") in states
    assert [e["t"] for e in events] == sorted(e["t"] for e in events)


def test_low_confidence_typo_and_dropped():
    assert all(
        e["person"]["confidence"] == 0.55
        for e in perturb(TIMELINE, "low_confidence")
        if "person" in e
    )
    typo = [e for e in perturb(TIMELINE, "typo") if "utterance" in e]
    assert typo[0]["utterance"]["confidence"] == 0.6
    dropped = perturb(TIMELINE, "dropped")
    assert [e["utterance"]["text"] for e in dropped if "utterance" in e] == ["I need the toilet."]
    with pytest.raises(ValueError):
        perturb(TIMELINE[:3], "dropped")


def test_variant_keeps_the_parents_labels(tmp_path):
    parent = SCENARIOS_DIR / "restroom-01.yaml"
    variant_id, text = variant_text(parent, "typo")
    path = tmp_path / f"{variant_id}.yaml"
    path.write_text(text)
    variant, original = load_scenario(path), load_scenario(parent)
    assert variant.noise_of == "restroom-01"
    assert variant.checkpoints == original.checkpoints
    assert variant.timeline != original.timeline
