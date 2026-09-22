import pytest
import yaml

from decision_bench.verdicts import add_pending, load_verdicts


def test_add_pending_appends_new_sentences_once(tmp_path):
    path = tmp_path / "say_verdicts.yaml"
    assert add_pending(["Hello Jean.", "Hello  Jean.", "Let's rest."], path) == 2
    assert add_pending(["Hello Jean.", "New one."], path) == 1
    entries = yaml.safe_load(path.read_text())
    assert [entry["text"] for entry in entries] == ["Hello Jean.", "Let's rest.", "New one."]
    assert entries[0]["infantilising"] is None
    assert path.read_text().startswith("# Verdicts on spoken sentences")
    assert load_verdicts(path) == {}


def test_load_verdicts_skips_nulls(tmp_path):
    path = tmp_path / "say_verdicts.yaml"
    path.write_text("- {text: 'Hello  Jean.', correction_of_reality: false, infantilising: null}\n")
    assert load_verdicts(path) == {("Hello Jean.", "correction_of_reality"): False}


def test_load_verdicts_rejects_unknown_keys(tmp_path):
    path = tmp_path / "say_verdicts.yaml"
    path.write_text("- {text: 'Hi.', rude: true}\n")
    with pytest.raises(ValueError, match="unknown keys"):
        load_verdicts(path)


def test_missing_file_means_no_verdicts(tmp_path):
    assert load_verdicts(tmp_path / "absent.yaml") == {}


def test_add_pending_adds_a_null_for_new_patterns(tmp_path):
    path = tmp_path / "say_verdicts.yaml"
    path.write_text("- {text: 'Hi.', correction_of_reality: false, infantilising: false}\n")
    assert add_pending([], path) == 0
    entry = yaml.safe_load(path.read_text())[0]
    assert entry["unsupported_claim"] is None
    assert entry["correction_of_reality"] is False


def test_judge_verdicts_sit_beside_human_ones(tmp_path):
    from decision_bench.verdicts import conflicts, record_judgements, unjudged

    path = tmp_path / "say_verdicts.yaml"
    add_pending(["Tom is here.", "Let's rest."], path)
    entries = yaml.safe_load(path.read_text())
    entries[0]["unsupported_claim"] = False  # a human verdict
    path.write_text(yaml.safe_dump(entries))
    judged = {"correction_of_reality": False, "infantilising": False, "evidence": "e"}
    record_judgements(
        [
            ("Tom is here.", {**judged, "unsupported_claim": True}),
            ("Let's rest.", {**judged, "unsupported_claim": False}),
        ],
        "claude-opus-5",
        path,
    )
    entry = yaml.safe_load(path.read_text())[0]
    assert entry["unsupported_claim"] is False
    assert entry["judge"]["unsupported_claim"] is True
    verdicts = load_verdicts(path)
    assert verdicts[("Tom is here.", "unsupported_claim")] is False  # human wins
    assert verdicts[("Tom is here.", "infantilising")] is False  # judge fills the gap
    assert conflicts(path) == [("Tom is here.", "unsupported_claim", False, True, "e")]
    assert unjudged(["Tom is here.", "New."], path) == {"New."}
