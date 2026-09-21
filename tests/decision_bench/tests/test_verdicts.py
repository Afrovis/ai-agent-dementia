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
    assert path.read_text().startswith("# Human verdicts")
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
