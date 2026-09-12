"""Person profile loading and prompt-shape tests (issue #16)."""

from pathlib import Path

from agent.profile import DEFAULT_PROFILE, PersonProfile, load_profile

EXAMPLE_PATH = Path(__file__).resolve().parents[3] / "config" / "person.example.yaml"


def test_shipped_example_contains_every_profile_field():
    profile = load_profile(EXAMPLE_PATH)
    assert profile == PersonProfile(
        name="Jean",
        preferred_address="Jean",
        caregiver_name="Tom",
        caregiver_relationship="son",
        night_themes=("Looks for a deceased spouse", "Believes the children are still young"),
        calming_things=(
            "A photo of the garden",
            'The phrase "Tom is nearby and everything is settled"',
        ),
        things_to_avoid=("Do not mention the hospital", "Avoid loud or urgent language"),
        physical_notes=("Uses a walker", "Is unsteady at night"),
        restroom_location=("The restroom is through the bedroom door and immediately to the left"),
    )


def test_load_profile_uses_env_path_and_accepts_string_caregiver(tmp_path):
    path = tmp_path / "custom.yaml"
    path.write_text("name: Jo\ncaregiver: Sam\ncalming_things: [Soft music]\n", encoding="utf-8")
    profile = load_profile(env={"PERSON_PATH": str(path)})
    assert profile.name == "Jo"
    assert profile.caregiver_name == "Sam"
    assert profile.calming_things == ("Soft music",)


def test_load_profile_uses_example_beside_missing_primary(tmp_path):
    (tmp_path / "person.example.yaml").write_text("name: Bea\n", encoding="utf-8")
    assert load_profile(tmp_path / "person.yaml").name == "Bea"


def test_load_profile_falls_back_safely_on_bad_shape(tmp_path):
    path = tmp_path / "person.yaml"
    path.write_text("night_themes: not-a-list\n", encoding="utf-8")
    assert load_profile(path) == DEFAULT_PROFILE


def test_prompt_data_is_json_safe_and_complete():
    data = PersonProfile(name="Jean", night_themes=("work",)).prompt_data()
    assert data == {
        "name": "Jean",
        "preferred_address": "",
        "caregiver_name": "your caregiver",
        "caregiver_relationship": "",
        "night_themes": ["work"],
        "calming_things": [],
        "things_to_avoid": [],
        "physical_notes": [],
        "restroom_location": "",
    }
