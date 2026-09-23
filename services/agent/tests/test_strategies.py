"""Tests for `agent.strategies`: the catalogue, `load_strategies`'s
fallback chain, and `StrategyEngine`'s ordering/cooldown/dwell/progress
behaviour."""

import wave
from dataclasses import replace
from datetime import datetime, timedelta
from pathlib import Path

import pytest
from dialogue_bench.checks import CheckContext, CheckStatus, conjunction_but, states_clock_time

from agent.rules import validate_say
from agent.strategies import (
    DEFAULT_PROFILE,
    DEFAULT_STRATEGIES,
    ESCALATE_PHONE_ID,
    FAMILIAR_VOICE_ID,
    PATH_LIGHT_ID,
    PersonProfile,
    StrategyEngine,
    load_strategies,
    render_template,
    spoken_time_words,
    time_as_words,
)

NOW = datetime(2026, 1, 1, 23, 0)
NAMED_PROFILE = PersonProfile(name="Jean", caregiver_name="Tom")

# The real, shipped example file, not a synthetic fixture: this is what a
# caregiver actually gets on a fresh checkout, so it must pass the same
# checks as the code defaults (the coordinator's "not just the code
# defaults" requirement).
EXAMPLE_STRATEGIES_PATH = Path(__file__).resolve().parents[3] / "config" / "strategies.example.yaml"


def strategies_by_order(*, dwell=100.0, cooldown=50.0):
    """A small, deterministic three-strategy catalogue for engine tests,
    independent of `DEFAULT_STRATEGIES`'s real dwell/cooldown values."""
    base = DEFAULT_STRATEGIES[0]
    return [
        replace(
            base, id="a", order=1, enabled=True, dwell_seconds=dwell, cooldown_seconds=cooldown
        ),
        replace(
            base, id="b", order=2, enabled=True, dwell_seconds=dwell, cooldown_seconds=cooldown
        ),
        replace(
            base, id="c", order=3, enabled=True, dwell_seconds=dwell, cooldown_seconds=cooldown
        ),
    ]


def write_wav(path: Path, *, seconds: float = 0.25) -> None:
    with wave.open(str(path), "wb") as audio:
        audio.setnchannels(1)
        audio.setsampwidth(2)
        audio.setframerate(8000)
        audio.writeframes(b"\0\0" * int(8000 * seconds))


# --- StrategyEngine: ordering, enable/disable, cooldown, dwell -----------


def test_spoken_time_phrases_rotate_by_hour_band_and_pass_speech_checks():
    expected = {
        22: ("late in the evening", "late at night", "night-time"),
        2: ("the middle of the night", "night-time", "still night-time"),
        5: ("very early in the morning", "still night-time", "nearly morning and still dark"),
        12: ("night-time", "night-time", "night-time"),
    }
    for hour, phrases in expected.items():
        now = NOW.replace(hour=hour)
        assert tuple(spoken_time_words(now, i) for i in range(3)) == phrases
        for phrase in phrases:
            sentence = f"You are home in your bedroom, and it is {phrase}."
            assert validate_say(sentence, seconds_since_last_say=None, min_gap_seconds=8).accepted
            ctx = CheckContext(
                profile={},
                utterance=None,
                time_words=phrase,
                scene_note=None,
                caregiver_phrase_template="",
                avoid_terms=(),
            )
            assert states_clock_time(sentence, ctx).status == CheckStatus.PASS
            assert conjunction_but(sentence, ctx).status == CheckStatus.PASS


def test_engine_selects_first_in_configured_order():
    engine = StrategyEngine(strategies_by_order())
    selected = engine.start(NOW)
    assert selected.id == "a"
    assert engine.index_of("a") == 0


def test_config_driven_order_actually_reorders_selection():
    strategies = strategies_by_order()
    # Reorder "c" to come first.
    reordered = [replace(s, order={"c": 1, "a": 2, "b": 3}[s.id]) for s in strategies]
    engine = StrategyEngine(reordered)
    selected = engine.start(NOW)
    assert selected.id == "c"
    assert engine.index_of("c") == 0
    assert engine.index_of("a") == 1


def test_disabled_strategy_is_skipped():
    strategies = strategies_by_order()
    strategies[0] = replace(strategies[0], enabled=False)
    engine = StrategyEngine(strategies)
    selected = engine.start(NOW)
    assert selected.id == "b"


def test_all_disabled_has_nothing_available():
    strategies = [replace(s, enabled=False) for s in strategies_by_order()]
    engine = StrategyEngine(strategies)
    assert engine.has_available(NOW) is False
    assert engine.start(NOW) is None


def test_familiar_voice_disabled_is_ineligible_even_with_a_clip(tmp_path):
    write_wav(tmp_path / "family-message.wav")
    strategy = replace(
        next(s for s in DEFAULT_STRATEGIES if s.id == FAMILIAR_VOICE_ID),
        enabled=False,
        clip_id="family-message",
    )

    assert StrategyEngine([strategy], voice_clip_dir=tmp_path).start(NOW) is None


@pytest.mark.parametrize("clip_id", [None, "", "../secret", "folder/clip", r"folder\clip", "Upper"])
def test_familiar_voice_rejects_missing_or_unsafe_clip_id(tmp_path, clip_id):
    strategy = replace(
        next(s for s in DEFAULT_STRATEGIES if s.id == FAMILIAR_VOICE_ID),
        enabled=True,
        clip_id=clip_id,
    )

    assert StrategyEngine([strategy], voice_clip_dir=tmp_path).start(NOW) is None


def test_familiar_voice_rejects_missing_and_unreadable_wav(tmp_path):
    base = next(s for s in DEFAULT_STRATEGIES if s.id == FAMILIAR_VOICE_ID)
    missing = replace(base, enabled=True, clip_id="missing")
    assert StrategyEngine([missing], voice_clip_dir=tmp_path).start(NOW) is None

    (tmp_path / "broken.wav").write_bytes(b"not a wave file")
    broken = replace(base, enabled=True, clip_id="broken")
    assert StrategyEngine([broken], voice_clip_dir=tmp_path).start(NOW) is None


def test_familiar_voice_uses_wav_duration_plus_fifteen_seconds(tmp_path):
    write_wav(tmp_path / "family-message.wav", seconds=0.25)
    strategy = replace(
        next(s for s in DEFAULT_STRATEGIES if s.id == FAMILIAR_VOICE_ID),
        enabled=True,
        clip_id="family-message",
        dwell_seconds=999,
    )
    engine = StrategyEngine([strategy], voice_clip_dir=tmp_path)

    selected = engine.start(NOW)

    assert selected is not None
    assert selected.dwell_seconds == pytest.approx(15.25)
    assert engine.maybe_advance(NOW + timedelta(seconds=15.24))[1] is False
    assert engine.maybe_advance(NOW + timedelta(seconds=15.26))[2] is True


def test_dwell_elapsed_with_no_progress_advances_to_next():
    engine = StrategyEngine(strategies_by_order(dwell=30.0))
    engine.start(NOW)
    new_current, changed, exhausted = engine.maybe_advance(NOW + timedelta(seconds=31))
    assert changed is True
    assert exhausted is False
    assert new_current.id == "b"


def test_dwell_not_elapsed_does_not_advance():
    engine = StrategyEngine(strategies_by_order(dwell=30.0))
    engine.start(NOW)
    new_current, changed, exhausted = engine.maybe_advance(NOW + timedelta(seconds=10))
    assert changed is False
    assert exhausted is False
    assert new_current.id == "a"


def test_progress_pauses_the_dwell_timer():
    engine = StrategyEngine(strategies_by_order(dwell=30.0))
    engine.start(NOW)
    # Progress right before the dwell would have elapsed...
    engine.note_progress(NOW + timedelta(seconds=29))
    # ...so 31s after the *original* start is still within a fresh 30s
    # window from the progress event, and must not have advanced.
    new_current, changed, _ = engine.maybe_advance(NOW + timedelta(seconds=31))
    assert changed is False
    assert new_current.id == "a"


def test_cooldown_prevents_reselection_and_expires():
    # A single-strategy ladder isolates the cooldown check to one id:
    # once "a" is the only strategy and it goes on cooldown, `has_available`
    # (which `_enter_engaged` would use to decide whether a fresh `ENGAGED`
    # entry has anything to show) reflects only whether *that* cooldown has
    # expired.
    strategy = replace(
        DEFAULT_STRATEGIES[0],
        id="a",
        order=1,
        enabled=True,
        dwell_seconds=10.0,
        cooldown_seconds=20.0,
    )
    engine = StrategyEngine([strategy])
    engine.start(NOW)
    new_current, changed, exhausted = engine.maybe_advance(NOW + timedelta(seconds=11))
    assert changed is True
    assert exhausted is True
    assert new_current is None

    # "a"'s cooldown started at +11s and lasts 20s, so it is not yet
    # available at +20s (9s into the cooldown)...
    assert engine.has_available(NOW + timedelta(seconds=20)) is False
    # ...but is available again once the cooldown has elapsed.
    assert engine.has_available(NOW + timedelta(seconds=32)) is True


def test_exhausting_every_strategy_reports_exhausted():
    engine = StrategyEngine(strategies_by_order(dwell=10.0, cooldown=1000.0))
    engine.start(NOW)  # "a"
    _, changed_1, exhausted_1 = engine.maybe_advance(NOW + timedelta(seconds=11))  # -> "b"
    assert changed_1 is True
    assert exhausted_1 is False
    _, changed_2, exhausted_2 = engine.maybe_advance(NOW + timedelta(seconds=22))  # -> "c"
    assert changed_2 is True
    assert exhausted_2 is False
    new_current, changed_3, exhausted_3 = engine.maybe_advance(NOW + timedelta(seconds=33))
    assert changed_3 is True
    assert exhausted_3 is True
    assert new_current is None
    assert engine.current() is None


def test_force_bypasses_enabled_and_cooldown():
    strategies = strategies_by_order()
    strategies[0] = replace(strategies[0], id=ESCALATE_PHONE_ID, enabled=False)
    engine = StrategyEngine(strategies)
    selected = engine.force(ESCALATE_PHONE_ID, NOW)
    assert selected is not None
    assert engine.current_id == ESCALATE_PHONE_ID


def test_reset_clears_current_strategy_and_cooldowns():
    engine = StrategyEngine(strategies_by_order(dwell=10.0, cooldown=1000.0))
    engine.start(NOW)
    engine.maybe_advance(NOW + timedelta(seconds=11))
    engine.reset()
    assert engine.current() is None
    assert engine.has_available(NOW) is True
    selected = engine.start(NOW)
    assert selected.id == "a"  # cooldown from before the reset is gone


# --- load_strategies: the fallback chain ---------------------------------


def test_load_strategies_falls_back_to_defaults_when_nothing_exists(tmp_path):
    missing = tmp_path / "does-not-exist.yaml"
    loaded = load_strategies(missing)
    assert [s.id for s in loaded] == [
        s.id for s in sorted(DEFAULT_STRATEGIES, key=lambda s: s.order)
    ]


def test_load_strategies_falls_back_to_defaults_on_malformed_yaml(tmp_path):
    path = tmp_path / "strategies.yaml"
    path.write_text("not: [valid, yaml", encoding="utf-8")
    loaded = load_strategies(path)
    assert [s.id for s in loaded] == [
        s.id for s in sorted(DEFAULT_STRATEGIES, key=lambda s: s.order)
    ]


def test_load_strategies_falls_back_to_defaults_on_wrong_shape(tmp_path):
    path = tmp_path / "strategies.yaml"
    path.write_text("just_a_string", encoding="utf-8")
    loaded = load_strategies(path)
    assert len(loaded) == len(DEFAULT_STRATEGIES)


def test_load_strategies_falls_back_to_example_file(tmp_path):
    example = tmp_path / "strategies.example.yaml"
    example.write_text(
        "strategies:\n  - id: ambient_orient\n    enabled: false\n", encoding="utf-8"
    )
    primary = tmp_path / "strategies.yaml"  # deliberately does not exist
    loaded = load_strategies(primary)
    by_id = {s.id: s for s in loaded}
    assert by_id["ambient_orient"].enabled is False


def test_load_strategies_applies_overrides_and_keeps_untouched_defaults(tmp_path):
    path = tmp_path / "strategies.yaml"
    path.write_text(
        "strategies:\n"
        "  - id: ambient_orient\n"
        "    order: 9\n"
        "    enabled: false\n"
        "    dwell_seconds: 5\n"
        "    cooldown_seconds: 5\n"
        "    say: 'Hello there.'\n",
        encoding="utf-8",
    )
    loaded = load_strategies(path)
    by_id = {s.id: s for s in loaded}
    overridden = by_id["ambient_orient"]
    assert overridden.order == 9
    assert overridden.enabled is False
    assert overridden.dwell_seconds == 5.0
    assert overridden.say_template == "Hello there."
    # A strategy not mentioned in the yaml keeps its code default untouched.
    default_greeting = next(s for s in DEFAULT_STRATEGIES if s.id == "soft_greeting")
    assert by_id["soft_greeting"] == default_greeting


def test_load_strategies_ignores_unknown_id(tmp_path):
    path = tmp_path / "strategies.yaml"
    path.write_text(
        "strategies:\n  - id: not_a_real_strategy\n    enabled: false\n", encoding="utf-8"
    )
    loaded = load_strategies(path)
    assert "not_a_real_strategy" not in {s.id for s in loaded}
    assert len(loaded) == len(DEFAULT_STRATEGIES)


def test_load_strategies_uses_env_path(tmp_path):
    path = tmp_path / "custom-strategies.yaml"
    path.write_text("strategies:\n  - id: ambient_orient\n    enabled: false\n", encoding="utf-8")
    loaded = load_strategies(env={"STRATEGIES_PATH": str(path)})
    by_id = {s.id: s for s in loaded}
    assert by_id["ambient_orient"].enabled is False


def test_load_strategies_reads_familiar_voice_clip_id(tmp_path):
    path = tmp_path / "strategies.yaml"
    path.write_text(
        "strategies:\n  - id: familiar_voice\n    enabled: true\n    clip_id: family-message\n",
        encoding="utf-8",
    )

    familiar = next(s for s in load_strategies(path) if s.id == FAMILIAR_VOICE_ID)

    assert familiar.enabled is True
    assert familiar.clip_id == "family-message"


# --- render_template / time_as_words -------------------------------------


def test_render_template_fills_known_placeholders():
    text = render_template(
        "Hello {name}, it's {time_words}.",
        PersonProfile(name="Jean"),
        time_words="3 o'clock at night",
    )
    assert text == "Hello Jean, it's 3 o'clock at night."


def test_render_template_prefers_configured_form_of_address():
    profile = PersonProfile(name="Jean Smith", preferred_address="Mum")
    assert render_template("Hello {name}.", profile) == "Hello Mum."
    assert render_template("Let's rest{name_vocative}.", profile) == "Let's rest, Mum."


def test_render_template_degrades_missing_field_instead_of_showing_placeholder():
    text = render_template("Hi {unknown_field}!", DEFAULT_PROFILE)
    assert "{" not in text
    assert "}" not in text


def test_render_template_uses_default_profile_safely():
    text = render_template("Hello {name}.", DEFAULT_PROFILE)
    assert text == f"Hello {DEFAULT_PROFILE.name}."
    assert "{name}" not in text


def test_time_as_words_is_a_fixed_phrase():
    assert time_as_words(datetime(2026, 1, 1, 3, 0)) == "3 o'clock at night"
    assert time_as_words(datetime(2026, 1, 1, 0, 0)) == "12 o'clock at night"
    assert time_as_words(datetime(2026, 1, 1, 13, 0)) == "1 o'clock at night"


# --- {name_vocative}: natural degradation, name-set and name-unset -------


def test_name_vocative_is_empty_when_no_name_is_configured():
    text = render_template("Let's rest now{name_vocative}.", DEFAULT_PROFILE)
    assert text == "Let's rest now."
    assert "there" not in text
    assert "{name_vocative}" not in text


def test_name_vocative_includes_the_name_when_configured():
    text = render_template("Let's rest now{name_vocative}.", NAMED_PROFILE)
    assert text == "Let's rest now, Jean."


def test_name_vocative_mid_sentence_both_ways():
    unset = render_template("It's alright{name_vocative}, let's rest.", DEFAULT_PROFILE)
    assert unset == "It's alright, let's rest."
    named = render_template("It's alright{name_vocative}, let's rest.", NAMED_PROFILE)
    assert named == "It's alright, Jean, let's rest."


# --- FIX2: every shipped template must be able to speak -------------------


def _rendered_texts(strategy, profile):
    """The three renderable fields of one `StrategyDef`/parsed strategy
    entry, with falsy ones (no headline/body text, no `say` at all --
    both legitimate "nothing to show/say" states) dropped rather than
    treated as failures."""
    extra = {"time_words": time_as_words(NOW)}
    texts = []
    for template in (strategy.headline_template, strategy.body_template, strategy.say_template):
        if not template:
            continue
        texts.append(render_template(template, profile, **extra))
    return texts


@pytest.mark.parametrize("strategy", DEFAULT_STRATEGIES, ids=lambda s: s.id)
@pytest.mark.parametrize("profile", [DEFAULT_PROFILE, NAMED_PROFILE], ids=["unset_name", "named"])
def test_every_default_strategy_template_passes_validate_say(strategy, profile):
    for text in _rendered_texts(strategy, profile):
        result = validate_say(text, seconds_since_last_say=None, min_gap_seconds=8.0)
        assert result.accepted is True, f"{strategy.id!r} produced {text!r}: {result.reason}"


@pytest.mark.parametrize(
    "strategy_id",
    [
        "acknowledge_pain",
        "comfort_pain",
        "acknowledge_progress",
        "acknowledge_feeling",
        "ask_need",
        "caregiver_alerted",
    ],
)
def test_new_reply_phrases_pass_validate_say(strategy_id):
    strategy = next(s for s in DEFAULT_STRATEGIES if s.id == strategy_id)
    for profile in (DEFAULT_PROFILE, NAMED_PROFILE):
        text = render_template(strategy.say_template, profile)
        assert validate_say(text, seconds_since_last_say=None, min_gap_seconds=8).accepted
        ctx = CheckContext(
            profile={"name": profile.name, "caregiver_name": profile.caregiver_name},
            utterance=None,
            time_words="the middle of the night",
            scene_note=None,
            caregiver_phrase_template="",
            avoid_terms=(),
        )
        assert states_clock_time(text, ctx).status == CheckStatus.PASS
        assert conjunction_but(text, ctx).status == CheckStatus.PASS
        assert "remember" not in text.lower()
        assert not any(phrase in text.lower() for phrase in ("you can't", "you're wrong"))


def test_caregiver_alerted_degrades_without_a_caregiver_name():
    strategy = next(s for s in DEFAULT_STRATEGIES if s.id == "caregiver_alerted")
    text = render_template(strategy.say_template, PersonProfile(caregiver_name=""))
    assert text == "I've let someone know, and help is on the way."


@pytest.mark.parametrize("profile", [DEFAULT_PROFILE, NAMED_PROFILE], ids=["unset_name", "named"])
def test_every_example_yaml_strategy_template_passes_validate_say(profile):
    strategies = load_strategies(EXAMPLE_STRATEGIES_PATH)
    assert len(strategies) == len(DEFAULT_STRATEGIES)  # sanity: the file actually parsed
    for strategy in strategies:
        for text in _rendered_texts(strategy, profile):
            result = validate_say(text, seconds_since_last_say=None, min_gap_seconds=8.0)
            assert result.accepted is True, f"{strategy.id!r} produced {text!r}: {result.reason}"


def test_no_shipped_say_reads_as_a_dropped_in_name_for_an_unset_name():
    # The concrete regression from the coordinator's report: with no name
    # configured, no *spoken* text should read like "..., there." mid- or
    # end-of-sentence -- that phrase is only natural as a greeting opener
    # ("Hello there, it's ..."), which is why `soft_greeting` is exempted
    # here (it uses the plain `{name}` placeholder by design, not
    # `{name_vocative}`; see `DEFAULT_STRATEGIES`).
    for strategy in DEFAULT_STRATEGIES:
        if strategy.id == "soft_greeting" or strategy.say_template is None:
            continue
        text = render_template(
            strategy.say_template, DEFAULT_PROFILE, time_words=time_as_words(NOW)
        )
        assert ", there" not in text, f"{strategy.id!r} produced {text!r}"
        assert not text.rstrip(".").endswith(" there"), f"{strategy.id!r} produced {text!r}"


# --- Regression tests for the five strategy-engine review findings --------
#
# Each of these fails against the engine as originally written. They are
# grouped here because they share one theme: the seam between the ordinary
# ladder and escalation, where a bug is invisible in ordinary testing
# because every individual piece behaves correctly on its own.


def test_the_ladder_never_selects_the_terminal_strategy():
    """`escalate_phone` is enabled with a zero cooldown, so before the
    `terminal` flag existed it was simply the sixth rung: running out of
    ordinary strategies selected it, the device told the person help was
    coming, and no caregiver had been notified at all. It must only ever
    arrive through `force`."""
    engine = StrategyEngine(list(DEFAULT_STRATEGIES))
    now = NOW
    seen = []
    current = engine.start(now)
    assert current is not None
    seen.append(current.id)
    for _ in range(20):
        now += timedelta(seconds=600)
        new_current, _changed, exhausted = engine.maybe_advance(now)
        if exhausted:
            break
        assert new_current is not None
        seen.append(new_current.id)
    assert ESCALATE_PHONE_ID not in seen
    assert seen == [
        "ambient_orient",
        "soft_greeting",
        "orient_time_place",
        "validate_and_redirect",
        "guided_return",
    ]


def test_path_light_is_goal_only_and_selectable_for_restroom():
    engine = StrategyEngine(list(DEFAULT_STRATEGIES))
    selected = engine.select_for_goal(PATH_LIGHT_ID, NOW)
    assert selected is not None
    assert selected.id == PATH_LIGHT_ID
    assert selected.goal_only is True
    current, changed, exhausted = engine.maybe_advance(NOW + timedelta(hours=1))
    assert current == selected
    assert changed is False
    assert exhausted is False


def test_running_out_of_ordinary_strategies_reports_exhausted():
    """The real catalogue, not a synthetic one. `escalate_phone` used to
    keep `_first_available_id` non-empty forever, which made the whole
    exhaustion branch unreachable in production while the synthetic
    fixtures kept it green."""
    engine = StrategyEngine(list(DEFAULT_STRATEGIES))
    now = NOW
    engine.start(now)
    exhausted = False
    for _ in range(20):
        now += timedelta(seconds=600)
        _current, _changed, exhausted = engine.maybe_advance(now)
        if exhausted:
            break
    assert exhausted is True
    assert engine.current() is None


def test_force_still_reaches_the_terminal_strategy():
    """The other half of the above: skipping terminal strategies in the
    ladder must not make them unreachable."""
    engine = StrategyEngine(list(DEFAULT_STRATEGIES))
    assert engine.force(ESCALATE_PHONE_ID, NOW) is not None
    assert engine.current_id == ESCALATE_PHONE_ID


def test_the_escalation_strategy_is_visible_and_awake():
    """It was `asleep` at brightness 0.15: dimmer and sleepier than every
    rung before it, at the one moment the person most needs to see the
    screen."""
    escalate = next(s for s in DEFAULT_STRATEGIES if s.id == ESCALATE_PHONE_ID)
    guided = next(s for s in DEFAULT_STRATEGIES if s.id == "guided_return")
    assert escalate.face == "speaking"
    assert escalate.brightness >= guided.brightness
    assert escalate.terminal is True


def test_progress_cannot_extend_a_strategy_forever():
    """Standing in the bed zone without getting in reports progress on
    every reading. Unbounded, that pinned the ladder on one strategy for
    the whole night."""
    engine = StrategyEngine(strategies_by_order(dwell=10.0, cooldown=1000.0))
    engine.start(NOW)
    assert engine.current_id == "a"

    # Progress every second, well past the three-dwell cap.
    advanced_at = None
    for i in range(1, 120):
        now = NOW + timedelta(seconds=i)
        engine.note_progress(now)
        _current, changed, _exhausted = engine.maybe_advance(now)
        if changed:
            advanced_at = i
            break
    assert advanced_at is not None, "ladder never advanced despite the cap"
    assert engine.current_id == "b"


def test_progress_still_pauses_the_dwell_up_to_the_cap():
    """The cap must not defeat the point of `note_progress`: a person
    genuinely climbing into bed still gets more than one bare dwell."""
    engine = StrategyEngine(strategies_by_order(dwell=10.0, cooldown=1000.0))
    engine.start(NOW)
    for i in range(1, 12):
        now = NOW + timedelta(seconds=i)
        engine.note_progress(now)
        _current, changed, _exhausted = engine.maybe_advance(now)
        assert changed is False, f"advanced at {i}s despite continuous progress"
    assert engine.current_id == "a"


@pytest.mark.parametrize("bad", ["Hello {name", "It is {0} now", "{}"])
def test_a_malformed_template_renders_empty_instead_of_raising(bad):
    """`_SafeFormatDict` only rescues unknown placeholder *names*. A
    template malformed as a format string still raised out of
    `format_map`, and `agent.main`'s loop has no handler, so a caregiver's
    typo crashed the service mid-session."""
    assert render_template(bad, DEFAULT_PROFILE) == ""


def test_a_malformed_template_in_yaml_keeps_the_code_default(tmp_path):
    """Better still: rejected at load time, while a good default is still
    available to fall back to."""
    path = tmp_path / "strategies.yaml"
    path.write_text('strategies:\n  - id: soft_greeting\n    say: "Hello {name"\n')
    loaded = {s.id: s for s in load_strategies(path)}
    default = next(s for s in DEFAULT_STRATEGIES if s.id == "soft_greeting")
    assert loaded["soft_greeting"].say_template == default.say_template


def test_a_valid_template_in_yaml_still_overrides(tmp_path):
    """The load-time check must reject only genuinely malformed templates,
    not ordinary caregiver edits."""
    path = tmp_path / "strategies.yaml"
    path.write_text('strategies:\n  - id: soft_greeting\n    say: "Good evening{name_vocative}."\n')
    loaded = {s.id: s for s in load_strategies(path)}
    assert loaded["soft_greeting"].say_template == "Good evening{name_vocative}."
