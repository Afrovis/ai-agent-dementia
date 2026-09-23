"""Director and batch checks with no Claude, Docker, or clock waiting."""

import asyncio
import json

from scene_lab.bugs import RunDir, harness_error, merge, render_merge_md
from scene_lab.director import Director
from scene_lab.run import run_hours
from scene_lab.scene import Scene


def card(n=1):
    return {
        "id": f"conversation-question-{n}",
        "category": "conversation",
        "persona_tag": "repeater",
        "start": "02:40",
        "duration_s": 60,
        "profile": "config/person.example.yaml",
        "strategies": "default",
        "persona": {"summary": "An adult asking for reassurance"},
        "opening": [{"at": 0, "say": "Is Tom here?"}],
        "stressors": ["repeated_question"],
        "mind": "claude",
    }


def test_director_valid_retry_and_prompt(tmp_path):
    run = RunDir("live", root=tmp_path)
    prompts = []
    answers = [
        {"scene": {**card(), "duration_s": 700}, "rationale": "test"},
        {"scene": card(), "rationale": "cover repetition"},
    ]

    def runner(system, prompt, schema, model, effort):
        prompts.append(prompt)
        assert model == "opus" and effort == "low" and system.exists()
        return {"structured_output": answers.pop(0)}

    state = {
        "run": run,
        "time_left_s": 900,
        "scenes": [
            {
                "id": "old",
                "category": "restroom",
                "persona_tag": "repeater",
                "stressors": ["repeated_question"],
                "noise": "none",
                "failures": [{"id": "TT-1", "severity": "major", "fingerprint": "TT-1|-|-|-|-"}],
                "harness_errors": [],
            }
        ],
        "bugs": {"TT-1|-|-|-|-": 2},
    }
    result = Director(runner=runner).next_scene(state)
    assert isinstance(result, Scene) and result.id == "conversation-question-1"
    assert "time_left_s" in prompts[0] and "900" in prompts[0]
    assert "TT-1" in prompts[0] and "counts" in prompts[0]
    assert "Validation error" in prompts[1]
    assert "cover repetition" in (run.path / "director.jsonl").read_text()


def test_director_fallback_logs_harness(tmp_path):
    run = RunDir("live", root=tmp_path)
    result = Director(runner=lambda *_: {"structured_output": {"bad": True}}).next_scene(
        {"run": run, "time_left_s": 900, "scenes": [], "bugs": {}}
    )
    assert result.mind == "claude" and result.opening
    assert "director failure" in (run.path / "bugs.jsonl").read_text()


class FakeClock:
    def __init__(self):
        self.t = 0

    def __call__(self):
        return self.t


class FakeStack:
    instances = []

    def __init__(self, root, env):
        self.env = env
        self.calls = []
        self.instances.append(self)

    def preflight(self):
        self.calls.append("preflight")
        return {"commit": "abc", "contention": False}

    def up(self):
        self.calls.append("up")

    def wait_ready(self):
        self.calls.append("ready")

    def down(self):
        self.calls.append("down")

    def unload_llm(self, model):
        self.calls.append("unload")
        return True

    def reset_llm(self, model):
        self.calls.append("reset")
        return {"unloaded": True, "loaded": True, "seconds": 0.0}


class FakeDirector:
    def __init__(self):
        self.calls = 0

    def next_scene(self, state):
        self.calls += 1
        return Scene.model_validate(card(self.calls))


def test_hours_deadline_and_index(tmp_path):
    clock = FakeClock()
    director = FakeDirector()

    async def runner(scene, source, run, stack, preflight, thresholds):
        assert source.exists() and clock.t + scene.duration_s <= 1200
        clock.t += 350
        return []

    path = asyncio.run(
        run_hours(
            1200 / 3600,
            runs_root=tmp_path,
            director=director,
            scene_runner=runner,
            stack_factory=FakeStack,
            triage_after=False,
            clock=clock,
        )
    )
    assert director.calls == 2
    assert len(list((path / "scenes").glob("*.yaml"))) == 2
    assert json.loads((tmp_path / "index.jsonl").read_text())["scene_count"] == 2
    # Unload before the stack starts, reset Ollama's cache before every scene
    # after the first, and leave nothing loaded at the end.
    assert FakeStack.instances[-1].calls == [
        "preflight",
        "unload",
        "up",
        "ready",
        "reset",
        "down",
        "unload",
    ]


def test_hours_interrupt_finishes(tmp_path):
    clock = FakeClock()
    director = FakeDirector()

    async def runner(scene, source, run, stack, preflight, thresholds):
        raise KeyboardInterrupt

    path = asyncio.run(
        run_hours(
            1,
            runs_root=tmp_path,
            director=director,
            scene_runner=runner,
            stack_factory=FakeStack,
            triage_after=False,
            clock=clock,
        )
    )
    assert director.calls == 1
    assert 'scene_count": 1' in (tmp_path / "index.jsonl").read_text()
    assert "KeyboardInterrupt" in (path / "bugs.md").read_text()


def test_hours_rechecks_deadline_after_director(tmp_path):
    clock = FakeClock()

    class SlowDirector(FakeDirector):
        def next_scene(self, state):
            clock.t += 101
            return super().next_scene(state)

    async def runner(*args):
        raise AssertionError("scene must not start with less than 600 seconds left")

    path = asyncio.run(
        run_hours(
            700 / 3600,
            runs_root=tmp_path,
            director=SlowDirector(),
            scene_runner=runner,
            stack_factory=FakeStack,
            triage_after=False,
            clock=clock,
        )
    )
    assert not (path / "scenes").exists()
    assert json.loads((tmp_path / "index.jsonl").read_text())["scene_count"] == 0


def test_bugs_context_and_three_run_merge(tmp_path):
    paths = []
    for n, checks in enumerate((["TT-1", "TT-3"], ["TT-1", "TT-2"], ["TT-1", "TT-2"])):
        path = tmp_path / f"2026-09-2{n + 1}T2200-live"
        path.mkdir()
        run = RunDir("live", root=tmp_path, path=path)
        entries = []
        for check in checks:
            item = harness_error(run.id, "scene", "first summary")
            item.check = check
            item.origin = "agent"
            item.fingerprint = f"{check}|ENGAGED|return_to_bed|orient|strategy_changed"
            item.evidence = ["Utterance at 12s", "pending say dropped at 18s"]
            entries.append(item)
        run.metadata = {
            "kind": "live",
            "commit": "abc",
            "model": "gemma",
            "contention": False,
            "scene_count": 1,
        }
        run.append(entries)
        paths.append(path)
    md = (paths[0] / "bugs.md").read_text()
    assert "phase: ENGAGED" in md and "drop reason: strategy_changed" in md
    assert "Evidence: Utterance at 12s" in md and "scenes run: 1" in md
    rows = merge(paths)
    statuses = {row["fingerprint"].split("|")[0]: row["status"] for row in rows}
    assert statuses == {"TT-1": "persisting", "TT-2": "new", "TT-3": "gone"}
    assert "pending say dropped at 18s" in render_merge_md(rows)


def test_hours_director_timeout_uses_fallback(tmp_path, monkeypatch):
    import time

    import scene_lab.run as run_module

    monkeypatch.setattr(run_module, "DIRECTOR_TIMEOUT_S", 0.05)
    clock = FakeClock()

    class HungDirector(FakeDirector):
        fallbacks = 0

        def next_scene(self, state):
            time.sleep(0.3)
            return super().next_scene(state)

        def fallback(self, state, error):
            assert "timed out" in error
            self.fallbacks += 1
            return Scene.model_validate(card(100 + self.fallbacks))

    director = HungDirector()

    async def runner(scene, source, run, stack, preflight, thresholds):
        clock.t += 700
        return []

    asyncio.run(
        run_hours(
            700 / 3600,
            runs_root=tmp_path,
            director=director,
            scene_runner=runner,
            stack_factory=FakeStack,
            triage_after=False,
            clock=clock,
        )
    )
    assert director.fallbacks == 1
    assert json.loads((tmp_path / "index.jsonl").read_text())["scene_count"] == 1


def test_director_schema_references_resolve_from_the_root():
    from scene_lab.director import output_schema

    schema = output_schema()
    refs = set()

    def walk(node):
        if isinstance(node, dict):
            if "$ref" in node:
                refs.add(node["$ref"])
            for value in node.values():
                walk(value)
        elif isinstance(node, list):
            for value in node:
                walk(value)

    walk(schema)
    assert refs, "Scene should reference nested models"
    for ref in refs:
        node = schema
        for part in ref.removeprefix("#/").split("/"):
            node = node[part]


def test_usage_limit_is_recognised():
    from scene_lab.bugs import usage_limited

    message = (
        'claude exited with status 1: {"api_error_status":429,'
        '"result":"You\'ve hit your session limit · resets 3:30am (America/New_York)"}'
    )
    assert usage_limited(message) == "resets 3:30am (America/New_York)"
    assert usage_limited("claude exited with status 1: bad JSON") is None
    assert usage_limited(None) is None


def test_hours_stops_on_director_usage_limit(tmp_path):
    clock = FakeClock()

    class LimitedDirector(FakeDirector):
        last_error = None

        def next_scene(self, state):
            self.last_error = '"api_error_status":429 You\'ve hit your session limit'
            return super().next_scene(state)

    async def runner(*args):
        raise AssertionError("no scene may start once the usage limit is reached")

    path = asyncio.run(
        run_hours(
            1,
            runs_root=tmp_path,
            director=LimitedDirector(),
            scene_runner=runner,
            stack_factory=FakeStack,
            triage_after=False,
            clock=clock,
        )
    )
    assert "usage limit" in (path / "bugs.md").read_text()
    assert json.loads((tmp_path / "index.jsonl").read_text())["scene_count"] == 0


def test_hours_writes_a_fix_list_at_the_end(tmp_path):
    clock = FakeClock()
    calls = []

    async def runner(scene, source, run, stack, preflight, thresholds):
        clock.t += 700
        return []

    def fake_triage(run_dir, repo_root, commit, *, quick_tests):
        calls.append((run_dir, commit, quick_tests))
        (run_dir / "fixes.md").write_text("# Fix list\n")
        return run_dir / "fixes.md"

    path = asyncio.run(
        run_hours(
            1200 / 3600,
            runs_root=tmp_path,
            director=FakeDirector(),
            scene_runner=runner,
            stack_factory=FakeStack,
            clock=clock,
            triage_quick_tests=False,
            triage_fn=fake_triage,
        )
    )
    assert calls == [(path, "abc", False)]
    assert (path / "fixes.md").exists()


def test_hours_skips_the_fix_list_after_the_usage_limit(tmp_path):
    class LimitedDirector(FakeDirector):
        def next_scene(self, state):
            if self.calls:
                self.last_error = '"api_error_status":429 You\'ve hit your session limit'
            return super().next_scene(state)

    clock = FakeClock()

    async def runner(scene, source, run, stack, preflight, thresholds):
        clock.t += 10
        return []

    def fake_triage(*args, **kwargs):
        raise AssertionError("no triage once the usage limit is reached")

    asyncio.run(
        run_hours(
            1,
            runs_root=tmp_path,
            director=LimitedDirector(),
            scene_runner=runner,
            stack_factory=FakeStack,
            clock=clock,
            triage_fn=fake_triage,
        )
    )
