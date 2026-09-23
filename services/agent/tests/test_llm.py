"""Tests for the local, bounded LLM seam; no Ollama or network is needed."""

import json
from types import SimpleNamespace

import pytest
from pydantic import ValidationError

from agent.llm import (
    ClaudeLLM,
    Composition,
    FakeLLM,
    FallbackLLM,
    Intent,
    Interpretation,
    OllamaLLM,
    OpenAICompatibleLLM,
    Plan,
    local_llm,
)


class _Response:
    def __init__(self, body: object, status: int = 200):
        self._body = body
        self.status = status

    def read(self):
        return json.dumps(self._body).encode()

    def __enter__(self):
        return self

    def __exit__(self, *_):
        return False


def test_ollama_interpret_posts_schema_and_parses_strict_json(monkeypatch):
    captured = {}

    def fake_open(request, timeout):
        captured["request"] = request
        captured["timeout"] = timeout
        return _Response({"response": '{"intent":"pain","distress":2}'})

    monkeypatch.setattr("agent.llm.urlopen", fake_open)
    client = OllamaLLM(ollama_url="http://ollama:11434", model="test", timeout_seconds=0.5)
    result = client.interpret("It hurts", ["one", "two", "three", "four"], {"name": "Jean"})

    assert result == Interpretation(intent=Intent.PAIN, distress=2)
    payload = json.loads(captured["request"].data)
    assert captured["request"].full_url == "http://ollama:11434/api/generate"
    assert captured["timeout"] == 0.5
    assert payload["model"] == "test"
    assert payload["stream"] is False
    assert payload["keep_alive"] == -1
    assert "properties" in payload["format"]
    # The allowed intents are spelled out, not only enforced by `format`.
    assert "JSON schema:" in payload["prompt"] and "need_restroom" in payload["prompt"]
    assert "wants_bed" in payload["prompt"]
    assert "back from the restroom" in payload["prompt"]
    assert '"last_turns": ["two", "three", "four"]' in payload["prompt"]


def test_ollama_rejects_invalid_or_extra_model_json(monkeypatch):
    monkeypatch.setattr(
        "agent.llm.urlopen",
        lambda *_: _Response({"response": '{"intent":"pain","distress":"2"}'}),
    )
    assert OllamaLLM(ollama_url="http://ollama").interpret("help", [], {}) is None

    monkeypatch.setattr(
        "agent.llm.urlopen",
        lambda *_: _Response({"response": '{"intent":"pain","distress":2,"extra":true}'}),
    )
    assert OllamaLLM(ollama_url="http://ollama").interpret("help", [], {}) is None


def _chat(content):
    return _Response({"choices": [{"message": {"role": "assistant", "content": content}}]})


def test_openai_compatible_posts_chat_with_schema_and_no_thinking(monkeypatch):
    captured = {}

    def fake_open(request, timeout):
        captured["request"] = request
        captured["timeout"] = timeout
        return _chat('<think>\n</think>\n```json\n{"intent":"pain","distress":2}\n```')

    monkeypatch.setattr("agent.llm.urlopen", fake_open)
    client = OpenAICompatibleLLM(base_url="http://mlx:8080/", model="m", timeout_seconds=0.5)
    result = client.interpret("It hurts", ["one", "two", "three", "four"], {"name": "Jean"})

    assert result == Interpretation(intent=Intent.PAIN, distress=2)
    payload = json.loads(captured["request"].data)
    assert captured["request"].full_url == "http://mlx:8080/v1/chat/completions"
    assert captured["timeout"] == 0.5
    assert payload["model"] == "m"
    assert payload["temperature"] == 0
    assert payload["chat_template_kwargs"] == {"enable_thinking": False}
    prompt = payload["messages"][0]["content"]
    assert '"last_turns": ["two", "three", "four"]' in prompt
    assert "JSON schema:" in prompt and '"distress"' in prompt


def test_openai_compatible_is_as_strict_as_ollama(monkeypatch):
    client = OpenAICompatibleLLM(base_url="http://mlx", model="m")
    for content in (
        '{"intent":"pain","distress":"2"}',
        '{"intent":"pain","distress":2,"extra":true}',
        "I think the person is in pain.",
        "",
        None,
    ):
        monkeypatch.setattr("agent.llm.urlopen", lambda *_, c=content: _chat(c))
        assert client.interpret("help", [], {}) is None
    monkeypatch.setattr("agent.llm.urlopen", lambda *_: _Response({"error": "boom"}))
    assert client.interpret("help", [], {}) is None
    monkeypatch.setattr(
        "agent.llm.urlopen", lambda *_: _chat('{"text": "No. You are wrong. Sit."}')
    )
    assert client.compose("s", "t", {}, "3 o'clock", None, "hi") is None


def test_compose_sends_goal_and_asks_for_a_reply_to_the_utterance(monkeypatch):
    captured = {}

    def fake_open(request, timeout):
        captured["request"] = request
        return _Response({"response": '{"text": "Jean, the restroom is just through the door."}'})

    monkeypatch.setattr("agent.llm.urlopen", fake_open)
    result = OllamaLLM(ollama_url="http://ollama").compose(
        "validate_and_redirect",
        "It's alright, Jean, let's rest now.",
        {"preferred_address": "Jean"},
        "3 o'clock at night",
        None,
        "I need the toilet",
        "restroom",
    )

    assert result == Composition(text="Jean, the restroom is just through the door.")
    prompt = json.loads(captured["request"].data)["prompt"]
    assert '"goal": "restroom"' in prompt
    assert "do not copy it word for word" in prompt
    assert "acknowledge what latest_utterance is about" in prompt
    assert "calming_things may be repeated as written" in prompt
    assert "latest_utterance or scene_note says so" in prompt


def test_local_llm_selects_backend():
    assert isinstance(local_llm("ollama", url="u", model="m", timeout_seconds=1), OllamaLLM)
    assert isinstance(
        local_llm("openai", url="u", model="m", timeout_seconds=1), OpenAICompatibleLLM
    )
    with pytest.raises(ValueError):
        local_llm("vllm", url="u", model="m", timeout_seconds=1)


def test_composition_and_plan_enforce_bounded_output_shapes():
    assert Composition(text="Let's sit down together.").text
    with pytest.raises(ValidationError):
        Composition(text="One sentence. Another sentence.")
    with pytest.raises(ValidationError):
        Plan(next_strategy="guided_return", goal_change="comfort", confidence=0.7)
    # JSON has a single number type; an integral confidence at the valid
    # upper bound is accepted and normalised to a float.
    assert Plan(next_strategy="guided_return", confidence=1).confidence == 1.0


def test_fake_llm_is_scriptable_without_ollama_and_records_calls():
    fake = FakeLLM(
        interpretations=[Interpretation(intent=Intent.NEED_RESTROOM, distress=1)],
        compositions=[Composition(text="Let's walk to the bathroom together.")],
        plans=[Plan(goal_change="restroom", confidence=0.8)],
    )

    result = fake.interpret("I need the toilet", [], {"name": "Jean"})
    assert result is not None
    assert result.intent == Intent.NEED_RESTROOM
    composition = fake.compose(
        "validate_and_redirect", "", {}, "two at night", None, "Where is my mother"
    )
    assert composition is not None
    assert composition.text.startswith("Let's")
    assert fake.calls[1][1]["latest_utterance"] == "Where is my mother"
    # This is deliberately only an advisory value; this test does not and
    # cannot cause a session change without Session.propose_goal/rules.
    plan = fake.plan({"goal": "return_to_bed"}, {"name": "Jean"})
    assert plan is not None
    assert plan.goal_change == "restroom"
    assert [name for name, _ in fake.calls] == ["interpret", "compose", "plan"]
    assert fake.plan({}, {}) is None


class _Messages:
    def __init__(self, text: str):
        self.text = text
        self.calls = []

    def create(self, **kwargs):
        self.calls.append(kwargs)
        return SimpleNamespace(content=[SimpleNamespace(type="text", text=self.text)])


def test_claude_uses_official_messages_shape_and_audits_exact_text_payload():
    messages = _Messages('{"intent":"need_restroom","distress":1}')
    recorded = []
    client = SimpleNamespace(messages=messages)
    cloud = ClaudeLLM(
        api_key="test",
        client=client,
        on_call=lambda task, model, payload: recorded.append((task, model, payload)),
    )

    result = cloud.interpret("I need the toilet", ["hello"], {"name": "Jean"})

    assert result == Interpretation(intent=Intent.NEED_RESTROOM, distress=1)
    assert recorded[0][0:2] == ("interpret", "claude-opus-5")
    assert recorded[0][2]["input"]["utterance"] == "I need the toilet"
    request = messages.calls[0]
    assert json.loads(request["messages"][0]["content"]) == recorded[0][2]
    assert request["output_config"]["format"]["type"] == "json_schema"


def test_fallback_interpret_waits_for_two_consecutive_local_unclear_results():
    local = FakeLLM(
        interpretations=[
            Interpretation(intent=Intent.UNCLEAR, distress=0),
            Interpretation(intent=Intent.UNCLEAR, distress=0),
        ]
    )
    cloud = FakeLLM(interpretations=[Interpretation(intent=Intent.NEED_RESTROOM, distress=1)])
    llm = FallbackLLM(local, cloud)

    assert llm.interpret("maybe", [], {}).intent == Intent.UNCLEAR
    assert cloud.calls == []
    assert llm.interpret("the toilet", ["maybe"], {}).intent == Intent.NEED_RESTROOM
    assert [name for name, _payload in cloud.calls] == ["interpret"]


def test_clear_interpretation_resets_the_cloud_threshold():
    local = FakeLLM(
        interpretations=[
            Interpretation(intent=Intent.UNCLEAR, distress=0),
            Interpretation(intent=Intent.FINE, distress=0),
            Interpretation(intent=Intent.UNCLEAR, distress=0),
        ]
    )
    cloud = FakeLLM()
    llm = FallbackLLM(local, cloud)

    for text in ("one", "two", "three"):
        llm.interpret(text, [], {})
    assert cloud.calls == []


def test_fallback_plan_uses_cloud_only_below_point_four_and_never_for_compose():
    local = FakeLLM(
        compositions=[Composition(text="Let's rest here together.")],
        plans=[
            Plan(next_strategy="guided_return", confidence=0.39),
            Plan(next_strategy="soft_greeting", confidence=0.4),
        ],
    )
    cloud = FakeLLM(plans=[Plan(goal_change="restroom", confidence=0.9)])
    llm = FallbackLLM(local, cloud)

    assert llm.compose("validate_and_redirect", "", {}, "two at night", None) is not None
    assert cloud.calls == []
    assert llm.plan({}, {}).goal_change == "restroom"
    assert llm.plan({}, {}).next_strategy == "soft_greeting"
    assert [name for name, _payload in cloud.calls] == ["plan"]


def test_ollama_warm_up_loads_the_model_and_keeps_it(monkeypatch):
    captured = {}

    def fake_open(request, timeout):
        captured["payload"] = json.loads(request.data)
        return _Response({"done": True})

    monkeypatch.setattr("agent.llm.urlopen", fake_open)

    assert OllamaLLM(ollama_url="http://ollama", model="m").warm_up() is True
    assert captured["payload"] == {"model": "m", "keep_alive": -1}


def test_ollama_warm_up_failure_is_harmless(monkeypatch):
    def fail(request, timeout):
        raise OSError("down")

    monkeypatch.setattr("agent.llm.urlopen", fail)

    assert OllamaLLM(ollama_url="http://ollama").warm_up() is False
