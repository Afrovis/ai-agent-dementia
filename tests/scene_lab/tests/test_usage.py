"""Per-call Claude usage rows and the per-role summary."""

import asyncio
import json

from scene_lab import usage
from scene_lab.mind import ClaudeMind
from scene_lab.scene import Persona

LIMIT = (
    'claude exited with status 1: {"is_error": true, "result": "You\'ve hit your session '
    'limit", "usage": {"input_tokens": 82, "cache_read_input_tokens": 4791179, '
    '"cache_creation_input_tokens": 167854, "output_tokens": 24654, '
    '"output_tokens_details": {"thinking_tokens": 15888}}, "modelUsage": '
    '{"claude-opus-5-5": {}}, "total_cost_usd": 2.79, "num_turns": 44}'
)


def test_failed_call_keeps_the_usage_in_its_error(tmp_path):
    log = tmp_path / "usage.jsonl"
    usage.record(log, "triage", error=LIMIT, requested_model="opus")
    row = json.loads(log.read_text())
    assert row["ok"] is False and row["model"] == "claude-opus-5-5"
    assert row["cache_read"] == 4791179 and row["thinking"] == 15888 and row["turns"] == 44
    table = usage.summarize(log)
    assert "| triage | claude-opus-5-5 | 1 | 1 |" in table and "4,791,179" in table


def test_mind_logs_each_call(tmp_path):
    log = tmp_path / "usage.jsonl"

    def runner(system, prompt, schema, model, effort):
        return {
            "structured_output": {"beats": [{"wait": 5}], "end_scene": False, "note": "n"},
            "usage": {"input_tokens": 900, "output_tokens": 40},
            "model_usage": {"claude-sonnet-5": {}},
        }

    mind = ClaudeMind(Persona(summary="x"), runner=runner, usage_log=log, scene="s-1")
    asyncio.run(mind.decide({"t": 0}))
    row = json.loads(log.read_text())
    assert row["role"] == "mind" and row["scene"] == "s-1" and row["input"] == 900
