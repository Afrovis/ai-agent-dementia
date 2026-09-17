"""Speed-test one realistic Night Companion turn across local LLM backends.

The task is a single bench scenario (default ``restroom-01``) pushed through the
agent's real ``interpret`` and ``compose`` prompts.  Prompts and output schemas
are captured from ``agent.llm.OllamaLLM`` itself, so they never drift from
production; outputs are checked with the same Pydantic models and
``agent.rules.validate_say``.

Model specs:
  mlx:<hf-repo>        mlx-lm (text models)
  mlx-vlm:<hf-repo>    mlx-vlm, text-only (e.g. Qwen3-VL)
  ollama:<tag>         Ollama /api/generate, same payload as the agent

Each spec runs in its own subprocess so memory is released between models.

  .venv-mlx/bin/python tools/llm_speedtest/speedtest.py \
      mlx-vlm:mlx-community/Qwen3-VL-8B-Instruct-4bit ollama:qwen3-vl:8b
"""

from __future__ import annotations

import argparse
import json
import os
import re
import statistics
import subprocess
import sys
import time
import urllib.request
import uuid
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from agent.llm import Composition, Interpretation, OllamaLLM, _prompt
from agent.rules import validate_say
from dialogue_bench.scenarios import load_scenarios

OLLAMA_URL = os.environ.get("OLLAMA_URL", "http://localhost:11434")
MAX_TOKENS = 128
RESULTS_DIR = Path(__file__).parent / "results"


@dataclass
class Call:
    task: str
    prompt: str
    output_type: type


def capture_calls(scenario_id: str) -> tuple[Any, list[Call]]:
    """Record the exact prompts the agent would send for one scenario."""
    calls: list[Call] = []

    class Recorder(OllamaLLM):
        def _call(self, task, payload, output_type):  # type: ignore[override]
            calls.append(Call(task, _prompt(task, payload), output_type))

    scenario = next(s for s in load_scenarios() if s.id == scenario_id)
    rec = Recorder()
    rec.interpret(scenario.utterance, scenario.turns, scenario.profile)
    rec.compose(
        "validate_and_redirect",
        scenario.caregiver_phrase_template,
        scenario.profile,
        scenario.time_words,
        scenario.scene_note,
        scenario.utterance,
    )
    return scenario, calls


def with_schema(call: Call) -> str:
    # Ollama enforces the schema via `format`; MLX has no constrained decoding
    # here, so the schema goes into the prompt instead.
    schema = json.dumps(call.output_type.model_json_schema(), separators=(",", ":"))
    return f"{call.prompt}\nJSON schema: {schema}"


@dataclass
class CallResult:
    task: str
    prompt_tokens: int = 0
    gen_tokens: int = 0
    ttft_s: float = 0.0
    gen_tps: float = 0.0
    total_s: float = 0.0
    text: str = ""
    note: str = ""


def fresh(call: Call) -> Call:
    """Unique prompt head, so no backend can serve the turn from a prefix cache."""
    return Call(
        call.task, f"Request {uuid.uuid4().hex[:8]}.\n{call.prompt}", call.output_type
    )


# --- backends ---------------------------------------------------------------


class MLXBackend:
    def __init__(self, repo: str) -> None:
        from mlx_lm import load

        self.tokenizer_kw = {"enable_thinking": False}
        self.model, self.tokenizer = load(repo)

    def run(self, call: Call) -> CallResult:
        import mlx.core as mx
        from mlx_lm import stream_generate
        from mlx_lm.sample_utils import make_sampler

        messages = [{"role": "user", "content": with_schema(call)}]
        prompt = self.tokenizer.apply_chat_template(
            messages, add_generation_prompt=True, tokenize=False, **self.tokenizer_kw
        )
        out = CallResult(call.task)
        started = time.perf_counter()
        last = None
        for resp in stream_generate(
            self.model,
            self.tokenizer,
            prompt,
            max_tokens=MAX_TOKENS,
            sampler=make_sampler(temp=0.0),
        ):
            if not out.ttft_s:
                out.ttft_s = time.perf_counter() - started
            out.text += resp.text
            last = resp
        out.total_s = time.perf_counter() - started
        if last is not None:
            out.prompt_tokens = last.prompt_tokens
            out.gen_tokens = last.generation_tokens
            out.gen_tps = last.generation_tps
        self.peak_gb = mx.get_peak_memory() / 1e9
        return out


class MLXVLMBackend:
    def __init__(self, repo: str) -> None:
        from mlx_vlm import load

        self.model, self.processor = load(repo)

    def run(self, call: Call) -> CallResult:
        import mlx.core as mx
        from mlx_vlm import stream_generate
        from mlx_vlm.prompt_utils import apply_chat_template

        prompt = apply_chat_template(
            self.processor,
            self.model.config,
            with_schema(call),
            num_images=0,
            enable_thinking=False,
        )
        out = CallResult(call.task)
        started = time.perf_counter()
        last = None
        for resp in stream_generate(
            self.model, self.processor, prompt, max_tokens=MAX_TOKENS, temperature=0.0
        ):
            if not out.ttft_s:
                out.ttft_s = time.perf_counter() - started
            out.text += resp.text
            last = resp
        out.total_s = time.perf_counter() - started
        if last is not None:
            out.prompt_tokens = last.prompt_tokens
            out.gen_tokens = last.generation_tokens
            out.gen_tps = last.generation_tps
        self.peak_gb = mx.get_peak_memory() / 1e9
        return out


def _ollama(path: str, body: dict[str, Any]):
    req = urllib.request.Request(
        OLLAMA_URL + path,
        json.dumps(body).encode(),
        {"Content-Type": "application/json"},
    )
    return urllib.request.urlopen(req, timeout=300)


class OllamaBackend:
    def __init__(self, tag: str) -> None:
        self.tag = tag
        self.peak_gb = 0.0
        # Force a load so load time is measured separately from the task.
        _ollama(
            "/api/generate", {"model": tag, "prompt": "", "keep_alive": "10m"}
        ).read()
        with urllib.request.urlopen(OLLAMA_URL + "/api/ps", timeout=10) as resp:
            for m in json.load(resp).get("models", []):
                if m["name"] == tag:
                    self.peak_gb = m.get("size_vram", m.get("size", 0)) / 1e9

    def run(self, call: Call) -> CallResult:
        body = {
            "model": self.tag,
            "prompt": call.prompt,
            "format": call.output_type.model_json_schema(),
            "stream": True,
            "think": False,
            "keep_alive": "10m",
            "options": {"temperature": 0, "num_predict": MAX_TOKENS},
        }
        out = CallResult(call.task)
        started = time.perf_counter()
        with _ollama("/api/generate", body) as resp:
            for line in resp:
                chunk = json.loads(line)
                piece = chunk.get("response") or chunk.get("thinking") or ""
                if piece and not chunk.get("response"):
                    out.note = (
                        "answer arrived in 'thinking'; the agent reads only 'response'"
                    )
                if piece and not out.ttft_s:
                    out.ttft_s = time.perf_counter() - started
                out.text += piece
                if chunk.get("done"):
                    out.prompt_tokens = chunk.get("prompt_eval_count", 0)
                    out.gen_tokens = chunk.get("eval_count", 0)
                    if chunk.get("eval_duration"):
                        out.gen_tps = out.gen_tokens / (chunk["eval_duration"] / 1e9)
        out.total_s = time.perf_counter() - started
        return out


BACKENDS = {"mlx": MLXBackend, "mlx-vlm": MLXVLMBackend, "ollama": OllamaBackend}


def unload_ollama() -> None:
    try:
        with urllib.request.urlopen(OLLAMA_URL + "/api/ps", timeout=5) as resp:
            names = [m["name"] for m in json.load(resp).get("models", [])]
        for name in names:
            _ollama("/api/generate", {"model": name, "keep_alive": 0}).read()
    except OSError:
        pass


# --- scoring ----------------------------------------------------------------


def parse(text: str, output_type: type):
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL)
    match = re.search(r"\{.*\}", text, flags=re.DOTALL)
    if not match:
        return None
    try:
        return output_type.model_validate_json(match.group(0))
    except ValueError:
        return None


@dataclass
class ModelReport:
    spec: str
    scenario: str
    load_s: float = 0.0
    peak_gb: float = 0.0
    runs: int = 0
    turn_s_median: float = 0.0
    turn_s_min: float = 0.0
    calls: dict[str, dict[str, float]] = field(default_factory=dict)
    intent: str | None = None
    intent_ok: bool = False
    distress: int | None = None
    say: str | None = None
    say_ok: bool = False
    say_reason: str | None = None
    notes: list[str] = field(default_factory=list)
    error: str | None = None


def run_one(spec: str, scenario_id: str, runs: int) -> ModelReport:
    kind, _, name = spec.partition(":")
    scenario, calls = capture_calls(scenario_id)
    report = ModelReport(spec, scenario_id)
    unload_ollama()  # cold start for every spec, and free memory for MLX

    started = time.perf_counter()
    backend = BACKENDS[kind](name)
    report.load_s = time.perf_counter() - started

    for call in calls:  # warm-up: first call pays graph compilation / cache setup
        backend.run(fresh(call))

    per_call: dict[str, list[CallResult]] = {c.task: [] for c in calls}
    turns = []
    for _ in range(runs):
        turn = 0.0
        for call in calls:
            result = backend.run(fresh(call))
            per_call[call.task].append(result)
            turn += result.total_s
        turns.append(turn)

    report.runs = runs
    report.turn_s_median = statistics.median(turns)
    report.turn_s_min = min(turns)
    report.peak_gb = getattr(backend, "peak_gb", 0.0)
    report.notes = sorted({r.note for rs in per_call.values() for r in rs if r.note})
    for (task, results), label in zip(per_call.items(), ("interpret", "compose")):
        report.calls[label] = {
            "prompt_tokens": results[-1].prompt_tokens,
            "gen_tokens": statistics.median(r.gen_tokens for r in results),
            "ttft_s": statistics.median(r.ttft_s for r in results),
            "gen_tps": statistics.median(r.gen_tps for r in results),
            "total_s": statistics.median(r.total_s for r in results),
        }

    interp_call, compose_call = calls
    interp = parse(per_call[interp_call.task][-1].text, Interpretation)
    if interp is not None:
        report.intent = str(interp.intent.value)
        report.distress = interp.distress
        report.intent_ok = interp.intent == scenario.expected_intent
    comp = parse(per_call[compose_call.task][-1].text, Composition)
    if comp is not None:
        report.say = comp.text
        verdict = validate_say(
            comp.text, seconds_since_last_say=None, min_gap_seconds=8
        )
        report.say_ok = verdict.accepted
        report.say_reason = verdict.reason
    else:
        report.say = per_call[compose_call.task][-1].text.strip()[:200]
        report.say_reason = "invalid composition JSON"
    return report


def print_table(reports: list[dict[str, Any]]) -> None:
    head = f"{'model':<52} {'load':>6} {'turn':>6} {'i.ptok':>6} {'i.ttft':>7} {'c.ttft':>7} {'tok/s':>6} {'mem':>5}  intent  say"
    print(head)
    print("-" * len(head))
    for r in reports:
        if r.get("error"):
            print(f"{r['spec']:<52} ERROR {r['error']}")
            continue
        i, c = r["calls"]["interpret"], r["calls"]["compose"]
        print(
            f"{r['spec']:<52} {r['load_s']:>5.1f}s {r['turn_s_median']:>5.2f}s {i['prompt_tokens']:>6} "
            f"{i['ttft_s']:>6.2f}s {c['ttft_s']:>6.2f}s {c['gen_tps']:>6.1f} {r['peak_gb']:>4.1f}G"
            f"  {'ok ' if r['intent_ok'] else 'BAD'}     {'ok ' if r['say_ok'] else 'BAD'} {r['say']!r}"
        )


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("specs", nargs="+")
    parser.add_argument("--scenario", default="restroom-01")
    parser.add_argument("--runs", type=int, default=5)
    parser.add_argument("--one", action="store_true", help=argparse.SUPPRESS)
    args = parser.parse_args()

    if args.one:
        try:
            report = asdict(run_one(args.specs[0], args.scenario, args.runs))
        except Exception as exc:  # noqa: BLE001 - report and keep the sweep going
            report = {"spec": args.specs[0], "error": f"{type(exc).__name__}: {exc}"}
        print("RESULT " + json.dumps(report))
        return

    RESULTS_DIR.mkdir(exist_ok=True)
    out_path = RESULTS_DIR / f"{time.strftime('%Y%m%d-%H%M%S')}.jsonl"
    reports = []
    for spec in args.specs:
        print(f"... {spec}", file=sys.stderr, flush=True)
        proc = subprocess.run(
            [
                sys.executable,
                __file__,
                spec,
                "--one",
                "--scenario",
                args.scenario,
                "--runs",
                str(args.runs),
            ],
            capture_output=True,
            text=True,
            check=False,
        )
        line = next(
            (l for l in proc.stdout.splitlines() if l.startswith("RESULT ")), None
        )
        report = (
            json.loads(line[7:])
            if line
            else {"spec": spec, "error": proc.stderr.strip()[-400:]}
        )
        reports.append(report)
        with out_path.open("a") as fh:
            fh.write(json.dumps(report) + "\n")
    print()
    print_table(reports)
    for r in reports:
        for note in r.get("notes") or []:
            print(f"note {r['spec']}: {note}")
    print(f"\nfull results: {out_path}")


if __name__ == "__main__":
    main()
