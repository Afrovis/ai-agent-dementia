"""Drive the real agent (Session + run_once + OllamaLLM) through one scripted night
moment and record every LLM request/response and every published event as JSON."""

from __future__ import annotations

import argparse
import json
import time
from datetime import datetime, timedelta

from agent import llm as llm_mod
from agent.config import AgentConfig
from agent.main import run_once
from agent.profile import load_profile
from agent.session import Session
from agent.strategies import load_strategies
from nc_shared.bus import FakeBus
from nc_shared.events import EVENT_STREAMS, PersonState, Utterance

LOG: list[dict] = []
CLOCK = {"now": None}

_orig_call = llm_mod.OllamaLLM._call
_orig_urlopen = llm_mod.urlopen
RAW = {"text": None}


class _Tee:
    def __init__(self, resp):
        self._resp = resp
        self.status = resp.status

    def read(self):
        data = self._resp.read()
        try:
            RAW["text"] = json.loads(data.decode("utf-8")).get("response")
        except Exception:
            RAW["text"] = None
        return data

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return self._resp.__exit__(*exc)


def _tee_urlopen(*a, **kw):
    return _Tee(_orig_urlopen(*a, **kw).__enter__())


llm_mod.urlopen = _tee_urlopen


def _logged_call(self, task, payload, output_type):
    t0 = time.perf_counter()
    RAW["text"] = None
    result = _orig_call(self, task, payload, output_type)
    LOG.append(
        {
            "kind": "llm",
            "at": CLOCK["now"].isoformat(),
            "call": output_type.__name__,
            "model": self._model,
            "latency_s": round(time.perf_counter() - t0, 2),
            "task": task,
            "input": json.loads(json.dumps(payload, default=str)),
            "prompt": llm_mod._prompt(task, payload, output_type),
            "output": result.model_dump(mode="json") if result is not None else None,
            "raw_response": RAW["text"],
        }
    )
    return result


llm_mod.OllamaLLM._call = _logged_call


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--script", required=True, help="JSON list of timed inputs")
    ap.add_argument("--out", required=True)
    ap.add_argument("--profile", default="config/person.example.yaml")
    ap.add_argument("--strategies", default="config/strategies.example.yaml")
    ap.add_argument("--model", default="gemma4:e4b-mlx")
    ap.add_argument("--ollama", default="http://127.0.0.1:11434")
    ap.add_argument("--start", default="2026-09-17T02:46:40")
    ap.add_argument("--tick", type=float, default=2.0)
    args = ap.parse_args()

    script = json.load(open(args.script))
    profile = load_profile(args.profile)
    strategies = load_strategies(args.strategies)
    config = AgentConfig(llm_model=args.model, llm_timeout_seconds=60.0)
    session = Session(config=config, strategies=strategies)
    client = llm_mod.OllamaLLM(ollama_url=args.ollama, model=args.model, timeout_seconds=60.0)

    bus = FakeBus()
    for stream in set(EVENT_STREAMS.values()):
        bus.ensure_group(stream, "agent")
        bus.ensure_group(stream, "capture")

    start = datetime.fromisoformat(args.start)
    CLOCK["now"] = start
    end = start + timedelta(seconds=max(s["t"] for s in script) + script[-1].get("hold", 20))
    pending = sorted(script, key=lambda s: s["t"])
    last_phase = None

    while CLOCK["now"] <= end:
        t = (CLOCK["now"] - start).total_seconds()
        while pending and pending[0]["t"] <= t:
            step = pending.pop(0)
            if step["type"] == "person":
                ev = PersonState(
                    source="perceive", ts=CLOCK["now"], state=step["state"],
                    confidence=step.get("confidence", 0.9), zone=step.get("zone", "bed"),
                    scene_note=step.get("scene_note"),
                )
                CLOCK["person"] = ev
            else:
                ev = Utterance(source="listen", ts=CLOCK["now"], text=step["text"],
                               confidence=step.get("confidence", 0.92),
                               duration_s=step.get("duration_s", 2.0))
            bus.publish(ev)
            LOG.append({"kind": "input", "at": CLOCK["now"].isoformat(),
                        "event": type(ev).__name__, "data": ev.model_dump(mode="json")})
        # perceive keeps reporting the current state every tick
        if "person" in CLOCK and not any(s["t"] <= t for s in pending):
            pass
        states = run_once(bus, session, now_fn=lambda: CLOCK["now"], profile=profile,
                          llm=client, block_ms=1)
        for st in states:
            if st.phase != last_phase:
                LOG.append({"kind": "phase", "at": CLOCK["now"].isoformat(),
                            "data": st.model_dump(mode="json")})
                last_phase = st.phase
        for stream in set(EVENT_STREAMS.values()):
            if stream in ("person", "speech_in", "session"):
                continue
            for _id, ev in bus.read(stream, "capture", "c", count=50, block_ms=1):
                LOG.append({"kind": "output", "at": CLOCK["now"].isoformat(),
                            "stream": stream, "event": type(ev).__name__,
                            "data": ev.model_dump(mode="json")})
        CLOCK["now"] += timedelta(seconds=args.tick)

    json.dump(LOG, open(args.out, "w"), indent=2, ensure_ascii=False)
    for e in LOG:
        if e["kind"] == "llm":
            print(e["at"][11:19], "LLM", e["call"], e["latency_s"], "s ->", e["output"],
                  "" if e["output"] else f"RAW={e['raw_response']!r}")
        elif e["kind"] == "output":
            d = e["data"]
            print(e["at"][11:19], e["event"], {k: d[k] for k in d if k in
                  ("text", "headline", "body", "level", "title", "strategy", "goal")})
        elif e["kind"] == "phase":
            print(e["at"][11:19], "PHASE", e["data"].get("phase"), e["data"].get("goal"))
        else:
            d = e["data"]
            print(e["at"][11:19], "IN", e["event"], d.get("state") or d.get("text"))


if __name__ == "__main__":
    main()
