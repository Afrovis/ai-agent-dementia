"""Print the raw reply and the validation error for every rejected composition.

why_rejected.py openai <model> [base-url]
why_rejected.py ollama <tag>
"""

from __future__ import annotations

import json
import sys

from agent import llm
from agent.llm import Composition, _json_object, local_llm
from dialogue_bench.scenarios import load_scenarios
from dialogue_bench.scoring import render_phrase
from pydantic import ValidationError


def main() -> None:
    backend, model = sys.argv[1], sys.argv[2]
    url = (
        sys.argv[3]
        if len(sys.argv) > 3
        else (
            "http://127.0.0.1:11435"
            if backend == "openai"
            else "http://127.0.0.1:11434"
        )
    )
    raw: list[str] = []
    real_urlopen = llm.urlopen

    def recording_urlopen(request, timeout):
        body = real_urlopen(request, timeout=timeout).read()
        data = json.loads(body)
        raw.append(
            data["choices"][0]["message"]["content"]
            if backend == "openai"
            else data["response"]
        )
        return _Response(body)

    llm.urlopen = recording_urlopen
    client = local_llm(backend, url=url, model=model, timeout_seconds=30)
    rejected = 0
    for scenario in load_scenarios():
        template = render_phrase(scenario.caregiver_phrase_template, scenario.profile)
        goal = (
            "restroom"
            if scenario.expected_intent.value == "need_restroom"
            else "return_to_bed"
        )
        result = client.compose(
            "validate_and_redirect",
            template,
            scenario.profile,
            scenario.time_words,
            scenario.scene_note,
            scenario.utterance,
            goal,
        )
        if result is not None:
            continue
        rejected += 1
        text = raw[-1]
        try:
            Composition.model_validate_json(_json_object(text) or "", strict=True)
            reason = "?"
        except ValidationError as exc:
            reason = exc.errors()[0]["msg"]
        print(f"{scenario.id}: {reason}\n    {text!r}")
    print(f"{rejected} rejected")


class _Response:
    def __init__(self, body: bytes) -> None:
        self._body = body
        self.status = 200

    def read(self) -> bytes:
        return self._body

    def __enter__(self):
        return self

    def __exit__(self, *_):
        return False


if __name__ == "__main__":
    main()
