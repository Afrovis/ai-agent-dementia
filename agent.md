# Agent Guide

Night Companion is a local, embodied AI agent that helps a person with dementia
return safely to bed at night. It is not a medical device or a substitute for
supervision.

## Before making changes

1. Read [HANDOFF.md](HANDOFF.md) for fixed decisions, event contracts, safety
   rules, and the definition of done.
2. Read [CLAUDE.md](CLAUDE.md) for local setup, testing, TLS, and operational
   guidance.
3. Consult [PLAN.md](PLAN.md) for design rationale and
   [ARCHITECTURE.md](ARCHITECTURE.md) for the generated as-built service map.

Treat `HANDOFF.md` as authoritative if this summary conflicts with it.

## Working rules

- Keep safety-critical state transitions, escalation timers, and limits in
  deterministic code. LLM output is advisory and must be validated.
- Never send camera frames or audio off-device or persist them unless the
  caregiver explicitly enables the documented debug option.
- Keep services independent and communicate only through Redis streams.
- Add or change event schemas in `shared/nc_shared/events.py` and update the
  event-contract documentation in the same change.
- Document every configuration key in `.env.example` or an example YAML file.
- Do not commit anything under `data/`.
- Preserve the person-facing language rules in `HANDOFF.md`, especially the
  one-sentence output limit and required silence between spoken prompts.

## Validate changes

Run the narrowest relevant tests first, then before handoff run:

```sh
ruff format --check .
ruff check .
pytest
```

Tests must not require a camera, microphone, Ollama, or a live Redis instance.
If an event, stream, subscription, or outside-world connection changes,
regenerate `ARCHITECTURE.md` as described in `CLAUDE.md`.
