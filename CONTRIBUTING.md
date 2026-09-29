# Contributing

Thanks for your interest in Night Companion. Bug reports, questions and pull
requests are all welcome.

## Before you start

- Read [HANDOFF.md](HANDOFF.md) for the fixed decisions, event contracts and
  safety rules, and [AGENTS.md](AGENTS.md) for how to run and test the stack.
  If a change would break a rule in `HANDOFF.md`, open an issue to discuss it
  first.
- For anything larger than a small fix, open an issue describing what you want
  to change and why, so the approach can be agreed before you invest time.

## Ground rules

These come from the project's safety and privacy design and are not up for
negotiation in a pull request:

- Safety-critical state transitions, escalation timers and limits stay in
  deterministic code. Language-model output is advisory and always validated.
- Camera frames and audio stay on the device and are not persisted unless the
  caregiver enables the documented debug option.
- Services communicate only through Redis streams, never by calling each other.
- Every service stays testable with no camera, microphone, Ollama or live Redis.
- Never commit recordings, frames, audio, transcripts from a real person, the
  SQLite database, certificates or a filled-in `.env`. Test fixtures must be
  synthetic or reviewed, text-only event logs.
- Changes to what the person sees or hears at night (phrasing, timing, voice,
  brightness) need a clear reason in the pull request. Keep to the
  person-facing language rules in `HANDOFF.md`.

## Pull requests

- Keep each pull request to one change, with tests.
- Run the checks for every service you touched in its own image (see
  "Validating changes" in `AGENTS.md`); `ruff check` and `ruff format --check`
  must pass.
- Document new configuration keys in `.env.example` or the relevant example
  YAML under `config/`.
- If you changed an event, stream or subscription, regenerate
  `ARCHITECTURE.md` with `python -m nc_shared.archdoc --write`.

By contributing you agree that your contributions are licensed under the
[GNU AGPL v3.0](LICENSE), the same license as the project.
