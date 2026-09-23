# The veto layer: what the MVP guarantees

`services/agent/agent/veto.py` is a pure function that runs before the agent
publishes a strategy's `Show` and `Say`, any `Say` text, a `Notify` or a
`LightCommand`. It allows the action, or denies it and logs one JSON line
with the rule id, the guideline clause and the action. It never picks a
replacement: a denial means that action is not published, and if everything
is denied the agent does nothing.

It exists because of one measurement. In the 2026-09-22 classification probe
(`docs/CLASSIFIER_BENCH.md`), `gemma4:e4b-mlx` called 16 of 124 forbidden
actions acceptable, every one at confidence 0.8-0.9. The model supplies
judgment; this layer supplies the prohibitions it is known to miss. Those 16
cases are regression tests in `services/agent/tests/test_veto.py`.

This page is the honest statement of what that buys. It is not a medical
device and this is not a general safety system.

## Enforced in code

| rule id | denies | when | clause |
| --- | --- | --- | --- |
| `no_redirect_from_toilet_need` | `guided_return` | the goal is `restroom`, or a recent utterance names a toilet need or an accident ("loo", "pee", "wet myself", ...) that has not been resolved by a return to bed | TOIL-01 |
| `no_redirect_from_stated_need` | `guided_return` | a recent utterance states another need: cold, pain, feeling unwell, a call for help | NICE-01 |
| `no_return_prompt_in_bed` | `guided_return`, `validate_and_redirect` | the latest person reading is `in_bed`; both strategies' fixed phrases direct the person toward bed | NICE-05 |
| `no_directions_from_floor` | `path_light` | the latest person reading is `on_floor` | FALL-01 |
| `no_orienting_a_settling_person` | `orient_time_place` | the person is in bed or has settled | NICE-05 |
| `no_night_orientation_by_day` | `orient_time_place` | outside the configured night window (its sentence says it is night-time) | AA-03 |
| `silence_when_settled` | any `Say` except the escalation sentence | the person is in bed and has not spoken since lying down | NICE-05 |
| `no_memory_question` | a `Say` | the text asks the person to recall ("do you remember", "don't you know", ...) | VAL-01 |
| `no_avoided_term` | a `Say` | the text contains a term the profile's things to avoid names ("Do not mention the hospital" → `hospital`) | NICE-04 |

"Recent utterances" are the session's last three complete utterances,
recorded while it is `OBSERVING`, `ENGAGED`, or `ESCALATED` and cleared when it returns to
`IDLE`. "Settled" is a perception fact kept by the session: the latest reading
is `in_bed`, and no utterance has arrived since the person lay down.

In `ESCALATED`, `validate_goal` forbids changing `wait_for_caregiver` to
`restroom`. A stated toilet need therefore keeps the caregiver alert and goal
active while the path light and `path_light` guidance are published (TOIL-01),
unless the person is on the floor; then the agent reassures them while help comes.

Each denial is logged as one WARNING line on the `agent` service's stdout, for
example
`{"service": "agent", "message": "vetoed strategy", "rule": "no_orienting_a_settling_person", "clause": "NICE-05", "action": "strategy:orient_time_place", ...}`.
The proposed sentence is never logged: it may paraphrase private speech.

## Already enforced elsewhere, before this layer

- `agent.rules.validate_say`: one sentence, a minimum silence gap, no "no",
  "you can't" or "you're wrong", and nothing in the form of a question.
- `agent.rules.validate_composition`: no "but", no invented caregiver
  presence claims, for model-composed text.
- The state machine: phase transitions, rule 5 escalation on `on_floor` or a
  long absence, and every escalation deadline and `Notify` level.
- A confirmed wish to return to bed receives a short acknowledgement and
  pauses ordinary ladder advances for the configured grace period; stated
  restroom needs, pain, distress and safety escalation still take priority.
- Recently spoken strategies are skipped on ladder advancement, and recent
  agent speech is filtered from incoming transcripts when it matches an echo.

## Left to the model's judgment, or not covered

These are the prohibitions the MVP does **not** guarantee.

- **Every other `guided_return` misuse.** Only a need stated in words, or a
  `restroom` goal, or a current in-bed reading blocks it. A silent person
  who needs the toilet and has not reached the bathroom path yet is not
  protected, and the need regexes are English keyword lists that will miss
  paraphrases ("I'm bursting").
- **When a need is met.** Rule 1 keeps blocking `guided_return` for as long
  as the utterance stays among the last three. That is conservative, but it
  also means a toilet trip that ended is not recognised as having ended.
- **Silence after speech in bed.** "Settled" requires no speech since lying
  down. A person who speaks in bed and then goes quiet for a long time is not
  settled by this definition until they get up and lie down again, so speech
  to them is left to the state machine's pacing and to `COOLDOWN`.
- **`correction_of_reality`, `infantilising`, `unsupported_claim`.** These
  are review-only patterns in the guideline pack. No string rule catches them
  reliably, so they stay with the model and with review.
- **Other wording patterns.** `conjunction_but` is caught only in composed
  text (`validate_composition`). `states_clock_time`, `invents_proper_noun`,
  `addresses_by_name`, `invents_directions` and `blunt_refusal` beyond
  `validate_say`'s phrases are not enforced at runtime.
- **Things to avoid that are not terms.** Only "do not mention X"-shaped
  entries become terms. "Avoid loud or urgent language" is style, and is left
  to the templates and the model.
- **Notify levels and deadlines.** No veto rule applies to `Notify` or
  `LightCommand`. Whether a missed or wrong-level notification is the veto's
  concern is an open question in `CHOICE_VETO_HANDOFF.md`; for now escalation
  timing and level belong to the state machine.
- **`familiar_voice` consent**, `path_light` hardware state, and anything
  about `phase` are unchanged and outside this layer.

## Known costs

A denied strategy publishes nothing, so the screen keeps what it last showed
and the strategy engine carries on as if the strategy ran: it dwells, then
advances. Denials are therefore visible in the logs, not on the dashboard.
