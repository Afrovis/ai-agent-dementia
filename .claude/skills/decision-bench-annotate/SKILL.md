---
name: decision-bench-annotate
description: Label decision_bench scenarios with the isolated Opus annotator, hand the labels to the human for review, and apply the reviewed labels to the fixtures while logging disagreements. Use whenever the user asks to annotate, label, re-label or review decision_bench scenarios or checkpoints, to run the annotator, or to apply reviewed labels, even if they only say "phase 3" or "label the pilots".
---

# decision_bench annotation

The workflow is in `tests/decision_bench/PLAN.md` ("Annotation workflow").
This skill runs it. All commands run from `tests/decision_bench` in an
environment where `decision_bench` is installed (see its README, "Running").

## Rules that keep the labels honest

- **You are not the annotator.** The annotator is a separate headless
  `claude -p` process with no tools, run from an empty temp directory, which
  sees only `guidelines.md`, the profile and the scenario. Never write or
  "improve" model labels yourself, and never pass the annotator anything from
  `config/` or `services/agent/`. If the annotator's output looks wrong, that
  is a finding for the human review, not something to fix quietly.
- **The human decides.** Do not edit `annotations/review/<id>.yaml` for the
  user, and do not set `reviewed: true`. You may explain a label, point at
  the clause it cites, or say what the current agent would do. Say that the
  last one is not evidence.
- Only clauses whose `Checked` box is ticked in `guidelines.md` can be cited.
  If a checkpoint needs an unchecked clause, tell the user which one to
  check against its source.
- Time limits (`escalate_by`) are caregiver thresholds, never evidence.

## Steps

1. **Annotate.** `python -m decision_bench annotate --scenario <id>` (repeat
   `--scenario`, or omit it to label every clean, unlabelled scenario). Use
   `--dry-run` first if the prompt or the guidelines changed, and check the
   prompt has no strategy-config or agent content. Each run costs Opus usage;
   do not re-run a labelled scenario without `--force` and a reason.
2. **Hand over for review.** Tell the user which files to review
   (`annotations/review/<id>.yaml`). For each scenario, summarise in a few
   lines what the annotator chose and quote its `uncertain` notes and
   `scenario_notes` (in `annotations/model/<id>.yaml`). The review file is
   self-contained: timeline, questions, labels and the annotator's doubts,
   each marked as written by the Claude annotator, not the local model. If
   its layout changes, rebuild it without new annotator calls:
   `python -m decision_bench annotate --review-only --force` (this discards
   any unsaved review edits). Then stop and wait.
3. **Apply.** When the user says the reviews are done:
   `python -m decision_bench apply <id> ...`. It refuses if `reviewed` is not
   true or a changed checkpoint has no `reason`; relay that rather than
   filling in a reason.
4. **Check.** `pytest -q` and `python -m decision_bench --backend stub` should
   now show the scenarios as labelled with pass rates. Summarise
   `annotations/disagreements.yaml`: what the human changed and why.
5. **Learn from disagreements.** Group them by cause: a prompt instruction,
   a missing or unclear clause, an action vocabulary gap, or a scenario
   problem. Propose concrete changes to `decision_bench/annotator_prompt.md`,
   `guidelines.md` (implications only; the clause text is the human's) or the
   schema, and record them in PLAN.md. Changing the prompt changes
   `prompt_sha256` in later annotations, which is how runs stay comparable.
