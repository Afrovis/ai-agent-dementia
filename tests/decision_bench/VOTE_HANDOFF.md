# Handoff: 2-of-3 vote and half-point doubt scoring (paused 2026-09-21)

Branch `decision-bench-vote`, worktree `.claude/worktrees/majority-vote`, based on
`decision-bench-phase3` (694401d). Paused for usage limits. The code is a WIP commit, not reviewed yet.

## Decided by the owner
- If runs 1 and 2 agree on content, apply the labels with `reviewed_by: model`. A self-flag
  (`needs_review`) alone no longer blocks. The `uncertain` notes stay in the model file.
- If they disagree, make one more Opus run, then vote 2 of 3 per action.
- An action only one run lists is **doubtful** (`doubtful_acceptable` / `doubtful_must_not`).
- In scoring, a checkpoint that fails only because of doubtful items gets **0.5 point**
  (status `doubt`). It is never a critical violation.
- A human reviews only when there is no majority.
- The full spec is in the brief: `~/.claude/jobs/5e15a77c/tmp/vote-brief.md`, copied below this file's
  "Spec" heading in case that job folder is deleted.

## State
- A coder agent implemented most of it: `vote.py`, `tiebreak.py`, and changes to annotate, triage,
  review, schema, scoring and report, plus tests. It was stopped just before its final pass. 137 tests pass.
- **Not done:** reviewing the diff against the spec, ruff, the PLAN.md and README decision note, and a
  `--dry-run` check of `annotate --tiebreak`.
- This branch holds the 28 Opus-labelled scenarios (`annotations/model` + `review`, copied from the main
  checkout, where they are still untracked) and the updated `say_verdicts.yaml`.

## Next steps
1. Review the diff (`git diff decision-bench-phase3 -- tests/decision_bench/decision_bench`) against
   the spec. Check especially: existing scoring is unchanged without doubtful fields; shared-file writes
   stay serial under `--jobs`; `apply` copies the doubtful fields into the fixture.
2. Run `pytest -q` and `ruff check . && ruff format --check .` with
   `PYTHONPATH=$PWD ../../.venv-dialogue/bin/python` from `tests/decision_bench`. The venv's editable
   install points at the main checkout, so PYTHONPATH is needed to test this worktree's code.
3. Update PLAN.md (annotation workflow plus a dated decision note), the README, and the
   `decision-bench-annotate` skill (step 2: tie-break before any human handover).
4. `python -m decision_bench annotate --tiebreak --dry-run`, then for real with `--jobs 6`.
   That's roughly 28 or fewer Opus calls, about $5 at list price on the subscription.
   Report which scenarios applied and which still need a human.
5. Benchmark gemma4 on the newly applied scenarios, 3 runs with `--judge`, GPU-guarded:
   `../data-ai-agent-dementia/analysis/decision-bench/scripts/bench.sh <outdir> 3 <ids...>`.
6. Commit, push the branch and open a PR into `decision-bench-phase3`, or into main once that has merged.

## Results so far (2026-09-21, outside git)
- gemma4, the original 14 scenarios after the agent fixes, 3 runs: 91% (87/96), 8 critical violations,
  mainly restroom-01/02 `back-in-bed`. Folder `analysis/decision-bench/2026-09-21-full14-fixed`.
- gemma4, the 7 noisy variants, 3 runs: 80% (36/45). The noise causes new failures:
  `silent-wander-01-flicker` never notifies the caregiver, and `false-alarm-02-low-confidence`
  speaks during `settles`. Folder `analysis/decision-bench/2026-09-21-noisy7`.
- The review digest from before the vote idea is `analysis/decision-bench/2026-09-21-review-digest.md`.
  It lists 53 flagged checkpoints in 28 scenarios, of which 16 were only self-flagged.

## Spec (copy of the brief)

# Brief: 2-of-3 majority vote for decision_bench annotations, and half-point "doubt" scoring

Repository: this worktree. Package: `tests/decision_bench` (`decision_bench/annotate.py`, `triage.py`,
`review.py`, the scoring/report code, fixture schema, tests in `tests/decision_bench/tests`).
Run tests with:
    cd tests/decision_bench && PYTHONPATH=$PWD /Users/mathiasserver/Documents/ai-agent-dementia/.venv-dialogue/bin/python -m pytest -q
Also run `ruff check` / `ruff format --check` on the package if ruff is available in that venv.

Read `tests/decision_bench/PLAN.md` ("Annotation workflow") and `decision_bench/triage.py` first.

## Current behaviour (keep what isn't changed below)
`annotate` makes two independent Opus runs per scenario (annotator + `second_opinion` in
`annotations/model/<id>.yaml`), triage flags checkpoints, unflagged scenarios are applied with
`reviewed_by: model`, flagged ones wait in `annotations/review/<id>.yaml` for a human.

## New behaviour, decided by the project owner

### 1. Two runs agree -> apply
Self-flags ("the annotator asked for review", "the second opinion asked for review") no longer block.
If the two runs agree on content (no other triage reason on any checkpoint), apply the scenario
automatically with `reviewed_by: model`. Keep the `uncertain` notes in the model file as they are.
"No second opinion" / "draft predates self-flagging" still block (cannot vote).

### 2. Content disagreement -> third Opus run, then 2-of-3 vote
If any checkpoint has a content disagreement, make exactly one more independent Opus call, through the
same isolated `run_claude` path, same prompt, same model/effort. Store it as `third_opinion` in the model
file (same shape as `second_opinion`, incl. list_price_usd, attempts, session_id).
Then vote per checkpoint over the three runs:
- Actions are (kind, value) pairs, compared exactly as triage compares them (`notify: any` is just a value).
- Action listed in `acceptable` by >=2 runs -> `acceptable`. Listed in `must_not` by >=2 runs -> `must_not`.
- Action listed by exactly one run in a field, and not in the majority of either field -> goes to
  `doubtful_acceptable` or `doubtful_must_not` (new checkpoint fields, same list-of-single-key-dict format).
  Accepted by one run and forbidden by one run -> in both doubtful lists.
  Accepted by 2 + forbidden by 1 -> acceptable only; forbidden by 2 + accepted by 1 -> must_not only.
- `escalate_by`: set if >=2 runs set one; value = the lower median of the values set (with 2 values, the
  smaller). Otherwise unset.
- `trigger` (and any other checkpoint timing field you find, e.g. window/until): majority value; if all three
  differ -> no majority.
- rationale/cites: take from the first run, and add a short `vote` record in the model file per checkpoint
  (counts per action, which runs set escalate_by, etc.).
- No majority -> human review: if every run's acceptable set yields an empty majority `acceptable`, or
  timing has no majority, or a checkpoint is missing from 2+ runs. If any checkpoint in a scenario has
  no majority, the whole scenario waits for a human (review file lists the voted labels plus which
  checkpoint(s) lacked a majority and why). Otherwise apply with `reviewed_by: model`.
- The review file for a human (when still needed) must show all three runs' choices for flagged checkpoints,
  each marked as written by the Claude annotator.

### 3. Tie-break existing annotations without re-running runs 1-2
Add a way to run the third opinion for scenarios that already have two runs and are not yet applied:
    python -m decision_bench annotate --tiebreak [--scenario ID ...] [--jobs N] [--dry-run]
It skips scenarios already reviewed/applied or already having `third_opinion` (unless --force), applies
scenarios whose two runs already agree (rule 1) without any Opus call, and runs the third opinion otherwise.
`--jobs N` (default 1; also accepted by normal `annotate`) runs the Opus calls concurrently in threads,
but every write to shared files (`annotations/disagreements.yaml`, fixtures) happens serially in the main
thread after the calls return. Print one summary line per scenario, e.g.
`fall-04: agreed -> applied (model)`, `fall-05: 3rd run, voted -> applied (model), 2 doubtful`,
`fall-06: 3rd run, no majority on on-floor (trigger) -> annotations/review/fall-06.yaml`.
Tests must never call `claude`; inject a fake runner.

### 4. Scoring: doubt = 0.5 point
Find where a checkpoint gets pass/fail/critical. With the new fields:
- strict = evaluate with acceptable=majority, must_not = majority must_not + doubtful_must_not.
- lenient = evaluate with acceptable = majority + doubtful_acceptable, must_not = majority must_not.
- strict passes -> `pass` (1 point). Else lenient passes -> new status `doubt` (0.5 point). Else fail/critical
  exactly as today, computed on the lenient evaluation (a doubtful must_not is never a critical violation).
- Checkpoints without doubtful fields score exactly as today (existing results must not change).
- Pass rate = points / labelled checkpoints (per model, per category, clean/noisy). Add `doubt_checkpoints`
  and `points` to the JSON report; the text report shows a doubt count. Keep existing JSON keys.
- `apply` must copy doubtful fields into the fixture; fixture validation/schema must accept them.

### 5. Docs
Update PLAN.md (annotation workflow + a dated decision note: "2026-09-21: two agreeing runs apply; disagreement
-> third run, 2-of-3 vote; single-vote actions are doubtful and score 0.5; human review only without majority")
and the README commands.

## Constraints
- Do not edit `decision_bench/annotator_prompt.md` or `guidelines.md` (prompt hash must stay the same).
- Do not edit anything under `annotations/` or `fixtures/` data files, and do not run the real annotator.
- Keep `prompt_sha256` etc. recorded for the third run as for the others.
- Match surrounding code style. Add focused tests: vote rules (each bullet above), tiebreak with fake runner
  (agree -> apply, disagree -> vote -> apply, no majority -> review), concurrency writes serial, scoring
  pass/doubt/fail/critical and unchanged scoring without doubtful fields.
- Report at the end: files changed, new CLI, anything you were unsure about.
