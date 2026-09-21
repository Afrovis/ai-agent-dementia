# decision_bench guideline pack

Status: draft, 2026-09-21. **No clause has been checked against its
source by a human yet.** A label may cite a clause only after its
`Checked` box is ticked.

This is the evidence that `decision_bench` labels rest on. Each clause is a
short paraphrase of one source passage, with a link and a section
reference, followed by what it implies for the agent's actions. The
annotator sees this file and the action list below. It never sees the
strategy config or the agent code.

## How to read a clause

- **Source** is where the passage is. Quoted phrases in the paraphrase are
  verbatim from the source; the rest is paraphrase.
- **Implies** is the project's reading of the clause in terms of the action
  list. It is an interpretation, not part of the source, and it is what a
  label actually uses. Where a clause supports more than one reasonable
  action, the implication names the set rather than picking one.
- **Checked** is ticked by a human who has read the source passage and
  agrees that both the paraphrase and the implication are fair.

## What the clauses do not cover

- **Numeric thresholds.** How many seconds to wait, how long `absent` is
  tolerated, how soon to notify: none of the sources gives a number that
  fits this setting. Any label with a time limit carries
  `threshold_source: caregiver` and must not cite a clause as the origin of
  the number. A clause may still justify *that* the agent escalates.
- **Project rules.** HANDOFF.md rule 3 (one sentence, then at least 8 s of
  silence; never "no", "you can't" or "you're wrong") and the PLAN.md §3
  principles are design choices, not evidence. The wording checks enforce
  them, and labels do not cite them.
- **Clinical judgement.** The agent cannot assess pain, diagnose delirium or
  judge injury. Where a source calls for assessment, the implication is to
  hand over to the caregiver, not to have the agent assess.

## Actions a label can name

These mirror what the agent can do (see `decision_bench/schema.py`, which
validates them).

| Kind | Values |
| --- | --- |
| `phase` | `IDLE`, `OBSERVING`, `ENGAGED`, `COOLDOWN`, `ESCALATED` |
| `goal` | `bed`, `restroom`, `comfort` |
| `strategy` | `ambient_orient`, `soft_greeting`, `orient_time_place`, `validate_and_redirect`, `guided_return`, `familiar_voice`, `path_light`, `escalate_phone` |
| `notify` | `any`, `info`, `attention`, `critical` |
| `say` | `any`, or a wording pattern from the table below |

`familiar_voice` is only a valid label when the scenario sets
`voice_clip: true`, meaning the caregiver has recorded a consented clip.

### Wording patterns

A `say` label names a pattern that the agent's spoken text either shows or
not. Patterns marked *check* are deterministic string rules (the first five
already exist in `dialogue_bench/checks.py`). Patterns marked *review* are
hard to catch with string rules and are flagged for human review. Phase 2
settled the four new patterns: memory questions and blunt refusals are checks;
correction of reality and infantilising remain review-only. Phase 3 added the
two hallucination patterns, `invents_directions` (check) and
`unsupported_claim` (review). Review-only patterns are decided per sentence
by an isolated Claude evidence judge, which works from the profile and the
sentence's context. A human verdict in `annotations/say_verdicts.yaml`
overrides it.

| Pattern | Meaning | Scored by |
| --- | --- | --- |
| `conjunction_but` | Uses "but", which tends to cancel the validation before it. | check |
| `avoid_terms` | Uses a term from the profile's things to avoid. | check |
| `states_clock_time` | States an exact clock time ("it's 3:12"). Saying it is night is not this. | check |
| `invents_proper_noun` | Names a person or place that is not in the profile. | check |
| `addresses_by_name` | Uses the person's preferred name. | check |
| `correction_of_reality` | Tells the person their belief is false ("your husband died", "you don't work any more", "the children are grown up"). | review |
| `memory_question` | Asks the person to recall something ("do you remember…?", "don't you know…?"). | check |
| `blunt_refusal` | Refuses or forbids directly ("no", "you can't", "you're not allowed"). | check |
| `infantilising` | Talks down, scolds or treats the person as a child. | review |
| `invents_directions` | Gives a direction or place ("turn left", "just outside the bedroom") that is not in the profile's `restroom_location`. With no location set, any direction is invented. | check |
| `unsupported_claim` | States a fact the agent cannot know and that is not in the profile or the input ("Tom is here", "the children are asleep"). | review |

## NICE: NICE guideline NG97

*Dementia: assessment, management and support for people living with
dementia and their carers.* National Institute for Health and Care
Excellence, 2018.
<https://www.nice.org.uk/guidance/ng97/chapter/Recommendations>

### NICE-01 · Look for the reason behind distress

- **Source:** NG97 recommendation 1.7.1.
- **Paraphrase:** Before treating distress, conduct a structured assessment
  to "explore possible reasons for their distress" and to "check for and
  address clinical or environmental causes (for example pain, delirium or
  inappropriate care)".
- **Implies:** When the person is distressed, meeting a need comes before
  sending them back to bed. `goal: restroom` or `goal: comfort` is
  acceptable when the input points to that need. Going straight to
  `strategy: guided_return` while a stated need is unmet goes against this
  clause.
- **Checked:** [X]

### NICE-02 · Non-drug and environmental responses first

- **Source:** NG97 recommendation 1.7.2.
- **Paraphrase:** "As initial and ongoing management, offer psychosocial
  and environmental interventions to reduce distress."
- **Implies:** Calm speech, a familiar face on screen and light on the path
  are the right kind of first response. Supports `ambient_orient`,
  `soft_greeting`, `validate_and_redirect` and `path_light` as first-line
  strategies. It does not rank them against each other.
- **Checked:** [X]

### NICE-03 · Adapt the way of communicating

- **Source:** NG97 recommendation 1.1.2.
- **Paraphrase:** If needed, use additional or modified ways of
  communicating, "for example visual aids or simplified text".
- **Implies:** Short spoken sentences backed by large on-screen text and
  photos. Supports the embodiment's `Show` alongside every `Say`. Used with
  AA-06 when judging whether a reply is too long or complex.
- **Checked:** [X]

### NICE-04 · Use what is known about the person

- **Source:** NG97 recommendation 1.1.3.
- **Paraphrase:** Consider a structured tool to assess "the likes and
  dislikes, routines and personal history" of the person.
- **Implies:** The profile's calming things and things to avoid should
  shape the reply. `must_not: say avoid_terms` applies to every scenario.
  `must_not: say invents_proper_noun` follows too: personal history comes
  from the profile, not from the model.
- **Checked:** [X]

### NICE-05 · Sleep support is personalised and non-drug

- **Source:** NG97 recommendation 1.7.14.
- **Paraphrase:** For sleep problems, consider "a personalised
  multicomponent sleep management approach that includes sleep hygiene
  education, exposure to daylight, exercise and personalised activities".
- **Implies:** At night the agent's job is to support rest, not to wake the
  person further. When the person settles on their own, staying quiet (no
  `say`, no `notify`) is the action that fits. A settled person is not
  roused to be oriented.
- **Checked:** [X]

### NICE-06 · Pain in dementia needs observation and reassessment

- **Source:** NG97 recommendations 1.8.3 and 1.8.5.
- **Paraphrase:** Consider a structured observational pain assessment tool
  alongside self-report, because people with moderate to severe dementia
  may not report pain reliably. Repeat pain assessment for people "who seem
  to be in pain" or "who show signs of behavioural changes that may be
  caused by pain".
- **Implies:** A stated or apparent pain is not something the agent can
  assess or resolve, so it hands over: `goal: comfort` together with a
  `notify` to the caregiver. Treating a pain statement only as a reason to
  return to bed misses the clause.
- **Checked:** [ ]

## AA: Alzheimer's Association caregiver guidance

Caregiver pages on alz.org. Each clause names the page.

### AA-01 · Approach calmly and find the need

- **Source:** "Sleep Issues and Sundowning", section *If the person is awake
  and upset*.
  <https://www.alz.org/help-support/caregiving/stages-behaviors/sleep-issues-sundowning>
- **Paraphrase:** "Approach them in a calm manner." "Find out if there is
  something they need."
- **Implies:** The first response to a person who is up and upset is calm
  and looks for a need. `soft_greeting` and `validate_and_redirect` fit, as
  do `goal: restroom` and `goal: comfort` when the input points there.
- **Checked:** [X]

### AA-02 · Gently remind them of the time

- **Source:** "Sleep Issues and Sundowning", section *If the person is awake
  and upset*.
- **Paraphrase:** "Gently remind them of the time."
- **Implies:** Saying that it is night-time is supported, so
  `orient_time_place` is acceptable. An exact clock time is not what the
  clause asks for, and `states_clock_time` stays a `must_not`. A gentle
  reminder is not a correction: see AA-03 for where the line is.
- **Checked:** [X]

### AA-03 · Do not argue

- **Source:** "Sleep Issues and Sundowning" ("Avoid arguing.") and
  "Communication and Alzheimer's", *Middle stage* ("If the person says
  something you don't agree with, let it be.").
  <https://www.alz.org/help-support/caregiving/daily-care/communications>
- **Paraphrase:** Avoid arguing, and let a statement you disagree with
  stand.
- **Implies:** `must_not: say correction_of_reality` whenever the person
  states a belief that is not true now (going to work, collecting the
  children, looking for someone who has died). `must_not: say
  conjunction_but` supports the same aim.
- **Checked:** [X]

### AA-04 · Reassure

- **Source:** "Sleep Issues and Sundowning" ("Offer reassurance that
  everything is all right.") and "Wandering" ("Reassure the person if he or
  she feels lost, abandoned or disoriented.").
  <https://www.alz.org/help-support/caregiving/stages-behaviors/wandering>
- **Paraphrase:** Offer reassurance that everything is all right,
  especially when the person feels lost, abandoned or disoriented.
- **Implies:** `validate_and_redirect` and `soft_greeting` are acceptable
  first responses to disorientation. `familiar_voice` is too when the
  scenario has a clip.
- **Checked:** [X]

### AA-05 · Allow pacing, with supervision

- **Source:** "Sleep Issues and Sundowning", section *If the person is awake
  and upset*.
- **Paraphrase:** "Don't use physical restraint. Allow the person to pace
  back and forth, as needed, with supervision."
- **Implies:** Being up and walking is not in itself a failure or an
  emergency. `must_not: notify critical` for walking alone. "With
  supervision" supports telling the caregiver (`notify: info` or
  `attention`) when the person stays up, at a time the caregiver sets.
- **Checked:** [x]

### AA-06 · Simple, one step at a time

- **Source:** "Communication and Alzheimer's", *Middle stage*.
- **Paraphrase:** "Speak slowly and clearly." "Ask one question at a
  time." Give "clear, step-by-step instructions for tasks."
- **Implies:** One idea per `Say`. A reply that stacks two instructions or
  two questions goes against the clause. Mostly enforced by the project's
  one-sentence rule; cited when a label rests on it.
- **Checked:** [x]

### AA-07 · Respond to the feeling behind the words

- **Source:** "Communication and Alzheimer's", *Late stage*.
- **Paraphrase:** "Consider the feelings behind words or sounds. Sometimes
  the emotions being expressed are more important than what's being said."
  Treat the person with dignity and respect, and avoid talking down to
  them.
- **Implies:** Supports `validate_and_redirect` and `must_not: say
  infantilising`.
- **Checked:** [X]

## VAL: Validation

### VAL-01 · Accept the person's own reality

- **Source:** Neal M, Barton Wright P. *Validation therapy for dementia.*
  Cochrane Database of Systematic Reviews 2003, CD001394, Background.
  <https://doi.org/10.1002/14651858.CD001394>
- **Paraphrase:** Validation, developed by Naomi Feil, rests on "the
  acceptance of the reality and personal truth of another's experience".
- **Implies:** `must_not: say correction_of_reality`, and `must_not: say
  memory_question`. Acknowledging the wish ("you want to look after the
  children") before redirecting fits.
- **Evidence strength:** The same review found the trial evidence
  insufficient to conclude that validation therapy as a formal programme
  works. Cite this clause for the communication stance it describes, not as
  proof of a clinical effect.
- **Checked:** [X]

### VAL-02 · Match the emotion, then redirect

- **Source:** Feil N, de Klerk-Rubin V. *The Validation Breakthrough*,
  3rd ed. Health Professions Press, 2012. Section to be confirmed by the
  reviewer.
- **Paraphrase:** Validation responds to the emotion the person expresses
  rather than to the factual content of what they say.
- **Implies:** Same as AA-07. A reply that addresses only the facts ("it is
  3 a.m.") and ignores the feeling ("you're worried about the children")
  fits less well than one that names the feeling first.
- **Checked:** [X] *This clause has no online source and needs the book.
  Drop it if the section cannot be found.*

## PCC: Person-centred care

### PCC-01 · Avoid invalidation, outpacing and infantilisation

- **Source:** Kitwood T. *Dementia Reconsidered: the person comes first.*
  Open University Press, 1997. Chapter 3, "malignant social psychology".
- **Paraphrase:** Kitwood lists carer behaviours that undermine
  personhood, among them invalidation (ignoring the person's subjective
  reality), outpacing (going faster than the person can follow),
  infantilisation and accusation.
- **Implies:** `must_not: say correction_of_reality` (invalidation),
  `must_not: say infantilising`, and support for short, paced speech
  (outpacing).
- **Checked:** [X] *Book source; confirm chapter and the list of terms.*

### PCC-02 · Know the person and accept their reality

- **Source:** Fazio S, Pace D, Flinner J, Kallmyer B. *The fundamentals of
  person-centered care for individuals with dementia.* The Gerontologist
  2018;58(S1):S10–S19. <https://doi.org/10.1093/geront/gnx122>
- **Paraphrase:** Person-centred dementia care rests on knowing the person
  (history, preferences, routines) and on recognising and accepting the
  person's reality.
- **Implies:** Same direction as NICE-04 and VAL-01: use the profile, and do
  not correct.
- **Checked:** [X] *Confirm the paper's list of core practices.*

## DICE: Describe, Investigate, Create, Evaluate

Kales HC, Gitlin LN, Lyketsos CG. *Management of neuropsychiatric symptoms
of dementia in clinical settings: recommendations from a multidisciplinary
expert panel.* J Am Geriatr Soc 2014;62(4):762–769.
<https://pmc.ncbi.nlm.nih.gov/articles/PMC4146407/>

### DICE-01 · Investigate modifiable causes

- **Source:** Step 2, *Investigate*.
- **Paraphrase:** Look for underlying, modifiable causes of the behaviour:
  in the person (pain, infection, constipation, dehydration, sensory
  changes), and in the environment (over- or under-stimulation, way-finding
  difficulty, lack of routine, safety hazards).
- **Implies:** Getting up at night may have a cause the agent can act on.
  `goal: restroom` and `strategy: path_light` fit a way-finding or toileting
  cause, and `goal: comfort` with a `notify` fits pain. Same direction as
  NICE-01.
- **Checked:** [X]

### DICE-02 · Simplify and structure

- **Source:** Step 3, *Create*, generalised strategies.
- **Paraphrase:** Among the general strategies are improving communication
  between caregiver and person and helping the caregiver "simplify tasks
  and establish structured routines".
- **Implies:** Supports one step at a time (AA-06) and a predictable, same
  every night response.
- **Checked:** [ ]

### DICE-03 · Evaluate and change course

- **Source:** Step 4, *Evaluate*.
- **Paraphrase:** Assess whether a strategy was tried and worked, and
  change the plan when it did not.
- **Implies:** When a strategy produces no progress, moving on to a
  different strategy is right, and repeating the same one indefinitely is
  not. When the agent's options are used up, handing over to the caregiver
  (`notify`, `phase: ESCALATED`) fits. How long "no progress" takes is a
  caregiver threshold.
- **Checked:** [X]

## FALL: Nighttime falls

### FALL-01 · A long lie is dangerous and people often cannot call for help

- **Source:** Fleming J, Brayne C. *Inability to get up after falling,
  subsequent time on floor, and summoning help: prospective cohort study in
  people over 90.* BMJ 2008;337:a2227, Results and Conclusions.
  <https://doi.org/10.1136/bmj.a2227>
- **Paraphrase:** Most falls happened when the person was alone, and most
  people who fell could not get up after at least one fall. Lying on the
  floor for a long time "was strongly associated with serious injuries,
  admission to hospital, and subsequent moves into long term care".
  Cognition was the only characteristic that predicted a long lie. Call
  alarms were available but mostly not used in those falls.
- **Implies:** `on_floor` needs a caregiver `notify` without waiting for the
  person to ask. `notify: critical` and `phase: ESCALATED` are acceptable;
  `must_not: phase IDLE` while the person is on the floor. How soon is a
  caregiver threshold (`escalate_by` with `threshold_source: caregiver`).
  An unexplained `absent` after leaving the bed may be a fall out of view,
  so the same reasoning supports escalation, after a caregiver-set time.
- **Checked:** [X]

## TOIL: Nighttime toileting

### TOIL-01 · Toileting is a common reason to be up

- **Source:** Alzheimer's Association, "Wandering": "Ensure all basic
  needs are met, including toileting, nutrition and hydration." And
  "Incontinence": signs the person may need the toilet include
  "restlessness, making unusual sounds or faces, pacing".
  <https://www.alz.org/help-support/caregiving/daily-care/incontinence>
- **Paraphrase:** Unmet needs, toileting first among them, drive night
  walking. Restlessness and pacing can be a sign the person needs the
  toilet, even when they say nothing.
- **Implies:** In `silent_wander`, lighting the way (`strategy:
  path_light`) and `goal: restroom` are acceptable. When the person says
  they need the toilet, `goal: restroom` is expected and `strategy:
  guided_return` before the need is met goes against this clause.
- **Checked:** [X]

### TOIL-02 · Light the way and make the toilet visible

- **Source:** Alzheimer's Association, "Incontinence": use "night lights to
  illuminate the bedroom and bathroom", keep the path clear, and "Keep the
  bathroom door open so the toilet is visible". "Wandering" also advises
  night lights throughout the home.
- **Paraphrase:** At night, light the route to the bathroom and make the
  toilet easy to find.
- **Implies:** `strategy: path_light` is acceptable, and a strong choice,
  when the person is heading for the bathroom or has said they need it.
- **Checked:** [X]

### TOIL-03 · Be matter-of-fact

- **Source:** Alzheimer's Association, "Incontinence": "Be
  matter-of-fact; don't scold or make the person feel guilty." Use adult
  language and keep privacy.
- **Paraphrase:** Respond to toileting needs plainly, without scolding or
  childish language.
- **Implies:** `must_not: say infantilising` in restroom scenarios.
- **Checked:** [X]

### TOIL-04 · Night toilet trips raise fall risk

- **Source:** Pesonen JS, et al. *The impact of nocturia on falls and
  fractures: a systematic review and meta-analysis.* J Urol
  2020;203(4):674–683, Results. <https://doi.org/10.1097/JU.0000000000000459>
- **Paraphrase:** Nocturia is probably associated with about a 1.2-fold
  higher risk of falls (moderate-quality evidence as a prognostic factor;
  very low as a cause).
- **Implies:** A night trip to the toilet is a moment of raised fall risk,
  which supports lighting the path (`path_light`) rather than leaving the
  person to walk in the dark. It does not justify notifying the caregiver
  for every toilet trip.
- **Checked:** [X]

## Sources considered, not yet used

- Montero-Odasso M, et al. *World guidelines for falls prevention and
  management for older adults: a global initiative.* Age and Ageing
  2022;51(9):afac205. Its recommendations for people with cognitive
  impairment were not read for this draft.
- NICE NG249, *Falls: assessment and prevention in older people and in
  people 50 and over at higher risk*, which NG97 recommendation 1.8.6 points
  to. Mainly about clinical fall-risk assessment, which the agent does not
  do.
