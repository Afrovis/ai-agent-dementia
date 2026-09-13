# Two-week volunteer dry run

This run checks Night Companion's timelines and decisions with a consenting
adult volunteer. It is not care, a medical test, or a substitute for
supervision. Do not use a person who may depend on alerts reaching a caregiver.

Keep the volunteer's name, transcripts, exports, room details, and observation
log out of Git. Report defects with synthetic or redacted examples only.

## Start the run

1. Record the volunteer's informed consent outside this repository and agree
   on a stop signal. Keep the camera indicator visible.
2. Copy `.env.example` to `.env`, complete the ordinary local configuration,
   and set `DRY_RUN=true`. This overrides `NTFY_URL`; no notification is sent.
3. Start the stack with `docker compose up -d --build`.
4. Open the authenticated dashboard System page and verify it says **Dry run
   active**. If it says **Live notification mode**, stop and correct the config.
5. Run the perception and dialogue benches described in `HANDOFF.md` before
   beginning night one. Record their results outside Git.
6. Set a start date and an end date fourteen nights later in the private run
   log. Do not count an unreviewed or incomplete night as a completed night.

`Notify` events still appear in Tonight and History. Only external delivery is
suppressed, so false alerts and missed expected alerts can be reviewed without
contacting a caregiver.

## Review every morning

Use Dashboard → History and review the previous night from start to finish.
Record this checklist in the private run log:

- date, operating interval, and whether the night was complete;
- expected versus observed wake-ups, state changes, and restroom trips;
- false positive and false negative detections;
- session start, settle, escalation, and strategy timing;
- whether every spoken sentence followed the language and silence rules;
- alerts that would have been sent, including severity and repetition;
- health faults, restarts, unavailable models, or missing timeline periods;
- volunteer comfort, stop-signal use, and any surprising behaviour;
- issue links created from the review.

Stop the run immediately for unwanted capture, privacy concerns, unsafe
physical behaviour, repeated distress, or any situation in which the volunteer
wants to stop. Export retained history from System only when it is necessary
for diagnosis, keep it local, and delete it when the diagnosis is complete.

## File an issue

Use one issue per independently fixable observation. Include:

- a short symptom and its impact;
- build commit and relevant non-secret configuration;
- redacted event types and timestamps needed to reproduce it;
- expected and actual behaviour;
- frequency (once, intermittent, or every attempt);
- a synthetic replay fixture or exact reproduction steps when possible.

Label the report as a dry-run finding if that label exists. Never attach real
frames, audio, raw transcripts, database files, exports, credentials, topic
URLs, names, or room details.

## Finish after fourteen reviewed nights

Summarize completed nights, false positives, false negatives, escalations,
faults, rule violations, volunteer stops, and open issue links. The dry run is
not a safety certification. Resolve any safety or privacy finding before a
pilot.

Set `DRY_RUN=false`, restart `notify` and `dashboard`, and verify the System
page says **Live notification mode**. Notification delivery must then be tested
with the intended caregiver during the separately supervised pilot.
