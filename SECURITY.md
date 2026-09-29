# Security and privacy

Night Companion runs a camera and microphone in someone's bedroom, so a
security or privacy flaw here can expose a vulnerable person. Please report
anything that could leak frames, audio, transcripts or caregiver data, bypass
the dashboard's authentication, or let an outsider influence what the bedside
screen says or does.

## Reporting a vulnerability

Do not open a public issue. Report it privately through GitHub:
**Security → Report a vulnerability** on this repository. Include what you
found, how to reproduce it, and which service it affects.

You should get an acknowledgement within a week. This is a small project
without a support team, so fixes are best effort, but privacy issues are
treated as the highest priority.

## Scope

In scope: the services under `services/`, the shared bus library in
`shared/`, and the volunteer site under `volunteer/`.

The stack is designed to run on a private home network or tailnet. Exposing
the dashboard or bedside page directly to the internet is not a supported
deployment.
