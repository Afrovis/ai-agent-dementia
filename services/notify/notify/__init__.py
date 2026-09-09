"""Night Companion `notify` service package.

Sends caregiver alerts through a pluggable backend (ntfy by default, see
`notify.backends`). Consumes the `Notify` event stream (HANDOFF.md section
5) and, per PLAN.md section 9, resends `critical` notifications every 60 s
until acknowledged. See `notify.main` for the consume loop and
`notify.backends` for the backend interface.
"""
