# Demo photos

Placeholder images shipped with the `embodiment` service. They exist so the
fake agent's `demo_room` and `demo_family` `photo_id`s resolve on a fresh
checkout, instead of making `docker compose up` log a 404 on every cycle.

Real photos are caregiver-uploaded into the gitignored `data/photos` tree
(`PHOTO_DIR`). `resolve_photo` searches that directory first, so an uploaded
photo always shadows a demo one of the same id.

These are not photographs of anyone. They are flat, generated washes, kept
dark and low contrast because they render behind the face at 0.6 opacity
under the night brightness overlay. Replace them freely.

Regenerate them with the stdlib-only writer in `../../tools`, which is
deterministic and rewrites these exact bytes:

```
python services/embodiment/tools/generate_demo_photos.py \
    services/embodiment/embodiment/demo_photos
```
