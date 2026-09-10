# Vendored third-party assets

`htmx.min.js` is served from this directory rather than a CDN. The
dashboard is a LAN-only page on a box that may have no internet access,
and a bedside system should not depend on a third-party host being
reachable at three in the morning.

| Field | Value |
| --- | --- |
| Library | htmx |
| Version | 1.9.12 |
| Source | https://cdn.jsdelivr.net/npm/htmx.org@1.9.12/dist/htmx.min.js |
| SHA-256 | `449317ade7881e949510db614991e195c3a099c4c791c24dacec55f9f4a2a452` |
| Licence | Zero-Clause BSD |

The file is byte-identical to that release. Re-verify it, or check a
replacement after an upgrade, with:

```sh
curl -sL https://cdn.jsdelivr.net/npm/htmx.org@1.9.12/dist/htmx.min.js \
  | shasum -a 256
shasum -a 256 services/dashboard/dashboard/static/htmx.min.js
```

Do not edit the file. Replace it wholesale with an upstream release and
update the version and hash above in the same change.
