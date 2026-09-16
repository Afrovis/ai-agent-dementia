#!/usr/bin/env python3
"""Idempotent Cloudflare setup for the volunteer site (HANDOFF.md section 7).

Standard library only. Reads `volunteer/.env`, creates or reuses the tunnel,
ingress rule, DNS record and Turnstile widget, and rewrites only the keys it
owns (`CLOUDFLARE_TUNNEL_TOKEN`, `TURNSTILE_SITE_KEY`, `TURNSTILE_SECRET_KEY`)
while preserving comments and every other line. Never prints a token or
secret value.

Run from `volunteer/`:

    python3 scripts/setup_cloudflare.py
"""

from __future__ import annotations

import json
import sys
import urllib.error
import urllib.request
from pathlib import Path

API_BASE = "https://api.cloudflare.com/client/v4"
TUNNEL_NAME = "nc-volunteer"
TURNSTILE_WIDGET_NAME = "nc-volunteer"
ENV_PATH = Path(__file__).resolve().parent.parent / ".env"

# Env keys this script is allowed to write back into .env. Anything else in
# the file is left untouched, comments included.
OWNED_KEYS = {"CLOUDFLARE_TUNNEL_TOKEN", "TURNSTILE_SITE_KEY", "TURNSTILE_SECRET_KEY"}


def read_env(path: Path) -> dict[str, str]:
    values: dict[str, str] = {}
    if not path.exists():
        return values
    for line in path.read_text().splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#") or "=" not in stripped:
            continue
        key, _, value = stripped.partition("=")
        values[key.strip()] = value.strip()
    return values


def write_env(path: Path, updates: dict[str, str]) -> None:
    """Rewrite only `OWNED_KEYS` lines that changed; preserve everything else."""
    lines = path.read_text().splitlines() if path.exists() else []
    seen: set[str] = set()
    new_lines: list[str] = []
    for line in lines:
        stripped = line.strip()
        if "=" in stripped and not stripped.startswith("#"):
            key = stripped.split("=", 1)[0].strip()
            if key in OWNED_KEYS and key in updates:
                new_lines.append(f"{key}={updates[key]}")
                seen.add(key)
                continue
        new_lines.append(line)
    for key, value in updates.items():
        if key not in seen:
            new_lines.append(f"{key}={value}")
    path.write_text("\n".join(new_lines) + "\n")


class CloudflareError(RuntimeError):
    pass


def api_request(
    method: str, path: str, token: str, *, body: dict | None = None
) -> dict:
    url = f"{API_BASE}{path}"
    data = json.dumps(body).encode() if body is not None else None
    request = urllib.request.Request(url, data=data, method=method)
    request.add_header("Authorization", f"Bearer {token}")
    request.add_header("Content-Type", "application/json")
    try:
        with urllib.request.urlopen(request, timeout=30) as response:
            payload = json.loads(response.read())
    except urllib.error.HTTPError as exc:
        payload = json.loads(exc.read())
        raise CloudflareError(
            f"{method} {path} -> {exc.code}: {payload.get('errors')}"
        ) from None
    if not payload.get("success"):
        raise CloudflareError(f"{method} {path} failed: {payload.get('errors')}")
    return payload["result"]


def get_zone_id(token: str, zone_name: str) -> str:
    result = api_request("GET", f"/zones?name={zone_name}", token)
    if not result:
        raise CloudflareError(f"no zone named {zone_name!r}")
    return result[0]["id"]


def get_or_create_tunnel(token: str, account_id: str) -> str:
    result = api_request(
        "GET",
        f"/accounts/{account_id}/cfd_tunnel?name={TUNNEL_NAME}&is_deleted=false",
        token,
    )
    if result:
        return result[0]["id"]
    result = api_request(
        "POST",
        f"/accounts/{account_id}/cfd_tunnel",
        token,
        body={"name": TUNNEL_NAME, "config_src": "cloudflare"},
    )
    return result["id"]


def get_tunnel_token(token: str, account_id: str, tunnel_id: str) -> str:
    result = api_request(
        "GET", f"/accounts/{account_id}/cfd_tunnel/{tunnel_id}/token", token
    )
    return result if isinstance(result, str) else result.get("token", result)


def set_ingress(token: str, account_id: str, tunnel_id: str, hostname: str) -> None:
    api_request(
        "PUT",
        f"/accounts/{account_id}/cfd_tunnel/{tunnel_id}/configurations",
        token,
        body={
            "config": {
                "ingress": [
                    {"hostname": hostname, "service": "http://web:8000"},
                    {"service": "http_status:404"},
                ]
            }
        },
    )


def ensure_dns_record(
    token: str, zone_id: str, hostname: str, tunnel_id: str
) -> None:
    content = f"{tunnel_id}.cfargotunnel.com"
    result = api_request("GET", f"/zones/{zone_id}/dns_records?name={hostname}", token)
    if result:
        record = result[0]
        if record["type"] != "CNAME":
            raise CloudflareError(
                f"{hostname} already has a {record['type']} record; "
                "refusing to replace it (HANDOFF.md section 7)"
            )
        if record["content"] == content and record.get("proxied") is True:
            return
        api_request(
            "PUT",
            f"/zones/{zone_id}/dns_records/{record['id']}",
            token,
            body={"type": "CNAME", "name": hostname, "content": content, "proxied": True},
        )
        return
    api_request(
        "POST",
        f"/zones/{zone_id}/dns_records",
        token,
        body={"type": "CNAME", "name": hostname, "content": content, "proxied": True},
    )


def ensure_turnstile_widget(
    token: str, account_id: str, hostname: str, existing_secret: str
) -> tuple[str, str]:
    """Returns (site_key, secret_key). Secret is empty when reusing an
    existing widget whose secret is already known (`existing_secret`)."""
    result = api_request(
        "GET", f"/accounts/{account_id}/challenges/widgets", token
    )
    widget = next((w for w in result if w["sitekey"] and w.get("name") == TURNSTILE_WIDGET_NAME), None)
    if widget is None:
        created = api_request(
            "POST",
            f"/accounts/{account_id}/challenges/widgets",
            token,
            body={
                "name": TURNSTILE_WIDGET_NAME,
                "domains": [hostname],
                "mode": "managed",
            },
        )
        return created["sitekey"], created["secret"]
    if existing_secret:
        return widget["sitekey"], existing_secret
    rotated = api_request(
        "POST",
        f"/accounts/{account_id}/challenges/widgets/{widget['sitekey']}/rotate_secret",
        token,
    )
    return widget["sitekey"], rotated["secret"]


def main() -> int:
    env = read_env(ENV_PATH)
    token = env.get("CLOUDFLARE_API_TOKEN")
    account_id = env.get("CLOUDFLARE_ACCOUNT_ID")
    zone_name = env.get("CLOUDFLARE_ZONE_NAME")
    hostname = env.get("PUBLIC_HOSTNAME")
    missing = [
        name
        for name, value in (
            ("CLOUDFLARE_API_TOKEN", token),
            ("CLOUDFLARE_ACCOUNT_ID", account_id),
            ("CLOUDFLARE_ZONE_NAME", zone_name),
            ("PUBLIC_HOSTNAME", hostname),
        )
        if not value
    ]
    if missing:
        print(f"missing required .env values: {', '.join(missing)}", file=sys.stderr)
        return 1

    try:
        zone_id = get_zone_id(token, zone_name)
        tunnel_id = get_or_create_tunnel(token, account_id)
        tunnel_token = get_tunnel_token(token, account_id, tunnel_id)
        set_ingress(token, account_id, tunnel_id, hostname)
        ensure_dns_record(token, zone_id, hostname, tunnel_id)
        site_key, secret_key = ensure_turnstile_widget(
            token, account_id, hostname, env.get("TURNSTILE_SECRET_KEY", "")
        )
    except CloudflareError as exc:
        print(f"Cloudflare setup failed: {exc}", file=sys.stderr)
        return 1

    write_env(
        ENV_PATH,
        {
            "CLOUDFLARE_TUNNEL_TOKEN": tunnel_token,
            "TURNSTILE_SITE_KEY": site_key,
            "TURNSTILE_SECRET_KEY": secret_key,
        },
    )
    print(f"tunnel {tunnel_id!r} ready, ingress set for {hostname}, DNS and Turnstile updated")
    print(f"wrote CLOUDFLARE_TUNNEL_TOKEN, TURNSTILE_SITE_KEY, TURNSTILE_SECRET_KEY to {ENV_PATH}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
