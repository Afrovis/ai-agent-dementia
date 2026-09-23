# TLS for the media bridge

The bedside page uses `getUserMedia`, which browsers only grant on a trustworthy
origin. A self-signed certificate is not enough: browsers restrict camera and
microphone access on pages with certificate errors, so the eyes render but the
bridge stays dead. Plain HTTP on `localhost` is the only exception.

Over Tailscale you can get a real Let's Encrypt certificate. Enable HTTPS
Certificates on the DNS page of the Tailscale admin console first, then:

```sh
tailscale status --json | python3 -c "import json,sys; print(json.load(sys.stdin)['Self']['DNSName'].rstrip('.'))"
tailscale cert --cert-file ts.crt --key-file ts.key <that-name>
```

On macOS the CLI is not on `PATH`. It lives at
`/Applications/Tailscale.app/Contents/MacOS/Tailscale`.

Two traps here.

The macOS Tailscale client is sandboxed. It ignores the paths you give it and
writes to `~/Library/Containers/io.tailscale.ipn.macos/Data`. Copy the files
out of there into `data/certs/lan.pem` and `data/certs/lan-key.pem`, then
`docker compose restart embodiment`.

The certificate covers the tailnet name only. Once installed, `localhost`,
the LAN address and the `.local` name all fail validation. Use the tailnet
name everywhere, including on the host itself, where MagicDNS resolves it
locally and validates cleanly.

Certificates last 90 days. Re-run `tailscale cert` to renew, and remember the
sandbox path again.

Verify a certificate is genuinely trusted with strict checks. `curl -k` or a
browser flag that ignores certificate errors hides exactly the failure this
page cares about:

```sh
curl -s -o /dev/null -w "%{http_code}\n" https://<name>:8443/
openssl s_client -connect <name>:8443 -servername <name> </dev/null 2>/dev/null | grep 'Verify return code'
```
