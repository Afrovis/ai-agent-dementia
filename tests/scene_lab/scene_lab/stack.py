"""Isolated compose stack for a synthetic bedside scene."""

from __future__ import annotations

import json
import os
import subprocess
import time
from pathlib import Path
from urllib.request import Request, urlopen

import redis

SERVICES = ("agent", "listen", "embodiment", "store", "light", "notify")

# Host Ollama, as seen from scene_lab on the host (the agent container reaches
# the same server through host.docker.internal).
OLLAMA_URL = os.environ.get("SCENE_LAB_OLLAMA_URL", "http://127.0.0.1:11434")


def _ollama_keep_alive(model: str, keep_alive: int, timeout: float) -> bool:
    """A prompt-less generate: keep_alive 0 unloads the model, -1 loads it."""
    body = json.dumps({"model": model, "keep_alive": keep_alive}).encode()
    request = Request(
        f"{OLLAMA_URL}/api/generate",
        data=body,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urlopen(request, timeout=timeout) as response:  # noqa: S310
            return response.status == 200
    except OSError:
        return False


class Stack:
    def __init__(
        self,
        repo_root: str | Path,
        project: str = "nightsim",
        env: dict | None = None,
        redis_port: int = 16379,
    ):
        self.repo_root = Path(repo_root).resolve()
        self.project = project
        self.redis_port = redis_port
        self.env = os.environ.copy() | (env or {})
        self.command = [
            "docker",
            "compose",
            "-p",
            project,
            "-f",
            "docker-compose.yml",
            "-f",
            "tests/scene_lab/compose.sim.yml",
        ]

    def _call(self, *args: str, capture: bool = False) -> str:
        result = subprocess.run(
            [*self.command, *args],
            cwd=self.repo_root,
            env=self.env,
            check=True,
            text=True,
            capture_output=capture,
        )
        return result.stdout if capture else ""

    def up(self, build: bool = True) -> None:
        self._call(
            "up",
            "-d",
            *(["--build"] if build else []),
            "bus",
            "store",
            "listen",
            "agent",
            "light",
            "embodiment",
            "notify",
        )

    def down(self) -> None:
        self._call("down")

    def restart(self, *services: str) -> None:
        self._call("restart", *services)

    def stop(self, *services: str) -> None:
        self._call("stop", *services)

    def start(self, *services: str) -> None:
        self._call("start", *services)

    def reset(self, services: tuple[str, ...] = SERVICES) -> None:
        """Fresh session state: stop consumers, flush the nightsim bus, start them again.

        Flushing while services run deletes their consumer groups under them (NOGROUP).
        """
        self.stop(*services)
        self.flush()
        self.start(*services)

    def logs(self, service: str, since: str) -> str:
        return self._call("logs", "--no-color", "--since", since, service, capture=True)

    def _redis(self):
        return redis.Redis(host="localhost", port=self.redis_port)

    def flush(self) -> None:
        if self.redis_port != 16379:
            raise ValueError("refusing FLUSHALL outside nightsim port 16379")
        self._redis().flushall()

    def wait_ready(self, timeout: float = 90) -> None:
        deadline = time.monotonic() + timeout
        last_error = "no response"
        while time.monotonic() < deadline:
            try:
                client = self._redis()
                # PING passes while Redis refuses writes (MISCONF); the services
                # need writes, so probe one.
                client.set("nightsim:ready", "1", ex=60)
                with urlopen("http://localhost:18443/", timeout=2) as response:
                    if response.status == 200:
                        return
                    last_error = f"embodiment HTTP {response.status}"
            except (OSError, redis.RedisError) as exc:
                last_error = f"{type(exc).__name__}: {str(exc)[:200]}"
            time.sleep(1)
        raise TimeoutError(f"nightsim Redis or embodiment did not become ready ({last_error})")

    def unload_llm(self, model: str, timeout: float = 60) -> bool:
        """Unload the agent's model from host Ollama.

        Ollama's MLX runner keeps a prefix cache that grows about 35 MB per
        request and only evicts near MLX's memory limit, about 16 GiB on the
        16 GB host. A night's worth of scenes pushed it there and the host
        swapped until requests hung (2026-09-23). Unloading frees it all.
        """
        return _ollama_keep_alive(model, 0, timeout)

    def reset_llm(self, model: str, timeout: float = 120) -> dict:
        """Unload and reload the model so a scene starts with an empty prefix
        cache and no cold start inside the scene (a load takes about 20 s)."""
        started = time.monotonic()
        unloaded = self.unload_llm(model)
        loaded = _ollama_keep_alive(model, -1, timeout)
        return {
            "unloaded": unloaded,
            "loaded": loaded,
            "seconds": round(time.monotonic() - started, 1),
        }

    def preflight(self) -> dict:
        def output(args):
            try:
                return subprocess.run(
                    args, cwd=self.repo_root, text=True, capture_output=True, check=True
                ).stdout.strip()
            except (OSError, subprocess.CalledProcessError) as exc:
                return f"unavailable: {exc}"

        names = output(["docker", "ps", "--format", "{{.Names}}"])
        other = [
            name
            for name in names.splitlines()
            if name.endswith("-agent-1") and not name.startswith(self.project)
        ]
        return {
            "commit": output(["git", "rev-parse", "HEAD"]),
            "ollama_ps": output(["ollama", "ps"]),
            "other_agent_stacks": other,
            "contention": bool(other),
        }


def main_models(repo_root: Path) -> Path:
    common = subprocess.check_output(
        ["git", "rev-parse", "--git-common-dir"], cwd=repo_root, text=True
    ).strip()
    return (repo_root / common).resolve().parent / "data/models"
