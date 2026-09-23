"""Isolated compose stack for a synthetic bedside scene."""

from __future__ import annotations

import os
import subprocess
import time
from pathlib import Path
from urllib.request import urlopen

import redis

SERVICES = ("agent", "listen", "embodiment", "store", "light", "notify")


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
        while time.monotonic() < deadline:
            try:
                if self._redis().ping():
                    with urlopen("http://localhost:18443/", timeout=2) as response:
                        if response.status == 200:
                            return
            except (OSError, redis.RedisError):
                pass
            time.sleep(1)
        raise TimeoutError("nightsim Redis or embodiment did not become ready")

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
