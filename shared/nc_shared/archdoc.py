"""Generate the as-built service and Redis-stream architecture document.

The event registry is the source of truth for streams and event types. Service
publishers and subscribers are discovered from Python source with :mod:`ast`;
no service modules are imported or executed while generating the document.
"""

from __future__ import annotations

import argparse
import ast
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from nc_shared.events import EVENT_STREAMS


class ArchitectureError(RuntimeError):
    """Raised when service source cannot be translated without guessing."""


@dataclass(frozen=True, order=True)
class Subscription:
    """One service consumer-group subscription."""

    stream: str
    group: str


@dataclass
class ServiceFacts:
    """Bus interactions discovered for one service."""

    publishes: set[str] = field(default_factory=set)
    subscribes: set[Subscription] = field(default_factory=set)


@dataclass(frozen=True)
class ExternalConnection:
    """A hand-declared connection that does not travel over the event bus."""

    service: str
    thing: str
    direction: str
    declared_at: str


CAPPED_STREAMS = {"audio_in": 50, "frames": 50, "frames_raw": 50}

EXTERNAL_CONNECTIONS = (
    ExternalConnection(
        "embodiment",
        "Browser page, websockets `/ws` and `/media`",
        "both",
        "services/embodiment/embodiment/app.py",
    ),
    ExternalConnection(
        "embodiment",
        "Photos under `data/photos` and demo photos",
        "reads",
        "services/embodiment/embodiment/app.py",
    ),
    ExternalConnection(
        "embodiment",
        "Piper voice model and ephemeral generated WAV cache",
        "reads/writes",
        "services/embodiment/embodiment/tts.py",
    ),
    ExternalConnection(
        "embodiment",
        "`config/strategies.yaml` and `config/person.yaml` for speech warming",
        "reads",
        "services/embodiment/embodiment/tts.py",
    ),
    ExternalConnection(
        "capture",
        "USB or RTSP camera, optional",
        "reads",
        "services/capture/capture/sources.py",
    ),
    ExternalConnection(
        "perceive",
        "Ollama `/api/generate` on the host",
        "calls",
        "services/perceive/perceive/vision.py",
    ),
    ExternalConnection(
        "perceive",
        "`config/zones.yaml` at startup",
        "reads",
        "services/perceive/perceive/zones.py",
    ),
    ExternalConnection(
        "listen",
        "faster-whisper model cache; downloads `small.en` on first use",
        "reads/writes",
        "services/listen/listen/transcribe.py",
    ),
    ExternalConnection(
        "agent",
        "`config/strategies.yaml` at startup",
        "reads",
        "services/agent/agent/strategies.py",
    ),
    ExternalConnection(
        "agent",
        "Ollama `/api/generate` on the host",
        "calls",
        "services/agent/agent/llm.py",
    ),
    ExternalConnection(
        "notify",
        "ntfy topic, or the log when `NTFY_URL` is empty",
        "calls",
        "services/notify/notify/backends.py",
    ),
    ExternalConnection(
        "store",
        "SQLite `data/night.db`",
        "writes",
        "services/store/store/main.py",
    ),
    ExternalConnection(
        "dashboard",
        "Caregiver browser, HTTP Basic auth",
        "both",
        "services/dashboard/dashboard/app.py",
    ),
    ExternalConnection(
        "dashboard",
        "`config/zones.yaml`",
        "writes",
        "services/dashboard/dashboard/app.py",
    ),
)


def _stream_events() -> dict[str, tuple[str, ...]]:
    streams: dict[str, list[str]] = {}
    for event_class, stream in EVENT_STREAMS.items():
        streams.setdefault(stream, []).append(event_class.__name__)
    return {stream: tuple(sorted(events)) for stream, events in sorted(streams.items())}


class _UnknownValue(Exception):
    pass


def _evaluate(node: ast.AST, values: dict[str, Any]) -> Any:
    """Evaluate the deliberately tiny constant language used by stream declarations."""
    if isinstance(node, ast.Constant) and isinstance(node.value, str):
        return node.value
    if isinstance(node, ast.Name):
        try:
            return values[node.id]
        except KeyError as exc:
            raise _UnknownValue(node.id) from exc
    if isinstance(node, (ast.Set, ast.List, ast.Tuple)):
        items = [_evaluate(item, values) for item in node.elts]
        return set(items) if isinstance(node, ast.Set) else items
    if (
        isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and isinstance(node.func.value, ast.Name)
        and node.func.value.id == "EVENT_STREAMS"
        and node.func.attr == "values"
        and not node.args
        and not node.keywords
    ):
        return list(EVENT_STREAMS.values())
    if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and len(node.args) <= 1:
        argument = _evaluate(node.args[0], values) if node.args else []
        if node.func.id == "set":
            return set(argument)
        if node.func.id == "sorted":
            return sorted(argument)
    if isinstance(node, ast.BinOp) and isinstance(node.op, ast.Sub):
        return set(_evaluate(node.left, values)) - set(_evaluate(node.right, values))
    raise _UnknownValue(ast.dump(node, include_attributes=False))


class _SourceVisitor(ast.NodeVisitor):
    def __init__(
        self,
        path: Path,
        values: dict[str, Any],
        event_names: set[str],
        event_aliases: dict[str, str],
    ) -> None:
        self.path = path
        self.values = values
        self.event_names = event_names
        self.event_aliases = event_aliases
        self.constructed: set[str] = set()
        self.subscriptions: set[Subscription] = set()
        self.has_publish = False

    def visit_Call(self, node: ast.Call) -> None:
        called_name: str | None = None
        if isinstance(node.func, ast.Name):
            called_name = self.event_aliases.get(node.func.id, node.func.id)
        elif isinstance(node.func, ast.Attribute):
            called_name = node.func.attr
        if called_name in self.event_names:
            self.constructed.add(called_name)

        if isinstance(node.func, ast.Attribute) and node.func.attr == "publish":
            self.has_publish = True
        if isinstance(node.func, ast.Attribute) and node.func.attr == "ensure_group":
            if len(node.args) < 2:
                self._fail(node, "ensure_group must have stream and group arguments")
            stream = self._resolve_string(node.args[0], node, "stream")
            group = self._resolve_string(node.args[1], node, "consumer group")
            self.subscriptions.add(Subscription(stream, group))
        self.generic_visit(node)

    def visit_For(self, node: ast.For) -> None:
        """Expand simple loops such as store's ``for stream in PERSISTED_STREAMS``."""
        if isinstance(node.target, ast.Name):
            try:
                items = _evaluate(node.iter, self.values)
            except _UnknownValue:
                items = None
            if isinstance(items, (list, tuple, set)):
                previous = self.values.get(node.target.id, _MISSING)
                for item in sorted(items) if isinstance(items, set) else items:
                    self.values[node.target.id] = item
                    for statement in node.body:
                        self.visit(statement)
                if previous is _MISSING:
                    self.values.pop(node.target.id, None)
                else:
                    self.values[node.target.id] = previous
                for statement in node.orelse:
                    self.visit(statement)
                return
        self.generic_visit(node)

    def _resolve_string(self, argument: ast.AST, call: ast.Call, label: str) -> str:
        try:
            value = _evaluate(argument, self.values)
        except _UnknownValue as exc:
            self._fail(call, f"cannot resolve ensure_group {label}: {exc}")
        if not isinstance(value, str):
            self._fail(call, f"ensure_group {label} did not resolve to one string")
        return value

    def _fail(self, node: ast.AST, message: str) -> None:
        raise ArchitectureError(f"{self.path}:{node.lineno}: {message}")


_MISSING = object()


def _module_values(tree: ast.Module) -> dict[str, Any]:
    values: dict[str, Any] = {}
    for statement in tree.body:
        if not isinstance(statement, (ast.Assign, ast.AnnAssign)):
            continue
        targets = statement.targets if isinstance(statement, ast.Assign) else [statement.target]
        value_node = statement.value
        if value_node is None:
            continue
        try:
            value = _evaluate(value_node, values)
        except _UnknownValue:
            continue
        for target in targets:
            if isinstance(target, ast.Name):
                values[target.id] = value
    return values


def _event_aliases(tree: ast.Module, event_names: set[str]) -> dict[str, str]:
    aliases: dict[str, str] = {}
    for statement in tree.body:
        if isinstance(statement, ast.ImportFrom) and statement.module == "nc_shared.events":
            for imported in statement.names:
                if imported.name in event_names:
                    aliases[imported.asname or imported.name] = imported.name
    return aliases


def inspect_services(services_root: Path) -> dict[str, ServiceFacts]:
    """Discover bus publications and subscriptions below ``services_root``."""
    event_names = {event_class.__name__ for event_class in EVENT_STREAMS}
    known_streams = set(_stream_events())
    facts: dict[str, ServiceFacts] = {}
    for service_dir in sorted(path for path in services_root.iterdir() if path.is_dir()):
        source_dir = service_dir / service_dir.name
        if not source_dir.is_dir():
            continue
        service = ServiceFacts()
        has_publish = False
        for path in sorted(source_dir.rglob("*.py")):
            tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
            visitor = _SourceVisitor(
                path, _module_values(tree), event_names, _event_aliases(tree, event_names)
            )
            visitor.visit(tree)
            has_publish = has_publish or visitor.has_publish
            service.publishes.update(visitor.constructed)
            service.subscribes.update(visitor.subscriptions)
        if has_publish and not service.publishes:
            raise ArchitectureError(
                f"service {service_dir.name} contains .publish(...) but constructs no "
                "registered event; cannot tell what it publishes"
            )
        for subscription in service.subscribes:
            if subscription.stream not in known_streams:
                raise ArchitectureError(
                    f"service {service_dir.name} subscribes to unknown stream "
                    f"{subscription.stream!r}"
                )
        facts[service_dir.name] = service
    return facts


def _publishers_by_stream(facts: dict[str, ServiceFacts]) -> dict[str, list[str]]:
    event_to_stream = {
        event_class.__name__: stream for event_class, stream in EVENT_STREAMS.items()
    }
    result: dict[str, set[str]] = {stream: set() for stream in _stream_events()}
    for service, service_facts in facts.items():
        for event_name in service_facts.publishes:
            result[event_to_stream[event_name]].add(service)
    return {stream: sorted(services) for stream, services in result.items()}


def _subscribers_by_stream(
    facts: dict[str, ServiceFacts],
) -> dict[str, list[tuple[str, str]]]:
    result: dict[str, list[tuple[str, str]]] = {stream: [] for stream in _stream_events()}
    for service, service_facts in facts.items():
        for subscription in service_facts.subscribes:
            result[subscription.stream].append((service, subscription.group))
    return {stream: sorted(consumers) for stream, consumers in result.items()}


def _main_diagram(facts: dict[str, ServiceFacts]) -> str:
    stream_events = _stream_events()
    publishers = _publishers_by_stream(facts)
    subscribers = _subscribers_by_stream(facts)
    lines = ["```mermaid", "flowchart LR"]
    lines.extend(
        [
            '  subgraph room["Room"]',
            '    browser["Night-screen browser<br/>camera, microphone, face, text"]',
            '    camera["Optional USB or RTSP camera"]',
            "  end",
            "",
            '  subgraph compose["docker compose"]',
        ]
    )
    for service in sorted(facts):
        lines.append(f'    s_{service}["{service}"]')
    for stream, events in stream_events.items():
        if stream != "health":
            lines.append(f'    q_{stream}[("{stream}<br/>{", ".join(events)}")]')
    lines.extend(
        [
            "  end",
            "",
            '  subgraph host["Host (macOS)"]',
            '    ollama["Ollama<br/>vision and text models"]',
            "  end",
            "",
            '  subgraph caregiver["Caregiver"]',
            '    phone["Phone via ntfy"]',
            '    cg_browser["Caregiver browser<br/>zones editor"]',
            "  end",
            "",
            '  subgraph disk["Disk"]',
            '    zones["config/zones.yaml"]',
            '    strategies["config/strategies.yaml"]',
            '    photos["data/photos and demo photos"]',
            "  end",
            "",
        ]
    )
    for stream in stream_events:
        if stream == "health":
            continue
        for service in publishers[stream]:
            lines.append(f"  s_{service} --> q_{stream}")
        for service, _group in subscribers[stream]:
            if service != "store":
                lines.append(f"  q_{stream} --> s_{service}")
    lines.extend(
        [
            '  browser <-->|"websockets /ws and /media"| s_embodiment',
            '  camera -.->|"frames"| s_capture',
            '  s_perceive -.->|"HTTP"| ollama',
            '  s_notify -.->|"ntfy"| phone',
            "  cg_browser <--> s_dashboard",
            '  zones -.->|"read at startup"| s_perceive',
            '  s_dashboard -.->|"writes"| zones',
            '  strategies -.->|"read at startup"| s_agent',
            '  photos -.->|"reads"| s_embodiment',
            "```",
        ]
    )
    return "\n".join(lines)


def _health_and_persistence_diagram(facts: dict[str, ServiceFacts]) -> str:
    publishers = _publishers_by_stream(facts)
    subscribers = _subscribers_by_stream(facts)
    persisted = [
        stream
        for stream, readers in subscribers.items()
        if any(service == "store" for service, _group in readers)
    ]
    lines = [
        "```mermaid",
        "flowchart LR",
        '  q_health[("health<br/>Health")]',
        '  s_store["store"]',
        '  sqlite[("data/night.db")]',
    ]
    for service in publishers["health"]:
        lines.append(f'  s_{service}["{service}"] --> q_health')
    lines.append("  q_health --> s_store")
    for stream in persisted:
        if stream != "health":
            lines.append(f'  q_p_{stream}[("{stream}")] --> s_store')
    lines.extend(['  s_store -.->|"writes"| sqlite', "```"])
    return "\n".join(lines)


def _markdown_cell(value: str) -> str:
    return value.replace("|", "\\|").replace("\n", " ")


def generate_document(repo_root: Path) -> str:
    """Return the complete generated ``ARCHITECTURE.md`` text."""
    repo_root = repo_root.resolve()
    for connection in EXTERNAL_CONNECTIONS:
        declared_path = repo_root / connection.declared_at
        if not declared_path.is_file():
            raise ArchitectureError(
                f"outside connection for {connection.service} points to missing file: "
                f"{connection.declared_at}"
            )

    facts = inspect_services(repo_root / "services")
    streams = _stream_events()
    publishers = _publishers_by_stream(facts)
    subscribers = _subscribers_by_stream(facts)

    parts = [
        "<!-- Generated by nc_shared.archdoc. Do not edit by hand. -->",
        "# As-built architecture",
        "",
        "This document is generated from the event registry and service source. "
        "Regenerate it with:",
        "",
        "```sh",
        "python -m nc_shared.archdoc --write",
        "```",
        "",
        "## Data flow",
        "",
        "Solid arrows carry runtime data. Dotted arrows are direct connections to the "
        "outside world.",
        "The diagram deliberately shows unconnected streams and services: they are not "
        "implemented yet.",
        "",
        _main_diagram(facts),
        "",
        "## Health and persistence",
        "",
        "Health traffic and persistence are separated from the main diagram to keep its "
        "data path readable.",
        "",
        _health_and_persistence_diagram(facts),
        "",
        "## Streams",
        "",
        "| Stream | Events | Published by | Read by (consumer group) | Capped |",
        "| --- | --- | --- | --- | --- |",
    ]
    for stream, events in streams.items():
        published = ", ".join(publishers[stream]) or "nobody yet"
        read = ", ".join(f"{service} (`{group}`)" for service, group in subscribers[stream])
        capped = f"yes, {CAPPED_STREAMS[stream]}" if stream in CAPPED_STREAMS else "no"
        parts.append(
            f"| `{stream}` | {', '.join(f'`{event}`' for event in events)} | "
            f"{published} | {read or 'nobody yet'} | {capped} |"
        )
    parts.extend(
        [
            "",
            "## Outside-world connections",
            "",
            "These connections do not use Redis streams, so they are declared in the "
            "generator and checked",
            "against the files that implement them.",
            "",
            "| Service | Outside thing | Direction | Declared at |",
            "| --- | --- | --- | --- |",
        ]
    )
    for connection in EXTERNAL_CONNECTIONS:
        parts.append(
            f"| `{connection.service}` | {_markdown_cell(connection.thing)} | "
            f"{connection.direction} | `{connection.declared_at}` |"
        )
    return "\n".join(parts) + "\n"


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    action = parser.add_mutually_exclusive_group()
    action.add_argument("--write", action="store_true", help="write ARCHITECTURE.md")
    action.add_argument("--check", action="store_true", help="fail if ARCHITECTURE.md is stale")
    args = parser.parse_args(argv)
    repo_root = _repo_root()
    generated = generate_document(repo_root)
    output_path = repo_root / "ARCHITECTURE.md"
    if args.write:
        output_path.write_text(generated, encoding="utf-8")
    elif args.check:
        if not output_path.is_file() or output_path.read_text(encoding="utf-8") != generated:
            print(
                "ARCHITECTURE.md is out of date. Run:  python -m nc_shared.archdoc --write",
                file=sys.stderr,
            )
            return 1
    else:
        print(generated, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
