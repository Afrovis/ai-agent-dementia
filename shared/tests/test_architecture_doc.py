from pathlib import Path

import pytest

from nc_shared.archdoc import (
    ArchitectureError,
    generate_document,
    inspect_services,
)

REPO_ROOT = Path(__file__).resolve().parents[2]


def test_architecture_document_is_current() -> None:
    expected = generate_document(REPO_ROOT)
    architecture_path = REPO_ROOT / "ARCHITECTURE.md"

    assert architecture_path.read_text(encoding="utf-8") == expected, (
        "ARCHITECTURE.md is out of date. Run:  python -m nc_shared.archdoc --write"
    )


def test_unresolvable_subscription_fails_loudly(tmp_path: Path) -> None:
    package = tmp_path / "example" / "example"
    package.mkdir(parents=True)
    (package / "main.py").write_text(
        "def connect(bus, configured_stream):\n"
        "    bus.ensure_group(configured_stream, 'example')\n",
        encoding="utf-8",
    )

    with pytest.raises(ArchitectureError, match=r"main\.py:2: cannot resolve ensure_group stream"):
        inspect_services(tmp_path)
