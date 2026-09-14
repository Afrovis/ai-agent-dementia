from pathlib import Path

from video_eval.__main__ import main
from video_eval.common import write_jsonl
from video_eval.zones import zone_reference_frame


def _manifest(root: Path, *, bridge_key: str = "bridge_path") -> Path:
    clip = root / "clips" / "clip"
    bridge = clip / ("bridge-letterbox" if bridge_key == "bridge_letterbox_path" else "bridge")
    bridge.mkdir(parents=True)
    frame = bridge / "f_000000.jpg"
    frame.write_bytes(b"jpeg")
    write_jsonl(
        clip / "frames.jsonl",
        [{"frame_index": 0, "t_s": 0.0, bridge_key: str(frame.relative_to(root))}],
    )
    return frame


def test_zone_reference_frame_returns_prepared_squash_frame(tmp_path):
    frame = _manifest(tmp_path)

    assert zone_reference_frame("clip", root=tmp_path) == frame.resolve()


def test_zones_cli_prints_letterbox_reference_frame(tmp_path, capsys):
    frame = _manifest(tmp_path, bridge_key="bridge_letterbox_path")

    status = main(
        [
            "--data-root",
            str(tmp_path),
            "zones",
            "--clip",
            "clip",
            "--variant",
            "letterbox",
        ]
    )

    assert status == 0
    assert capsys.readouterr().out.strip() == str(frame.resolve())


def test_zone_reference_frame_requires_prepared_variant(tmp_path):
    _manifest(tmp_path)

    try:
        zone_reference_frame("clip", root=tmp_path, variant="letterbox")
    except RuntimeError as exc:
        assert "prepare --variant letterbox" in str(exc)
    else:
        raise AssertionError("expected missing variant to be rejected")
