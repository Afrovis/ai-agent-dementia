from pathlib import Path

import yaml
from PIL import Image

from video_eval.common import read_jsonl
from video_eval.prepare import BRIDGE_SIZE, prepare_video, produce_bridge_frames


def _image(path: Path, size: tuple[int, int] = (160, 90), color: str = "white") -> None:
    Image.new("RGB", size, color).save(path, format="JPEG")


def test_produce_bridge_frames_matches_squash_and_letterbox_formats(tmp_path):
    review = tmp_path / "f_000000.jpg"
    _image(review)

    squash = tmp_path / "squash"
    letterbox = tmp_path / "letterbox"
    produce_bridge_frames([review], squash, "squash")
    produce_bridge_frames([review], letterbox, "letterbox")

    with Image.open(squash / review.name) as image:
        assert image.size == BRIDGE_SIZE
        assert image.getpixel((160, 0))[0] > 240
    with Image.open(letterbox / review.name) as image:
        assert image.size == BRIDGE_SIZE
        assert max(image.getpixel((160, 0))) < 10
        assert min(image.getpixel((160, 120))) > 240


def test_prepare_writes_card_manifest_meta_index_and_skips_matching_run(tmp_path):
    source = tmp_path / "source.mov"
    source.write_bytes(b"not decoded by injected extractor")
    data_root = tmp_path / "private"
    calls = 0

    def fake_extract(video: Path, destination: Path) -> None:
        nonlocal calls
        calls += 1
        assert video == data_root / "raw" / "2026-09-13_test.mov"
        destination.mkdir(parents=True)
        _image(destination / "f_000000.jpg")
        _image(destination / "f_000001.jpg", color="gray")

    probe = {
        "duration_s": 1.0,
        "width": 1920,
        "height": 1080,
        "frame_rate": "30/1",
        "codec": "h264",
        "rotation": 0,
    }

    def fake_probe(video: Path):
        assert video == data_root / "raw" / "2026-09-13_test.mov"
        return probe

    first = prepare_video(
        source,
        "2026-09-13_test",
        root=data_root,
        probe_fn=fake_probe,
        extract_fn=fake_extract,
    )
    second = prepare_video(
        source,
        "2026-09-13_test",
        root=data_root,
        probe_fn=fake_probe,
        extract_fn=fake_extract,
    )

    assert first == {"status": "complete", "frames": 2}
    assert second == {"status": "skipped", "frames": 2}
    assert calls == 1
    manifest = read_jsonl(data_root / "clips" / "2026-09-13_test" / "frames.jsonl")
    assert [row["t_s"] for row in manifest] == [0.0, 0.5]
    assert all("bridge_path" in row for row in manifest)
    card = yaml.safe_load((data_root / "clips" / "2026-09-13_test" / "clip.yaml").read_text())
    assert card["script"] == []
    assert card["video"] == probe
    assert (data_root / "clips" / "2026-09-13_test" / "prepare.meta.json").exists()
    index = yaml.safe_load((data_root / "index.yaml").read_text())
    assert index["clips"]["2026-09-13_test"]["prepare_squash"] == "complete"


def test_second_variant_is_added_to_existing_manifest(tmp_path):
    source = tmp_path / "source.mov"
    source.write_bytes(b"video")
    data_root = tmp_path / "private"

    def fake_extract(_video: Path, destination: Path) -> None:
        destination.mkdir(parents=True)
        _image(destination / "f_000000.jpg")

    kwargs = {
        "root": data_root,
        "probe_fn": lambda _: {"duration_s": 0.5},
        "extract_fn": fake_extract,
    }
    prepare_video(source, "clip", **kwargs)
    prepare_video(source, "clip", variant="letterbox", **kwargs)

    row = read_jsonl(data_root / "clips" / "clip" / "frames.jsonl")[0]
    assert "bridge_path" in row
    assert "bridge_letterbox_path" in row


def test_prepare_refuses_to_replace_a_raw_clip_without_force(tmp_path):
    first = tmp_path / "first.mov"
    second = tmp_path / "second.mov"
    first.write_bytes(b"first")
    second.write_bytes(b"second")
    data_root = tmp_path / "private"

    def fake_extract(_video: Path, destination: Path) -> None:
        destination.mkdir(parents=True)
        _image(destination / "f_000000.jpg")

    kwargs = {
        "root": data_root,
        "probe_fn": lambda _: {"duration_s": 0.5},
        "extract_fn": fake_extract,
    }
    prepare_video(first, "clip", **kwargs)

    try:
        prepare_video(second, "clip", **kwargs)
    except RuntimeError as exc:
        assert "different content" in str(exc)
    else:
        raise AssertionError("expected immutable raw clip check")
