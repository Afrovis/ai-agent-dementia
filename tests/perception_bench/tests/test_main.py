"""End-to-end CLI test: `python -m perception_bench --ir-manifest ...`
actually runs tier 3 through to a report without a camera, mic, Ollama, or
network (this issue's hard constraint) -- see also `tests/test_infrared.py`
for `run_tier3` unit coverage."""

from pathlib import Path

from perception_bench.__main__ import main


def test_main_runs_tier3_against_a_manifest_with_no_clips(tmp_path, capsys):
    manifest = tmp_path / "manifest.yaml"
    manifest.write_text("clips: []\n", encoding="utf-8")

    exit_code = main(
        [
            "--skip-daylight",
            "--ir-manifest",
            str(manifest),
        ]
    )

    assert exit_code == 0
    out = capsys.readouterr().out
    assert "Tier 3" in out
    assert "NOT MEASURED" in out


def test_main_json_report_includes_tier3_section(tmp_path, capsys):
    exit_code = main(
        [
            "--json",
            "--skip-daylight",
            "--ir-manifest",
            str(Path("/nonexistent/manifest.yaml")),
        ]
    )

    assert exit_code == 0
    out = capsys.readouterr().out
    assert '"tier3"' in out
    assert '"missing_reason"' in out
