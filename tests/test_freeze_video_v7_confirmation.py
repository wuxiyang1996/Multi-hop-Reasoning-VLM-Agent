import json

from scripts.freeze_video_v7_confirmation import main


def test_frozen_video_confirmation_is_disjoint(tmp_path, monkeypatch):
    source = tmp_path / "source.json"
    output = tmp_path / "output.json"
    source.write_text(json.dumps({
        "counts_per_family": {"held_out": 3},
        "benchmarks": {
            "b": {
                "families": ["x", "y"],
                "splits": {"held_out": ["x0", "x1", "x2", "y0", "y1", "y2"]},
            },
        },
    }))
    monkeypatch.setattr("sys.argv", [
        "freeze", "--input", str(source), "--output", str(output),
        "--confirmation-per-family", "2",
    ])
    main()
    frozen = json.loads(output.read_text())
    splits = frozen["benchmarks"]["b"]["splits"]
    assert splits["confirmation"] == ["x0", "x1", "y0", "y1"]
    assert splits["reserve"] == ["x2", "y2"]
    assert not set(splits["confirmation"]) & set(splits["reserve"])
    assert frozen["outcomes_or_answers_read"] is False
