from pathlib import Path

from motif_transfer.portable_paths import resolve_repo_artifact


def test_relative_artifact_is_checkout_relative(tmp_path: Path) -> None:
    repo = tmp_path / "Multi-hop-Reasoning-VLM-Agent-two-agent-clean"
    assert resolve_repo_artifact("runs/example/report.json", repo) == (
        repo.resolve() / "runs/example/report.json"
    )


def test_historical_absolute_artifact_is_remapped_to_active_checkout(
    tmp_path: Path,
) -> None:
    repo = tmp_path / "Multi-hop-Reasoning-VLM-Agent-two-agent-clean"
    historical = Path(
        "/fs/gamma-projects/vlm-robot/"
        "Multi-hop-Reasoning-VLM-Agent-two-agent-clean/runs/example/report.json"
    )
    assert resolve_repo_artifact(historical, repo) == (
        repo.resolve() / "runs/example/report.json"
    )


def test_unrelated_absolute_artifact_is_not_rewritten(tmp_path: Path) -> None:
    repo = tmp_path / "Multi-hop-Reasoning-VLM-Agent-two-agent-clean"
    external = Path("/datasets/official/video.mp4")
    assert resolve_repo_artifact(external, repo) == external
