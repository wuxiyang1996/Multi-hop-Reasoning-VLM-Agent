from pathlib import Path

import pytest

from scripts.merge_completion_caches import merge_caches


def test_merge_caches_deduplicates_identical_entries(tmp_path: Path) -> None:
    first = tmp_path / "first.json"
    second = tmp_path / "second.json"
    first.write_text('{"backend_identity_sha256":"x","entries":{"a":{"completion":"1"}}}')
    second.write_text(
        '{"backend_identity_sha256":"x","entries":'
        '{"a":{"completion":"1"},"b":{"completion":"2"}}}'
    )
    assert set(merge_caches([first, second])["entries"]) == {"a", "b"}


def test_merge_caches_rejects_identity_mismatch(tmp_path: Path) -> None:
    first = tmp_path / "first.json"
    second = tmp_path / "second.json"
    first.write_text('{"backend_identity_sha256":"x","entries":{}}')
    second.write_text('{"backend_identity_sha256":"y","entries":{}}')
    with pytest.raises(ValueError, match="different backend identities"):
        merge_caches([first, second])
