from __future__ import annotations

from pathlib import Path

import pytest

from harness.provider_clients import load_literal_secret


def test_load_literal_secret_reads_only_named_literal(tmp_path: Path) -> None:
    path = tmp_path / "keys.py"
    path.write_text('OTHER = "ignore"\nOPENAI_API_KEY = "secret-value"\n')
    assert load_literal_secret(path, "OPENAI_API_KEY") == "secret-value"


def test_load_literal_secret_refuses_executable_expression(tmp_path: Path) -> None:
    path = tmp_path / "keys.py"
    path.write_text('OPENAI_API_KEY = get_secret()\n')
    with pytest.raises(ValueError, match="literal string"):
        load_literal_secret(path, "OPENAI_API_KEY")
