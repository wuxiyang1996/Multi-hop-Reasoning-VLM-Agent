from __future__ import annotations

import importlib.util
from pathlib import Path


REPO = Path(__file__).resolve().parents[1]
SCRIPT = REPO / "scripts/recover_alfworld_target_acquisition_py311_v24.py"


def _load_module():
    spec = importlib.util.spec_from_file_location("v24_recovery", SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_selection_schema_alias_preserves_serialized_content() -> None:
    module = _load_module()
    original = {
        "selection_used_identity_only": True,
        module.NEW_FIELD: False,
    }
    aliased = module._SelectionSchemaAlias(original)

    assert aliased[module.OLD_FIELD] is False
    assert dict(aliased) == original
    assert module.OLD_FIELD not in aliased


def test_selection_schema_alias_does_not_hide_unrelated_missing_keys() -> None:
    module = _load_module()
    aliased = module._SelectionSchemaAlias({module.NEW_FIELD: False})

    try:
        aliased["not_a_selection_field"]
    except KeyError as error:
        assert error.args == ("not_a_selection_field",)
    else:
        raise AssertionError("an unrelated missing field must still fail closed")
