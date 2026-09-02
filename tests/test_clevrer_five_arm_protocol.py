import importlib.util
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def _module(name, relative):
    spec = importlib.util.spec_from_file_location(name, ROOT / relative)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_one_sided_exact_significance():
    module = _module("clevrer_eval", "scripts/evaluate_clevrer_five_arm_predictions_v1.py")
    assert module._one_sided(10, 0) < 0.05
    assert module._one_sided(0, 10) == 1.0
    assert module._one_sided(0, 0) == 1.0


def test_paired_counts_exact():
    module = _module("clevrer_eval_pair", "scripts/evaluate_clevrer_five_arm_predictions_v1.py")
    rows = [
        {"correct": {"a": True, "b": False}},
        {"correct": {"a": False, "b": True}},
        {"correct": {"a": True, "b": True}},
    ]
    value = module._paired(rows, "a", "b")
    assert value["wins"] == 1 and value["losses"] == 1 and value["ties"] == 1
