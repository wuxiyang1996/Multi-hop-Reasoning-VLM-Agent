from pathlib import Path
import sys

from motif_transfer.clevrer_compact_event_graph import compact_event_graph


ROOT = Path("/fs/gamma-projects/vlm-robot/datasets/CLEVRER-official")


def test_compact_graph_contains_only_prediction_facts():
    sys.path.insert(0, str(ROOT / "executor"))
    from executor import Executor
    from simulation import Simulation
    path = ROOT / "off_the_shelf_nsdr/propnet_preds/with_edge_supervision_old/sim_10000.json"
    text = compact_event_graph(Executor(Simulation(str(path), use_event_ann=True)))
    assert "OBJECTS:" in text and "OBSERVED_EVENTS:" in text
    assert "MOTION_STATES" in text and "COUNTERFACTUAL_REMOVE_" in text
    assert "answer" not in text.casefold() and "question" not in text.casefold()


def test_actor_schema_rejects_choice_indices():
    import importlib.util
    path = Path(__file__).resolve().parents[1] / "scripts/collect_clevrer_qwen9b_graph_actor_v1.py"
    spec = importlib.util.spec_from_file_location("collector", path)
    module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
    import pytest
    with pytest.raises(ValueError):
        module._extract('{"q":"2"}', {"q": 4})
    assert module._extract('{"q":"1010"}', {"q": 4}) == {"q": "1010"}
