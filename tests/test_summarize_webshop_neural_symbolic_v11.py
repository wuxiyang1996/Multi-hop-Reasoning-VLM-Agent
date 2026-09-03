from scripts.summarize_webshop_neural_symbolic_v11 import CONTROLS


def test_v11_requires_all_non_authentic_controls() -> None:
    assert set(CONTROLS) == {
        "target_only",
        "target_native_myopic",
        "shuffled_source_plus_target",
        "source_marginal_plus_target",
    }
