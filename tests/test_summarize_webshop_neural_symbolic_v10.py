from scripts.summarize_webshop_neural_symbolic_v10 import exact_binomial_two_sided


def test_exact_binomial_two_sided_requires_six_unopposed_wins_for_significance() -> None:
    assert exact_binomial_two_sided(5, 0) == 0.0625
    assert exact_binomial_two_sided(6, 0) == 0.03125
    assert exact_binomial_two_sided(0, 0) == 1.0
