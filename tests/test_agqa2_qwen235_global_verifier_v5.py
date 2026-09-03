from scripts.collect_agqa2_qwen235_global_verifier_v5 import _direct_boolean


def test_direct_boolean_reads_only_canonical_first_token():
    assert _direct_boolean("Yes, they did.") == "yes"
    assert _direct_boolean("No.") == "no"
    assert _direct_boolean("a dish") is None
