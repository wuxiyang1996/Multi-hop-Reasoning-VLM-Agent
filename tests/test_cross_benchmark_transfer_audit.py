from motif_transfer.contracts import stable_hash
from motif_transfer.cross_benchmark_transfer_audit import (
    TransferAuditError,
    validate_self_hash,
)


def test_validate_self_hash_accepts_exact_payload():
    body = {"value": 3}
    validate_self_hash(body | {"summary_sha256": stable_hash(body)}, "summary_sha256")


def test_validate_self_hash_rejects_mutation():
    body = {"value": 3}
    payload = body | {"summary_sha256": stable_hash(body)}
    payload["value"] = 4
    try:
        validate_self_hash(payload, "summary_sha256")
    except TransferAuditError:
        pass
    else:
        raise AssertionError("mutated evidence was accepted")
