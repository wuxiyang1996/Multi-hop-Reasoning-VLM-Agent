import json

import pytest

from motif_transfer.bank import MotifBank
from motif_transfer.contracts import Lifecycle, MotifCandidate, TransferReport
from motif_transfer.legacy_import import load_jsonl


def test_bank_does_not_promote_from_text_only():
    bank = MotifBank()
    bank.add_candidate(MotifCandidate("m", (), (), (), untrusted_description="excellent skill"))
    with pytest.raises(ValueError):
        bank.mark_source_supported("m", ())
    with pytest.raises(ValueError):
        bank.apply_transfer_report("m", TransferReport(Lifecycle.POSITIVE_TRANSFER, "agent says yes"))


def test_legacy_import_has_no_execution_authority(tmp_path):
    source = tmp_path / "legacy.jsonl"
    source.write_text(json.dumps({"skill": "COLLECT"}) + "\n", encoding="utf-8")
    row = next(iter(load_jsonl(source)))
    assert row.authority == "LINEAGE_RETRIEVAL_ONLY"
    assert not hasattr(row, "action")
