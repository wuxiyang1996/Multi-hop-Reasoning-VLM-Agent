import sys
from pathlib import Path


SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(SCRIPTS))

from collect_agqa2_frame_grounding_v2 import _answer_matches  # noqa: E402


def test_closed_set_answers_accept_explanatory_suffixes():
    assert _answer_matches("No, the event was shorter.", "no")
    assert _answer_matches("after going from standing to sitting", "after")
    assert _answer_matches("before holding a book", "before")
    assert not _answer_matches("before opening the laptop", "after")


def test_free_object_answers_remain_exact_after_article_normalization():
    assert _answer_matches("a chair", "chair")
    assert not _answer_matches("a dish or object", "dish")
