import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from wattbot.scoring import score_answer_value, score_citation_f1, score_abstention, score_submission
import pandas as pd


def test_exact_numeric():
    assert score_answer_value("53", "53") == 1.0


def test_numeric_tolerance():
    assert score_answer_value("53.05", "53") == 1.0
    assert score_answer_value("54", "53") == 0.0


def test_is_blank_match():
    assert score_answer_value("is_blank", "is_blank") == 1.0


def test_is_blank_mismatch():
    assert score_answer_value("42", "is_blank") == 0.0
    assert score_answer_value("is_blank", "42") == 0.0


def test_range_bracket():
    assert score_answer_value("119", "[116.9,121.7]") == 1.0
    assert score_answer_value("115", "[116.9,121.7]") == 0.0


def test_true_false():
    assert score_answer_value("1", "1") == 1.0
    assert score_answer_value("0", "0") == 1.0
    assert score_answer_value("1", "0") == 0.0


def test_categorical():
    assert score_answer_value("Alphabet", "Alphabet") == 1.0
    assert score_answer_value("alphabet", "Alphabet") == 1.0


def test_citation_f1_exact():
    assert score_citation_f1("luccioni2025c", "['luccioni2025c']") == 1.0


def test_citation_f1_partial():
    f1 = score_citation_f1("patterson2021", "['patterson2021', 'jegham2025']")
    assert 0.6 < f1 < 0.7  # precision=1, recall=0.5, F1=0.667


def test_citation_f1_extra():
    f1 = score_citation_f1("patterson2021, jegham2025, extra2024", "['patterson2021', 'jegham2025']")
    assert 0.7 < f1 < 0.9  # precision=2/3, recall=1, F1=0.8


def test_citation_f1_blank():
    assert score_citation_f1("is_blank", "is_blank") == 1.0
    assert score_citation_f1("something", "is_blank") == 0.0


def test_abstention_correct():
    pred = {"answer_value": "is_blank", "ref_id": "is_blank", "ref_url": "is_blank", "supporting_materials": "is_blank"}
    gold = {"answer_value": "is_blank"}
    assert score_abstention(pred, gold) == 1.0


def test_abstention_missed():
    pred = {"answer_value": "42", "ref_id": "x", "ref_url": "x", "supporting_materials": "x"}
    gold = {"answer_value": "is_blank"}
    assert score_abstention(pred, gold) == 0.0


def test_abstention_non_na_gold():
    pred = {"answer_value": "42", "ref_id": "x", "ref_url": "x", "supporting_materials": "x"}
    gold = {"answer_value": "42"}
    assert score_abstention(pred, gold) == 1.0


def test_self_score_perfect():
    gold_df = pd.DataFrame([
        {"id": "q1", "answer_value": "53", "ref_id": "['luccioni2025c']",
         "ref_url": "url", "supporting_materials": "quote"},
        {"id": "q2", "answer_value": "is_blank", "ref_id": "is_blank",
         "ref_url": "is_blank", "supporting_materials": "is_blank"},
    ])
    result = score_submission(gold_df, gold_df)
    assert result["wattbot_score"] == 1.0
