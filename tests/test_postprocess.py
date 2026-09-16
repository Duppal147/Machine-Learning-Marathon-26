import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from wattbot.postprocess import (
    clean_list_string,
    normalize_answer_value,
    enforce_is_blank,
    postprocess_row,
)


def test_clean_single_ref_id():
    assert clean_list_string("['luccioni2025c']") == "luccioni2025c"


def test_clean_multi_ref_id():
    assert clean_list_string("['patterson2021', 'jegham2025']") == "patterson2021, jegham2025"


def test_clean_bare_url():
    assert clean_list_string("https://arxiv.org/pdf/2506.15572") == "https://arxiv.org/pdf/2506.15572"


def test_clean_list_url():
    result = clean_list_string("['https://arxiv.org/pdf/2104.10350', 'https://arxiv.org/pdf/2505.09598']")
    assert "https://arxiv.org/pdf/2104.10350" in result
    assert "https://arxiv.org/pdf/2505.09598" in result


def test_clean_is_blank():
    assert clean_list_string("is_blank") == "is_blank"
    assert clean_list_string("") == "is_blank"
    assert clean_list_string("nan") == "is_blank"


def test_normalize_true_false():
    assert normalize_answer_value("True") == "1"
    assert normalize_answer_value("TRUE") == "1"
    assert normalize_answer_value("False") == "0"
    assert normalize_answer_value("FALSE") == "0"


def test_normalize_numeric():
    assert normalize_answer_value("53") == "53"
    assert normalize_answer_value("~119") == "119"
    assert normalize_answer_value("0.3") == "0.3"


def test_normalize_is_blank():
    assert normalize_answer_value("is_blank") == "is_blank"
    assert normalize_answer_value("") == "is_blank"
    assert normalize_answer_value("NA") == "is_blank"


def test_normalize_range_preserved():
    assert normalize_answer_value("[116.9,121.7]") == "[116.9,121.7]"
    assert normalize_answer_value("(1,5)") == "(1,5)"


def test_enforce_is_blank_blanks_evidence():
    row = {
        "answer_value": "is_blank",
        "ref_id": "luccioni2025c",
        "ref_url": "https://example.com",
        "supporting_materials": "some quote",
    }
    result = enforce_is_blank(row)
    assert result["ref_id"] == "is_blank"
    assert result["ref_url"] == "is_blank"
    assert result["supporting_materials"] == "is_blank"


def test_enforce_is_blank_keeps_evidence():
    row = {
        "answer_value": "53",
        "ref_id": "luccioni2025c",
        "ref_url": "https://example.com",
        "supporting_materials": "some quote",
    }
    result = enforce_is_blank(row)
    assert result["ref_id"] == "luccioni2025c"


def test_postprocess_row_full():
    row = {
        "answer_value": "True",
        "ref_id": "['luccioni2025c']",
        "ref_url": "['https://arxiv.org/pdf/2506.15572']",
        "supporting_materials": "quote here",
        "explanation": "reasoning here",
    }
    valid_ids = {"luccioni2025c", "patterson2021"}
    result = postprocess_row(row, valid_ids)
    assert result["answer_value"] == "1"
    assert result["ref_id"] == "luccioni2025c"


def test_postprocess_row_invalid_ref_id():
    row = {
        "answer_value": "42",
        "ref_id": "nonexistent_paper",
        "ref_url": "https://example.com",
        "supporting_materials": "quote",
        "explanation": "test",
    }
    valid_ids = {"luccioni2025c"}
    result = postprocess_row(row, valid_ids)
    assert result["ref_id"] == "is_blank"


def test_postprocess_row_empty_explanation():
    row = {
        "answer_value": "42",
        "ref_id": "luccioni2025c",
        "ref_url": "https://example.com",
        "supporting_materials": "quote",
        "explanation": "",
    }
    result = postprocess_row(row)
    assert result["explanation"] == "No explanation provided."
