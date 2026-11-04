import json
import subprocess
import sys
from pathlib import Path

import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from wattbot.eval.errors import answer_type, classify

SCORER = ROOT / "WattBot2026" / "Score.py"


@pytest.mark.parametrize("args,bucket", [
    ((True, True, True, 1.0, False, False), "correct_na"),
    ((True, False, False, 0.0, False, False), "answered_unanswerable"),
    ((False, False, True, 1.0, True, True), "correct"),
    ((False, False, True, 0.5, True, True), "citation_off"),
    ((False, True, False, 0.0, True, True), "abstain_had_evidence"),
    ((False, True, False, 0.0, False, True), "abstain_gold_doc_only"),
    ((False, True, False, 0.0, False, False), "abstain_no_gold_doc"),
    ((False, False, False, 1.0, True, True), "wrong_had_evidence"),
    ((False, False, False, 1.0, False, True), "wrong_gold_doc_only"),
    ((False, False, False, 0.0, False, False), "wrong_no_gold_doc"),
])
def test_classify(args, bucket):
    assert classify(*args) == bucket


def test_answer_type():
    assert [answer_type(v) for v in ["is_blank", "(2,5)", "1", "0", "4,096", "1.5e6", "Alphabet"]] == \
        ["unanswerable", "list/range", "true/false", "true/false", "number", "number", "text"]


def _gold(id, value, ref, evidence, unit="kWh"):
    return {"id": id, "question": f"question {id}", "answer": value, "answer_value": value, "answer_unit": unit,
            "ref_id": ref, "ref_url": "x", "supporting_materials": evidence, "explanation": "e", "Cohort": "2026",
            "Corrected": "0", "Quote": "1", "Table": "0", "Figure": "0", "Math": "0", "is_NA": "0",
            "CrossPaper": "0", "Reconcile": "0"}


def _record(id, value, refs, passages):
    blank = value == "is_blank"
    row = {"id": id, "question": f"question {id}", "answer": "a", "answer_value": value, "answer_unit": "kWh",
           "ref_id": "is_blank" if blank else json.dumps(refs), "ref_url": "is_blank" if blank else "[]",
           "supporting_materials": "is_blank" if blank else "q", "explanation": "because"}
    return {"question": {"id": id, "question": row["question"], "answer_unit": "kWh"}, "row": row,
            "passages": passages, "passage_ids": []}


@pytest.mark.skipif(not SCORER.exists(), reason="WattBot data not available")
def test_report_matches_official_score(tmp_path):
    """Per-question scores must average to exactly what Score.py reports for the run."""
    evidence_text = "the data center consumed 1.9 billion gallons of water per day in total"
    has_it = [{"ref_id": "a2024", "text": evidence_text}]
    other_doc = [{"ref_id": "z2024", "text": "unrelated text about something else entirely here"}]
    gold_rows = [
        _gold("q1", "1.9", "['a2024']", evidence_text),                    # correct
        _gold("q2", "1.9", "['a2024']", evidence_text),                    # abstained with evidence
        _gold("q3", "1.9", "['a2024']", evidence_text),                    # wrong, no gold doc
        _gold("q4", "1.9", "['a2024', 'b2024']", evidence_text),           # right value, half the citations
        _gold("q5", "is_blank", "is_blank", "is_blank"),                   # clean refusal
        _gold("q6", "is_blank", "is_blank", "is_blank"),                   # invented an answer
    ]
    records = [_record("q1", "1.9", ["a2024"], has_it), _record("q2", "is_blank", [], has_it),
               _record("q3", "7", ["z2024"], other_doc), _record("q4", "1.9", ["a2024"], has_it),
               _record("q5", "is_blank", [], other_doc), _record("q6", "3", ["z2024"], other_doc)]
    (tmp_path / "answers").mkdir()
    for record in records:
        (tmp_path / "answers" / f"{record['question']['id']}.json").write_text(json.dumps(record))
    gold_csv = tmp_path / "gold.csv"
    pd.DataFrame(gold_rows).to_csv(gold_csv, index=False)

    result = subprocess.run([sys.executable, str(ROOT / "scripts" / "analyze_errors.py"), str(tmp_path),
                             "--gold", str(gold_csv), "--scorer", str(SCORER)],
                            capture_output=True, text=True, cwd=ROOT)
    assert result.returncode == 0, result.stderr
    details = pd.read_csv(tmp_path / "errors_by_question.csv", keep_default_na=False).set_index("id")
    assert details.loc["q1", "bucket"] == "correct"
    assert details.loc["q2", "bucket"] == "abstain_had_evidence"
    assert details.loc["q3", "bucket"] == "wrong_no_gold_doc"
    assert details.loc["q4", "bucket"] == "citation_off"
    assert details.loc["q5", "bucket"] == "correct_na"
    assert details.loc["q6", "bucket"] == "answered_unanswerable"

    from wattbot.eval.errors import load_scorer
    submission = pd.DataFrame([r["row"] for r in records])
    official = load_scorer(SCORER).score(pd.DataFrame(gold_rows), submission, "id", verbose=False)
    assert details["score"].mean() == pytest.approx(official)
    assert f"score {official:.4f}" in result.stdout
