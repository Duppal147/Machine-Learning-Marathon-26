"""Sort a generation run's mistakes into buckets that each suggest a different fix.

Every question is scored with the official Score.py functions, so the points lost per
bucket add up to the gap between the run's score and 1.0. Whether the evidence reached
the LLM is judged on the passages saved in the run's checkpoint, with the same fuzzy
evidence match as the retrieval metrics.
"""
from __future__ import annotations

import importlib.util
import json
from dataclasses import dataclass
from pathlib import Path

from .retrieval import PASSAGE_THRESHOLD, _parse_ref_ids, evidence_coverage, extract_evidence

# Bucket -> what it points at. Order is how the report lists them when points tie.
BUCKETS = {
    "correct": "value and citations right",
    "correct_na": "unanswerable, abstained cleanly",
    "citation_off": "value right, citations partly wrong: citation logic",
    "abstain_had_evidence": "abstained though the evidence was in the passages: prompt, abstention policy",
    "abstain_gold_doc_only": "abstained, right paper but evidence not located: figures, tables, more passages",
    "abstain_no_gold_doc": "abstained, right paper not in the passages: retrieval",
    "wrong_had_evidence": "wrong value though the evidence was in the passages: reading, math, format, model",
    "wrong_gold_doc_only": "wrong value, right paper but evidence not located: figures, tables, more passages",
    "wrong_no_gold_doc": "wrong value, right paper not in the passages: retrieval",
    "answered_unanswerable": "answered a question the corpus can't answer: abstention policy",
}


def classify(gold_is_na: bool, predicted_blank: bool, value_ok: bool, ref_f1: float,
             has_evidence: bool, has_gold_doc: bool) -> str:
    if gold_is_na:
        return "correct_na" if value_ok else "answered_unanswerable"
    if value_ok:
        return "correct" if ref_f1 == 1.0 else "citation_off"
    prefix = "abstain" if predicted_blank else "wrong"
    if has_evidence:
        return f"{prefix}_had_evidence"
    return f"{prefix}_gold_doc_only" if has_gold_doc else f"{prefix}_no_gold_doc"


def answer_type(value: str) -> str:
    value = value.strip()
    if value == "is_blank":
        return "unanswerable"
    if value.startswith(("(", "[")):
        return "list/range"
    if value in ("0", "1"):
        return "true/false"
    try:
        float(value.replace(",", ""))
        return "number"
    except ValueError:
        return "text"


def load_scorer(path: Path):
    spec = importlib.util.spec_from_file_location("wattbot_score", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@dataclass
class ErrorRow:
    id: str
    bucket: str
    score: float
    value_ok: bool
    ref_f1: float
    has_evidence: bool
    has_gold_doc: bool
    question: str
    gold_value: str
    predicted_value: str
    gold_refs: str
    predicted_refs: str
    answer_type: str
    cohort: str
    flags: list[str]
    explanation: str


FLAGS = ("Table", "Figure", "Math", "CrossPaper", "Reconcile", "Quote")


def analyze_answer(gold: dict, record: dict, scorer) -> ErrorRow:
    """Score one checkpoint against its gold row and put it in a bucket."""
    row = record["row"]
    payload = {key: row[key] for key in ("answer_value", "answer_unit", "ref_id")}
    payload.update({key: row.get(key, "") for key in ("ref_url", "supporting_materials")})
    bits = scorer.row_bits(sol={key: gold[key] for key in ("answer_value", "answer_unit", "ref_id")}, sub=payload)
    _, _, f1 = scorer.ref_prf(gold["ref_id"], row["ref_id"])
    gold_refs = _parse_ref_ids(gold["ref_id"])
    in_gold = [p for p in record["passages"] if p.get("ref_id") in gold_refs]
    evidence = extract_evidence(gold.get("supporting_materials") or "")
    has_evidence = any(evidence_coverage(seg, p["text"]) >= PASSAGE_THRESHOLD for seg in evidence for p in in_gold)
    gold_is_na = scorer.is_blank(gold["answer_value"])
    bucket = classify(gold_is_na, scorer.is_blank(row["answer_value"]), bool(bits["val"]), f1,
                      has_evidence, bool(in_gold))
    return ErrorRow(
        id=gold["id"], bucket=bucket, score=0.75 * bool(bits["val"]) + 0.20 * f1 + 0.05 * bool(bits["na"]),
        value_ok=bool(bits["val"]), ref_f1=f1, has_evidence=has_evidence, has_gold_doc=bool(in_gold),
        question=gold["question"], gold_value=gold["answer_value"], predicted_value=row["answer_value"],
        gold_refs=" ".join(gold_refs), predicted_refs=" ".join(json.loads(row["ref_id"]))
        if row["ref_id"].startswith("[") else row["ref_id"],
        answer_type=answer_type(gold["answer_value"]), cohort=str(gold.get("Cohort", "")),
        flags=[flag for flag in FLAGS if gold.get(flag) == "1"], explanation=row.get("explanation", ""))


def load_records(run_dir: Path) -> list[dict]:
    """All completed checkpoints; works on an unfinished run too."""
    return [json.loads(path.read_text()) for path in sorted(Path(run_dir, "answers").glob("*.json"))]
