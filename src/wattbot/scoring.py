"""
Unofficial local scorer implementing the WattBot Score formula.

WattBot Score = 0.75 * answer_accuracy + 0.20 * citation_f1 + 0.05 * abstention_accuracy

Use the official Score.py from Kaggle for leaderboard-accurate scoring.
This is for quick sanity checks only.
"""

import re
import pandas as pd


def _parse_numeric(val: str) -> float | None:
    if not isinstance(val, str):
        return None
    val = val.strip().replace(",", "")
    try:
        return float(val)
    except ValueError:
        return None


def _parse_range(val: str) -> tuple[float, float] | None:
    match = re.match(r"^[\[\(]\s*([\d.eE\+\-]+)\s*,\s*([\d.eE\+\-]+)\s*[\]\)]$", val.strip())
    if match:
        try:
            return (float(match.group(1)), float(match.group(2)))
        except ValueError:
            return None
    return None


def _normalize(val: str) -> str:
    if not isinstance(val, str):
        return ""
    val = val.strip().lower().replace(",", "")
    if val in ("is_blank", "na", "nan", "", "none"):
        return "is_blank"
    if val == "true":
        return "1"
    if val == "false":
        return "0"
    return val


def score_answer_value(pred: str, gold: str) -> float:
    pred_n = _normalize(pred)
    gold_n = _normalize(gold)

    if gold_n == "is_blank":
        return 1.0 if pred_n == "is_blank" else 0.0

    if pred_n == "is_blank":
        return 0.0

    gold_range = _parse_range(gold)
    if gold_range is not None:
        pred_num = _parse_numeric(pred_n)
        if pred_num is not None:
            lo, hi = gold_range
            return 1.0 if lo <= pred_num <= hi else 0.0
        return 0.0

    gold_num = _parse_numeric(gold_n)
    pred_num = _parse_numeric(pred_n)
    if gold_num is not None and pred_num is not None:
        if gold_num == 0:
            return 1.0 if pred_num == 0 else 0.0
        rel_err = abs(pred_num - gold_num) / abs(gold_num)
        return 1.0 if rel_err <= 0.001 else 0.0

    return 1.0 if pred_n == gold_n else 0.0


def _parse_ref_ids(val: str) -> set[str]:
    if not isinstance(val, str):
        return set()
    val = val.strip().lower()
    if val in ("is_blank", "na", "nan", "", "none"):
        return set()
    val = val.strip("[]")
    ids = {p.strip().strip("'\"") for p in val.split(",")}
    return {i for i in ids if i and i not in ("is_blank", "nan", "none")}


def score_citation_f1(pred_refs: str, gold_refs: str) -> float:
    pred_set = _parse_ref_ids(pred_refs)
    gold_set = _parse_ref_ids(gold_refs)

    if not gold_set and not pred_set:
        return 1.0
    if not gold_set or not pred_set:
        return 0.0

    tp = len(pred_set & gold_set)
    precision = tp / len(pred_set) if pred_set else 0.0
    recall = tp / len(gold_set) if gold_set else 0.0

    if precision + recall == 0:
        return 0.0
    return 2 * precision * recall / (precision + recall)


def score_abstention(pred_row: dict, gold_row: dict) -> float:
    gold_av = _normalize(str(gold_row.get("answer_value", "")))
    if gold_av != "is_blank":
        return 1.0

    pred_av = _normalize(str(pred_row.get("answer_value", "")))
    if pred_av != "is_blank":
        return 0.0

    for col in ("ref_id", "ref_url", "supporting_materials"):
        pv = _normalize(str(pred_row.get(col, "")))
        if pv != "is_blank":
            return 0.0

    return 1.0


def score_submission(pred_df: pd.DataFrame, gold_df: pd.DataFrame) -> dict:
    merged = gold_df.merge(pred_df, on="id", suffixes=("_gold", "_pred"), how="left")

    answer_scores = []
    citation_scores = []
    abstention_scores = []

    for _, row in merged.iterrows():
        av_pred = str(row.get("answer_value_pred", "is_blank"))
        av_gold = str(row.get("answer_value_gold", "is_blank"))
        answer_scores.append(score_answer_value(av_pred, av_gold))

        ref_pred = str(row.get("ref_id_pred", "is_blank"))
        ref_gold = str(row.get("ref_id_gold", "is_blank"))
        citation_scores.append(score_citation_f1(ref_pred, ref_gold))

        pred_row = {c.replace("_pred", ""): row[c] for c in row.index if c.endswith("_pred")}
        gold_row = {c.replace("_gold", ""): row[c] for c in row.index if c.endswith("_gold")}
        abstention_scores.append(score_abstention(pred_row, gold_row))

    n = len(answer_scores) or 1
    avg_answer = sum(answer_scores) / n
    avg_citation = sum(citation_scores) / n
    avg_abstention = sum(abstention_scores) / n

    wattbot_score = 0.75 * avg_answer + 0.20 * avg_citation + 0.05 * avg_abstention

    return {
        "wattbot_score": round(wattbot_score, 4),
        "answer_accuracy": round(avg_answer, 4),
        "citation_f1": round(avg_citation, 4),
        "abstention_accuracy": round(avg_abstention, 4),
        "n_questions": n,
    }
