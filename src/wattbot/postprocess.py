"""
Postprocessor: clean and normalize generated answers for submission.

Handles:
  - ref_id list-string cleaning: "['luccioni2025c']" -> "luccioni2025c"
  - ref_url list-string cleaning: "['https://...']" -> "https://..."
  - True/False -> 1/0 in answer_value
  - is_blank consistency: if answer_value is is_blank, all evidence columns must be too
  - Strip units from answer_value (keep only the number)
  - Validate ref_ids against metadata.csv
"""

import re
import pandas as pd


def clean_list_string(val: str) -> str:
    if not isinstance(val, str):
        return str(val) if val is not None else ""
    val = val.strip()
    if val.lower() in ("is_blank", "na", "nan", "", "none"):
        return "is_blank"
    val = val.strip("[]")
    parts = [p.strip().strip("'\"") for p in val.split(",")]
    parts = [p for p in parts if p and p.lower() not in ("nan", "none")]
    if not parts:
        return "is_blank"
    return ", ".join(parts)


def normalize_answer_value(val: str) -> str:
    if not isinstance(val, str):
        val = str(val) if val is not None else ""
    val = val.strip()

    if val.lower() in ("is_blank", "na", "nan", "", "none", "unable to answer"):
        return "is_blank"

    low = val.lower()
    if low == "true":
        return "1"
    if low == "false":
        return "0"

    range_match = re.match(r"^[\[\(]([\d.,\s\-]+)[\]\)]$", val)
    if range_match:
        return val

    number_match = re.match(r"^[~≈]?\s*([\d,]+\.?\d*)", val)
    if number_match:
        return number_match.group(1).replace(",", "")

    return val


def enforce_is_blank(row: dict) -> dict:
    evidence_cols = ["ref_id", "ref_url", "supporting_materials"]
    av = str(row.get("answer_value", "")).strip().lower()

    if av == "is_blank":
        for col in evidence_cols:
            row[col] = "is_blank"
    else:
        for col in evidence_cols:
            v = str(row.get(col, "")).strip().lower()
            if v in ("is_blank", "na", "nan", "", "none"):
                row[col] = "is_blank"

    return row


def validate_ref_ids(ref_id_str: str, valid_ids: set) -> str:
    if ref_id_str == "is_blank":
        return ref_id_str
    parts = [p.strip() for p in ref_id_str.split(",")]
    valid = [p for p in parts if p in valid_ids]
    if not valid:
        return "is_blank"
    return ", ".join(valid)


def postprocess_row(row: dict, valid_doc_ids: set | None = None) -> dict:
    row["answer_value"] = normalize_answer_value(str(row.get("answer_value", "")))
    row["ref_id"] = clean_list_string(str(row.get("ref_id", "")))
    row["ref_url"] = clean_list_string(str(row.get("ref_url", "")))

    if valid_doc_ids:
        row["ref_id"] = validate_ref_ids(row["ref_id"], valid_doc_ids)

    row = enforce_is_blank(row)

    if not row.get("explanation", "").strip():
        row["explanation"] = "No explanation provided."

    return row


def postprocess_dataframe(df: pd.DataFrame, metadata_df: pd.DataFrame | None = None) -> pd.DataFrame:
    valid_ids = None
    if metadata_df is not None:
        valid_ids = set(metadata_df["id"].dropna().astype(str).tolist())

    rows = df.to_dict("records")
    processed = [postprocess_row(r, valid_ids) for r in rows]
    return pd.DataFrame(processed)
