"""
Pipeline orchestrator: question CSV -> retrieve -> generate -> postprocess -> submission CSV.
"""

import os
import pandas as pd
from tqdm import tqdm

from . import config
from .retriever import retrieve
from .generator import generate
from .postprocess import postprocess_dataframe


def load_questions(path: str) -> pd.DataFrame:
    return pd.read_csv(path, encoding="utf-8-sig")


def load_metadata(path: str | None = None) -> pd.DataFrame:
    if path is None:
        path = os.path.join(config.DATA_DIR, "metadata.csv")
    return pd.read_csv(path, encoding="utf-8-sig")


def run_pipeline(
    questions_path: str,
    output_path: str,
    metadata_path: str | None = None,
    limit: int | None = None,
    collection=None,
):
    questions_df = load_questions(questions_path)
    metadata_df = load_metadata(metadata_path)

    if limit:
        questions_df = questions_df.head(limit)

    results = []

    for _, row in tqdm(questions_df.iterrows(), total=len(questions_df), desc="Processing"):
        qid = row["id"]
        question = row["question"]
        unit_hint = str(row.get("answer_unit", "")) if pd.notna(row.get("answer_unit")) else ""

        chunks = retrieve(question, collection=collection)
        answer = generate(question, chunks, unit_hint=unit_hint)

        result = {
            "id": qid,
            "question": question,
            **answer,
        }
        results.append(result)

    results_df = pd.DataFrame(results)
    results_df = postprocess_dataframe(results_df, metadata_df)

    for col in config.SUBMISSION_COLUMNS:
        if col not in results_df.columns:
            results_df[col] = ""

    results_df = results_df[config.SUBMISSION_COLUMNS]
    results_df.to_csv(output_path, index=False)
    print(f"Submission written to {output_path} ({len(results_df)} rows)")
    return results_df
