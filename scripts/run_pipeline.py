#!/usr/bin/env python3
"""CLI entrypoint for the WattBot RAG pipeline."""

import argparse
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))


def main():
    parser = argparse.ArgumentParser(description="WattBot 2026 RAG Pipeline")
    sub = parser.add_subparsers(dest="command")

    # --- ingest ---
    p_ingest = sub.add_parser("ingest", help="Download PDFs, extract text, chunk, embed, store")
    p_ingest.add_argument("--metadata", default=None, help="Path to metadata.csv")
    p_ingest.add_argument("--limit", type=int, default=None, help="Process only first N documents")

    # --- run ---
    p_run = sub.add_parser("run", help="Run the full RAG pipeline on a question CSV")
    p_run.add_argument("--questions", required=True, help="Path to question CSV (test_Q.csv or train_QA.csv)")
    p_run.add_argument("--output", default="submission.csv", help="Output submission CSV path")
    p_run.add_argument("--metadata", default=None, help="Path to metadata.csv")
    p_run.add_argument("--limit", type=int, default=None, help="Process only first N questions")

    # --- score ---
    p_score = sub.add_parser("score", help="Score a submission against gold answers")
    p_score.add_argument("--gold", required=True, help="Path to gold CSV (train_QA.csv)")
    p_score.add_argument("--pred", required=True, help="Path to prediction CSV")

    args = parser.parse_args()

    if args.command == "ingest":
        from wattbot.ingest import ingest
        ingest(metadata_path=args.metadata, limit=args.limit)

    elif args.command == "run":
        from wattbot.pipeline import run_pipeline
        run_pipeline(
            questions_path=args.questions,
            output_path=args.output,
            metadata_path=args.metadata,
            limit=args.limit,
        )

    elif args.command == "score":
        import pandas as pd
        from wattbot.scoring import score_submission
        gold_df = pd.read_csv(args.gold, encoding="utf-8-sig")
        pred_df = pd.read_csv(args.pred, encoding="utf-8-sig")
        results = score_submission(pred_df, gold_df)
        print(f"\nWattBot Score: {results['wattbot_score']}")
        print(f"  Answer accuracy (0.75): {results['answer_accuracy']}")
        print(f"  Citation F1    (0.20): {results['citation_f1']}")
        print(f"  Abstention     (0.05): {results['abstention_accuracy']}")
        print(f"  Questions scored: {results['n_questions']}")

    else:
        parser.print_help()


if __name__ == "__main__":
    main()
