#!/usr/bin/env python3
"""Show where a generation run on labeled questions loses points, and why.

Usage:
    python scripts/analyze_errors.py documents/runs/train-v1 [--exclude-ids holdout_ids.txt]

Reads the checkpoints of a scripts/generate_answers.py run (a partial run is fine), scores each
question with the official WattBot2026/Score.py, and reports points lost per error bucket, plus
breakdowns by cohort, answer type and question flag. Per-question details go to
<run-dir>/errors_by_question.csv. --exclude-ids skips a sealed holdout (one id per line).
"""
import argparse
import csv
import dataclasses
import os
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from wattbot.eval.errors import BUCKETS, analyze_answer, load_records, load_scorer


def breakdown(title, rows, key):
    groups = defaultdict(list)
    for row in rows:
        for name in key(row):
            groups[name].append(row)
    print(f"\n{title:<18}{'n':>5}{'score':>8}{'value ok':>10}{'abstained':>11}")
    for name, group in sorted(groups.items()):
        n = len(group)
        abstained = sum(r.predicted_value == "is_blank" for r in group)
        print(f"{name:<18}{n:>5}{sum(r.score for r in group) / n:>8.3f}"
              f"{sum(r.value_ok for r in group) / n:>10.3f}{abstained / n:>11.3f}")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("run_dir", type=Path)
    parser.add_argument("--gold", type=Path, default=Path("WattBot2026/train_QA.csv"))
    parser.add_argument("--scorer", type=Path, default=Path("WattBot2026/Score.py"))
    parser.add_argument("--exclude-ids", type=Path, help="ids to skip, one per line (a sealed holdout)")
    args = parser.parse_args()

    with open(args.gold, encoding="utf-8-sig", newline="") as f:
        gold = {row["id"]: row for row in csv.DictReader(f)}
    excluded = set(args.exclude_ids.read_text().split()) if args.exclude_ids else set()
    scorer = load_scorer(args.scorer)
    records = [r for r in load_records(args.run_dir) if r["question"]["id"] not in excluded]
    if not records:
        sys.exit(f"no completed answers in {args.run_dir}/answers")
    missing = [r["question"]["id"] for r in records if r["question"]["id"] not in gold]
    if missing:
        sys.exit(f"{len(missing)} answers have no gold row (is this a test run?): {missing[:5]}")
    rows = [analyze_answer(gold[r["question"]["id"]], r, scorer) for r in records]

    n = len(rows)
    print(f"{n} questions analyzed, score {sum(r.score for r in rows) / n:.4f}, "
          f"excluded {len(excluded)} ids" if excluded else
          f"{n} questions analyzed, score {sum(r.score for r in rows) / n:.4f}")
    groups = defaultdict(list)
    for row in rows:
        groups[row.bucket].append(row)
    print(f"\n{'bucket':<24}{'n':>4}{'share':>7}{'points lost':>13}  examples")
    for bucket in sorted(BUCKETS, key=lambda b: -sum(1 - r.score for r in groups[b])):
        group = groups[bucket]
        if group:
            lost = sum(1 - r.score for r in group) / n
            print(f"{bucket:<24}{len(group):>4}{len(group) / n:>7.1%}{lost:>13.4f}  "
                  f"{' '.join(r.id for r in group[:5])}")
    print()
    for bucket, meaning in BUCKETS.items():
        if groups[bucket]:
            print(f"  {bucket}: {meaning}")

    breakdown("cohort", rows, lambda r: [r.cohort])
    breakdown("answer type", rows, lambda r: [r.answer_type])
    breakdown("flag", rows, lambda r: r.flags)

    out = args.run_dir / "errors_by_question.csv"
    with open(out, "w", newline="", encoding="utf-8") as f:
        fields = [field.name for field in dataclasses.fields(rows[0])]
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for row in sorted(rows, key=lambda r: (r.bucket, r.id)):
            writer.writerow({**dataclasses.asdict(row), "flags": " ".join(row.flags)})
    print(f"\nper-question details -> {out}")


if __name__ == "__main__":
    main()
