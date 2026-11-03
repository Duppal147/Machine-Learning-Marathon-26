"""Offline tests for the PR description generator and the eval summaries it reads."""
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from wattbot.eval.summary import git_state, load_summary, save_summary
from wattbot.pr_body import build_body

HEAD = "abc1234"


def _stamp(**extra):
    return {"git_commit": HEAD, "git_dirty": False, "created_at": "2026-11-02T10:00:00+00:00", **extra}


def _retrieval(**changes):
    base = {"chunks": 6033, "doc_recall@5": 0.925, "passage_recall@1": 0.522, "passage_recall@5": 0.819,
            "passage_recall@10": 0.879, "passage_recall@20": 0.967, "passage_mrr": 0.651}
    fig = {**base, "chunks": 7132, "passage_recall@5": 0.805, **changes}
    return _stamp(retriever="rrf_bm25+chroma_qwen", questions=240, passage_questions=182,
                  strategies={"hybrid": base, "hybrid_fig": fig})


def _e2e(oracle=False):
    def prompt(answer1, wrong):
        return {"questions": 10, "gold_doc_read@1": 0.9, "gold_doc_read@3": 1.0, "answer@1": answer1, "answer@3": 0.8,
                "strict_answer@1": answer1, "strict_answer@3": 0.8, "wrong_number@1": wrong, "read_errors": 1}
    prompts = {"default": prompt(0.6 if oracle else 0.5, 0.2)}
    if not oracle:
        prompts["scope"] = prompt(0.5, 0.1)
    return _stamp(strategy="hybrid_fig", oracle_doc=oracle, reads=3, tolerance=0.0, model="qwen3.8-27b",
                  prompts=prompts)


def _facts(**changes):
    facts = {
        "head": HEAD,
        "readings": {"count": 1260, "model": "qwen3.8-27b", "charts": 1100, "dropped": 160, "invalid": 0},
        "chunks": {"figure": 1099, "text": 6033, "identical": True},
        "files": ["src/wattbot/chunking/loaders.py", "src/wattbot/snapshot.py", "src/wattbot/figures.py",
                  "tests/test_figures.py"],
        "tests": {"passed": 97, "failed": 0},
        "retrieval": _retrieval(),
        "figures": _stamp(retriever="rrf_bm25+chroma_qwen", strategies={
            "hybrid": {"figure_chunks": 0, "figure_questions": 14, "value_questions": 10, "figure_recall@1": 0.286,
                       "figure_recall@5": 0.5, "figure_recall@10": 0.643, "value_recall@1": 0.4,
                       "value_recall@5": 0.5, "value_recall@10": 0.5},
            "hybrid_fig": {"figure_chunks": 1099, "figure_questions": 14, "value_questions": 10,
                           "figure_recall@1": 0.214, "figure_recall@5": 0.571, "figure_recall@10": 0.643,
                           "value_recall@1": 0.1, "value_recall@5": 0.7, "value_recall@10": 0.8}}),
        "e2e": _e2e(),
        "e2e_oracle": _e2e(oracle=True),
    }
    facts.update(changes)
    return facts


def test_a_complete_set_of_results_gives_a_body_with_no_warnings():
    body, warnings = build_body(_facts())
    assert warnings == []
    assert "1260 charts were read with `qwen3.8-27b`: 1100 indexed as figure chunks, 160 logos/photos dropped" in body
    assert "the 6033 text chunks of `hybrid_fig` are identical to `hybrid`'s, plus 1099 figure chunks" in body
    # the regression table carries the measured values and the change row
    assert "| `hybrid` | 6033 | 0.925 | 0.522 | 0.819 |" in body
    assert "| change | +1099 | +0.000 | +0.000 | -0.014 |" in body
    # the end-to-end table has a column per prompt and per oracle prompt
    assert "| metric | default | scope | default, oracle doc |" in body
    assert "right paper is the first chart's source for 9/10 questions, and 10/10 within the top 3" in body
    assert "- [x] `pytest tests/`: 97 passed" in body and "- [x] `hybrid_fig` text chunks identical" in body
    assert "- [ ] One approval from someone other than the author" in body     # manual items stay unticked
    assert "`pyproject.toml` and `uv.lock` are unchanged." in body
    assert "`src/wattbot/chunking/loaders.py`, `src/wattbot/snapshot.py`" in body
    assert "tests/test_figures.py" not in body.split("## Shared files touched")[1].split("## Results")[0]


def test_missing_results_are_flagged_with_the_command_and_never_invented():
    body, warnings = build_body(_facts(retrieval=None, e2e=None, e2e_oracle=None, figures=None))
    assert any("text-question regression check" in w for w in warnings)
    assert "scripts/eval_retrieval.py --retriever hybrid --strategies hybrid hybrid_fig" in body
    assert "scripts/eval_figures_e2e.py --reads 3 --prompt default scope" in body
    assert "scripts/eval_figures.py --retriever hybrid" in body
    assert "0.805" not in body and "0.819" not in body

    only_one, warnings = build_body(_facts(retrieval={**_retrieval(), "strategies": {"hybrid": {"chunks": 1}}}))
    assert "needs both `hybrid` and `hybrid_fig`" in only_one and warnings

    _, warnings = build_body(_facts(e2e_oracle=None))
    assert any("oracle-document run" in w for w in warnings)


def test_results_from_another_commit_or_a_dirty_tree_are_flagged():
    stale = _retrieval()
    stale["git_commit"] = "old0000"
    body, warnings = build_body(_facts(retrieval=stale))
    assert "produced at `old0000` but the PR is at `abc1234`" in body
    assert any("text-question check" in w and "old0000" in w for w in warnings)

    dirty = _e2e()
    dirty["git_dirty"] = True
    body, warnings = build_body(_facts(e2e=dirty))
    assert "produced with uncommitted changes" in body and any("uncommitted" in w for w in warnings)


def test_chunk_identity_failures_and_test_failures_are_loud():
    body, warnings = build_body(_facts(chunks={"figure": 5, "text": 100, "identical": False}))
    assert "text chunks of `hybrid_fig` differ from `hybrid`'s" in body and "text chunks of hybrid_fig" in " ".join(warnings)
    assert "- [ ] `hybrid_fig` text chunks identical" in body

    body, warnings = build_body(_facts(chunks={"figure": 5, "text": 100, "identical": None}))
    assert "identity with `hybrid` not checked" in body and warnings == []

    body, warnings = build_body(_facts(tests={"passed": 90, "failed": 2}))
    assert "90 passed, 2 failed" in body and "tests did not all pass" in warnings

    body, warnings = build_body(_facts(tests=None))
    assert "(not run by the generator)" in body and "tests were not run" in warnings


def test_a_body_with_no_figure_chunks_warns_instead_of_sounding_fine():
    body, warnings = build_body(_facts(chunks={"figure": 0, "text": 6033, "identical": True}))
    assert "`hybrid_fig` has no figure chunks" in body and "hybrid_fig has no figure chunks" in warnings
    assert "plus 0 figure chunks" not in body


def test_shared_files_include_a_changed_lockfile():
    body, _ = build_body(_facts(files=["pyproject.toml", "src/wattbot/config.py", "docs/x.md"]))
    section = body.split("## Shared files touched")[1].split("## Results")[0]
    assert "`pyproject.toml`" in section and "`src/wattbot/config.py`" in section
    assert "`uv.lock` is unchanged." in section and "docs/x.md" not in section


def test_summaries_round_trip_with_numpy_values_and_nan(tmp_path):
    path = tmp_path / "sub" / "s.json"
    save_summary(path, "retrieval", {"a": np.float64(0.5), "n": np.int64(3), "bad": float("nan"),
                                     "nested": {"x": [np.float32(1.5)]}})
    loaded = load_summary(path)
    assert loaded["a"] == 0.5 and loaded["n"] == 3 and loaded["bad"] is None and loaded["nested"] == {"x": [1.5]}
    assert loaded["kind"] == "retrieval" and loaded["created_at"] and "git_commit" in loaded
    assert not list(path.parent.glob("*.tmp"))
    assert load_summary(tmp_path / "missing.json") is None
    (tmp_path / "bad.json").write_text("{truncated")
    assert load_summary(tmp_path / "bad.json") is None
    commit, dirty = git_state()
    assert isinstance(commit, str) and isinstance(dirty, bool)
