import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from wattbot.chunking import Chunk, CorpusMetadataEnricher, Document
from wattbot.eval import Question, TfidfRetriever, evaluate, evidence_coverage, extract_evidence, load_questions


def test_extract_evidence_prefers_long_quotes_and_splits_elisions():
    s = 'refx2024: "The first quoted sentence has plenty of words ... and a second part with enough words too"'
    assert extract_evidence(s) == ["The first quoted sentence has plenty of words",
                                   "and a second part with enough words too"]


def test_extract_evidence_falls_back_when_quotes_are_short():
    s = 'These effects can be large for so-called "general-purpose technologies" such as electricity.'
    assert extract_evidence(s) == [s.rstrip(".")]


def test_extract_evidence_splits_inline_ref_markers():
    s = ("wpf2026: 'the largest hyperscale data centers use up to 1,000 GWh' -- orders of magnitude beyond "
         "typical facilities. amazon2023: AWS reports a fleet water usage effectiveness of 0.18 L/kWh.")
    segs = extract_evidence(s)
    assert "the largest hyperscale data centers use up to 1,000 GWh" in segs
    assert "AWS reports a fleet water usage effectiveness of 0.18 L/kWh" in segs


def test_evidence_coverage_ignores_spacing_and_case():
    chunk = "Intro.  The  rest  of  the  paper  is  organized  as  follows. Section III covers X."
    assert evidence_coverage("the rest of the paper is organized as follows", chunk) == 1.0
    assert evidence_coverage("completely unrelated words about something else", chunk) < 0.3


def test_corpus_metadata_enricher(tmp_path):
    csv_path = tmp_path / "meta.csv"
    csv_path.write_text("﻿id,type,title,year,url,local_path\n"
                        "amazon2023,report,Amazon Report,2023,https://x,documents/pdfs/Amazon_abc.pdf\n")
    enricher = CorpusMetadataEnricher(csv_path)
    doc = Document("Amazon_abc", "t", [])
    chunk = enricher.enrich(Chunk("Amazon_abc", "text", (), [1], "text", []), doc)
    assert chunk.metadata["ref_id"] == "amazon2023" and chunk.metadata["year"] == "2023"
    unknown = enricher.enrich(Chunk("other", "text", (), [1], "text", []), doc)
    assert "ref_id" not in unknown.metadata
    assert CorpusMetadataEnricher(tmp_path / "missing.csv").by_stem == {}


def _chunk(i, ref, text):
    return Chunk("d", text, (), [1], "text", [], chunk_id=f"c{i}", metadata={"ref_id": ref})


def test_evaluate_scores_doc_and_passage_ranks():
    chunks = [
        _chunk(0, "a2024", "Solar panels convert sunlight into electricity with modest efficiency."),
        _chunk(1, "a2024", "The data center consumed 1.9 billion gallons of water per day in total."),
        _chunk(2, "b2024", "Wind turbines generate power from moving air across large blades."),
    ]
    questions = [
        Question("q1", "How many gallons of water per day did the data center consume?", ["a2024"],
                 ["data center consumed 1.9 billion gallons of water per day"]),
        Question("q2", "How do wind turbines generate power?", ["b2024"], ["paraphrased evidence not in any chunk"]),
        Question("q3", "Out of corpus question", ["zzz2020"], ["whatever words are here for sure"]),
    ]
    summary, results = evaluate(chunks, questions, TfidfRetriever(), ks=(1, 2))

    assert summary["questions"] == 2          # q3's document is not in the corpus
    assert summary["passage_questions"] == 1  # q2's evidence can't be located
    assert summary["doc_recall@1"] == 1.0
    assert summary["passage_recall@1"] == 1.0
    assert results[0].passage_rank == 1 and results[0].top_chunk_ids[0] == "c1"


@pytest.mark.skipif(not Path("WattBot2026/train_QA.csv").exists(), reason="WattBot data not available")
def test_load_questions_real_file():
    qs = load_questions()
    assert len(qs) > 200
    assert all(q.ref_ids for q in qs)
    assert sum(bool(q.evidence) for q in qs) / len(qs) > 0.8
