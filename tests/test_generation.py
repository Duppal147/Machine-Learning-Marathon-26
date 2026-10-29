"""Offline answer-generation and runner tests. Never use gateway credentials or models."""
import csv
import importlib.util
import json
from pathlib import Path
import sys
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from wattbot import config, generation
from wattbot.chunking import Chunk, ChunkStore
from wattbot.generation import GenerationValidationError, load_input_questions, validate_answer, validate_predictions

spec = importlib.util.spec_from_file_location(
    "answer_runner", Path(__file__).resolve().parents[1] / "scripts/generate_answers.py")
runner = importlib.util.module_from_spec(spec)
spec.loader.exec_module(runner)


@pytest.fixture(autouse=True)
def no_live_client(monkeypatch):
    def forbidden():
        raise AssertionError("Tests must not construct a live gateway client")
    monkeypatch.setattr(generation, "create_embedding_client", forbidden)
    monkeypatch.setattr(runner, "create_embedding_client", forbidden)


@pytest.fixture
def question():
    return {"id": "q1", "question": "How much training energy was used?", "answer_unit": "MWh"}


@pytest.fixture
def chunks():
    return [Chunk("paper", "Training used 1.25 MWh in 2023.", (), [3], "text", [], chunk_id="c1",
                  metadata={"ref_id": "paper2023", "url": "https://example.org/paper.pdf"})]


@pytest.fixture
def payload():
    return {"answer": "Training used 1.25 MWh.", "answer_value": "1.25",
            "explanation": "The cited passage states the training electricity use.",
            "evidence": [{"chunk_id": "c1", "quote": "Training used 1.25 MWh in 2023."}]}


def write_questions(path, rows):
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def test_input_loader_keeps_all_rows_without_gold_fields(tmp_path, question):
    path = tmp_path / "train.csv"
    rows = [{**question, "answer_value": "SECRET_GOLD", "ref_id": "SECRET_REF", "is_NA": "0"},
            {**question, "id": "q2", "answer_value": "is_blank", "ref_id": "is_blank", "is_NA": "1"}]
    write_questions(path, rows)
    assert load_input_questions(path) == [question, {**question, "id": "q2"}]


@pytest.mark.parametrize("rows", [[], [{"id": "x", "question": "Q"}],
    [{"id": "", "question": "Q", "answer_unit": ""}],
    [{"id": "x", "question": " ", "answer_unit": ""}],
    [{"id": "x", "question": "Q", "answer_unit": ""}] * 2])
def test_invalid_question_files_are_rejected(tmp_path, rows):
    path = tmp_path / "questions.csv"
    if rows:
        write_questions(path, rows)
    else:
        path.write_text("id,question,answer_unit\n")
    with pytest.raises(ValueError):
        load_input_questions(path)


def test_sources_are_derived_and_deduplicated(question, payload, chunks):
    payload["ref_id"] = "invented_source"  # model cannot choose exported refs or URLs directly
    payload["evidence"] *= 2
    row = validate_answer(question, json.dumps(payload), chunks)
    assert row["answer_value"] == "1.25"
    assert json.loads(row["ref_id"]) == ["paper2023"]
    assert json.loads(row["ref_url"]) == ["https://example.org/paper.pdf"]
    assert row["answer_unit"] == question["answer_unit"]
    validate_predictions([question], [row])


def test_abstention_preserves_unit_and_blanks_evidence(question):
    payload = {"answer": "Insufficient evidence.", "answer_value": "is_blank",
               "explanation": "No supplied passage supports the requested measurement.", "evidence": []}
    row = validate_answer(question, json.dumps(payload), [])
    assert row["answer_unit"] == "MWh"
    assert all(row[field] == "is_blank" for field in ("answer_value", "ref_id", "ref_url", "supporting_materials"))


@pytest.mark.parametrize("raw", ["not JSON", "```json\n{}\n```", "[]", "null", "{}", None])
def test_malformed_responses_are_rejected(question, chunks, raw):
    with pytest.raises(GenerationValidationError):
        validate_answer(question, raw, chunks)


@pytest.mark.parametrize("field,value", [("answer", ""), ("answer_value", 1.25),
    ("answer_value", "NaN"), ("answer_value", "1e999"), ("explanation", None), ("evidence", {}), ("evidence", []),
    ("evidence", [{"chunk_id": "invented", "quote": "Training"}]),
    ("evidence", [{"chunk_id": "c1", "quote": "Invented quotation"}]),
    ("evidence", [{"chunk_id": "c1", "quote": " "}]),
    ("evidence", [{"chunk_id": []}]), ("evidence", [None]), ("answer_value", "is_blank")])
def test_invalid_answer_fields_and_citations(question, chunks, payload, field, value):
    payload[field] = value
    with pytest.raises(GenerationValidationError):
        validate_answer(question, json.dumps(payload), chunks)


def test_whitespace_normalized_literal_quotes_are_accepted(question, payload, chunks):
    chunks[0].text = "Training  used\n1.25 MWh in 2023."
    assert validate_answer(question, json.dumps(payload), chunks)["answer_value"] == "1.25"


@pytest.mark.parametrize("field", ["ref_id", "url"])
def test_missing_citation_metadata_is_not_fabricated(question, payload, chunks, field):
    chunks[0].metadata.pop(field)
    with pytest.raises(GenerationValidationError):
        validate_answer(question, json.dumps(payload), chunks)


def mock_client(raw, finish="stop"):
    response = SimpleNamespace(choices=[SimpleNamespace(finish_reason=finish,
                                                       message=SimpleNamespace(content=raw))])
    return SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=Mock(return_value=response))))


def test_chat_call_is_structured_and_never_receives_gold(question, payload, chunks):
    client = mock_client(json.dumps(payload))
    row, raw = generation.generate_answer({**question, "answer_value": "SECRET_GOLD", "is_NA": "1"},
                                          chunks, model="fake-chat", client=client, max_tokens=1234)
    args = client.chat.completions.create.call_args.kwargs
    assert args["model"] == "fake-chat" and args["temperature"] == 0
    assert args["max_tokens"] == 1234 and args["response_format"] == {"type": "json_object"}
    content = json.loads(args["messages"][1]["content"])
    assert set(content) == {"question", "answer_unit", "passages"}
    assert "SECRET_GOLD" not in json.dumps(args)
    assert row == validate_answer(question, raw, chunks)


@pytest.mark.parametrize("finish", ["length", "content_filter", None])
def test_incomplete_chat_responses_fail(question, payload, chunks, finish):
    with pytest.raises(GenerationValidationError, match="finish_reason") as caught:
        generation.generate_answer(question, chunks, model="fake", client=mock_client(json.dumps(payload), finish))
    assert caught.value.raw_response == json.dumps(payload)


def test_gateway_errors_do_not_become_abstentions(question, chunks):
    client = mock_client("{}")
    client.chat.completions.create.side_effect = ValueError("gateway failure")
    with pytest.raises(ValueError, match="gateway failure"):
        generation.generate_answer(question, chunks, model="fake", client=client)


@pytest.mark.parametrize("change", ["duplicate", "missing", "extra", "unit", "blank", "type"])
def test_submission_validation_is_strict(question, payload, chunks, change):
    row = validate_answer(question, json.dumps(payload), chunks)
    rows = [row]
    if change == "duplicate":
        rows *= 2
    elif change == "missing":
        rows = []
    elif change == "extra":
        rows += [{**row, "id": "unexpected"}]
    elif change == "unit":
        row["answer_unit"] = "kWh"
    elif change == "blank":
        row["answer_value"] = "is_blank"
    else:
        row["explanation"] = None
    with pytest.raises(GenerationValidationError):
        validate_predictions([question], rows)


class FakeRetriever:
    def __init__(self):
        self.fits = self.searches = 0

    def fit(self, chunks):
        self.fits += 1

    def search(self, queries, k):
        self.searches += 1
        return [[0] for _ in queries]


def test_failure_checkpoint_resume_and_complete_export(tmp_path, monkeypatch, question, chunks, payload):
    questions = [question, {**question, "id": "q2"}]
    settings = {"top_k": 1, "questions": questions}
    monkeypatch.setattr(config, "API_KEY", "offline-secret")
    def generate(q, context):
        return validate_answer(q, json.dumps(payload), context), json.dumps(payload)
    generate_mock = Mock(side_effect=[generate(question, chunks), RuntimeError("offline-secret")])
    retriever = FakeRetriever()
    assert runner.run_questions(questions, chunks, retriever, generate_mock, tmp_path, settings) == 1
    assert not (tmp_path / "predictions.csv").exists()
    assert len(list((tmp_path / "answers").glob("*.json"))) == 1
    assert "offline-secret" not in (tmp_path / "errors.json").read_text()
    retry = Mock(side_effect=generate)
    assert runner.run_questions(questions, chunks, FakeRetriever(), retry, tmp_path, settings, resume=True) == 0
    assert retry.call_count == 1 and retry.call_args.args[0]["id"] == "q2"
    with (tmp_path / "predictions.csv").open(newline="") as stream:
        assert [row["id"] for row in csv.DictReader(stream)] == ["q1", "q2"]
    assert json.loads((tmp_path / "errors.json").read_text()) == {}
    complete = FakeRetriever()
    unused = Mock(side_effect=AssertionError("Completed runs must not call the API"))
    assert runner.run_questions(questions, chunks, complete, unused, tmp_path, settings, resume=True) == 0
    assert complete.fits == complete.searches == unused.call_count == 0
    with pytest.raises(ValueError, match="already exists"):
        runner.run_questions(questions, chunks, complete, unused, tmp_path, settings)
    with pytest.raises(ValueError, match="changed"):
        runner.run_questions(questions, chunks, complete, unused, tmp_path, {**settings, "model": "new"}, resume=True)


def test_checkpoint_tampering_fails_loudly(tmp_path, question, chunks, payload):
    settings = {"top_k": 1}
    raw = json.dumps(payload)
    generate = lambda q, context: (validate_answer(q, raw, context), raw)
    runner.run_questions([question], chunks, FakeRetriever(), generate, tmp_path, settings)
    checkpoint = next((tmp_path / "answers").glob("*.json"))
    record = json.loads(checkpoint.read_text())
    record["row"]["answer_value"] = "999"
    checkpoint.write_text(json.dumps(record))
    with pytest.raises(ValueError, match="differs"):
        runner.run_questions([question], chunks, FakeRetriever(), generate, tmp_path, settings, resume=True)


def test_cli_bm25_to_predictions_entirely_offline(tmp_path, monkeypatch, question, chunks, payload):
    inputs, corpus, output = tmp_path / "train.csv", tmp_path / "chunks.jsonl", tmp_path / "run"
    write_questions(inputs, [{**question, "answer_value": "SECRET_GOLD", "is_NA": "1"}])
    ChunkStore(chunks).save(corpus)
    client = mock_client(json.dumps(payload))
    factory = Mock()
    factory.with_options.return_value = client
    monkeypatch.setattr(runner, "create_embedding_client", lambda: factory)
    assert runner.main(["--questions", str(inputs), "--chunks", str(corpus), "--run-dir", str(output),
                        "--chat-model", "fake", "--retriever", "bm25", "--no-rerank"]) == 0
    assert "SECRET_GOLD" not in json.dumps(client.chat.completions.create.call_args.kwargs)
    manifest = json.loads((output / "run.json").read_text())
    assert manifest["settings"]["chat_model"] == "fake"
    assert manifest["settings"]["length_retries"] == 1
    checkpoint = json.loads(next((output / "answers").glob("*.json")).read_text())
    assert checkpoint["generation"]["attempts"][0]["max_output_tokens"] == 8192
    assert "api_key" not in json.dumps(manifest).lower()
    assert (output / "predictions.csv").is_file()


@pytest.mark.parametrize("args", [["--limit", "0"], ["--candidates", "-1"], [],
                                 ["--length-retries", "-1"], ["--length-retries", "3"],
                                 ["--questions", "x", "--run-dir", "y", "--chat-model", config.EMBEDDING_MODEL]])
def test_invalid_cli_arguments_fail_before_api_access(args):
    with pytest.raises(SystemExit):
        runner.main(args)


def test_cross_document_evidence_exports_both_sources(question, payload, chunks):
    chunks += [Chunk("other", "Cooling used 2 liters of water.", (), [9], "text", [], chunk_id="c2",
                     metadata={"ref_id": "other2024", "url": "https://example.org/other.pdf"})]
    payload["evidence"].append({"chunk_id": "c2", "quote": "Cooling used 2 liters of water."})
    row = validate_answer(question, json.dumps(payload), chunks)
    assert json.loads(row["ref_id"]) == ["paper2023", "other2024"]
    assert "other2024" in row["supporting_materials"]


def test_missing_chat_choices_is_an_explicit_failure(question, chunks):
    client = mock_client("{}")
    client.chat.completions.create.return_value = SimpleNamespace(choices=[])
    with pytest.raises(GenerationValidationError):
        generation.generate_answer(question, chunks, model="fake", client=client)


def test_technical_failure_stops_requests_and_leaves_pending_ids(tmp_path, question, chunks):
    questions = [question, {**question, "id": "q2"}]
    generate = Mock(side_effect=RuntimeError("connection unavailable"))
    assert runner.run_questions(questions, chunks, FakeRetriever(), generate, tmp_path, {"top_k": 1}) == 1
    assert generate.call_count == 1
    status = json.loads((tmp_path / "status.json").read_text())
    assert status == {"requested": 2, "completed": 0, "failed": 1, "pending": 1, "complete": False}


def test_invalid_response_keeps_other_questions_and_raw_error(tmp_path, question, chunks, payload):
    questions = [question, {**question, "id": "q2"}]
    raw = json.dumps(payload)
    error = GenerationValidationError("malformed response")
    error.raw_response = "not JSON"
    generate = Mock(side_effect=[error, (validate_answer(questions[1], raw, chunks), raw)])
    assert runner.run_questions(questions, chunks, FakeRetriever(), generate, tmp_path, {"top_k": 1}) == 1
    assert generate.call_count == 2
    assert json.loads((tmp_path / "errors.json").read_text())["q1"]["raw_response"] == "not JSON"


def test_interruption_releases_lock_and_keeps_completed_answers(tmp_path, question, chunks, payload):
    questions = [question, {**question, "id": "q2"}]
    raw = json.dumps(payload)
    generate = Mock(side_effect=[(validate_answer(question, raw, chunks), raw), KeyboardInterrupt()])
    with pytest.raises(KeyboardInterrupt):
        runner.run_questions(questions, chunks, FakeRetriever(), generate, tmp_path, {"top_k": 1})
    assert not (tmp_path / ".lock").exists()
    assert len(list((tmp_path / "answers").glob("*.json"))) == 1
    assert not (tmp_path / "predictions.csv").exists()


def test_lock_excludes_a_second_writer(tmp_path):
    with runner.run_lock(tmp_path):
        with pytest.raises(ValueError, match="locked"):
            with runner.run_lock(tmp_path):
                pytest.fail("second writer acquired the lock")
    assert not (tmp_path / ".lock").exists()


def test_stale_document_vectors_are_not_rebuilt(monkeypatch, chunks):
    from wattbot import vectorstore
    collection = SimpleNamespace(get=Mock(return_value={"ids": [], "metadatas": []}))
    store = SimpleNamespace(model="fake-embedding", collection=collection, index=Mock())
    monkeypatch.setattr(vectorstore, "ChunkVectorStore", lambda strategy: store)
    args = SimpleNamespace(retriever="chroma", strategy="hybrid", no_query_instruction=False, no_rerank=True)
    with pytest.raises(ValueError, match="missing or stale"):
        runner.build_retriever(args, chunks, "corpus-hash")
    store.index.assert_not_called()


def test_length_retry_keeps_the_prompt_and_records_budgets(question, chunks, payload):
    client = mock_client(json.dumps(payload))
    success = client.chat.completions.create.return_value
    incomplete = SimpleNamespace(choices=[SimpleNamespace(finish_reason="length",
                                                         message=SimpleNamespace(content=None))])
    client.chat.completions.create.side_effect = [incomplete, success]
    trace = {}
    row, raw = generation.generate_answer(question, chunks, model="fake", client=client, trace=trace)
    calls = client.chat.completions.create.call_args_list
    assert [call.kwargs["max_tokens"] for call in calls] == [8192, 16384]
    assert calls[0].kwargs["messages"] == calls[1].kwargs["messages"]
    assert [attempt["finish_reason"] for attempt in trace["attempts"]] == ["length", "stop"]
    assert row == validate_answer(question, raw, chunks)


def test_second_length_retry_can_complete_a_long_answer(question, chunks, payload):
    client = mock_client(json.dumps(payload))
    success = client.chat.completions.create.return_value
    incomplete = SimpleNamespace(choices=[SimpleNamespace(finish_reason="length",
                                                         message=SimpleNamespace(content=None))])
    client.chat.completions.create.side_effect = [incomplete, incomplete, success]
    trace = {}
    row, raw = generation.generate_answer(question, chunks, model="fake", client=client,
                                          length_retries=2, trace=trace)
    calls = client.chat.completions.create.call_args_list
    assert [call.kwargs["max_tokens"] for call in calls] == [8192, 16384, 32768]
    assert all(call.kwargs["messages"] == calls[0].kwargs["messages"] for call in calls)
    assert [attempt["finish_reason"] for attempt in trace["attempts"]] == ["length", "length", "stop"]
    assert row == validate_answer(question, raw, chunks)


@pytest.mark.parametrize("retries,expected_calls", [(0, 1), (1, 2), (2, 3)])
def test_length_retries_are_bounded(question, chunks, payload, retries, expected_calls):
    client = mock_client(json.dumps(payload), finish="length")
    with pytest.raises(GenerationValidationError):
        generation.generate_answer(question, chunks, model="fake", client=client, length_retries=retries)
    assert client.chat.completions.create.call_count == expected_calls


def test_nonlength_incomplete_response_is_not_retried(question, chunks, payload):
    client = mock_client(json.dumps(payload), finish="content_filter")
    with pytest.raises(GenerationValidationError):
        generation.generate_answer(question, chunks, model="fake", client=client)
    assert client.chat.completions.create.call_count == 1
