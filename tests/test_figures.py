"""Offline tests for chart reading, figure chunks and the chart retrieval metrics; a fake client stands in for the gateway."""
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from wattbot import figures
from wattbot.chunking import Block, Chunk, DoclingJsonLoader, HybridChunker, build_pipeline
from wattbot.eval import FigureQuestion, evaluate_figures
from wattbot.eval.figures import answer_number, figure_numbers, is_caption_of, load_figure_questions, numbers_in

GOOD_REPLY = json.dumps({
    "kind": "bar_chart", "title": "Energy per query", "x_axis": "Model", "y_axis": "Wh",
    "legend": ["Wh"], "data": "| Model | Wh |\n| --- | --- |\n| Alpha | 3 |\n| Beta | 7 |",
    "summary": "Beta uses the most energy at 7 Wh.",
})


class FakeVision:
    """Stands in for the OpenAI client: returns canned chat replies and records the requests."""

    def __init__(self, reply=GOOD_REPLY, models=("qwen3-vl-embedding-8b", "qwen3-vl-8b-instruct", "llama-3"),
                 finish_reason="stop", reasoning=None):
        self.reply, self.calls, self.finish_reason, self.reasoning = reply, [], finish_reason, reasoning
        self.chat = SimpleNamespace(completions=SimpleNamespace(create=self._create))
        self.models = SimpleNamespace(list=lambda: SimpleNamespace(data=[SimpleNamespace(id=m) for m in models]))

    def _create(self, **kwargs):
        self.calls.append(kwargs)
        message = SimpleNamespace(content=self.reply, reasoning_content=self.reasoning)
        return SimpleNamespace(choices=[SimpleNamespace(message=message, finish_reason=self.finish_reason)])


def _png(path, size=(200, 120)):
    from PIL import Image
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", size, "white").save(path)
    return path


def _doc():
    """Section "1 Intro" with a chart (picture 0, with labels), a logo (picture 1) and a bare picture (2)."""
    def text(i, label, s, parent="#/body"):
        return {"self_ref": f"#/texts/{i}", "parent": {"$ref": parent}, "children": [], "content_layer": "body",
                "label": label, "prov": [{"page_no": 2}], "text": s, "orig": s}

    texts = [
        text(0, "section_header", "My Paper"),
        text(1, "section_header", "1 Intro"),
        text(2, "text", "Intro body about the paper's contributions and what we measure."),
        text(3, "caption", "Figure 1: Energy per query.", parent="#/pictures/0"),
        text(4, "text", "Alpha", parent="#/pictures/0"),
        text(5, "text", "Beta", parent="#/pictures/0"),
        text(6, "text", "7", parent="#/pictures/0"),
        text(7, "text", "Figure 1: Energy per query.", parent="#/pictures/0"),
        text(8, "caption", "Figure 2: Our logo.", parent="#/pictures/1"),
    ]
    picture = lambda i, caps, kids: {"self_ref": f"#/pictures/{i}", "content_layer": "body", "prov": [{"page_no": 2}],
                                     "captions": [{"$ref": f"#/texts/{c}"} for c in caps],
                                     "children": [{"$ref": f"#/texts/{k}"} for k in kids]}
    return {
        "name": "d", "texts": texts, "groups": [], "tables": [],
        "body": {"children": [{"$ref": r} for r in
                              ["#/texts/0", "#/texts/1", "#/texts/2", "#/pictures/0", "#/pictures/1", "#/pictures/2"]]},
        "pictures": [picture(0, [3], [4, 5, 6, 7]), picture(1, [8], []), picture(2, [], [])],
    }


@pytest.fixture
def parsed(tmp_path):
    path = tmp_path / "d.json"
    path.write_text(json.dumps(_doc()))
    return path


# --- reading figures --------------------------------------------------------------------------

def test_picture_helpers_exclude_the_caption_from_labels():
    doc = _doc()
    chart = doc["pictures"][0]
    assert figures.picture_index(chart) == 0
    assert figures.picture_caption(doc, chart) == "Figure 1: Energy per query."
    assert figures.picture_labels(doc, chart) == ["Alpha", "Beta", "7"]


def test_parse_vlm_json_tolerates_fences_and_chatter():
    assert figures.parse_vlm_json(GOOD_REPLY)["kind"] == "bar_chart"
    assert figures.parse_vlm_json(f"```json\n{GOOD_REPLY}\n```")["kind"] == "bar_chart"
    assert figures.parse_vlm_json(f"Here you go: {GOOD_REPLY} Hope that helps!")["kind"] == "bar_chart"
    assert figures.parse_vlm_json("no json here") is None
    assert figures.parse_vlm_json("{broken") is None
    assert figures.parse_vlm_json("[1, 2]") is None


def test_normalize_result_handles_list_data_and_unparseable_replies():
    result = figures.normalize_result({"kind": "Line_Chart", "data": [{"x": 1, "y": 2}, {"x": 3, "y": 4}],
                                       "legend": "Wh"}, "")
    assert result["kind"] == "line_chart" and result["legend"] == ["Wh"]
    assert result["data"].splitlines()[0] == "| x | y |" and "| 3 | 4 |" in result["data"]
    bad = figures.normalize_result(None, "I cannot read this")
    assert bad["parse_error"] and bad["summary"] == "I cannot read this"


def test_render_description_skips_logos_and_keeps_the_caption_on_its_own_line():
    result = figures.normalize_result(json.loads(GOOD_REPLY), GOOD_REPLY)
    text = figures.render_description(result, "Figure 1: Energy per query.")
    assert text.splitlines()[:2] == ["Figure (bar chart)", "Figure 1: Energy per query."]
    assert "| Beta | 7 |" in text and "Summary: Beta uses the most energy" in text
    assert figures.render_description({"kind": "logo", "summary": "a logo"}, "Figure 2") is None
    assert figures.render_description(figures.normalize_result({"kind": "bar_chart"}, ""), "") is None  # nothing to index


def test_render_description_truncates_long_data():
    data = "\n".join(f"| r{i} | {i} |" for i in range(figures.MAX_DATA_ROWS + 20))
    text = figures.render_description({"kind": "table", "data": data}, "")
    assert "... (20 more rows)" in text and f"| r{figures.MAX_DATA_ROWS} |" not in text


def test_describe_image_sends_the_image_and_context(tmp_path):
    image = _png(tmp_path / "f.png")
    client = FakeVision()
    record = figures.describe_image(client, "vlm", image, "Figure 1: Energy per query.", ["Alpha", "Beta"])

    assert record["result"]["kind"] == "bar_chart" and record["model"] == "vlm"
    assert record["image_sha256"] == figures.file_sha256(image)
    call = client.calls[0]
    assert call["model"] == "vlm" and call["temperature"] == 0
    text_part, image_part = call["messages"][0]["content"]
    assert "Figure 1: Energy per query." in text_part["text"] and "Alpha; Beta" in text_part["text"]
    assert image_part["image_url"]["url"].startswith("data:image/png;base64,")


def test_large_images_are_downscaled(tmp_path):
    import base64
    import io
    from PIL import Image
    image = _png(tmp_path / "big.png", size=(4000, 1000))
    url, _ = figures.image_data_url(image, max_side=1000)
    sent = Image.open(io.BytesIO(base64.b64decode(url.split(",", 1)[1])))
    assert sent.size == (1000, 250)


def test_reasoning_prefix_is_stripped(tmp_path):
    reply = '<think>The bars are {3, 7}. Let me answer.\n</think>\n' + GOOD_REPLY
    record = figures.describe_image(FakeVision(reply=reply), "vlm", _png(tmp_path / "f.png"))
    assert record["result"]["kind"] == "bar_chart" and "parse_error" not in record["result"]
    out = figures.answer_from_figure(FakeVision(reply="<think>hm</think>7 Wh"), "vlm", "q", _png(tmp_path / "g.png"))
    assert out == "7 Wh"
    with pytest.raises(figures.VisionModelError, match="no usable reply"):
        figures.describe_image(FakeVision(reply="<think>never finished</think>"), "vlm", _png(tmp_path / "h.png"))


def test_a_reply_that_ran_out_of_tokens_is_an_error_not_a_cached_fragment(tmp_path):
    image = _png(tmp_path / "f.png")
    # the gateway splits reasoning into its own field and the budget ran out before any answer
    client = FakeVision(reply=None, finish_reason="length", reasoning="x" * 5000)
    with pytest.raises(figures.VisionModelError, match=r"finish_reason=length.*separate reasoning 5000 chars"):
        figures.describe_image(client, "vlm", image)
    # cut off while still thinking: the unclosed <think> text must not become the figure's summary
    with pytest.raises(figures.VisionModelError, match="no usable reply"):
        figures.describe_image(FakeVision(reply="<think>The bars are", finish_reason="length"), "vlm", image)
    # cut off in the middle of the JSON
    with pytest.raises(figures.VisionModelError, match="truncated"):
        figures.describe_image(FakeVision(reply=GOOD_REPLY[:60], finish_reason="length"), "vlm", image)
    # a normal stop with a non-JSON reply is still recorded, flagged
    record = figures.describe_image(FakeVision(reply="a chart of energy"), "vlm", image)
    assert record["result"]["parse_error"]


def test_token_budget_and_thinking_switch_reach_the_request(tmp_path):
    image = _png(tmp_path / "f.png")
    client = FakeVision()
    figures.describe_image(client, "vlm", image)
    assert client.calls[0]["max_tokens"] == figures.MAX_TOKENS and "extra_body" not in client.calls[0]
    figures.describe_image(client, "vlm", image, max_tokens=123, extra_body=figures.NO_THINKING)
    assert client.calls[1]["max_tokens"] == 123
    assert client.calls[1]["extra_body"] == {"chat_template_kwargs": {"enable_thinking": False}}


def test_empty_reply_is_an_error(tmp_path):
    with pytest.raises(figures.VisionModelError):
        figures.describe_image(FakeVision(reply="  "), "vlm", _png(tmp_path / "f.png"))


def test_answer_from_figure_includes_question_and_context(tmp_path):
    client = FakeVision(reply="7 Wh\nRead from the Beta bar.")
    out = figures.answer_from_figure(client, "vlm", "Which model uses most energy?", _png(tmp_path / "f.png"), "ctx")
    assert out.startswith("7 Wh")
    prompt = client.calls[0]["messages"][0]["content"][0]["text"]
    assert "Which model uses most energy?" in prompt and "ctx" in prompt


def test_resolve_vision_model(monkeypatch):
    monkeypatch.setattr(figures.config, "VISION_MODEL", "")
    assert figures.resolve_vision_model(FakeVision()) == "qwen3-vl-8b-instruct"  # embedding model is not a candidate
    assert figures.resolve_vision_model(FakeVision(), "explicit") == "explicit"
    with pytest.raises(figures.VisionModelError, match="VISION_MODEL"):
        figures.resolve_vision_model(FakeVision(models=("qwen3-vl-8b", "qwen2.5-vl-72b")))
    with pytest.raises(figures.VisionModelError, match="No model name contains 'vl'"):
        figures.resolve_vision_model(FakeVision(models=("churro-3b", "qwen3-vl-embedding-8b", "qwen3.8-27b")))
    monkeypatch.setattr(figures.config, "VISION_MODEL", "from-env")
    assert figures.resolve_vision_model(FakeVision()) == "from-env"


def test_cache_round_trip_and_staleness(tmp_path):
    path = figures.figure_text_path("d", 3, tmp_path)
    assert path == tmp_path / "d" / "figure_03.json" and figures.load_record(path) is None
    record = {"prompt_version": figures.PROMPT_VERSION, "image_sha256": "abc", "result": {"kind": "bar_chart"}}
    figures.save_record(path, record)
    assert figures.load_record(path) == record and not list(path.parent.glob("*.tmp"))
    assert figures.record_is_current(record, "abc")
    assert not figures.record_is_current(record, "different image")
    assert not figures.record_is_current({**record, "prompt_version": figures.PROMPT_VERSION + 1}, "abc")
    path.write_text("{truncated")
    assert figures.load_record(path) is None  # a corrupt file is treated as unread


# --- figure chunks -----------------------------------------------------------------------------

def test_default_loader_still_keeps_only_captions(parsed):
    doc = DoclingJsonLoader().load(parsed)
    assert [b.kind for b in doc.blocks if "Figure" in b.text] == ["caption", "caption"]
    assert not any(b.kind == "figure" for b in doc.blocks)


def test_label_source_emits_figure_blocks_only_for_pictures_with_labels(parsed):
    doc = DoclingJsonLoader(figure_text=figures.LabelFigureText()).load(parsed)
    figs = [b for b in doc.blocks if b.kind == "figure"]
    assert len(figs) == 1
    assert figs[0].text == "Figure\nFigure 1: Energy per query.\nText in figure: Alpha; Beta; 7"
    assert figs[0].metadata == {"figure_index": 0} and "#/pictures/0" in figs[0].source_refs
    assert figs[0].page == 2 and figs[0].heading_path == ("1 Intro",)
    # the text flow is the default loader's: both captions stay, whether or not the picture got a figure block
    assert [b.text for b in doc.blocks if b.kind == "caption"] == ["Figure 1: Energy per query.", "Figure 2: Our logo."]


def test_vlm_source_uses_cached_readings_and_falls_back(parsed, tmp_path):
    figures.save_record(figures.figure_text_path("d", 0, tmp_path),
                        {"result": figures.normalize_result(json.loads(GOOD_REPLY), GOOD_REPLY)})
    figures.save_record(figures.figure_text_path("d", 1, tmp_path),
                        {"result": figures.normalize_result({"kind": "logo"}, "")})
    doc = DoclingJsonLoader(figure_text=figures.VlmFigureText(tmp_path)).load(parsed)

    figs = [b for b in doc.blocks if b.kind == "figure"]
    assert len(figs) == 1 and "| Beta | 7 |" in figs[0].text
    assert any(b.kind == "caption" and b.text == "Figure 2: Our logo." for b in doc.blocks)  # logo -> caption only


def test_hybrid_chunker_adds_figure_chunks_without_touching_the_text_chunks(parsed):
    plain = HybridChunker().chunk(DoclingJsonLoader().load(parsed).blocks)
    chunks = HybridChunker().chunk(DoclingJsonLoader(figure_text=figures.LabelFigureText()).load(parsed).blocks)

    assert [c.kind for c in chunks] == ["text", "figure"]          # the figure goes after the section's text
    assert [c.text for c in chunks if c.kind == "text"] == [c.text for c in plain]
    fig = chunks[1]
    assert fig.metadata["figure_index"] == 0 and fig.pages == [2] and "Alpha; Beta; 7" in fig.text


def test_a_figure_between_paragraphs_does_not_split_the_text_chunk():
    block = lambda kind, text: Block(doc_id="d", kind=kind, text=text, heading_path=("S",), page=1)
    blocks = [block("text", "First paragraph."), block("figure", "Figure\nbars"), block("text", "Second paragraph.")]
    chunks = HybridChunker().chunk(blocks)
    assert [c.kind for c in chunks] == ["text", "figure"]
    assert chunks[0].text == "First paragraph.\n\nSecond paragraph."  # a table here would have split it in two


def test_figure_strategies_are_registered_and_leave_hybrid_unchanged(parsed, monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)  # no figure_text dir here: hybrid_fig falls back to captions
    plain = build_pipeline("hybrid").run([parsed])
    assert [c.kind for c in plain] == ["text"] and "Figure 1: Energy per query." in plain[0].text

    with_labels = build_pipeline("hybrid_figlabels").run([parsed])
    assert [c.kind for c in with_labels] == ["text", "figure"]
    assert with_labels[0].text == plain[0].text and with_labels[0].embed_text == plain[0].embed_text
    assert with_labels[1].chunk_id == "d::hybrid_figlabels::0001"
    assert with_labels[1].embed_text.startswith("d > 1 Intro\n\nFigure")  # heading prefix reaches the embedding

    assert [c.kind for c in build_pipeline("hybrid_fig").run([parsed])] == ["text"]


# --- chart retrieval metrics -------------------------------------------------------------------

def test_figure_reference_and_number_helpers():
    assert figure_numbers('Figure 8 caption: "28.5%"; see Fig. 2(d) and Figures 3') == {"8", "2", "3"}
    assert numbers_in("84% of 1,234.5 kWh and -3") == {84.0, 1234.5, -3.0}
    assert answer_number("28.5", "percent") == 28.5 and answer_number("1,200", "MW") == 1200.0
    assert answer_number("1", "is_blank") is None and answer_number("Claude", "is_blank") is None
    assert answer_number("[1,2]", "kWh") is None


def _chunk(cid, text, ref="paper1", kind="text"):
    return Chunk(doc_id=ref, text=text, heading_path=(), pages=[1], kind=kind, source_refs=[], chunk_id=cid,
                 metadata={"ref_id": ref})


def test_is_caption_of_requires_a_line_start():
    assert is_caption_of(_chunk("a", "Figure (bar chart)\nFig. 3. Energy.\nData: ..."), {"3"})
    assert is_caption_of(_chunk("a", "Intro text.\n\nFigure 3: Energy."), {"3", "4"})
    assert not is_caption_of(_chunk("a", "As shown in Figure 3, energy rises."), {"3"})
    assert not is_caption_of(_chunk("a", "Figure 30: Other."), {"3"})  # "30" is not "3"
    assert not is_caption_of(_chunk("a", "Figure 3: Energy."), set())


class FixedRetriever:
    name = "fixed"

    def __init__(self, order):
        self.order = order

    def fit(self, chunks):
        self.ids = {c.chunk_id: i for i, c in enumerate(chunks)}

    def search(self, queries, k):
        return [[self.ids[c] for c in self.order][:k] for _ in queries]


def test_evaluate_figures_scores_caption_and_value_ranks():
    chunks = [
        _chunk("noise", "Unrelated 28.5 mention", ref="other"),              # wrong doc: never counts
        _chunk("prose", "Energy fell a lot.", kind="text"),
        _chunk("caption", "Figure 8. Optimization reduced energy.", kind="text"),
        _chunk("chart", "Figure (bar chart)\nFigure 8. Optimization.\nData: | y | 28.5 |", kind="figure"),
    ]
    questions = [
        FigureQuestion("q1", "how much did energy fall", ["paper1"], {"8"}, 28.5),
        FigureQuestion("q2", "true or false?", ["paper1"], set(), None),
    ]
    summary, results = evaluate_figures(chunks, questions, FixedRetriever(["noise", "prose", "caption", "chart"]))
    q1 = results[0]
    assert (q1.figure_rank, q1.value_rank) == (3, 4)  # the wrong-doc chunk at rank 1 is ignored
    assert summary["questions"] == 2 and summary["figure_questions"] == 1 and summary["value_questions"] == 1
    assert summary["figure_chunks"] == 1
    assert summary["figure_recall@1"] == 0 and summary["figure_recall@5"] == 1 and summary["value_recall@5"] == 1

    summary, results = evaluate_figures(chunks, questions, FixedRetriever(["chart", "caption", "prose", "noise"]))
    assert (results[0].figure_rank, results[0].value_rank) == (1, 1)
    assert summary["figure_recall@1"] == 1 and summary["value_recall@1"] == 1


def test_evaluate_figures_rejects_questions_outside_the_corpus():
    with pytest.raises(ValueError):
        evaluate_figures([_chunk("a", "x")], [FigureQuestion("q", "?", ["missing"])], FixedRetriever(["a"]))


def test_load_figure_questions(tmp_path):
    csv_path = tmp_path / "q.csv"
    csv_path.write_text(
        "id,question,answer_value,answer_unit,ref_id,supporting_materials,Figure,is_NA\n"
        'q1,How much?,28.5,percent,[\'wu2021a\'],"Figure 8 caption: ""28.5%""",1,0\n'
        "q2,Which?,Claude,is_blank,['jegham2025'],Table 2,0,0\n"
        "q3,True?,1,is_blank,['dodge2022'],Section 6; Figure 3,1,0\n")
    qs = load_figure_questions(csv_path)
    assert [q.id for q in qs] == ["q1", "q3"]
    assert qs[0].ref_ids == ["wu2021a"] and qs[0].figure_numbers == {"8"} and qs[0].answer_number == 28.5
    assert qs[1].answer_number is None  # boolean answers aren't value-scored


# --- searching and reading charts, end to end -----------------------------------------------------

from wattbot.eval import evaluate_e2e
from wattbot.figure_qa import FigureSearcher, numbers_match, read_with_retry, reply_numbers


def _fig(cid, text, ref="paper1", index=0):
    return Chunk(doc_id=ref, text=text, heading_path=(), pages=[1], kind="figure", source_refs=[], chunk_id=cid,
                 metadata={"ref_id": ref, "figure_index": index})


def test_reply_numbers_ignore_figure_references_and_later_lines():
    assert reply_numbers("28.5 percent\nRead from Fig. 8, the 2020 bar.") == [28.5]
    assert reply_numbers("Figure 3: 1,200 MW") == [1200.0]
    assert reply_numbers("\n\n7.1 and 9 Wh") == [7.1, 9.0]
    assert reply_numbers("not in figure") == []
    assert reply_numbers("") == []


def test_numbers_match_at_the_gold_answers_precision():
    assert numbers_match("28.5%", 28.5) and numbers_match("28.54 percent", 28.5)
    assert not numbers_match("28.4 percent", 28.5) and not numbers_match("27.5 percent", 28.5)
    assert numbers_match("2007", 2007) and not numbers_match("2008", 2007)  # 1% of a year is not rounding
    assert numbers_match("4 categories", 4) and not numbers_match("5", 4)
    assert numbers_match("2.9x", 2.9) and not numbers_match("3", 2.9)
    assert not numbers_match("no number here", 5)
    # a relative tolerance is opt-in, for values that can only be estimated off the axis
    assert numbers_match("29", 28.5, tolerance=0.02) and not numbers_match("29", 28.5)


class FakeStore:
    """Dense side of the search: a fixed ranking, filtered the way Chroma applies `where`."""

    def __init__(self, order, refs):
        self.order, self.refs, self.wheres = order, refs, []

    def search(self, question, k, where):
        self.wheres.append(where)
        wanted = next((c["ref_id"] for c in where.get("$and", []) if "ref_id" in c), None)
        return [SimpleNamespace(chunk_id=c) for c in self.order if wanted in (None, self.refs[c])][:k]


def test_figure_searcher_fuses_bm25_and_dense_and_filters_by_document():
    chunks = [_fig("a", "energy per query bar chart", "p1"), _fig("b", "water use pie chart", "p2"),
              _fig("c", "energy footprint reduction", "p2")]
    assert [c.chunk_id for c in FigureSearcher(chunks).search("energy reduction", 3)] == ["c", "a"]  # BM25 only

    store = FakeStore(["b", "c", "a"], {"a": "p1", "b": "p2", "c": "p2"})
    fused = FigureSearcher(chunks, store).search("energy reduction", 3)
    assert fused[0].chunk_id == "c"                      # both rankings agree on it
    assert store.wheres == [{"kind": "figure"}]

    only_p2 = FigureSearcher(chunks, store).search("energy reduction", 3, ref_id="p2")
    assert {c.chunk_id for c in only_p2} <= {"b", "c"} and "a" not in [c.chunk_id for c in only_p2]
    assert store.wheres[-1] == {"$and": [{"kind": "figure"}, {"ref_id": "p2"}]}


def test_read_with_retry_doubles_the_budget_once(tmp_path):
    from test_figures import FakeVision  # noqa: F401  (same module; keeps the helper visible here)
    image = _png(tmp_path / "f.png")

    class Budget:
        def __init__(self, enough):
            self.enough, self.budgets = enough, []
            self.chat = SimpleNamespace(completions=SimpleNamespace(create=self.create))

        def create(self, **kw):
            self.budgets.append(kw["max_tokens"])
            ok = kw["max_tokens"] >= self.enough
            return SimpleNamespace(choices=[SimpleNamespace(
                message=SimpleNamespace(content="28.5%" if ok else None, reasoning_content="x" * 99),
                finish_reason="stop" if ok else "length")])

    retried = []
    client = Budget(16000)
    assert read_with_retry(client, "m", "q", image, "ctx", 8000, None, on_retry=retried.append) == "28.5%"
    assert client.budgets == [8000, 16000] and retried == [16000]

    client = Budget(0)
    assert read_with_retry(client, "m", "q", image, "ctx", 8000, None) == "28.5%" and client.budgets == [8000]

    with pytest.raises(figures.VisionModelError):
        read_with_retry(Budget(10**9), "m", "q", image, "ctx", 8000, None)


def test_evaluate_e2e_scores_numbers_and_separates_coincidences():
    chunks = {"right": _fig("right", "Fig. 8 reduction", "wu", 7), "near": _fig("near", "Fig. 6 average", "wu", 5),
              "other": _fig("other", "someone else's chart", "zz", 1), "broken": _fig("broken", "x", "wu", 9)}
    questions = [
        FigureQuestion("q1", "How much did energy fall?", ["wu"], {"8"}, 28.5),
        FigureQuestion("q2", "Which year?", ["wu"], set(), 2007.0),
        FigureQuestion("q3", "True or false?", ["wu"], set(), None),   # not numeric: not scored
    ]
    order = {"q1": ["near", "right", "other"], "q2": ["other", "broken", "near"]}
    replies = {("q1", "near"): "20 percent\nFig. 6 shows the average.", ("q1", "right"): "28.5 percent\nFig. 8.",
               ("q1", "other"): "28.5\nsomewhere else", ("q2", "other"): "2007", ("q2", "near"): "2008"}

    def read(q, chunk):
        if (q.id, chunk.chunk_id) not in replies:
            raise figures.VisionModelError("no usable reply")
        return replies[(q.id, chunk.chunk_id)]

    summary, results = evaluate_e2e(questions, lambda q: [chunks[c] for c in order[q.id]], read, n_reads=3)

    assert [r.id for r in results] == ["q1", "q2"] and summary["questions"] == 2
    q1, q2 = results
    assert [r.correct for r in q1.reads] == [False, True, True]
    assert q1.reads[0].numbers == [20.0] and q1.first_correct() == 2
    assert [r.from_gold_doc for r in q1.reads] == [True, True, False]
    assert q2.reads[0].correct and not q2.reads[0].from_gold_doc        # right number, wrong document
    assert q2.first_correct(strict=False) == 1 and q2.first_correct() is None
    assert q2.reads[1].reply.startswith("[error:") and q2.reads[1].numbers == []

    assert summary["read_errors"] == 1
    assert summary["answer@1"] == 0.5 and summary["answer@3"] == 1.0           # q2's coincidence counts here
    assert summary["strict_answer@1"] == 0 and summary["strict_answer@3"] == 0.5
    assert summary["wrong_number@1"] == 0.5                                    # q1's first read said 20
    assert summary["gold_doc_read@1"] == 0.5 and summary["gold_doc_read@3"] == 1.0


def test_scope_prompt_is_selectable_and_default_is_unchanged(tmp_path):
    image = _png(tmp_path / "f.png")
    client = FakeVision(reply="28.5 percent")
    figures.answer_from_figure(client, "vlm", "How much?", image, "ctx")
    figures.answer_from_figure(client, "vlm", "How much?", image, "ctx", prompt="scope")
    default_text, scope_text = (c["messages"][0]["content"][0]["text"] for c in client.calls)

    assert default_text == figures.ANSWER_PROMPT.replace("{context}", "ctx").replace("{question}", "How much?")
    assert "How much?" in scope_text and "ctx" in scope_text and "{question}" not in scope_text
    assert "is the one the question is about" in scope_text and "is the one the question is about" not in default_text
    assert set(figures.ANSWER_PROMPTS) == {"default", "scope"}
    with pytest.raises(KeyError):
        figures.answer_from_figure(client, "vlm", "q", image, prompt="nonsense")
    # read_with_retry carries the choice through, including on the retried call
    read_with_retry(client, "vlm", "How much?", image, "ctx", 100, None, prompt="scope")
    assert "is the one the question is about" in client.calls[-1]["messages"][0]["content"][0]["text"]
