"""Turn chart images into searchable text, and read exact values back off them.

Docling saves every picture as documents/figures/<doc_id>/figure_NN.png (NN is the index into
the parsed JSON's `pictures`). `scripts/describe_figures.py` sends each image to a vision-language
model and caches its structured reading in documents/figure_text/<doc_id>/figure_NN.json. At
chunking time `VlmFigureText` (or the model-free `LabelFigureText`) turns a picture into the text
of a "figure" chunk; `answer_from_figure` re-reads the image to answer one question.

Nothing here imports the chunking package (the pipeline imports this module), and the OpenAI
client is passed in, so everything is testable offline with a fake client.
"""
from __future__ import annotations

import base64
import hashlib
import io
import json
import re
from pathlib import Path
from typing import Any

from . import config

FIGURES_DIR = Path("documents/figures")
FIGURE_TEXT_DIR = Path("documents/figure_text")

PROMPT_VERSION = 1  # bump when FIGURE_PROMPT changes so cached readings are redone
MAX_IMAGE_SIDE = 2048  # larger images are downscaled before upload
MAX_DATA_ROWS = 80  # rows of the extracted data table kept in the chunk text
MAX_LABEL_CHARS = 1500
MAX_TOKENS = 8000  # a reasoning model can spend thousands of tokens thinking before it answers
NO_THINKING = {"chat_template_kwargs": {"enable_thinking": False}}  # vLLM/Qwen3 switch; opt-in via --no-thinking

# kinds the model assigns to images that carry no information worth indexing
NON_CONTENT_KINDS = frozenset({"photo", "logo", "icon", "decorative", "portrait", "background"})

FIGURE_PROMPT = """You are reading one figure from a research paper or report so that its contents can be searched and used to answer questions that need exact numbers.

Caption: {caption}
Text the PDF text layer found inside the figure (may be incomplete or out of order): {labels}

Reply with a single JSON object and nothing else, with these keys:
  "kind": one of "bar_chart", "line_chart", "scatter_plot", "pie_chart", "stacked_chart", "map", "table", "diagram", "infographic", "other_chart", "photo", "logo", "decorative"
  "title": the chart's own title, or ""
  "x_axis": the x axis label with units, or ""
  "y_axis": the y axis label with units, or ""
  "legend": a list of the series or category names
  "data": a markdown table with one row per data point (or category) and one column per series, copying every value exactly as printed on the figure. Mark a value you can only estimate from the axis as "~value". Use "" if the figure has no data to extract.
  "summary": two or three sentences saying what the figure shows, naming the maximum, the minimum and the overall trend, with units.
Never invent values. Prefer numbers printed on the figure over estimates from the axis."""

ANSWER_PROMPT = """Answer the question using only the figure image.

Caption and text extracted from the figure: {context}

Question: {question}

Read values off the figure carefully. Give the exact number(s) as printed on the figure, with units, on the first line. On the second line say in one sentence which part of the figure you read it from. If the figure does not contain the answer, reply "not in figure"."""


class VisionModelError(RuntimeError):
    """The vision model cannot be chosen or its reply cannot be used."""


# ---------------------------------------------------------------------------
# Paths and Docling lookups
# ---------------------------------------------------------------------------

def figure_name(index: int) -> str:
    return f"figure_{index:02d}"  # must match scripts/parse_pdfs.py


def figure_image_path(doc_id: str, index: int, root: Path = FIGURES_DIR) -> Path:
    return Path(root) / doc_id / f"{figure_name(index)}.png"


def figure_text_path(doc_id: str, index: int, root: Path = FIGURE_TEXT_DIR) -> Path:
    return Path(root) / doc_id / f"{figure_name(index)}.json"


def picture_index(picture: dict) -> int:
    return int(picture["self_ref"].rsplit("/", 1)[1])  # "#/pictures/12"


def _ref_text(doc: dict, ref: str) -> str:
    _, collection, idx = ref.split("/")
    return " ".join(doc[collection][int(idx)].get("text", "").split())


def picture_caption(doc: dict, picture: dict) -> str:
    texts = (_ref_text(doc, c["$ref"]) for c in picture.get("captions", []))
    return "\n".join(t for t in texts if t)


def picture_labels(doc: dict, picture: dict) -> list[str]:
    """Text Docling read out of the picture (axis labels, series names, printed values), caption excluded."""
    caption = {" ".join(line.split()) for line in picture_caption(doc, picture).split("\n")}
    seen, labels = set(), []
    for child in picture.get("children", []):
        text = _ref_text(doc, child["$ref"])
        if text and text not in caption and text not in seen:
            seen.add(text)
            labels.append(text)
    return labels


def image_size(path: Path) -> tuple[int, int] | None:
    """(width, height) without decoding the image, or None if Pillow is unavailable or the file is unreadable."""
    try:
        from PIL import Image
        with Image.open(path) as img:
            return img.size
    except Exception:
        return None


# ---------------------------------------------------------------------------
# Talking to the vision model
# ---------------------------------------------------------------------------

def file_sha256(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def image_data_url(path: Path, max_side: int = MAX_IMAGE_SIDE) -> tuple[str, str]:
    """Return (data URL, sha256 of the file) for an image, downscaled if its longest side exceeds max_side."""
    raw = Path(path).read_bytes()
    digest = hashlib.sha256(raw).hexdigest()
    payload = raw
    try:
        from PIL import Image
        with Image.open(io.BytesIO(raw)) as img:
            if max(img.size) > max_side:
                scale = max_side / max(img.size)
                img = img.convert("RGB").resize((round(img.width * scale), round(img.height * scale)),
                                                Image.LANCZOS)
                buf = io.BytesIO()
                img.save(buf, format="PNG")
                payload = buf.getvalue()
    except ImportError:
        pass
    return f"data:image/png;base64,{base64.b64encode(payload).decode()}", digest


def resolve_vision_model(client: Any, configured: str = "") -> str:
    """The configured model, else the gateway's only vision-language chat model."""
    configured = configured or config.VISION_MODEL
    if configured:
        return configured
    ids = [m.id for m in client.models.list().data]
    candidates = [i for i in ids if "vl" in i.lower() and not any(w in i.lower() for w in ("embed", "rerank"))]
    if len(candidates) == 1:
        return candidates[0]
    raise VisionModelError(
        "set VISION_MODEL to a vision chat model the gateway serves. "
        + (f"Candidates: {', '.join(candidates)}. " if candidates else
           "No model name contains 'vl'; if a general chat model is multimodal, set VISION_MODEL to it and "
           "run scripts/check_vision_model.py to test image input. ")
        + f"All models: {', '.join(sorted(ids))}")


_THINK = re.compile(r"<think>.*?</think>", re.DOTALL)  # reasoning models may prefix their answer with this


def _chat(client: Any, model: str, prompt: str, data_url: str, max_tokens: int = MAX_TOKENS,
          extra_body: dict | None = None) -> tuple[str, str | None]:
    """Send one image and prompt. Returns (reply with any reasoning removed, finish_reason).

    Raises VisionModelError when there is nothing usable to keep: an empty reply, or one cut off while
    still inside a <think> block. Callers must not cache those, so a rerun retries them.
    """
    from .vectorstore import call_with_retry  # lazy: vectorstore imports the chunking package

    def call():
        kwargs = {"extra_body": extra_body} if extra_body else {}
        return client.chat.completions.create(
            model=model,
            temperature=0,
            max_tokens=max_tokens,
            messages=[{"role": "user", "content": [
                {"type": "text", "text": prompt},
                {"type": "image_url", "image_url": {"url": data_url}},
            ]}],
            **kwargs,
        )

    choice = call_with_retry(call).choices[0]
    raw = choice.message.content or ""
    content = _THINK.sub("", raw).strip()
    finish = getattr(choice, "finish_reason", None)
    if not content or "<think>" in content:  # empty, or truncated before the answer began
        reasoning = (getattr(choice.message, "reasoning_content", None) or getattr(choice.message, "reasoning", None)
                     or "")
        raise VisionModelError(
            f"no usable reply (finish_reason={finish}, content {len(raw)} chars, separate reasoning "
            f"{len(reasoning)} chars, max_tokens={max_tokens}). A reasoning model can spend the whole budget "
            "thinking: raise --max-tokens, or try --no-thinking")
    return content, finish


def build_prompt(caption: str, labels: list[str]) -> str:
    label_text = "; ".join(labels)[:MAX_LABEL_CHARS] or "(none)"
    return FIGURE_PROMPT.replace("{caption}", caption or "(none)").replace("{labels}", label_text)


_FENCE = re.compile(r"^```[a-zA-Z]*\s*|\s*```$")


def parse_vlm_json(raw: str) -> dict | None:
    """The JSON object in a model reply, tolerating code fences and chatter around it."""
    text = _FENCE.sub("", raw.strip())
    start, end = text.find("{"), text.rfind("}")
    if start < 0 or end <= start:
        return None
    try:
        parsed = json.loads(text[start:end + 1])
    except json.JSONDecodeError:
        return None
    return parsed if isinstance(parsed, dict) else None


def _data_to_markdown(data: Any) -> str:
    if isinstance(data, str):
        return data.strip()
    if isinstance(data, list) and data and all(isinstance(r, dict) for r in data):
        cols = list(dict.fromkeys(k for r in data for k in r))
        lines = ["| " + " | ".join(cols) + " |", "|" + " --- |" * len(cols)]
        lines += ["| " + " | ".join(str(r.get(c, "")) for c in cols) + " |" for r in data]
        return "\n".join(lines)
    return json.dumps(data, ensure_ascii=False) if data else ""


def normalize_result(parsed: dict | None, raw: str) -> dict:
    """Fill defaults so downstream code can rely on every key; keep the raw reply if it wasn't JSON."""
    if parsed is None:
        return {"kind": "unknown", "title": "", "x_axis": "", "y_axis": "", "legend": [], "data": "",
                "summary": raw.strip(), "parse_error": True}
    legend = parsed.get("legend") or []
    return {
        "kind": str(parsed.get("kind") or "unknown").strip().lower(),
        "title": str(parsed.get("title") or "").strip(),
        "x_axis": str(parsed.get("x_axis") or "").strip(),
        "y_axis": str(parsed.get("y_axis") or "").strip(),
        "legend": [str(x) for x in legend] if isinstance(legend, list) else [str(legend)],
        "data": _data_to_markdown(parsed.get("data")),
        "summary": str(parsed.get("summary") or "").strip(),
    }


def describe_image(client: Any, model: str, image_path: Path, caption: str = "",
                   labels: list[str] | None = None, max_tokens: int = MAX_TOKENS,
                   extra_body: dict | None = None) -> dict:
    """Ask the vision model to read one figure. Returns the record that is cached on disk."""
    data_url, digest = image_data_url(image_path)
    raw, finish = _chat(client, model, build_prompt(caption, labels or []), data_url, max_tokens, extra_body)
    parsed = parse_vlm_json(raw)
    if parsed is None and finish == "length":
        # cut off mid-JSON: caching it would index a fragment as the figure's summary, so fail and let a rerun retry
        raise VisionModelError(f"reply truncated at max_tokens={max_tokens} before the JSON was complete; "
                               "raise --max-tokens")
    return {"prompt_version": PROMPT_VERSION, "model": model, "image_sha256": digest,
            "result": normalize_result(parsed, raw)}


def answer_from_figure(client: Any, model: str, question: str, image_path: Path, context: str = "",
                       max_tokens: int = MAX_TOKENS, extra_body: dict | None = None) -> str:
    """Re-read the original image to answer one question (more exact than the cached reading)."""
    data_url, _ = image_data_url(image_path)
    prompt = ANSWER_PROMPT.replace("{context}", context or "(none)").replace("{question}", question)
    return _chat(client, model, prompt, data_url, max_tokens, extra_body)[0]


# ---------------------------------------------------------------------------
# Cache on disk
# ---------------------------------------------------------------------------

def save_record(path: Path, record: dict) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(record, indent=2, ensure_ascii=False))
    tmp.replace(path)  # atomic, so an interrupted run never leaves a half-written file


def load_record(path: Path) -> dict | None:
    path = Path(path)
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text())
    except json.JSONDecodeError:
        return None


def record_is_current(record: dict | None, image_sha256: str) -> bool:
    return bool(record) and record.get("prompt_version") == PROMPT_VERSION \
        and record.get("image_sha256") == image_sha256


# ---------------------------------------------------------------------------
# Chunk text
# ---------------------------------------------------------------------------

def render_description(result: dict, caption: str = "") -> str | None:
    """The text of a figure chunk, or None for images that carry no content (logos, photos)."""
    kind = result.get("kind", "unknown")
    if kind in NON_CONTENT_KINDS:
        return None
    lines = [f"Figure ({kind.replace('_', ' ')})"]
    if caption:
        lines.append(caption)  # on its own line(s): eval/figures.py finds "Fig. N" captions at line start
    if result.get("title"):
        lines.append(f"Title: {result['title']}")
    if result.get("x_axis"):
        lines.append(f"X axis: {result['x_axis']}")
    if result.get("y_axis"):
        lines.append(f"Y axis: {result['y_axis']}")
    if result.get("legend"):
        lines.append("Legend: " + ", ".join(result["legend"]))
    if result.get("data"):
        rows = result["data"].splitlines()
        if len(rows) > MAX_DATA_ROWS:
            rows = rows[:MAX_DATA_ROWS] + [f"... ({len(rows) - MAX_DATA_ROWS} more rows)"]
        lines += ["Data:", *rows]
    if result.get("summary"):
        lines.append(f"Summary: {result['summary']}")
    return "\n".join(lines) if len(lines) > 1 else None


class VlmFigureText:
    """Figure text from the cached vision-model readings (documents/figure_text).

    Pictures with no cached reading, or read as a logo or photo, return None, so the loader
    keeps just their caption as before.
    """

    def __init__(self, root: Path = FIGURE_TEXT_DIR):
        self.root = Path(root)

    def __call__(self, doc_id: str, picture: dict, doc: dict, caption: str) -> str | None:
        record = load_record(figure_text_path(doc_id, picture_index(picture), self.root))
        if not record:
            return None
        return render_description(record["result"], caption)


class LabelFigureText:
    """Model-free baseline: caption plus the labels Docling read out of the picture.

    Gives BM25 and the embedder the series names and printed numbers, but not which value
    belongs to which series, so it is the floor the vision-model reading should beat.
    """

    def __init__(self, min_labels: int = 3, max_chars: int = MAX_LABEL_CHARS):
        self.min_labels = min_labels
        self.max_chars = max_chars

    def __call__(self, doc_id: str, picture: dict, doc: dict, caption: str) -> str | None:
        labels = picture_labels(doc, picture)
        if len(labels) < self.min_labels:
            return None
        lines = ["Figure"] + ([caption] if caption else [])
        lines.append("Text in figure: " + "; ".join(labels)[:self.max_chars])
        return "\n".join(lines)
