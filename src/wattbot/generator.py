"""
Generator: prompt augmentation + LLM answer generation.

Stages:
  1. Prompt augmentation: combine question + retrieved context into a structured prompt
  2. Generation: LLM produces a JSON answer with all submission fields
  3. Parse: extract structured fields from LLM response
"""

import json
import re

from openai import OpenAI

from . import config


def _get_client():
    return OpenAI(base_url=config.BASE_URL, api_key=config.API_KEY, timeout=config.REQUEST_TIMEOUT)


SYSTEM_PROMPT = """\
You are WattBot, a retrieval-augmented question-answering system about AI's environmental impact.
Answer questions using ONLY the provided context passages. Each passage is tagged with its document ID.

Rules:
- If the context does not contain enough information to answer, set answer_value to "is_blank"
  and leave ref_id, ref_url, and supporting_materials as "is_blank".
- For True/False questions, answer_value must be 1 (True) or 0 (False).
- For numeric answers, provide just the number without units in answer_value.
- For ranges stated by a source, use (low,high) format.
- Cite ONLY documents that directly support your answer in ref_id.
- supporting_materials must contain the verbatim quote, table reference, or figure reference.
- explanation must connect the evidence to the answer.

Respond with valid JSON only, no markdown fencing."""

ANSWER_SCHEMA = """\
{
  "answer": "<concise natural language answer>",
  "answer_value": "<number, term, 1, 0, or is_blank>",
  "answer_unit": "<unit or is_blank>",
  "ref_id": "<comma-separated doc IDs or is_blank>",
  "ref_url": "<comma-separated URLs or is_blank>",
  "supporting_materials": "<verbatim evidence or is_blank>",
  "explanation": "<reasoning connecting evidence to answer>"
}"""


def build_prompt(question: str, chunks: list[dict], unit_hint: str = "") -> str:
    context_parts = []
    for c in chunks:
        doc_id = c["metadata"].get("doc_id", "unknown")
        context_parts.append(f"[Document: {doc_id}]\n{c['text']}")

    context_block = "\n\n---\n\n".join(context_parts)

    user_msg = f"Context:\n{context_block}\n\nQuestion: {question}"
    if unit_hint and unit_hint != "is_blank":
        user_msg += f"\n(Expected unit: {unit_hint})"
    user_msg += f"\n\nRespond with this JSON schema:\n{ANSWER_SCHEMA}"

    return user_msg


def generate(question: str, chunks: list[dict], unit_hint: str = "", client: OpenAI | None = None) -> dict:
    if client is None:
        client = _get_client()

    user_msg = build_prompt(question, chunks, unit_hint)

    resp = client.chat.completions.create(
        model=config.CHAT_MODEL,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_msg},
        ],
        temperature=0.0,
        max_tokens=1000,
    )

    raw = resp.choices[0].message.content.strip()
    return parse_response(raw)


def parse_response(raw: str) -> dict:
    cleaned = raw.strip()
    cleaned = re.sub(r"^```(?:json)?\s*", "", cleaned)
    cleaned = re.sub(r"\s*```$", "", cleaned)

    try:
        parsed = json.loads(cleaned)
    except json.JSONDecodeError:
        json_match = re.search(r"\{[^{}]*\}", cleaned, re.DOTALL)
        if json_match:
            try:
                parsed = json.loads(json_match.group())
            except json.JSONDecodeError:
                parsed = {}
        else:
            parsed = {}

    defaults = {
        "answer": "",
        "answer_value": "is_blank",
        "answer_unit": "is_blank",
        "ref_id": "is_blank",
        "ref_url": "is_blank",
        "supporting_materials": "is_blank",
        "explanation": "Unable to parse model response.",
    }

    for key, default in defaults.items():
        if key not in parsed or not parsed[key]:
            parsed[key] = default

    return parsed
