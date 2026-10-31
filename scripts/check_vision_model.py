#!/usr/bin/env python3
"""Live check that the gateway's vision model can read a chart (run this before describe_figures.py).

Usage:
    python scripts/check_vision_model.py [--model NAME] [--probe-image-embedding]

Lists the models the gateway serves, draws a small bar chart with known values, asks the model to
read it with the same prompt describe_figures.py uses, and passes if all three values come back.
--probe-image-embedding also embeds three different charts and three text captions with the embedding
model and checks that each image lands nearest its own caption (a comparison arm: Qwen3-VL embeddings
put text and images in one space, if the gateway really processes image input).
Needs OPENAI_API_KEY and the UW VPN.
"""
import argparse
import os
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from wattbot import config, figures
from wattbot.embeddings import EmbeddingConfigurationError, create_embedding_client

BARS = [("Alpha", 3), ("Beta", 7), ("Gamma", 5)]


def draw_chart(path: Path) -> None:
    from PIL import Image, ImageDraw, ImageFont
    try:
        font = ImageFont.load_default(size=22)
    except TypeError:  # Pillow < 10.1 has a single fixed size
        font = ImageFont.load_default()
    img = Image.new("RGB", (640, 420), "white")
    draw = ImageDraw.Draw(img)
    draw.text((140, 15), "Energy per query (Wh)", fill="black", font=font)
    draw.line([(60, 60), (60, 350), (600, 350)], fill="black", width=2)
    for i, (name, value) in enumerate(BARS):
        left = 110 + i * 170
        top = 350 - value * 35
        draw.rectangle([left, top, left + 100, 350], fill="#3b6ea5")
        draw.text((left + 38, top - 28), str(value), fill="black", font=font)
        draw.text((left + 15, 360), name, fill="black", font=font)
    img.save(path)


def _canvas(title: str):
    from PIL import Image, ImageDraw, ImageFont
    try:
        font = ImageFont.load_default(size=22)
    except TypeError:
        font = ImageFont.load_default()
    img = Image.new("RGB", (640, 420), "white")
    draw = ImageDraw.Draw(img)
    draw.text((140, 15), title, fill="black", font=font)
    return img, draw


def draw_line_chart(path: Path) -> None:
    img, draw = _canvas("Capacity over time")
    draw.line([(60, 60), (60, 350), (600, 350)], fill="black", width=2)
    draw.line([(80, 320), (200, 270), (320, 200), (440, 140), (560, 80)], fill="#c0392b", width=5)
    img.save(path)


def draw_pie_chart(path: Path) -> None:
    img, draw = _canvas("Share of electricity")
    for start, end, color in ((0, 200, "#3b6ea5"), (200, 290, "#e0a030"), (290, 360, "#4a9d5b")):
        draw.pieslice([160, 70, 480, 390], start, end, fill=color)
    img.save(path)


PROBE_CHARTS = {
    "bar": (draw_chart, "a bar chart with three vertical bars of different heights"),
    "line": (draw_line_chart, "a line chart with a line rising from the lower left to the upper right"),
    "pie": (draw_pie_chart, "a pie chart divided into three colored slices"),
}


def probe_image_embedding(client, tmp: Path) -> bool:
    """Embed three different charts and three plain-text captions; the image must land nearest its own caption.

    A successful call alone proves nothing: a gateway that ignores image input would embed the base64
    text of the data URL and still return a vector of the right size. Matching each image to its caption
    only works if the image was actually seen.
    """
    from wattbot.embeddings import cosine_similarity, embed_texts
    names = list(PROBE_CHARTS)
    urls = []
    for name in names:
        path = tmp / f"probe_{name}.png"
        PROBE_CHARTS[name][0](path)
        urls.append(figures.image_data_url(path)[0])

    print("\nimage embedding probe:")
    try:
        response = client.embeddings.create(model=config.EMBEDDING_MODEL, input=urls)
        image_vecs = [item.embedding for item in sorted(response.data, key=lambda d: d.index)]
        text_vecs = embed_texts([PROBE_CHARTS[n][1] for n in names], client)
    except Exception as error:
        print(f"  failed ({type(error).__name__}: {str(error)[:200]})")
        return False

    sims = [[cosine_similarity(i, t) for t in text_vecs] for i in image_vecs]
    print("  cosine similarity, rows = images, columns = captions")
    print("          " + "".join(f"{n:>8}" for n in names))
    for name, row in zip(names, sims):
        print(f"  {name:<8}" + "".join(f"{s:>8.3f}" for s in row))
    matched = all(max(range(len(names)), key=row.__getitem__) == i for i, row in enumerate(sims))
    image_to_image = max(cosine_similarity(image_vecs[i], image_vecs[j])
                         for i in range(len(names)) for j in range(i + 1, len(names)))
    print(f"  each image is closest to its own caption: {matched}")
    print(f"  highest image-to-image similarity: {image_to_image:.3f} (near 1.0 suggests the data URL is being "
          "embedded as text)")
    print("  verdict: " + ("the gateway embeds images" if matched else
                           "NOT convincing; the gateway may be embedding the data URL as text"))
    return matched


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model", default="")
    parser.add_argument("--probe-image-embedding", action="store_true")
    args = parser.parse_args()

    try:
        client = create_embedding_client()
        print("models served by the gateway:")
        for model in sorted(m.id for m in client.models.list().data):
            print(f"  {model}")
        model = figures.resolve_vision_model(client, args.model)
    except (EmbeddingConfigurationError, figures.VisionModelError) as error:
        sys.exit(f"error: {error}")
    print(f"\nusing vision model: {model}")

    with tempfile.TemporaryDirectory() as tmp:
        image = Path(tmp) / "chart.png"
        draw_chart(image)
        record = figures.describe_image(client, model, image, caption="Figure 1. Energy per query by model.")
        result = record["result"]
        print("\nreading:\n" + (figures.render_description(result, "Figure 1. Energy per query by model") or "(none)"))
        text = result["data"] + " " + result["summary"]
        missing = [str(v) for _, v in BARS if str(v) not in text]
        ok = not result.get("parse_error") and not missing
        print(f"\nJSON reply parsed: {not result.get('parse_error')}; values missing from reading: {missing or 'none'}")
        if args.probe_image_embedding:
            probe_image_embedding(client, Path(tmp))  # informational: does not change PASS/FAIL
    print("\noverall: " + ("PASS" if ok else "FAIL"))
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
