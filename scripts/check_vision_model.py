#!/usr/bin/env python3
"""Live check that the gateway's vision model can read a chart (run this before describe_figures.py).

Usage:
    python scripts/check_vision_model.py [--model NAME] [--probe-image-embedding]

Lists the models the gateway serves, draws a small bar chart with known values, asks the model to
read it with the same prompt describe_figures.py uses, and passes if all three values come back.
--probe-image-embedding also tries to embed that image with the embedding model (a comparison arm:
Qwen3-VL embeddings put text and images in one space, if the gateway accepts image input).
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


def probe_image_embedding(client, image: Path) -> None:
    data_url, _ = figures.image_data_url(image)
    attempts = {
        "input=[data URL]": {"model": config.EMBEDDING_MODEL, "input": [data_url]},
        "chat-style messages": {"model": config.EMBEDDING_MODEL, "messages": [
            {"role": "user", "content": [{"type": "image_url", "image_url": {"url": data_url}}]}]},
    }
    print("\nimage embedding probe:")
    for label, body in attempts.items():
        try:
            response = client.post("/embeddings", cast_to=object, body=body)
            print(f"  {label}: OK, dimension {len(response['data'][0]['embedding'])}")
        except Exception as error:
            print(f"  {label}: failed ({type(error).__name__}: {str(error)[:150]})")


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
            probe_image_embedding(client, image)
    print("\noverall: " + ("PASS" if ok else "FAIL"))
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
