"""
scripts/generate_synthetic_data.py
-----------------------------------
Generates a synthetic DocVQA-style dataset for fine-tuning the VLM.

Each sample = { "image": <PIL Image of fake ID>, "question": str, "answer": str }

Fake data is generated using the Faker library with Indian locale.
Images are programmatically rendered via Pillow (no real PII involved).

Usage:
    python scripts/generate_synthetic_data.py --num_samples 5000 --output data/synthetic
"""

import argparse
import json
import random
import uuid
from pathlib import Path
from typing import Dict, List, Tuple, Any

from PIL import Image, ImageDraw, ImageFont
from faker import Faker

fake = Faker("en_IN")
random.seed(42)

# ── Document templates ──────────────────────────────────────────────────────

DOC_COLORS = {
    "aadhaar": {"bg": (255, 241, 200), "header": (99, 40, 138), "text": (30, 30, 30)},
    "pan":     {"bg": (255, 255, 255), "header": (10, 60, 130), "text": (20, 20, 20)},
    "passport": {"bg": (220, 235, 255), "header": (0, 40, 100), "text": (10, 10, 10)},
}

QA_TEMPLATES: Dict[str, List[Tuple[str, str]]] = {
    "name":       [("What is the name on this document?", "{name}"),
                   ("Who does this ID card belong to?", "{name}")],
    "dob":        [("What is the date of birth?", "{dob}"),
                   ("When was this person born?", "{dob}")],
    "doc_number": [("What is the document number?", "{doc_number}"),
                   ("What is the ID number on this card?", "{doc_number}")],
    "doc_type":   [("What type of document is this?", "{doc_type}"),
                   ("Identify the document type.", "{doc_type}")],
}


def _make_aadhaar_number() -> str:
    return " ".join(str(random.randint(1000, 9999)) for _ in range(3))


def _make_pan_number() -> str:
    letters = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
    return (
        "".join(random.choices(letters, k=5))
        + "".join(random.choices("0123456789", k=4))
        + random.choice(letters)
    )


def _make_passport_number() -> str:
    return random.choice("ABCDEFGHIJKLMNOPQRSTUVWXYZ") + str(
        random.randint(1000000, 9999999)
    )


DOC_NUMBER_FN = {
    "aadhaar": _make_aadhaar_number,
    "pan": _make_pan_number,
    "passport": _make_passport_number,
}

DOC_LABELS = {
    "aadhaar": "Aadhaar Card",
    "pan": "PAN Card",
    "passport": "Passport",
}


def render_id_card(
    doc_type: str,
    name: str,
    dob: str,
    doc_number: str,
    size: Tuple[int, int] = (600, 380),
) -> Image.Image:
    """Render a synthetic ID card image using Pillow."""
    colors = DOC_COLORS[doc_type]
    img = Image.new("RGB", size, color=colors["bg"])
    draw = ImageDraw.Draw(img)

    # Header bar
    draw.rectangle([0, 0, size[0], 60], fill=colors["header"])
    draw.text((20, 15), DOC_LABELS[doc_type].upper(), fill="white")
    if doc_type == "aadhaar":
        draw.text((size[0] - 160, 15), "Government of India", fill=(220, 220, 220))

    # Face placeholder (grey box)
    draw.rectangle([20, 80, 140, 200], fill=(200, 200, 200), outline=(150, 150, 150))
    draw.text((50, 130), "PHOTO", fill=(100, 100, 100))

    # Fields
    col_x = 160
    fields = [
        ("Name", name),
        ("Date of Birth", dob),
        (f"{DOC_LABELS[doc_type]} No.", doc_number),
    ]
    for i, (label, value) in enumerate(fields):
        y = 90 + i * 50
        draw.text((col_x, y), label + ":", fill=(100, 100, 100))
        draw.text((col_x, y + 18), value, fill=colors["text"])

    # Bottom strip / watermark
    draw.rectangle([0, size[1] - 30, size[0], size[1]], fill=colors["header"])
    draw.text((20, size[1] - 22), f"ID: {uuid.uuid4().hex[:12].upper()}", fill=(200, 200, 200))

    return img


def generate_sample(doc_type: str) -> Dict:
    """Generate one (image, QA pairs) sample."""
    name = fake.name()
    dob = fake.date_of_birth(minimum_age=18, maximum_age=70).strftime("%d/%m/%Y")
    doc_number = DOC_NUMBER_FN[doc_type]()

    image = render_id_card(doc_type, name, dob, doc_number)

    context = {
        "name": name,
        "dob": dob,
        "doc_number": doc_number,
        "doc_type": DOC_LABELS[doc_type],
    }

    # Pick one QA pair per field
    qa_pairs = []
    for field_key, templates in QA_TEMPLATES.items():
        q_tmpl, a_tmpl = random.choice(templates)
        qa_pairs.append({
            "question": q_tmpl,
            "answer": a_tmpl.format(**context),
        })

    return {"image": image, "doc_type": doc_type, "qa_pairs": qa_pairs, "meta": context}


def main(num_samples: int, output_dir: Path, split: str):
    output_dir.mkdir(parents=True, exist_ok=True)
    images_dir = output_dir / "images"
    images_dir.mkdir(exist_ok=True)
    
    # Use deterministic seed for benchmark
    if split == "benchmark":
        random.seed(9999)
    else:
        random.seed(42)

    doc_types = list(DOC_NUMBER_FN.keys())
    records = []

    for i in range(num_samples):
        doc_type = random.choice(doc_types)
        sample = generate_sample(doc_type)

        img_path = images_dir / f"{i:05d}_{doc_type}.jpg"
        sample["image"].save(img_path, "JPEG", quality=90)

        if split == "benchmark":
            # Flat schema for fair benchmark evaluation
            records.append({
                "id": f"{i:05d}",
                "image_path": str(img_path.relative_to(output_dir)),
                "document_type": doc_type,
                "ground_truth": {
                    "name": sample["meta"]["name"],
                    "dob": sample["meta"]["dob"],
                    "doc_number": sample["meta"]["doc_number"],
                    "doc_type": sample["meta"]["doc_type"],
                    "gender": None,
                    "address": None
                }
            })
        else:
            # DocVQA QA-pair schema for training
            for qa in sample["qa_pairs"]:
                records.append({
                    "id": f"{i:05d}",
                    "image_path": str(img_path.relative_to(output_dir)),
                    "doc_type": doc_type,
                    "question": qa["question"],
                    "answer": qa["answer"],
                })

        if (i + 1) % 500 == 0:
            print(f"  Generated {i + 1}/{num_samples} samples...")

    # Save JSONL
    jsonl_path = output_dir / f"{split}.jsonl"
    with open(jsonl_path, "w") as f:
        for rec in records:
            f.write(json.dumps(rec) + "\n")

    print(f"\n✅ Done. {len(records)} QA pairs → {jsonl_path}")
    print(f"   Images saved to: {images_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate synthetic ID card dataset")
    parser.add_argument("--num_samples", type=int, default=5000)
    parser.add_argument("--output", type=Path, default=Path("data/synthetic"))
    parser.add_argument("--split", type=str, choices=["train", "val", "benchmark"], default="train", help="Which dataset split to generate.")
    args = parser.parse_args()
    main(args.num_samples, args.output, args.split)
