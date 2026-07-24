"""
src/training/dataset.py
------------------------
Builds the MLX-LM compatible training dataset from synthetic ID card data.

MLX-LM expects JSONL files in chat format:
  {"messages": [{"role": "user", "content": "..."}, {"role": "assistant", "content": "..."}]}

This script:
  1. Generates synthetic ID card QA pairs (no images needed for text-only LLM)
  2. Formats them into the MLX-LM chat template
  3. Splits into train/valid (85/15)
  4. Saves to data/mlx_train/

Usage:
    python -m src.training.dataset --num_samples 3000
"""

from __future__ import annotations

import argparse
import json
import random
import re
from pathlib import Path
from typing import Dict, List, Tuple

from faker import Faker

fake = Faker("en_IN")
random.seed(42)

# ── Synthetic field generators ───────────────────────────────────────────────

DOC_TYPES = {
    "aadhaar": "Aadhaar Card",
    "pan": "PAN Card",
    "passport": "Passport",
    "driving_license": "Driving License",
}


def _aadhaar_number() -> str:
    return " ".join(str(random.randint(1000, 9999)) for _ in range(3))


def _pan_number() -> str:
    chars = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
    return (
        "".join(random.choices(chars, k=5))
        + "".join(random.choices("0123456789", k=4))
        + random.choice(chars)
    )


def _passport_number() -> str:
    return random.choice("ABCDEFGHIJKLMNOPQRSTUVWXYZ") + str(random.randint(1000000, 9999999))


def _dl_number() -> str:
    state = random.choice(["MH", "DL", "KA", "TN", "UP", "GJ", "RJ"])
    return f"{state}{random.randint(10,99)}{random.randint(20180000, 20259999):08d}"


DOC_NUMBER_FN = {
    "aadhaar": _aadhaar_number,
    "pan": _pan_number,
    "passport": _passport_number,
    "driving_license": _dl_number,
}

GENDER_OPTIONS = ["Male", "Female", "Other"]

# ── OCR simulation: raw text that OCR would extract ─────────────────────────

OCR_TEMPLATES = {
    "aadhaar": (
        "Government of India | Unique Identification Authority of India | "
        "AADHAAR | {name} | DOB: {dob} | {gender} | "
        "{address} | {doc_number} | Download Date: 24/07/2025"
    ),
    "pan": (
        "INCOME TAX DEPARTMENT | GOVT. OF INDIA | PERMANENT ACCOUNT NUMBER | "
        "Name: {name} | Father's Name: {father_name} | "
        "Date of Birth: {dob} | {doc_number}"
    ),
    "passport": (
        "REPUBLIC OF INDIA | PASSPORT | Type: P | Country Code: IND | "
        "Surname: {surname} | Given Name: {given_name} | "
        "Nationality: Indian | Date of Birth: {dob} | Sex: {gender} | "
        "Place of Birth: {pob} | Date of Issue: {issue_date} | "
        "Date of Expiry: {expiry_date} | Passport No.: {doc_number}"
    ),
    "driving_license": (
        "MOTOR VEHICLES ACT | DRIVING LICENCE | "
        "Name: {name} | S/D/W of: {father_name} | DOB: {dob} | "
        "Address: {address} | DL No.: {doc_number} | "
        "Valid Till: {expiry_date} | Vehicle Class: LMV, MCWG"
    ),
}

# ── QA pair templates ────────────────────────────────────────────────────────

QA_TEMPLATES: List[Tuple[str, str]] = [
    # Name
    ("What is the name on this document?",
     '{{"name": "{name}"}}'),
    ("Who does this ID belong to?",
     '{{"name": "{name}"}}'),
    # DOB
    ("What is the date of birth?",
     '{{"dob": "{dob}"}}'),
    ("Extract the date of birth from this document.",
     '{{"dob": "{dob}"}}'),
    # Doc number
    ("What is the document number?",
     '{{"doc_number": "{doc_number}"}}'),
    ("Extract the ID number from this document.",
     '{{"doc_number": "{doc_number}"}}'),
    ("What is the Aadhaar/PAN/Passport number shown?",
     '{{"doc_number": "{doc_number}"}}'),
    # Doc type
    ("What type of identity document is this?",
     '{{"doc_type": "{doc_type_label}"}}'),
    ("Identify the document type.",
     '{{"doc_type": "{doc_type_label}"}}'),
    # Gender
    ("What is the gender of the person?",
     '{{"gender": "{gender}"}}'),
    # Multi-field extraction
    ("Extract all key fields from this document as JSON.",
     '{{"name": "{name}", "dob": "{dob}", "doc_number": "{doc_number}", "doc_type": "{doc_type_label}"}}'),
    ("Parse this identity document and return a structured JSON with name, dob, and document number.",
     '{{"name": "{name}", "dob": "{dob}", "doc_number": "{doc_number}"}}'),
]

SYSTEM_PROMPT = (
    "You are a document intelligence assistant specializing in Indian identity documents. "
    "You will receive the OCR-extracted text from an identity document and a question. "
    "Respond ONLY with a valid JSON object containing the requested field(s). "
    'Example: {"name": "Ravi Sharma"} or {"dob": "23/04/1990"}. '
    "If a field is not found in the text, use null as the value."
)


def _inject_ocr_noise(text: str, noise_prob: float = 0.05) -> str:
    """Simulate real-world OCR errors like character swapping, dropping, and spacing issues."""
    if random.random() > 0.8:  # 20% of documents are perfectly clean
        return text

    noisy_chars = []
    # Common OCR confusions
    confusions = {
        '0': 'O', 'O': '0',
        '1': 'I', 'I': '1', 'l': '1',
        'S': '5', '5': 'S',
        'B': '8', '8': 'B',
        'Z': '2', '2': 'Z',
        'A': '4',
    }
    
    for char in text:
        if random.random() < noise_prob:
            action = random.choice(["confuse", "drop", "space", "garbage"])
            if action == "confuse" and char in confusions:
                noisy_chars.append(confusions[char])
            elif action == "drop":
                continue # Skip character
            elif action == "space":
                noisy_chars.append(char + " ")
            elif action == "garbage":
                noisy_chars.append(random.choice("!@#$%^&*()_+-=[]{}|;:,.<>?~`"))
            else:
                noisy_chars.append(char)
        else:
            noisy_chars.append(char)
            
    return "".join(noisy_chars)


def _make_record(doc_type: str) -> Dict:
    """Generate one synthetic identity record."""
    name = fake.name()
    parts = name.split()
    surname = parts[-1]
    given_name = " ".join(parts[:-1])
    dob = fake.date_of_birth(minimum_age=18, maximum_age=70).strftime("%d/%m/%Y")
    gender = random.choice(GENDER_OPTIONS)
    address = fake.address().replace("\n", ", ")
    father_name = fake.name_male()
    doc_number = DOC_NUMBER_FN[doc_type]()
    issue_date = fake.date_between(start_date="-10y", end_date="today").strftime("%d/%m/%Y")
    expiry_date = fake.date_between(start_date="today", end_date="+10y").strftime("%d/%m/%Y")
    pob = fake.city()

    context = {
        "name": name,
        "surname": surname,
        "given_name": given_name,
        "dob": dob,
        "gender": gender,
        "address": address,
        "father_name": father_name,
        "doc_number": doc_number,
        "doc_type": doc_type,
        "doc_type_label": DOC_TYPES[doc_type],
        "issue_date": issue_date,
        "expiry_date": expiry_date,
        "pob": pob,
    }

    # Simulate OCR output
    ocr_text = OCR_TEMPLATES[doc_type].format(**context)
    
    # Inject realistic OCR noise
    noisy_ocr_text = _inject_ocr_noise(ocr_text, noise_prob=0.04)
    
    return {"context": context, "ocr_text": noisy_ocr_text}


def _format_message(ocr_text: str, question: str, answer: str) -> Dict:
    """Format a single QA pair into MLX-LM chat format."""
    user_content = f"Document OCR text:\n{ocr_text}\n\nQuestion: {question}"
    return {
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_content},
            {"role": "assistant", "content": answer},
        ]
    }


def build_dataset(num_samples: int, output_dir: Path):
    """
    Generate dataset and save as train.jsonl + valid.jsonl in MLX-LM format.
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    all_samples = []
    doc_types = list(DOC_TYPES.keys())

    print(f"Generating {num_samples} QA samples across {len(doc_types)} document types...")

    for i in range(num_samples):
        doc_type = doc_types[i % len(doc_types)]
        record = _make_record(doc_type)

        # Pick random QA template
        q_tmpl, a_tmpl = random.choice(QA_TEMPLATES)
        question = q_tmpl
        answer = a_tmpl.format(**record["context"])

        sample = _format_message(record["ocr_text"], question, answer)
        all_samples.append(sample)

        if (i + 1) % 500 == 0:
            print(f"  {i+1}/{num_samples} samples generated")

    # Shuffle and split
    random.shuffle(all_samples)
    split_idx = int(len(all_samples) * 0.85)
    train_data = all_samples[:split_idx]
    valid_data = all_samples[split_idx:]

    # Write JSONL
    train_path = output_dir / "train.jsonl"
    valid_path = output_dir / "valid.jsonl"

    with open(train_path, "w") as f:
        for sample in train_data:
            f.write(json.dumps(sample) + "\n")

    with open(valid_path, "w") as f:
        for sample in valid_data:
            f.write(json.dumps(sample) + "\n")

    print(f"\n✅ Dataset ready:")
    print(f"   Train: {len(train_data)} samples → {train_path}")
    print(f"   Valid: {len(valid_data)} samples → {valid_path}")

    # Show a sample
    print("\n── Sample entry ──────────────────────────────")
    sample = train_data[0]
    print(f"User:      {sample['messages'][1]['content'][:120]}...")
    print(f"Assistant: {sample['messages'][2]['content']}")

    return train_path, valid_path


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--num_samples", type=int, default=3000,
                        help="Total number of QA samples to generate")
    parser.add_argument("--output", type=Path, default=Path("data/mlx_train"))
    args = parser.parse_args()
    build_dataset(args.num_samples, args.output)
