# VerifyLens 🔍

**Multimodal KYC verification pipeline combining document intelligence, VLM-based field extraction, and facial verification.**

> Fine-tuned `Qwen2-VL-2B` on document Q&A for structured field extraction · ArcFace-based face matching · FastAPI serving

---

## Overview

VerifyLens is an end-to-end identity verification system designed to automate KYC (Know Your Customer) onboarding. It processes a document image and a live selfie, then returns extracted fields, a face match score, and an overall verification verdict.

```
Input: [ID Document Image] + [Selfie Image]
         │                        │
         ▼                        ▼
  ┌─────────────┐         ┌──────────────┐
  │ Doc Pipeline│         │ Face Pipeline│
  │             │         │              │
  │ CLIP Classif│         │ RetinaFace   │
  │ PaddleOCR   │         │ ArcFace      │
  │ VLM Extract │         │ Cosine Match │
  └──────┬──────┘         └──────┬───────┘
         │                        │
         └──────────┬─────────────┘
                    ▼
           ┌─────────────────┐
           │  Fusion & Verdict│
           │  confidence score│
           └─────────────────┘
```

## Features

- **Document Classification** — CLIP-based zero-shot classifier (Aadhaar, PAN, Passport, Driver's License)
- **OCR Field Extraction** — PaddleOCR + layout-aware post-processing
- **VLM Field Extraction** — QLoRA fine-tuned `Qwen2-VL-2B-Instruct` for structured JSON output
- **Face Verification** — RetinaFace detection + ArcFace embeddings + cosine similarity
- **REST API** — FastAPI server with async inference and confidence scores
- **Evaluation Suite** — Field-level accuracy, face match ROC/AUC benchmarks

## Project Structure

```
VerifyLens/
├── src/
│   ├── document/
│   │   ├── classifier.py       # CLIP document type classifier
│   │   ├── ocr.py              # PaddleOCR wrapper
│   │   └── field_extractor.py  # VLM-based structured extraction
│   ├── face/
│   │   ├── detector.py         # RetinaFace face detection
│   │   ├── embedder.py         # ArcFace embedding model
│   │   └── matcher.py          # Cosine similarity + threshold
│   ├── training/
│   │   ├── dataset.py          # DocVQA-style dataset builder
│   │   ├── train.py            # QLoRA fine-tuning script
│   │   └── evaluate.py         # Post-training evaluation
│   ├── api/
│   │   ├── server.py           # FastAPI app
│   │   ├── models.py           # Pydantic request/response schemas
│   │   └── pipeline.py         # End-to-end orchestration
│   └── utils/
│       ├── image_utils.py
│       └── config.py
├── data/
│   ├── synthetic/              # Synthetic ID images (generated)
│   └── README.md
├── notebooks/
│   ├── 01_data_exploration.ipynb
│   ├── 02_ocr_benchmarks.ipynb
│   └── 03_vlm_finetuning_demo.ipynb
├── scripts/
│   ├── generate_synthetic_data.py
│   └── run_eval.py
├── tests/
├── configs/
│   └── training_config.yaml
├── requirements.txt
├── setup.py
└── README.md
```

## Quickstart

```bash
git clone https://github.com/sunset0331/VerifyLens
cd VerifyLens
pip install -e .
uvicorn src.api.server:app --reload
```

## Testing the VLM Locally (Apple Silicon)

If you are on an M-Series Mac and encounter `Pillow` or `torch` C-binding compilation issues with the standard vision libraries (`facenet-pytorch`, `paddleocr`), you can still natively test the **fine-tuned VLM extraction engine** using the MLX backend. 

We provide a standalone testing script that isolates the LLM and runs it against simulated noisy OCR text to demonstrate its extraction capabilities without requiring PyTorch:

```bash
# Run the VLM on default heavily degraded synthetic OCR text
python test_vlm_only.py

# Or pass your own messy text string:
python test_vlm_only.py --text "P A 5 5 P 0 R T | Name: J0hn D0e | D0B: 01/01/90 | Numb3r: A1234567"
```

## Model Card

| Component | Model | Size |
|-----------|-------|------|
| Document Classifier | CLIP ViT-L/14 | Zero-shot |
| OCR | PaddleOCR v4 | — |
| Field Extractor | Qwen2.5-1.5B (MLX LoRA) | 1.5B (4-bit) |
| Face Detector | MTCNN | ~1.6MB |
| Face Embedder | InceptionResnetV1 | ~166MB |

## Training Details & Synthetic Data

To build a robust field extractor without compromising PII (Personally Identifiable Information), the model was fine-tuned exclusively on locally-generated synthetic data.

- **Dataset Volume**: 3,000 synthetic QA pairs (85% train / 15% validation split).
- **Synthetic Parameters**:
  - **Generator**: `Faker` (en_IN locale) to generate realistic Indian names, addresses, DOBs, and ID numbers (Aadhaar, PAN, Passport, DL).
  - **Noise Injection**: Simulated real-world OCR degradation with a 4% noise probability per character. This injects typical OCR confusions (e.g., `0` vs `O`, `8` vs `B`), random dropped characters, and garbage symbols to simulate blurry or heavily degraded ID photos.
- **Fine-Tuning Configuration (LoRA)**:
  - **Target Layers**: The last 8 transformer layers (`--num-layers 8`).
  - **Method**: QLoRA (4-bit quantization) on Apple Silicon via MLX (`mlx-lm.lora`).
  - **Hyperparameters**: 600 iterations, batch size 4, learning rate 1e-4.

## Benchmarks: Real-World OCR Noise Resistance

We evaluated the model on extracting structured JSON from simulated raw OCR text with injected real-world noise (e.g. typos, garbage characters, missing spaces) across validation samples. The fine-tuned LoRA adapter shows massive improvements, particularly in extracting true field values despite corrupted inputs.

| Metric / Field | Base Model (Qwen2.5) | Fine-Tuned (LoRA) | Improvement |
|----------------|----------------------|-------------------|-------------|
| **Exact Match** | 26.0% | **80.0%** | 🟢 +54.0% |
| **Valid JSON Rate**| 100.0% | **100.0%** | ⚪ 0.0% |
| **Doc Type** | 0.0% | **100.0%** | 🟢 +100.0% |
| **Doc Number** | 0.0% | **80.0%** | 🟢 +80.0% |
| **DOB** | 57.9% | **89.5%** | 🟢 +31.6% |
| **Name** | 78.6% | **85.7%** | 🟢 +7.1% |

> *Note: The baseline accuracy dropped heavily (to 26%) when real-world OCR noise was introduced. The fine-tuned model successfully learns to "denoise" the text to reliably extract document numbers and dates despite OCR errors.*

## Status

> 🚧 Active development — see branch `dev` for latest work.
