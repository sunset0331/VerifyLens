# VerifyLens

**Multimodal KYC verification pipeline combining document intelligence, OCR-to-JSON language modeling, direct Vision-Language Models (VLMs), and facial verification.**

VerifyLens automates identity document verification and KYC (Know Your Customer) workflows. Real-world document extraction is often brittle due to noisy OCR (Optical Character Recognition) resulting from poor lighting, blur, and low-resolution uploads. VerifyLens solves this by offering a dual-extraction backend: 
1. **Approach 1 (OCR + LLM):** Feeds raw, noisy PaddleOCR output into a locally-hosted, fine-tuned Language Model (Qwen2.5-1.5B) that acts as an intelligent parser.
2. **Approach 2 (Direct VLM):** Bypasses OCR entirely, reading document pixels directly via a Vision-Language Model (Qwen2.5-VL) to emit structured JSON.

A parallel facial verification pipeline independently confirms the user's identity.

---

## Architecture

```text
Input: [ID Document Image] + [Selfie Image]
         │                        │
         ▼                        ▼
  ┌─────────────┐         ┌──────────────┐
  │ Doc Pipeline│         │ Face Pipeline│
  │ (Dynamic)   │         │              │
  │             │         │ MTCNN        │
  │ ┌─────────┐ │         │ (detection)  │
  │ │VLM Extr.│ │         │              │
  │ └────┬────┘ │         │              │
  │      OR     │         │ Inception    │
  │ ┌─────────┐ │         │ ResnetV1     │
  │ │OCR+LLM  │ │         │ (embeddings) │
  │ └────┬────┘ │         └──────┬───────┘
  └──────┼──────┘                │
         └─────────┬─────────────┘
         └─────────┬─────────────┘
                   ▼
          ┌─────────────────┐
          │     Fusion      │
          │(Cosine matching,│
          │ field validation)│
          └─────────────────┘
                   ▼
           [Final KYC Verdict]
```

## Key Components

1. **Document Extraction Router**: Dynamically routes document processing to either the `vlm` or `ocr_llm` backend based on configuration.
2. **Vision-Language Model (VLM)**: Qwen2.5-VL extracts identity fields directly from document images with robust layout understanding (Approach 2).
3. **OCR-to-JSON Language Model**: PaddleOCR 3.7.0 combined with a fine-tuned Qwen2.5-1.5B model to act as a resilient OCR parser (Approach 1).
4. **Face Verification**: MTCNN for bounding box detection followed by InceptionResnetV1 for high-dimensional facial embeddings, compared via cosine similarity.
5. **REST API**: A FastAPI backend that orchestrates the execution of these pipelines to return a standardized verification verdict.

## Model Details

| Pipeline | Role | Model | Framework |
|----------|------|-------|-----------|
| Document | VLM Extractor | Qwen2.5-VL-3B-Instruct-4bit | MLX (mlx-vlm) |
| Document | OCR Backend | PaddleOCR 3.7.0 | PaddlePaddle |
| Document | OCR-LLM Extractor | Qwen2.5-1.5B-Instruct-4bit | MLX (mlx-lm) |
| Face | Detector | MTCNN | PyTorch |
| Face | Embedder | InceptionResnetV1 | PyTorch |

## Training & Fine-Tuning Approach

The language model was fine-tuned to act as a resilient OCR text parser using LoRA fine-tuning on a 4-bit quantized Qwen2.5-1.5B model using MLX.

- **Base Model**: `mlx-community/Qwen2.5-1.5B-Instruct-4bit`
- **Methodology**: Low-Rank Adaptation (LoRA) targeting the last 8 transformer layers over 600 iterations.
- **Optimization Techniques**: 4-bit quantization, LoRA, and native MLX inference allows the entire generative extraction pipeline to run highly efficiently on local Apple Silicon unified memory without requiring massive VRAM footprint.

### Synthetic Data Generation
Due to the highly sensitive nature of PII (Personally Identifiable Information) on identity documents, the entire fine-tuning dataset was generated synthetically using the Python `Faker` library configured for the `en_IN` locale to mimic Aadhaar, PAN, and Indian Passports.

### OCR Noise Augmentation
To bridge the domain gap between pristine synthetic text and real-world mobile scans, we implemented an OCR degradation pipeline. This pipeline injects a 4% character-level noise probability, swapping visually similar characters (e.g., `0` vs `O`, `8` vs `B`) and arbitrarily dropping or adding whitespace to mimic poor alignment and bounding-box overlap.

## Evaluation Methodology & Results

The fine-tuned model was evaluated on a held-out dataset of 50 synthetic validation samples subjected to the same simulated OCR noise. The primary metric was exact-match accuracy against the ground-truth JSON fields.

| Metric / Field | Base Model (Qwen2.5) | Fine-Tuned (LoRA) | Improvement |
|----------------|----------------------|-------------------|-------------|
| **Overall Exact Match** | 26.0% | **80.0%** | +54.0 percentage points |
| **Valid JSON Rate**| 100.0% | **100.0%** | 0.0% |
| **Doc Type** | 0.0% | **100.0%** | +100.0% |
| **Doc Number** | 0.0% | **80.0%** | +80.0% |
| **DOB** | 57.9% | **89.5%** | +31.6% |
| **Name** | 78.6% | **85.7%** | +7.1% |
| **Gender** | 50.0% | 50.0% | 0.0% |

> *Conclusion: The fine-tuned adapter achieved **80% exact-match accuracy on 50 synthetic validation samples**, demonstrating a massive +54.0 percentage points improvement in handling OCR noise compared to the base model.*

## Example OCR → JSON Flow

Here is a real example of the pipeline running successfully:

**Raw OCR Output (from PaddleOCR):**
```text
PASSPORT Name: Kai Gole PHOTO Date of Birth: 30/05/2007 Passport No.: D1419610
```

**Fine-Tuned Extractor Output (Qwen2.5 + LoRA):**
```json
{
    "name": "Kai Gole",
    "dob": "30/05/2007",
    "doc_number": "D1419610"
}
```
*(PaddleOCR Confidence: 0.996)*

## Installation & Setup

1. **Clone the repository:**
   ```bash
   git clone https://github.com/sunset0331/VerifyLens.git
   cd VerifyLens
   ```

2. **Create a Python 3.12 environment:**
   ```bash
   python -m venv .venv312
   source .venv312/bin/activate
   ```

3. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

## Usage Instructions

### Running Training
To fine-tune the LoRA adapter locally using MLX:
```bash
python -m src.training.train --iters 600 --batch-size 4
```

### Running Evaluation
To generate metrics against the validation set:
```bash
python -m src.training.evaluate
```

### Running the API
Boot the FastAPI server for end-to-end KYC verification:
```bash
uvicorn src.api.server:app --reload
```
You can then POST form data (`id_image`, `selfie_image`) to `http://localhost:8000/verify`.

## Project Structure

```text
VerifyLens/
├── src/
│   ├── document/
│   │   ├── ocr.py              # PaddleOCR wrapper
│   │   └── field_extractor.py  # Qwen2.5 OCR-to-JSON extractor via mlx_lm
│   ├── face/
│   │   ├── detector.py         # MTCNN face detection
│   │   ├── embedder.py         # InceptionResnetV1 embedding model
│   │   └── matcher.py          # Cosine similarity logic
│   ├── training/
│   │   ├── train.py            # MLX QLoRA fine-tuning script
│   │   └── evaluate.py         # Diagnostic evaluation script
│   └── api/
│       ├── server.py           # FastAPI app routes
│       └── pipeline.py         # Concurrent orchestration pipeline
├── checkpoints/
│   └── verifylens-adapter/     # Fine-tuned LoRA weights
└── data/                       # Synthetic data pipelines
```

## Limitations

- **Synthetic Validation:** The current validation dataset is entirely synthetic. Real-world physical documents possess distinct photometric distortions (glare, holograms) that this dataset may not perfectly capture.
- **Small Evaluation Sample:** The current reported evaluation leverages only 50 validation samples.
- **OCR Distribution Shift:** Real-world OCR error distributions may differ from the programmatic 4% character-swap noise applied during training.
- **Field Performance Variation:** Model performance is not uniform across all fields. For instance, the `gender` extraction showed no improvement in the current evaluation split.
- **Face Verification Baseline:** The facial verification backend currently uses MTCNN + InceptionResnetV1, which is effective but serves as a baseline compared to modern ArcFace/RetinaFace architectures.
- **Platform Constraints:** This repository is heavily optimized for Apple Silicon (macOS) via MLX and currently assumes it as the intended local inference environment.

## Future Improvements

1. **ArcFace/RetinaFace Upgrade**: Replace the MTCNN/InceptionResnetV1 pipeline with state-of-the-art RetinaFace detection and ArcFace embeddings to increase facial recognition robustness under severe poses and varied lighting.
2. **Hybrid VLM + LoRA Architecture**: Explore fusing Approach 1 and Approach 2. Benchmark data indicates VLM excels at spatial layout (`doc_type`), while OCR+LoRA excels at high-fidelity identity numbers (`doc_number`).
3. **Hardware Agnosticism**: Standardize the extraction pipeline utilizing PyTorch/vLLM so the API can scale seamlessly across Linux/CUDA environments in production.
