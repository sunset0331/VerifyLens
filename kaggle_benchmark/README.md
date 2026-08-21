# VerifyLens Kaggle Benchmark

Self-contained GPU benchmark package for the VerifyLens document extraction system.

## Why Kaggle?

The 200-sample VerifyLens benchmark runs heavy VLM and OCR+LLM inference.
On a Mac, processing all 200 images causes severe memory exhaustion and laptop shutdown.
This package moves the benchmark computation to a **Kaggle GPU machine** (T4/P100/A100)
where it can run safely and reliably.

---

## Required GPU

A CUDA GPU is **mandatory**. The script will fail clearly if CUDA is unavailable.

| Mode | Minimum GPU | Recommended |
|---|---|---|
| `--model vlm` | T4 (16 GB) | P100 / A100 |
| `--model base` | T4 (16 GB) | T4 is sufficient |

**Enable GPU on Kaggle**: Settings → Accelerator → GPU T4 x2 (or P100/V100/A100)

---

## Package Structure

```
kaggle_benchmark/
├── README.md              ← This file
├── requirements.txt       ← GPU/CUDA dependencies (NO MLX)
├── config.yaml            ← Benchmark configuration
├── run_benchmark.py       ← Main entry point (one script to run)
├── benchmark/
│   ├── benchmark.jsonl    ← 200-sample benchmark (copied verbatim from production)
│   └── images/            ← 200 JPEG document images (identical to production)
├── models/
│   └── README.md          ← Model download sizes, LoRA conversion requirements
├── results/               ← Output directory (written during benchmark)
│   ├── predictions.jsonl  ← Per-sample predictions (written incrementally)
│   ├── benchmark_results.json
│   ├── benchmark_results.csv
│   └── benchmark_report.md
└── verifylens_kaggle/     ← Self-contained benchmark library
    ├── normalizer.py      ← Copied from src/evaluation/normalizer.py
    ├── metrics.py         ← Copied from src/evaluation/metrics.py
    ├── hardware.py        ← GPU/CPU detection
    ├── dataset.py         ← Dataset loader
    ├── checkpoint.py      ← Resume logic
    ├── ocr_extractor.py   ← GPU PaddleOCR
    ├── llm_extractor.py   ← HuggingFace Qwen2.5-1.5B (base, no LoRA)
    ├── vlm_extractor.py   ← HuggingFace Qwen2.5-VL-3B (BF16 CUDA)
    ├── reporter.py        ← Generates all result files
    └── json_utils.py      ← JSON parsing helpers
```

---

## Installation

```bash
pip install -r requirements.txt
```

On Kaggle, PyTorch with CUDA is pre-installed. The above installs the remaining dependencies.

If flash-attention is available on your GPU (A100/H100), installing it is optional but speeds up VLM inference:
```bash
pip install flash-attn --no-build-isolation   # optional
```

---

## Dataset

The `benchmark/` directory contains the **exact** benchmark used in production:

- **`benchmark.jsonl`**: 200 samples, one per line
- **`images/`**: 200 JPEG synthetic ID card images (600×380 px)
- **Seed**: `random.seed(9999)` (from `scripts/generate_synthetic_data.py`)
- **Distribution**: ~64 Aadhaar, ~73 PAN, ~63 Passport (deterministic from seed)

The benchmark data is synthetic (generated with Faker + Pillow, no real PII).

Ground truth fields per sample: `name`, `dob`, `doc_number`, `doc_type`, `gender=null`, `address=null`

> **Note**: `gender` and `address` are `null` for 100% of samples. They always score 100% correct (null == null) and are flagged as non-discriminative in the report.

---

## Model Downloads

Models are downloaded automatically on first run via HuggingFace Hub.

| Mode | Model | Download Size | GPU Memory |
|---|---|---|---|
| `--model vlm` | `Qwen/Qwen2.5-VL-3B-Instruct` | ~7 GB | ~8–10 GB |
| `--model base` | `Qwen/Qwen2.5-1.5B-Instruct` | ~3 GB | ~4–5 GB |

No credentials or tokens are required (both are public models).

---

## 5-Sample Smoke Test

Before running the full benchmark, validate GPU compatibility with 5 samples:

```bash
# VLM smoke test (5 samples, ~1-2 minutes including download)
python run_benchmark.py --model vlm --limit 5

# Base LLM smoke test
python run_benchmark.py --model base --limit 5
```

Check `results/benchmark_report.md` after completion.

---

## 200-Sample Benchmark

```bash
# Full VLM benchmark (~15-30 minutes on T4)
python run_benchmark.py --model vlm --limit 200

# Full base LLM benchmark (~5-10 minutes on T4)
python run_benchmark.py --model base --limit 200
```

---

## LoRA Evaluation

The production LoRA adapter (trained with `mlx-lm` on Apple Silicon) **cannot be directly evaluated on Kaggle** without a conversion step.

```bash
# See full compatibility report:
python run_benchmark.py --model lora
```

This prints a detailed explanation and exits cleanly — it does NOT silently substitute the base model.

See `models/README.md` for conversion requirements.

---

## Resume Behavior

The benchmark writes predictions incrementally to `results/predictions.jsonl` after each sample.

If the Kaggle session is interrupted, simply re-run the same command:

```bash
python run_benchmark.py --model vlm --limit 200
```

Completed sample IDs are detected automatically and skipped. Only remaining samples are processed.

---

## Results Location

After the benchmark completes:

```
results/
├── predictions.jsonl       ← Per-sample: id, ground_truth, prediction, latency_ms, error
├── benchmark_results.json  ← Full structured results + hardware + memory stats
├── benchmark_results.csv   ← Flat metric table
└── benchmark_report.md     ← Human-readable Markdown report
```

---

## Known Differences from Mac MLX Implementation

| Aspect | Mac Production | Kaggle Benchmark |
|---|---|---|
| **VLM model** | `mlx-community/Qwen2.5-VL-3B-Instruct-4bit` | `Qwen/Qwen2.5-VL-3B-Instruct` |
| **VLM runtime** | `mlx-vlm` (Apple MLX) | HuggingFace Transformers |
| **VLM precision** | 4-bit MLX quantization | BF16 (bfloat16) |
| **LLM model** | `mlx-community/Qwen2.5-1.5B-Instruct-4bit` | `Qwen/Qwen2.5-1.5B-Instruct` |
| **LLM runtime** | `mlx-lm` (Apple MLX) | HuggingFace Transformers |
| **LLM precision** | 4-bit MLX quantization | BF16 (bfloat16) |
| **LoRA adapter** | ✅ Loaded | ❌ Cannot load (format incompatible) |
| **OCR device** | CPU | GPU (CUDA) |
| **Platform** | Apple Silicon (M-series) | NVIDIA GPU (CUDA) |

The VLM benchmark evaluates the **same model family and weights** through a **GPU-compatible runtime**. Results are comparable directionally but are **not bit-identical** to Mac production results.

The benchmark report explicitly states this for every run.

---

## Benchmark Modes Reference

| Command | Description |
|---|---|
| `python run_benchmark.py --model vlm --limit 5` | VLM smoke test |
| `python run_benchmark.py --model vlm --limit 20` | VLM medium test |
| `python run_benchmark.py --model vlm --limit 200` | VLM full benchmark |
| `python run_benchmark.py --model base --limit 5` | Base LLM smoke test |
| `python run_benchmark.py --model base --limit 200` | Base LLM full benchmark |
| `python run_benchmark.py --model lora` | LoRA compatibility report |

---

## Hardware Detection Output

At startup, the script prints:

```
============================================================
  HARDWARE DETECTION
============================================================
  CUDA available : True
  CUDA version   : 12.1
  GPU count      : 1
  GPU name       : NVIDIA Tesla T4
  GPU memory     : 15.78 GB
  CPU            : x86_64
  RAM            : 29.0 GB
  Python         : 3.10.x
  Platform       : Linux-...
============================================================
```

If CUDA is unavailable, the script exits immediately with instructions to enable GPU on Kaggle.
