#!/usr/bin/env python3
"""
run_benchmark.py
-----------------
VerifyLens Kaggle GPU Benchmark — Main Entry Point

Usage
-----
# Smoke test (5 samples):
    python run_benchmark.py --model vlm --limit 5

# Full VLM benchmark (200 samples):
    python run_benchmark.py --model vlm --limit 200

# Base LLM benchmark:
    python run_benchmark.py --model base --limit 200

# LoRA evaluation (prints adapter compatibility warning and exits):
    python run_benchmark.py --model lora

# Resume interrupted run:
    python run_benchmark.py --model vlm --limit 200
    (completed samples are automatically detected and skipped)

Modes
-----
  base   : OCR + Qwen2.5-1.5B-Instruct (BF16, CUDA, NO adapter)
  lora   : Prints LoRA adapter incompatibility report and stops
  vlm    : Qwen2.5-VL-3B-Instruct (BF16, CUDA) — primary benchmark

Requirements
------------
See requirements.txt. GPU with CUDA is mandatory.
The script will fail clearly if CUDA is not available.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path
from typing import Any, Dict, Optional

# ── LORA incompatibility message (printed before any imports) ────────────────
_LORA_MESSAGE = """
╔══════════════════════════════════════════════════════════════════╗
║      LoRA ADAPTER — KAGGLE COMPATIBILITY REPORT                 ║
╠══════════════════════════════════════════════════════════════════╣
║                                                                  ║
║  The existing VerifyLens LoRA adapter CANNOT be evaluated on    ║
║  Kaggle without a conversion step.                               ║
║                                                                  ║
║  WHY                                                             ║
║  ─────────────────────────────────────────────────────────────── ║
║  • Trained with: mlx-lm (Apple MLX)                             ║
║  • Base model  : mlx-community/Qwen2.5-1.5B-Instruct-4bit      ║
║                  (MLX 4-bit quantized format)                    ║
║  • Adapter     : checkpoints/verifylens-adapter/                ║
║                  adapters.safetensors                            ║
║                                                                  ║
║  • HuggingFace PEFT expects different layer naming conventions   ║
║  • MLX 4-bit base model ≠ HuggingFace full-precision base model ║
║  • Direct loading would cause shape mismatches / wrong results  ║
║                                                                  ║
║  WHAT WOULD BE NEEDED FOR CONVERSION                             ║
║  ─────────────────────────────────────────────────────────────── ║
║  1. Map MLX LoRA layer names → HuggingFace layer names           ║
║  2. Convert 4-bit MLX tensors → float32/BF16                    ║
║  3. Wrap in HuggingFace PEFT PeftModel format                    ║
║  4. Export adapter_config.json in PEFT format                    ║
║                                                                  ║
║  This conversion has NOT been implemented in this package.       ║
║                                                                  ║
║  WHAT YOU CAN DO NOW                                             ║
║  ─────────────────────────────────────────────────────────────── ║
║  Run the BASE model benchmark (no adapter):                      ║
║    python run_benchmark.py --model base --limit 5               ║
║                                                                  ║
║  This evaluates Qwen2.5-1.5B-Instruct zero-shot capability      ║
║  through the same OCR+LLM pipeline, without the LoRA adapter.  ║
║                                                                  ║
║  DO NOT compare base model results directly to the production   ║
║  LoRA results — they use different model configurations.        ║
║                                                                  ║
╚══════════════════════════════════════════════════════════════════╝
"""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="VerifyLens Kaggle GPU Benchmark",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python run_benchmark.py --model vlm --limit 5         # Smoke test
  python run_benchmark.py --model vlm --limit 200       # Full VLM benchmark
  python run_benchmark.py --model base --limit 200      # Base LLM benchmark
  python run_benchmark.py --model lora                  # See LoRA compat report
        """,
    )
    parser.add_argument(
        "--model",
        choices=["base", "lora", "vlm", "hybrid"],
        required=True,
        help=(
            "base: OCR+Qwen2.5-1.5B (no LoRA). "
            "lora: Print adapter incompatibility report and exit. "
            "vlm: Qwen2.5-VL-3B (BF16 CUDA). "
            "hybrid: VLM doc_type + PEFT OCR+LoRA."
        ),
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=200,
        help="Number of samples to process (default: 200). Use 5 or 20 for smoke tests.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=1,
        help=(
            "Batch size for inference (default: 1). "
            "VLM is memory-intensive — do NOT increase beyond 1 unless you have 40+ GB VRAM."
        ),
    )
    parser.add_argument(
        "--benchmark-dir",
        type=str,
        default="benchmark",
        help="Directory containing benchmark.jsonl and images/ (default: benchmark/).",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="results",
        help="Directory to write results into (default: results/).",
    )
    parser.add_argument(
        "--model-id",
        type=str,
        default=None,
        help=(
            "Override default HuggingFace model ID. "
            "Default for vlm: Qwen/Qwen2.5-VL-3B-Instruct. "
            "Default for base: Qwen/Qwen2.5-1.5B-Instruct."
        ),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    # ── Handle --model lora immediately (no CUDA check needed) ───────────────
    if args.model == "lora":
        print(_LORA_MESSAGE)
        sys.exit(0)

    # ── Import benchmark modules ──────────────────────────────────────────────
    from verifylens_kaggle.hardware import detect_hardware, reset_peak_memory_stats, get_memory_stats
    from verifylens_kaggle.dataset import load_benchmark, count_benchmark, validate_benchmark_images
    from verifylens_kaggle.checkpoint import CheckpointManager
    from verifylens_kaggle.reporter import save_results

    # ── Hardware detection (fails if no CUDA) ────────────────────────────────
    hw_info = detect_hardware(require_cuda=True)

    # ── Validate benchmark dataset ────────────────────────────────────────────
    benchmark_dir = Path(args.benchmark_dir)
    jsonl_path = benchmark_dir / "benchmark.jsonl"

    if not jsonl_path.exists():
        print(f"\nERROR: Benchmark file not found: {jsonl_path}")
        print(f"Expected structure:")
        print(f"  {args.benchmark_dir}/")
        print(f"    benchmark.jsonl")
        print(f"    images/")
        sys.exit(1)

    print("\nValidating benchmark dataset...")
    validation = validate_benchmark_images(str(jsonl_path))
    counts = count_benchmark(str(jsonl_path))
    print(f"  Total samples : {counts['total']}")
    for dt in ["aadhaar", "pan", "passport"]:
        print(f"  {dt:10s}    : {counts.get(dt, 0)}")

    if not validation["ok"]:
        print(f"\nERROR: {len(validation['missing'])} benchmark images are missing:")
        for p in validation["missing"][:5]:
            print(f"  {p}")
        if len(validation["missing"]) > 5:
            print(f"  ... and {len(validation['missing']) - 5} more")
        sys.exit(1)
    print(f"  Image validation: OK ({validation['total']} images found)")

    # ── Checkpoint / resume ───────────────────────────────────────────────────
    predictions_path = str(Path(args.output_dir) / "predictions.jsonl")
    ckpt = CheckpointManager(predictions_path)
    completed_ids = ckpt.load_completed_ids()

    if completed_ids:
        print(f"\nResume detected: {len(completed_ids)} samples already completed.")
        print(f"  Skipping completed samples and continuing from where we left off.")

    # ── Model loading (ONCE) ──────────────────────────────────────────────────
    print(f"\n{'=' * 60}")
    print(f"  MODEL LOADING: {args.model.upper()}")
    print(f"{'=' * 60}")

    model_load_start = time.perf_counter()
    model_info: Dict[str, Any] = {}
    extractor = None

    if args.model == "vlm":
        from verifylens_kaggle.vlm_extractor import VLMExtractor
        kwargs: Dict[str, Any] = {}
        if args.model_id:
            kwargs["model_id"] = args.model_id
        extractor = VLMExtractor(**kwargs)
        model_info = extractor.model_info

    elif args.model == "base":
        from verifylens_kaggle.llm_extractor import LLMExtractor
        kwargs = {}
        if args.model_id:
            kwargs["model_id"] = args.model_id
        extractor = LLMExtractor(**kwargs)
        model_info = extractor.model_info

    elif args.model == "hybrid":
        from verifylens_kaggle.hybrid_extractor import HybridExtractor
        extractor = HybridExtractor()
        model_info = extractor.model_info

    model_load_time_s = time.perf_counter() - model_load_start
    print(f"\nModel loaded in {model_load_time_s:.1f}s")

    # ── Reset CUDA memory peak stats before benchmark run ────────────────────
    reset_peak_memory_stats()

    # ── Warmup (single dummy image, no grading) ───────────────────────────────
    print("\nWarming up model...")
    try:
        from PIL import Image as PILImage
        dummy = PILImage.new("RGB", (300, 200), color=(240, 240, 240))
        extractor.extract(dummy)
        print("  Warmup complete.")
    except Exception as e:
        print(f"  Warmup skipped (non-fatal): {e}")

    # ── Benchmark loop ────────────────────────────────────────────────────────
    print(f"\n{'=' * 60}")
    print(f"  BENCHMARK: {args.limit} samples (mode={args.model})")
    if completed_ids:
        print(f"  Resuming: {len(completed_ids)} already done, continuing...")
    print(f"{'=' * 60}\n")

    from PIL import Image as PILImage

    processed = len(completed_ids)
    target = min(args.limit, counts["total"])
    errors_this_run = 0

    dataset_iter = load_benchmark(
        str(jsonl_path),
        limit=target,
        skip_ids=completed_ids,
    )

    for sample in dataset_iter:
        sample_id = sample["id"]
        doc_type = sample["document_type"]
        gt = sample["ground_truth"]
        img_path = sample["image_abs_path"]

        processed += 1
        total_remaining = target - processed + 1
        print(f"[{processed:3d}/{target}] {sample_id} ({doc_type})", end="  ", flush=True)

        # ── Load image ──────────────────────────────────────────────────────
        try:
            img = PILImage.open(img_path).convert("RGB")
        except Exception as e:
            err_msg = f"Image load failed: {e}"
            print(f"ERROR: {err_msg}")
            ckpt.append(
                sample_id=sample_id,
                document_type=doc_type,
                ground_truth=gt,
                prediction=None,
                json_valid=False,
                latency_ms=None,
                error=err_msg,
            )
            errors_this_run += 1
            continue

        # ── Inference ───────────────────────────────────────────────────────
        try:
            result = extractor.extract(img)
        except RuntimeError as e:
            # Check for CUDA OOM
            if "out of memory" in str(e).lower() or "CUDA out of memory" in str(e):
                ckpt.handle_oom(
                    sample_id=sample_id,
                    gpu_info=hw_info,
                    model_config=model_info,
                    batch_size=args.batch_size,
                )
                # handle_oom calls sys.exit(2)
            else:
                err_msg = f"RuntimeError: {e}"
                print(f"ERROR: {err_msg}")
                ckpt.append(
                    sample_id=sample_id,
                    document_type=doc_type,
                    ground_truth=gt,
                    prediction=None,
                    json_valid=False,
                    latency_ms=None,
                    error=err_msg,
                )
                errors_this_run += 1
                continue
        except Exception as e:
            err_msg = f"{type(e).__name__}: {e}"
            print(f"ERROR: {err_msg}")
            ckpt.append(
                sample_id=sample_id,
                document_type=doc_type,
                ground_truth=gt,
                prediction=None,
                json_valid=False,
                latency_ms=None,
                error=err_msg,
            )
            errors_this_run += 1
            continue

        # ── Checkpoint ──────────────────────────────────────────────────────
        ckpt.append(
            sample_id=sample_id,
            document_type=doc_type,
            ground_truth=gt,
            prediction=result.get("prediction"),
            json_valid=result.get("json_valid", False),
            latency_ms=result.get("latency_ms"),
            ocr_latency_ms=result.get("ocr_latency_ms"),
            llm_latency_ms=result.get("llm_latency_ms"),
            error=result.get("error"),
        )

        # ── Progress print ───────────────────────────────────────────────────
        lat_str = f"{result.get('latency_ms', '?')} ms"
        jv_str = "✓" if result.get("json_valid") else "✗"
        err_str = f" ERR:{result['error'][:40]}" if result.get("error") else ""
        print(f"json={jv_str}  lat={lat_str}{err_str}")

        # ── Release per-sample image memory ─────────────────────────────────
        del img

    # ── Collect final memory stats BEFORE unloading model ────────────────────
    mem_stats = get_memory_stats()

    # ── Final reporting ───────────────────────────────────────────────────────
    print(f"\n{'=' * 60}")
    print(f"  BENCHMARK COMPLETE")
    print(f"{'=' * 60}")
    print(f"  Processed this run : {processed - len(completed_ids)}")
    print(f"  Total completed    : {processed}")
    print(f"  Errors this run    : {errors_this_run}")
    print(f"  Peak GPU memory    : {mem_stats.get('peak_allocated_gb', 'N/A')} GB allocated")

    # Load all predictions (including previously completed ones)
    all_predictions = ckpt.load_all_predictions()

    config = {
        "model": args.model,
        "model_id": model_info.get("model_id", "unknown"),
        "limit": args.limit,
        "batch_size": args.batch_size,
        "benchmark_dir": args.benchmark_dir,
        "output_dir": args.output_dir,
        "total_completed": len(all_predictions),
    }

    save_results(
        output_dir=args.output_dir,
        model_mode=args.model,
        model_info=model_info,
        hardware_info=hw_info,
        memory_stats=mem_stats,
        predictions=all_predictions,
        model_load_time_s=model_load_time_s,
        config=config,
    )

    print(f"\nDone. Check {args.output_dir}/benchmark_report.md for the full report.")


if __name__ == "__main__":
    main()
