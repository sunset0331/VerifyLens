"""
src/training/train.py
----------------------
LoRA fine-tuning of Qwen2.5-1.5B-Instruct using MLX-LM.

This is a wrapper around `mlx_lm.lora` that:
  1. Downloads the quantized base model from mlx-community
  2. Runs LoRA fine-tuning on our synthetic ID document QA dataset
  3. Saves the adapter weights to checkpoints/

Why MLX-LM?
- Designed for Apple Silicon (M1/M2/M3/M4) with unified memory
- 4-bit quantized model + LoRA fits in 8GB RAM
- Native Metal GPU acceleration, no CUDA needed

Usage:
    python -m src.training.train [--iters 600] [--model <hf-id>]

Or equivalently via mlx-lm CLI directly:
    .venv/bin/mlx_lm.lora \\
        --model mlx-community/Qwen2.5-1.5B-Instruct-4bit \\
        --train --data data/mlx_train \\
        --iters 600 --batch-size 4 \\
        --num-layers 8 \\
        --adapter-path checkpoints/verifylens-adapter
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


# ── Defaults ─────────────────────────────────────────────────────────────────

DEFAULT_MODEL = "mlx-community/Qwen2.5-1.5B-Instruct-4bit"
DEFAULT_DATA_DIR = "data/mlx_train"
DEFAULT_ADAPTER_PATH = "checkpoints/verifylens-adapter"
DEFAULT_ITERS = 600      # ~10-15 min on M-series with 8GB
DEFAULT_BATCH_SIZE = 4
DEFAULT_LORA_LAYERS = 8  # fine-tune last 8 transformer layers


def run_training(
    model: str = DEFAULT_MODEL,
    data_dir: str = DEFAULT_DATA_DIR,
    adapter_path: str = DEFAULT_ADAPTER_PATH,
    iters: int = DEFAULT_ITERS,
    batch_size: int = DEFAULT_BATCH_SIZE,
    lora_layers: int = DEFAULT_LORA_LAYERS,
    learning_rate: float = 1e-4,
    val_batches: int = 25,
    save_every: int = 100,
    grad_checkpoint: bool = True,
):
    """
    Launch MLX-LM LoRA fine-tuning as a subprocess.

    Using subprocess keeps the training log clean and allows real-time
    output streaming, which is useful for monitoring loss curves.
    """
    # Ensure checkpoint dir exists
    Path(adapter_path).parent.mkdir(parents=True, exist_ok=True)
    Path(data_dir).mkdir(parents=True, exist_ok=True)

    cmd = [
        sys.executable, "-m", "mlx_lm", "lora",
        "--model", model,
        "--train",
        "--data", data_dir,
        "--iters", str(iters),
        "--batch-size", str(batch_size),
        "--num-layers", str(lora_layers),
        "--learning-rate", str(learning_rate),
        "--val-batches", str(val_batches),
        "--adapter-path", adapter_path,
        "--save-every", str(save_every),
    ]

    if grad_checkpoint:
        cmd.append("--grad-checkpoint")

    print("=" * 60)
    print("VerifyLens — LoRA Fine-tuning")
    print("=" * 60)
    print(f"Base model   : {model}")
    print(f"Data dir     : {data_dir}")
    print(f"Adapter path : {adapter_path}")
    print(f"Iterations   : {iters}")
    print(f"Batch size   : {batch_size}")
    print(f"LoRA layers  : {lora_layers}")
    print(f"LR           : {learning_rate}")
    print("=" * 60)
    print("Starting training...\n")

    result = subprocess.run(cmd, check=False)

    if result.returncode == 0:
        print("\n✅ Training complete!")
        print(f"   Adapter saved to: {adapter_path}/")
        print("\nNext steps:")
        print("  1. Evaluate: python -m src.training.evaluate")
        print("  2. Fuse:     python -m src.training.train --fuse")
        print("  3. Run API:  uvicorn src.api.server:app --reload")
    else:
        print(f"\n❌ Training failed with exit code {result.returncode}")
        sys.exit(result.returncode)


def fuse_adapter(
    model: str = DEFAULT_MODEL,
    adapter_path: str = DEFAULT_ADAPTER_PATH,
    output_path: str = "checkpoints/verifylens-fused",
):
    """
    Merge LoRA adapter weights into the base model for faster inference.
    Run this after training is complete.
    """
    cmd = [
        sys.executable, "-m", "mlx_lm.fuse",
        "--model", model,
        "--adapter-path", adapter_path,
        "--save-path", output_path,
        "--de-quantize",  # saves in float16 for inference
    ]

    print(f"Fusing adapter into base model → {output_path}")
    subprocess.run(cmd, check=True)
    print(f"✅ Fused model saved to: {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="VerifyLens LoRA fine-tuning via MLX-LM")
    parser.add_argument("--model", default=DEFAULT_MODEL,
                        help="HuggingFace model ID (must be an mlx-community quantized model)")
    parser.add_argument("--data", default=DEFAULT_DATA_DIR)
    parser.add_argument("--adapter-path", default=DEFAULT_ADAPTER_PATH)
    parser.add_argument("--iters", type=int, default=DEFAULT_ITERS)
    parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE)
    parser.add_argument("--lora-layers", type=int, default=DEFAULT_LORA_LAYERS)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--fuse", action="store_true",
                        help="Fuse adapter into base model (run after training)")
    args = parser.parse_args()

    if args.fuse:
        fuse_adapter(model=args.model, adapter_path=args.adapter_path)
    else:
        run_training(
            model=args.model,
            data_dir=args.data,
            adapter_path=args.adapter_path,
            iters=args.iters,
            batch_size=args.batch_size,
            lora_layers=args.lora_layers,
            learning_rate=args.lr,
        )
