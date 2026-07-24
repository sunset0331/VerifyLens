"""
src/training/evaluate.py
-------------------------
Post-training evaluation of the fine-tuned VerifyLens adapter.

Metrics computed:
  - Exact Match (EM): JSON key-value pairs match perfectly
  - Field-level Accuracy: per-field (name, dob, doc_number, doc_type)
  - Valid JSON Rate: % of outputs that parse as valid JSON
  - Avg inference time per sample

Compares base model vs fine-tuned adapter side by side.

Usage:
    python -m src.training.evaluate \\
        --data data/mlx_train/valid.jsonl \\
        --adapter-path checkpoints/verifylens-adapter \\
        --num-samples 100
"""

from __future__ import annotations

import argparse
import json
import re
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from mlx_lm import load, generate


# ── JSON parsing ─────────────────────────────────────────────────────────────

def parse_json_output(raw: str) -> Optional[Dict]:
    """Try to extract JSON from model output."""
    raw = raw.strip()
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        pass
    # Find first {...} block
    match = re.search(r"\{.*?\}", raw, re.DOTALL)
    if match:
        try:
            return json.loads(match.group())
        except json.JSONDecodeError:
            pass
    return None


def exact_match(pred: Optional[Dict], gold: Dict) -> bool:
    """True if all gold key-value pairs appear in prediction."""
    if pred is None:
        return False
    return all(
        str(pred.get(k, "")).strip().lower() == str(v).strip().lower()
        for k, v in gold.items()
    )


def field_accuracy(preds: List[Optional[Dict]], golds: List[Dict]) -> Dict[str, float]:
    """Per-field accuracy across the validation set."""
    all_fields = set()
    for g in golds:
        all_fields.update(g.keys())

    results = {}
    for field in all_fields:
        correct = 0
        total = 0
        for pred, gold in zip(preds, golds):
            if field in gold:
                total += 1
                pred_val = str(pred.get(field, "") if pred else "").strip().lower()
                gold_val = str(gold[field]).strip().lower()
                if pred_val == gold_val:
                    correct += 1
        results[field] = correct / total if total > 0 else 0.0

    return results


# ── Inference ─────────────────────────────────────────────────────────────────

def run_inference(
    model,
    tokenizer,
    messages: List[Dict],
    max_tokens: int = 128,
) -> str:
    """Run chat inference using mlx_lm.generate."""
    prompt = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    return generate(model, tokenizer, prompt=prompt, max_tokens=max_tokens, verbose=False)


# ── Main evaluation loop ─────────────────────────────────────────────────────

def evaluate(
    model_path: str,
    data_path: Path,
    adapter_path: Optional[str],
    num_samples: int,
) -> Dict[str, Any]:
    """
    Load model (with or without adapter) and evaluate on validation set.
    Returns a metrics dict.
    """
    print(f"\nLoading model: {model_path}")
    if adapter_path and Path(adapter_path).exists():
        print(f"Adapter: {adapter_path}")
        model, tokenizer = load(model_path, adapter_path=adapter_path)
    else:
        print("No adapter — evaluating base model")
        model, tokenizer = load(model_path)

    # Load validation data
    samples = []
    with open(data_path) as f:
        for line in f:
            samples.append(json.loads(line.strip()))
            if len(samples) >= num_samples:
                break

    print(f"\nEvaluating on {len(samples)} samples...")

    preds = []
    golds = []
    latencies = []
    valid_json_count = 0

    for i, sample in enumerate(samples):
        messages = sample["messages"][:-1]  # system + user only, no assistant
        gold_raw = sample["messages"][-1]["content"]
        gold = parse_json_output(gold_raw) or {}

        t0 = time.time()
        raw_output = run_inference(model, tokenizer, messages)
        latency = time.time() - t0

        pred = parse_json_output(raw_output)
        if pred is not None:
            valid_json_count += 1

        preds.append(pred)
        golds.append(gold)
        latencies.append(latency)

        if (i + 1) % 20 == 0:
            print(f"  {i + 1}/{len(samples)}")

    # Compute metrics
    em_score = sum(exact_match(p, g) for p, g in zip(preds, golds)) / len(samples)
    valid_json_rate = valid_json_count / len(samples)
    field_acc = field_accuracy(preds, golds)
    avg_latency = sum(latencies) / len(latencies)

    return {
        "exact_match": round(em_score, 4),
        "valid_json_rate": round(valid_json_rate, 4),
        "field_accuracy": {k: round(v, 4) for k, v in field_acc.items()},
        "avg_latency_s": round(avg_latency, 3),
        "num_samples": len(samples),
    }


def print_report(label: str, metrics: Dict):
    print(f"\n{'='*55}")
    print(f"  {label}")
    print(f"{'='*55}")
    print(f"  Exact Match      : {metrics['exact_match']:.1%}")
    print(f"  Valid JSON Rate  : {metrics['valid_json_rate']:.1%}")
    print(f"  Avg Latency      : {metrics['avg_latency_s']:.2f}s / sample")
    print(f"\n  Field-level Accuracy:")
    for field, acc in sorted(metrics["field_accuracy"].items()):
        bar = "█" * int(acc * 20) + "░" * (20 - int(acc * 20))
        print(f"    {field:<15} {bar} {acc:.1%}")
    print(f"{'='*55}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="mlx-community/Qwen2.5-1.5B-Instruct-4bit")
    parser.add_argument("--data", type=Path, default=Path("data/mlx_train/valid.jsonl"))
    parser.add_argument("--adapter-path", default="checkpoints/verifylens-adapter",
                        help="Path to LoRA adapter (or leave empty for base model only)")
    parser.add_argument("--num-samples", type=int, default=100)
    parser.add_argument("--compare-base", action="store_true",
                        help="Also evaluate base model (no adapter) for comparison")
    args = parser.parse_args()

    # Evaluate fine-tuned model
    print("\n🔍 Evaluating FINE-TUNED model...")
    ft_metrics = evaluate(args.model, args.data, args.adapter_path, args.num_samples)
    print_report("Fine-tuned (LoRA adapter)", ft_metrics)

    # Optionally compare with base
    if args.compare_base:
        print("\n🔍 Evaluating BASE model (no adapter)...")
        base_metrics = evaluate(args.model, args.data, None, args.num_samples)
        print_report("Base model (no fine-tuning)", base_metrics)

        print(f"\n{'='*55}")
        print("  Improvement Summary")
        print(f"{'='*55}")
        em_delta = ft_metrics["exact_match"] - base_metrics["exact_match"]
        json_delta = ft_metrics["valid_json_rate"] - base_metrics["valid_json_rate"]
        print(f"  Exact Match:     {base_metrics['exact_match']:.1%} → {ft_metrics['exact_match']:.1%}  ({em_delta:+.1%})")
        print(f"  Valid JSON Rate: {base_metrics['valid_json_rate']:.1%} → {ft_metrics['valid_json_rate']:.1%}  ({json_delta:+.1%})")

    # Save results
    results_path = Path("checkpoints/eval_results.json")
    results_path.parent.mkdir(exist_ok=True)
    with open(results_path, "w") as f:
        json.dump({"fine_tuned": ft_metrics}, f, indent=2)
    print(f"\n📄 Results saved to {results_path}")
