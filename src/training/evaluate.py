"""
src/training/evaluate.py
-------------------------
Post-training evaluation of the fine-tuned VerifyLens adapter.

Metrics computed:
  - Exact Match (EM): JSON key-value pairs match perfectly
  - Field-level Accuracy: per-field (name, dob, doc_number, doc_type, gender)
  - Valid JSON Rate: % of outputs that parse as valid JSON
  - Avg inference time per sample
  - Per-document-type exact match (where doc type is determinable)
  - Error breakdown: most-missed fields

Compares base model vs fine-tuned adapter side by side.

Usage:
    # Evaluate fine-tuned model only (default):
    python -m src.training.evaluate

    # Evaluate fine-tuned AND base model (saves both, prints comparison):
    python -m src.training.evaluate --compare-base

    # Evaluate base model only (no adapter):
    python -m src.training.evaluate --adapter-path ""

    # Save per-sample predictions for inspection:
    python -m src.training.evaluate --compare-base --save-predictions
"""

from __future__ import annotations

import argparse
import json
import re
import time
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from mlx_lm import load, generate
from mlx_lm.sample_utils import make_sampler


# ── Constants ─────────────────────────────────────────────────────────────────

# Keyword markers used to infer document type from the OCR text in the user
# message.  These are matched case-sensitively against the noisy OCR string.
# Matching is done with a short, distinctive substring so that mild OCR noise
# (extra spaces, single-char drops) still matches.  When a sample's OCR text
# matches none of these (heavy noise), the sample's doc type is left as
# "unknown" and it is excluded from per-doc-type tallies.
_DOC_MARKERS: Dict[str, List[str]] = {
    "Aadhaar Card":    ["AADHAAR", "UIDAI", "Unique Identification"],
    "PAN Card":        ["PERMANENT ACCOUNT", "INCOME TAX"],
    "Passport":        ["PASSPORT", "REPUBLIC OF INDIA | PASS"],
    "Driving License": ["DRIVING LICEN", "MOTOR VEHICLES ACT"],
}


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
    """True if all gold key-value pairs appear in prediction.

    Definition is unchanged from Phase 1: gold-anchored, case-insensitive,
    whitespace-stripped. pred=None always returns False.
    """
    if pred is None:
        return False
    return all(
        str(pred.get(k, "")).strip().lower() == str(v).strip().lower()
        for k, v in gold.items()
    )


def field_accuracy(preds: List[Optional[Dict]], golds: List[Dict]) -> Dict[str, float]:
    """Per-field accuracy across the validation set."""
    all_fields: set = set()
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


# ── Document-type inference ───────────────────────────────────────────────────

def infer_doc_type(sample: Dict) -> Optional[str]:
    """Return the document type label for a sample, or None if not determinable.

    Strategy (in priority order):
      1. If the gold answer already contains a doc_type key, use that.
      2. Search the user's OCR text for known distinctive keywords.
      3. If neither works, return None (sample excluded from per-type tallies).
    This avoids inventing labels through unreliable heuristics.
    """
    gold_raw = sample["messages"][-1]["content"]
    gold = parse_json_output(gold_raw) or {}
    if "doc_type" in gold and gold["doc_type"]:
        return str(gold["doc_type"])

    user_text = sample["messages"][1]["content"]
    for label, markers in _DOC_MARKERS.items():
        for marker in markers:
            if marker in user_text:
                return label
    return None


# ── Inference ─────────────────────────────────────────────────────────────────

def run_inference(
    model,
    tokenizer,
    messages: List[Dict],
    max_tokens: int = 128,
) -> str:
    """Run deterministic (greedy) chat inference using mlx_lm.generate.

    sampler=make_sampler(temp=0) passes a greedy argmax callable into
    generate_step via **kwargs through stream_generate.  This is the correct
    kwarg for the installed mlx_lm version (generate_step accepts 'sampler',
    not 'temp' directly).
    """
    prompt = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    return generate(
        model,
        tokenizer,
        prompt=prompt,
        max_tokens=max_tokens,
        verbose=False,
        sampler=make_sampler(temp=0),  # greedy decoding: deterministic, reproducible
    )


# ── Main evaluation loop ─────────────────────────────────────────────────────

def evaluate(
    model_path: str,
    data_path: Path,
    adapter_path: Optional[str],
    num_samples: int,
) -> Tuple[Dict[str, Any], List[Dict]]:
    """
    Load model (with or without adapter) and evaluate on validation set.

    Returns
    -------
    metrics : dict   — aggregate metrics (unchanged structure from Phase 1)
    sample_records : list — per-sample detail records for diagnostics
    """
    print(f"\nLoading model: {model_path}")
    if adapter_path and Path(adapter_path).exists():
        print(f"Adapter: {adapter_path}")
        model, tokenizer = load(model_path, adapter_path=adapter_path)
    else:
        if adapter_path and not Path(adapter_path).exists():
            print(f"[WARNING] Adapter path not found: '{adapter_path}' — evaluating base model")
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

    print(f"\nEvaluating on {len(samples)} samples... (deterministic, temp=0)")

    preds: List[Optional[Dict]] = []
    golds: List[Dict] = []
    latencies: List[float] = []
    raw_outputs: List[str] = []
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
        raw_outputs.append(raw_output)

        if (i + 1) % 20 == 0:
            print(f"  {i + 1}/{len(samples)}")

    # ── Aggregate metrics (structure unchanged from Phase 1) ─────────────────
    em_score = sum(exact_match(p, g) for p, g in zip(preds, golds)) / len(samples)
    valid_json_rate = valid_json_count / len(samples)
    field_acc = field_accuracy(preds, golds)
    avg_latency = sum(latencies) / len(latencies)

    metrics: Dict[str, Any] = {
        "exact_match": round(em_score, 4),
        "valid_json_rate": round(valid_json_rate, 4),
        "field_accuracy": {k: round(v, 4) for k, v in field_acc.items()},
        "avg_latency_s": round(avg_latency, 3),
        "num_samples": len(samples),
    }

    # ── Per-document-type exact match ─────────────────────────────────────────
    per_type_correct: Dict[str, int] = {}
    per_type_total: Dict[str, int] = {}
    unknown_count = 0

    for sample, pred, gold in zip(samples, preds, golds):
        doc_type = infer_doc_type(sample)
        if doc_type is None:
            unknown_count += 1
            continue
        per_type_total[doc_type] = per_type_total.get(doc_type, 0) + 1
        if exact_match(pred, gold):
            per_type_correct[doc_type] = per_type_correct.get(doc_type, 0) + 1

    if per_type_total:
        metrics["per_doc_type_exact_match"] = {
            dt: {
                "exact_match": round(per_type_correct.get(dt, 0) / per_type_total[dt], 4),
                "samples": per_type_total[dt],
            }
            for dt in sorted(per_type_total)
        }
        metrics["per_doc_type_note"] = (
            f"{unknown_count} sample(s) excluded: doc type not determinable from "
            "gold or OCR text (heavy OCR noise corrupted type keywords)."
            if unknown_count > 0
            else "All samples assigned a document type."
        )
    else:
        metrics["per_doc_type_exact_match"] = None
        metrics["per_doc_type_note"] = (
            "Per-document-type evaluation unavailable from current validation data."
        )

    # ── Error breakdown ───────────────────────────────────────────────────────
    incorrect_field_counts: Counter = Counter()
    exact_match_count = 0
    non_exact_count = 0

    for pred, gold in zip(preds, golds):
        if exact_match(pred, gold):
            exact_match_count += 1
        else:
            non_exact_count += 1
            for key, gold_val in gold.items():
                pred_val = str(pred.get(key, "") if pred else "").strip().lower()
                if pred_val != str(gold_val).strip().lower():
                    incorrect_field_counts[key] += 1

    metrics["error_breakdown"] = {
        "total_samples": len(samples),
        "exact_matches": exact_match_count,
        "non_exact_matches": non_exact_count,
        "incorrect_field_counts": dict(incorrect_field_counts.most_common()),
    }

    # ── Per-sample records for --save-predictions ─────────────────────────────
    sample_records: List[Dict] = []
    for i, (sample, pred, gold, raw) in enumerate(zip(samples, preds, golds, raw_outputs)):
        field_correctness = {
            key: (str(pred.get(key, "") if pred else "").strip().lower()
                  == str(val).strip().lower())
            for key, val in gold.items()
        }
        sample_records.append({
            "index": i,
            "gold": gold,
            "prediction": pred,
            "raw_output": raw,
            "exact_match": exact_match(pred, gold),
            "field_correctness": field_correctness,
        })

    return metrics, sample_records


# ── Printing ──────────────────────────────────────────────────────────────────

def print_report(label: str, metrics: Dict):
    print(f"\n{'='*60}")
    print(f"  {label}")
    print(f"{'='*60}")
    print(f"  Exact Match      : {metrics['exact_match']:.1%}")
    print(f"  Valid JSON Rate  : {metrics['valid_json_rate']:.1%}")
    print(f"  Avg Latency      : {metrics['avg_latency_s']:.2f}s / sample")
    print(f"\n  Field-level Accuracy:")
    for field, acc in sorted(metrics["field_accuracy"].items()):
        bar = "█" * int(acc * 20) + "░" * (20 - int(acc * 20))
        print(f"    {field:<15} {bar} {acc:.1%}")

    # Per-doc-type
    if metrics.get("per_doc_type_exact_match"):
        print(f"\n  Per-Document-Type Exact Match:")
        for dt, info in metrics["per_doc_type_exact_match"].items():
            bar = "█" * int(info["exact_match"] * 20) + "░" * (20 - int(info["exact_match"] * 20))
            print(f"    {dt:<20} {bar} {info['exact_match']:.1%}  (n={info['samples']})")
        if metrics.get("per_doc_type_note"):
            print(f"    Note: {metrics['per_doc_type_note']}")
    else:
        print(f"\n  Per-Document-Type: {metrics.get('per_doc_type_note', 'unavailable')}")

    # Error breakdown
    eb = metrics.get("error_breakdown", {})
    if eb:
        print(f"\n  Error Breakdown:")
        print(f"    Total samples       : {eb['total_samples']}")
        print(f"    Exact matches       : {eb['exact_matches']}")
        print(f"    Non-exact matches   : {eb['non_exact_matches']}")
        if eb["incorrect_field_counts"]:
            print(f"    Incorrect field counts (most frequent first):")
            for field, cnt in eb["incorrect_field_counts"].items():
                print(f"      {field:<15} : {cnt}")

    print(f"{'='*60}")


def print_field_comparison(base_metrics: Dict, ft_metrics: Dict):
    """Print a side-by-side field accuracy table for --compare-base."""
    all_fields = sorted(
        set(base_metrics["field_accuracy"]) | set(ft_metrics["field_accuracy"])
    )
    print(f"\n{'='*60}")
    print("  Field-level Accuracy Comparison")
    print(f"{'='*60}")
    print(f"  {'FIELD':<18} {'BASE':>10} {'FINE-TUNED':>12}")
    print(f"  {'-'*18} {'-'*10} {'-'*12}")
    for field in all_fields:
        base_acc = base_metrics["field_accuracy"].get(field)
        ft_acc = ft_metrics["field_accuracy"].get(field)
        base_str = f"{base_acc:.1%}" if base_acc is not None else "  —"
        ft_str = f"{ft_acc:.1%}" if ft_acc is not None else "  —"
        print(f"  {field:<18} {base_str:>10} {ft_str:>12}")
    print(f"{'='*60}")


def print_comparison_summary(base_metrics: Dict, ft_metrics: Dict):
    """Print the improvement summary using correct percentage-point language."""
    base_em = base_metrics["exact_match"]
    ft_em = ft_metrics["exact_match"]
    delta_pp = (ft_em - base_em) * 100  # percentage points
    base_json = base_metrics["valid_json_rate"]
    ft_json = ft_metrics["valid_json_rate"]

    print(f"\n{'='*60}")
    print("  Improvement Summary")
    print(f"{'='*60}")
    print(f"  Exact Match:      {base_em:.1%} → {ft_em:.1%}")
    print(f"  Improvement:      +{delta_pp:.1f} percentage points")
    print(f"  Valid JSON Rate:  {base_json:.1%} → {ft_json:.1%}")
    print(f"{'='*60}")


# ── Saving ────────────────────────────────────────────────────────────────────

def save_results(
    results_path: Path,
    model: str,
    data: str,
    num_samples: int,
    adapter_path: Optional[str],
    ft_metrics: Optional[Dict] = None,
    base_metrics: Optional[Dict] = None,
):
    """
    Save evaluation results with full metadata to JSON.

    Preserves the existing base_model / fine_tuned / improvement structure.
    Adds evaluation_config and extended diagnostics.
    """
    output: Dict[str, Any] = {
        "evaluation_config": {
            "model": model,
            "adapter_path": adapter_path if adapter_path else None,
            "data": data,
            "num_samples": num_samples,
            "deterministic": True,
            "temperature": 0,
            "timestamp": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        }
    }

    if base_metrics is not None:
        output["base_model"] = base_metrics

    if ft_metrics is not None:
        output["fine_tuned"] = ft_metrics

    if base_metrics is not None and ft_metrics is not None:
        base_em = base_metrics["exact_match"]
        ft_em = ft_metrics["exact_match"]
        output["improvement"] = {
            "exact_match": round(ft_em - base_em, 4),
            "exact_match_percentage_points": round((ft_em - base_em) * 100, 1),
            "valid_json_rate": round(
                ft_metrics["valid_json_rate"] - base_metrics["valid_json_rate"], 4
            ),
        }

    results_path.parent.mkdir(exist_ok=True)
    with open(results_path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\n📄 Results saved to {results_path}")


def save_predictions(
    predictions_path: Path,
    label: str,
    records: List[Dict],
):
    """Save per-sample prediction records to a JSON file."""
    output = {
        "label": label,
        "num_samples": len(records),
        "predictions": records,
    }
    predictions_path.parent.mkdir(exist_ok=True)
    with open(predictions_path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"📄 Predictions saved to {predictions_path}")


# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Evaluate VerifyLens fine-tuned and/or base model."
    )
    parser.add_argument("--model", default="mlx-community/Qwen2.5-1.5B-Instruct-4bit")
    parser.add_argument("--data", type=Path, default=Path("data/mlx_train/valid.jsonl"))
    parser.add_argument(
        "--adapter-path",
        default="checkpoints/verifylens-adapter",
        help="Path to LoRA adapter. Pass '' to evaluate base model only.",
    )
    parser.add_argument(
        "--num-samples",
        type=int,
        default=50,
        help="Number of validation samples to evaluate (default: 50).",
    )
    parser.add_argument(
        "--compare-base",
        action="store_true",
        help="Also evaluate base model (no adapter). Both results saved and compared.",
    )
    parser.add_argument(
        "--save-predictions",
        action="store_true",
        help="Save per-sample gold/prediction/correctness records to checkpoints/eval_predictions.json.",
    )
    args = parser.parse_args()

    results_path = Path("checkpoints/eval_results.json")
    predictions_path = Path("checkpoints/eval_predictions.json")
    adapter = args.adapter_path if args.adapter_path else None

    if args.compare_base:
        # ── Evaluate fine-tuned model ──────────────────────────────────────
        print("\n🔍 Evaluating FINE-TUNED model...")
        ft_metrics, ft_records = evaluate(args.model, args.data, adapter, args.num_samples)
        print_report("Fine-tuned (LoRA adapter)", ft_metrics)

        # ── Evaluate base model ────────────────────────────────────────────
        print("\n🔍 Evaluating BASE model (no adapter)...")
        base_metrics, base_records = evaluate(args.model, args.data, None, args.num_samples)
        print_report("Base model (no fine-tuning)", base_metrics)

        # ── Field comparison table ─────────────────────────────────────────
        print_field_comparison(base_metrics, ft_metrics)

        # ── Summary ───────────────────────────────────────────────────────
        print_comparison_summary(base_metrics, ft_metrics)

        # ── Save results ──────────────────────────────────────────────────
        save_results(
            results_path,
            model=args.model,
            data=str(args.data),
            num_samples=args.num_samples,
            adapter_path=str(adapter) if adapter else None,
            ft_metrics=ft_metrics,
            base_metrics=base_metrics,
        )

        if args.save_predictions:
            save_predictions(predictions_path, "base_model", base_records)
            ft_pred_path = predictions_path.with_stem("eval_predictions_finetuned")
            save_predictions(ft_pred_path, "fine_tuned", ft_records)

    elif adapter:
        # ── Fine-tuned only ────────────────────────────────────────────────
        print("\n🔍 Evaluating FINE-TUNED model...")
        ft_metrics, ft_records = evaluate(args.model, args.data, adapter, args.num_samples)
        print_report("Fine-tuned (LoRA adapter)", ft_metrics)
        save_results(
            results_path,
            model=args.model,
            data=str(args.data),
            num_samples=args.num_samples,
            adapter_path=str(adapter),
            ft_metrics=ft_metrics,
        )
        if args.save_predictions:
            ft_pred_path = predictions_path.with_stem("eval_predictions_finetuned")
            save_predictions(ft_pred_path, "fine_tuned", ft_records)

    else:
        # ── Base model only ────────────────────────────────────────────────
        print("\n🔍 Evaluating BASE model (no adapter)...")
        base_metrics, base_records = evaluate(args.model, args.data, None, args.num_samples)
        print_report("Base model (no fine-tuning)", base_metrics)
        save_results(
            results_path,
            model=args.model,
            data=str(args.data),
            num_samples=args.num_samples,
            adapter_path=None,
            base_metrics=base_metrics,
        )
        if args.save_predictions:
            save_predictions(predictions_path, "base_model", base_records)
