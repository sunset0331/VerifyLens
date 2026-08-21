"""
verifylens_kaggle/reporter.py
-------------------------------
Produces all benchmark output files:

  results/
    benchmark_results.json    — Full structured results
    benchmark_results.csv     — Per-metric summary table
    predictions.jsonl         — Per-sample predictions (written incrementally)
    benchmark_report.md       — Human-readable Markdown report

The report explicitly states the production vs. Kaggle model difference.
"""

from __future__ import annotations

import csv
import json
import platform
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

from verifylens_kaggle.metrics import BenchmarkMetrics
from verifylens_kaggle.normalizer import normalize_dict


NON_DISCRIMINATIVE_FIELDS = {"gender", "address"}
"""
These fields are null for 100% of benchmark samples.
They always score 100% correct (null == null) but provide no signal.
The report explicitly flags them.
"""


def _pct(n: int, d: int) -> str:
    if d == 0:
        return "N/A"
    return f"{round(n / d * 100, 2):.2f}%"


def build_metrics_from_predictions(
    predictions: List[Dict[str, Any]],
) -> BenchmarkMetrics:
    """
    Re-build BenchmarkMetrics from a list of prediction records
    (as loaded from predictions.jsonl).
    """
    metrics = BenchmarkMetrics()

    for rec in predictions:
        if rec.get("error") and rec.get("prediction") is None:
            # Runtime error — record as invalid JSON with null predictions
            doc_type = rec.get("document_type", "unknown")
            gt = rec.get("ground_truth", {})
            gt_norm = normalize_dict(gt)
            pred_norm = normalize_dict({k: None for k in gt})
            metrics.record_sample(
                doc_type=doc_type,
                ground_truth=gt_norm,
                predicted=pred_norm,
                json_valid=False,
                latency_ms=rec.get("latency_ms"),
                parse_error=rec.get("error"),
            )
        else:
            doc_type = rec.get("document_type", "unknown")
            gt = rec.get("ground_truth", {})
            pred = rec.get("prediction") or {}
            gt_norm = normalize_dict(gt)
            pred_norm = normalize_dict(pred)
            metrics.record_sample(
                doc_type=doc_type,
                ground_truth=gt_norm,
                predicted=pred_norm,
                json_valid=rec.get("json_valid", False),
                latency_ms=rec.get("latency_ms"),
                parse_error=rec.get("parse_error"),
            )

    return metrics


def save_results(
    output_dir: str,
    model_mode: str,
    model_info: Dict[str, Any],
    hardware_info: Dict[str, Any],
    memory_stats: Dict[str, Any],
    predictions: List[Dict[str, Any]],
    model_load_time_s: float,
    config: Dict[str, Any],
) -> None:
    """
    Save all result files to output_dir.

    Parameters
    ----------
    output_dir    : Path to write results into
    model_mode    : "base", "vlm", etc.
    model_info    : dict from extractor.model_info
    hardware_info : dict from hardware.detect_hardware()
    memory_stats  : dict from hardware.get_memory_stats()
    predictions   : list of prediction records from checkpoint
    model_load_time_s : float
    config        : benchmark config dict
    """
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    metrics = build_metrics_from_predictions(predictions)
    summary = metrics.compute_summary()
    summary["model_load_time_s"] = round(model_load_time_s, 1)

    # ── 1. benchmark_results.json ────────────────────────────────────────────
    full_results = {
        "timestamp": datetime.utcnow().isoformat() + "Z",
        "model_mode": model_mode,
        "model_info": model_info,
        "hardware": hardware_info,
        "memory_stats": memory_stats,
        "config": config,
        "summary": summary,
        "errors": metrics.errors,
    }
    with open(out / "benchmark_results.json", "w") as f:
        json.dump(full_results, f, indent=2)

    # ── 2. benchmark_results.csv ─────────────────────────────────────────────
    _save_csv(out / "benchmark_results.csv", summary, model_mode)

    # ── 3. benchmark_report.md ───────────────────────────────────────────────
    _save_markdown(
        out / "benchmark_report.md",
        model_mode=model_mode,
        model_info=model_info,
        hardware_info=hardware_info,
        memory_stats=memory_stats,
        summary=summary,
        errors=metrics.errors,
        predictions=predictions,
        config=config,
    )

    print(f"\n[Reporter] Results saved to: {out}")
    print(f"  benchmark_results.json")
    print(f"  benchmark_results.csv")
    print(f"  benchmark_report.md")
    print(f"  predictions.jsonl  (written incrementally)")


def _save_csv(path: Path, summary: Dict[str, Any], model_mode: str) -> None:
    rows = [
        ["metric", "value"],
        ["model_mode", model_mode],
        ["total_samples", summary.get("total_samples", 0)],
        ["json_valid_rate", summary.get("json_valid_rate", 0)],
        ["exact_match_rate", summary.get("exact_match_rate", 0)],
        ["exact_match_no_doctype_rate", summary.get("exact_match_no_doctype_rate", 0)],
        ["core_identity_exact_match_rate", summary.get("core_identity_exact_match_rate", 0)],
    ]

    for field, acc in summary.get("field_accuracies", {}).items():
        rows.append([f"field_accuracy_{field}", acc])

    lat = summary.get("latency_ms", {})
    for stat in ["mean", "median", "p95", "min", "max"]:
        rows.append([f"latency_ms_{stat}", lat.get(stat, "")])

    rows.append(["model_load_time_s", summary.get("model_load_time_s", "")])

    with open(path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerows(rows)


def _save_markdown(
    path: Path,
    model_mode: str,
    model_info: Dict[str, Any],
    hardware_info: Dict[str, Any],
    memory_stats: Dict[str, Any],
    summary: Dict[str, Any],
    errors: List[Dict[str, Any]],
    predictions: List[Dict[str, Any]],
    config: Dict[str, Any],
) -> None:

    total = summary.get("total_samples", 0)
    lat = summary.get("latency_ms", {})
    fa = summary.get("field_accuracies", {})
    dt = summary.get("doc_types", {})

    # Error categorization
    error_types: Dict[str, int] = {}
    for e in errors:
        t = e.get("type", "unknown")
        error_types[t] = error_types.get(t, 0) + 1

    runtime_errors = [p for p in predictions if p.get("error")]
    oom_errors = [p for p in predictions if "OutOfMemory" in str(p.get("error", ""))]

    lines = []
    lines.append("# VerifyLens Kaggle Benchmark Report\n")
    lines.append(f"Generated: {datetime.utcnow().strftime('%Y-%m-%d %H:%M UTC')}\n")

    # ── Hardware ──────────────────────────────────────────────────────────────
    lines.append("## Hardware\n")
    lines.append(f"| Property | Value |")
    lines.append(f"|---|---|")
    lines.append(f"| GPU | {hardware_info.get('gpu_name', 'N/A')} |")
    lines.append(f"| GPU Memory | {hardware_info.get('gpu_memory_gb', 'N/A')} GB |")
    lines.append(f"| CUDA Version | {hardware_info.get('cuda_version', 'N/A')} |")
    lines.append(f"| GPU Count | {hardware_info.get('gpu_count', 'N/A')} |")
    lines.append(f"| CPU | {hardware_info.get('cpu', 'N/A')} |")
    lines.append(f"| RAM | {hardware_info.get('ram_gb', 'N/A')} GB |")
    lines.append(f"| Python | {hardware_info.get('python_version', 'N/A')} |")
    lines.append("")

    # ── Memory ───────────────────────────────────────────────────────────────
    lines.append("## GPU Memory Usage\n")
    lines.append(f"| Metric | Value |")
    lines.append(f"|---|---|")
    lines.append(f"| Peak allocated | {memory_stats.get('peak_allocated_gb', 'N/A')} GB |")
    lines.append(f"| Peak reserved | {memory_stats.get('peak_reserved_gb', 'N/A')} GB |")
    lines.append("")

    # ── Models ───────────────────────────────────────────────────────────────
    lines.append("## Model Information\n")
    lines.append(f"| Property | Value |")
    lines.append(f"|---|---|")
    lines.append(f"| Mode | `{model_mode}` |")
    lines.append(f"| Kaggle model | `{model_info.get('model_id', 'N/A')}` |")
    lines.append(f"| Precision | {model_info.get('dtype', 'N/A')} |")
    lines.append(f"| Device | {model_info.get('device', 'N/A')} |")
    lines.append(f"| LoRA adapter | {model_info.get('adapter', 'none')} |")
    lines.append(f"| Model load time | {summary.get('model_load_time_s', 'N/A')} s |")
    lines.append("")

    # ── CRITICAL: Comparability Warning ──────────────────────────────────────
    lines.append("## ⚠️  Comparability Warning\n")
    lines.append("**These results are NOT directly comparable to the Mac production benchmark.**\n")
    lines.append("")
    lines.append("| | Mac Production | Kaggle Benchmark |")
    lines.append("|---|---|---|")

    if model_mode == "vlm":
        lines.append("| Model | `mlx-community/Qwen2.5-VL-3B-Instruct-4bit` | `Qwen/Qwen2.5-VL-3B-Instruct` |")
        lines.append("| Runtime | `mlx-vlm` (Apple MLX) | HuggingFace Transformers |")
        lines.append("| Precision | 4-bit MLX quantization | BF16 (bfloat16) |")
        lines.append("| Platform | Apple Silicon (M-series) | NVIDIA GPU (CUDA) |")
        lines.append("")
        lines.append(
            "> **Same model family and weights (Qwen2.5-VL-3B). "
            "NOT bit-identical due to different precision and runtime framework.**\n"
        )
    else:
        lines.append("| Model | `mlx-community/Qwen2.5-1.5B-Instruct-4bit` + LoRA adapter | `Qwen/Qwen2.5-1.5B-Instruct` (base only) |")
        lines.append("| Runtime | `mlx-lm` (Apple MLX) | HuggingFace Transformers |")
        lines.append("| LoRA Adapter | ✅ Loaded (`verifylens-adapter`) | ❌ Not loaded |")
        lines.append("| Precision | 4-bit MLX quantization | BF16 (bfloat16) |")
        lines.append("")
        lines.append(
            "> **Base model only — LoRA adapter not evaluated. "
            "The MLX LoRA adapter cannot be directly loaded on Kaggle without conversion. "
            "See models/README.md for conversion requirements.**\n"
        )

    # ── Dataset ───────────────────────────────────────────────────────────────
    lines.append("## Dataset\n")
    lines.append(f"| Property | Value |")
    lines.append(f"|---|---|")
    lines.append(f"| Total samples | {total} |")
    lines.append(f"| Benchmark JSONL | `benchmark/benchmark.jsonl` |")
    lines.append(f"| Images | `benchmark/images/` (200 JPEG files) |")
    lines.append(f"| Generation seed | 9999 |")
    lines.append(f"| Sample limit | {config.get('limit', 200)} |")
    for doc_type, dstats in dt.items():
        lines.append(f"| {doc_type} samples | {dstats.get('samples', 0)} |")
    lines.append("")

    lines.append("> **⚠️ Non-discriminative fields**: `gender` and `address` are `null`")
    lines.append("> for 100% of samples. They always score 100% correct (null == null)")
    lines.append("> and do NOT reflect model capability for those fields.\n")

    # ── Results ───────────────────────────────────────────────────────────────
    lines.append("## Results Summary\n")
    lines.append("| Metric | Value |")
    lines.append("|---|---|")
    lines.append(f"| JSON valid | {summary.get('json_valid_rate', 0):.2f}% |")
    lines.append(f"| All-field exact match | {summary.get('exact_match_rate', 0):.2f}% |")
    lines.append(f"| Exact match excl. doc_type | {summary.get('exact_match_no_doctype_rate', 0):.2f}% |")
    lines.append(f"| Core identity exact match (name+dob+doc_number) | {summary.get('core_identity_exact_match_rate', 0):.2f}% |")
    lines.append("")

    lines.append("## Field Accuracies\n")
    lines.append("| Field | Accuracy | Note |")
    lines.append("|---|---|---|")
    for field, acc in fa.items():
        note = "⚠️ non-discriminative (all null)" if field in NON_DISCRIMINATIVE_FIELDS else ""
        lines.append(f"| {field} | {acc:.2f}% | {note} |")
    lines.append("")

    # ── Document Type Breakdown ───────────────────────────────────────────────
    lines.append("## Document Type Breakdown\n")
    lines.append("| Type | Samples | All-field EM | Core EM | doc_type acc |")
    lines.append("|---|---|---|---|---|")
    for doc_type, dstats in dt.items():
        dfa = dstats.get("field_accuracies", {})
        lines.append(
            f"| {doc_type} "
            f"| {dstats.get('samples', 0)} "
            f"| {dstats.get('exact_match_rate', 0):.2f}% "
            f"| {dstats.get('core_identity_exact_match_rate', 0):.2f}% "
            f"| {dfa.get('doc_type', 0):.2f}% |"
        )
    lines.append("")

    # ── Latency ───────────────────────────────────────────────────────────────
    lines.append("## Latency (per sample, milliseconds)\n")
    if lat:
        lines.append("| Stat | Value (ms) |")
        lines.append("|---|---|")
        for stat in ["mean", "median", "p95", "min", "max"]:
            lines.append(f"| {stat.upper()} | {lat.get(stat, 'N/A')} |")
    else:
        lines.append("No latency data available.\n")
    lines.append("")

    # ── Failures ─────────────────────────────────────────────────────────────
    lines.append("## Failures\n")
    lines.append(f"| Category | Count |")
    lines.append(f"|---|---|")
    for etype, count in error_types.items():
        lines.append(f"| {etype} | {count} |")
    lines.append(f"| runtime errors | {len(runtime_errors)} |")
    lines.append(f"| OOM errors | {len(oom_errors)} |")
    lines.append("")

    # ── Config ────────────────────────────────────────────────────────────────
    lines.append("## Configuration\n")
    lines.append("```yaml")
    for k, v in config.items():
        lines.append(f"{k}: {v}")
    lines.append("```\n")

    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
