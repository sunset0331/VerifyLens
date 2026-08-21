"""
tests/test_lightweight.py
--------------------------
Lightweight tests for the Kaggle benchmark package.
These tests run WITHOUT loading any models.

Tests:
  1. Benchmark JSONL parsing — 200 samples load correctly
  2. Image path resolution — all 200 images exist on disk
  3. Normalizer — date, None, whitespace handling
  4. Metrics — field accuracy calculations
  5. Checkpoint — resume logic detects completed IDs
  6. Hardware module — CPU-only path (no CUDA check)
  7. CLI args — all modes parse correctly
  8. JSON utils — parse_json, validate_fields
  9. Dataset counting — document type distribution
 10. Reporter — metrics-from-predictions

Run from kaggle_benchmark/ directory:
    python -m pytest tests/test_lightweight.py -v
or without pytest:
    python tests/test_lightweight.py
"""

from __future__ import annotations

import json
import os
import sys
import tempfile
from pathlib import Path

# Ensure kaggle_benchmark/ is on sys.path
HERE = Path(__file__).parent.parent
sys.path.insert(0, str(HERE))

# ──────────────────────────────────────────────────────────────────────────────
# Minimal test harness (no pytest required for basic run)
# ──────────────────────────────────────────────────────────────────────────────

_PASS = []
_FAIL = []


def _test(name: str, fn):
    try:
        fn()
        _PASS.append(name)
        print(f"  ✓ {name}")
    except Exception as e:
        _FAIL.append((name, str(e)))
        print(f"  ✗ {name}: {e}")


# ──────────────────────────────────────────────────────────────────────────────
# 1. Benchmark JSONL parsing
# ──────────────────────────────────────────────────────────────────────────────

BENCHMARK_JSONL = HERE / "benchmark" / "benchmark.jsonl"


def test_benchmark_loads():
    assert BENCHMARK_JSONL.exists(), f"benchmark.jsonl not found: {BENCHMARK_JSONL}"
    samples = []
    with open(BENCHMARK_JSONL) as f:
        for line in f:
            if line.strip():
                samples.append(json.loads(line))
    assert len(samples) == 200, f"Expected 200 samples, got {len(samples)}"
    # Check first sample structure
    s = samples[0]
    assert "id" in s
    assert "image_path" in s
    assert "document_type" in s
    assert "ground_truth" in s
    gt = s["ground_truth"]
    assert "name" in gt
    assert "dob" in gt
    assert "doc_number" in gt
    assert "doc_type" in gt
    assert "gender" in gt
    assert "address" in gt


def test_ground_truth_null_fields():
    """gender and address must be null for all 200 samples."""
    with open(BENCHMARK_JSONL) as f:
        for line in f:
            if not line.strip():
                continue
            s = json.loads(line)
            gt = s["ground_truth"]
            assert gt["gender"] is None, f"Expected gender=null for {s['id']}"
            assert gt["address"] is None, f"Expected address=null for {s['id']}"


def test_document_type_distribution():
    from verifylens_kaggle.dataset import count_benchmark
    counts = count_benchmark(str(BENCHMARK_JSONL))
    assert counts["total"] == 200
    assert "aadhaar" in counts
    assert "pan" in counts
    assert "passport" in counts
    total_typed = counts.get("aadhaar", 0) + counts.get("pan", 0) + counts.get("passport", 0)
    assert total_typed == 200, f"Unexpected doc types: {counts}"


# ──────────────────────────────────────────────────────────────────────────────
# 2. Image path resolution
# ──────────────────────────────────────────────────────────────────────────────

def test_all_images_exist():
    from verifylens_kaggle.dataset import validate_benchmark_images
    result = validate_benchmark_images(str(BENCHMARK_JSONL))
    assert result["ok"], f"Missing {len(result['missing'])} images: {result['missing'][:3]}"
    assert result["total"] == 200


# ──────────────────────────────────────────────────────────────────────────────
# 3. Normalizer
# ──────────────────────────────────────────────────────────────────────────────

def test_normalizer_basic():
    from verifylens_kaggle.normalizer import normalize_string, normalize_dict

    assert normalize_string(None) is None
    assert normalize_string("") is None
    assert normalize_string("  ") is None
    assert normalize_string("null") is None
    assert normalize_string("none") is None
    assert normalize_string("Ravi Kumar") == "ravi kumar"
    assert normalize_string("  Ravi  Kumar  ") == "ravi kumar"


def test_normalizer_dates():
    from verifylens_kaggle.normalizer import normalize_string

    # DD-MM-YYYY → DD/MM/YYYY
    assert normalize_string("07-01-1985") == "07/01/1985"
    # DD.MM.YYYY → DD/MM/YYYY
    assert normalize_string("07.01.1985") == "07/01/1985"
    # already correct
    assert normalize_string("07/01/1985") == "07/01/1985"


def test_normalize_dict():
    from verifylens_kaggle.normalizer import normalize_dict

    d = {"name": "Ravi Kumar", "dob": None, "doc_type": "PAN CARD"}
    nd = normalize_dict(d)
    assert nd["name"] == "ravi kumar"
    assert nd["dob"] is None
    assert nd["doc_type"] == "pan card"


# ──────────────────────────────────────────────────────────────────────────────
# 4. Metrics
# ──────────────────────────────────────────────────────────────────────────────

def test_metrics_basic():
    from verifylens_kaggle.metrics import BenchmarkMetrics

    m = BenchmarkMetrics()
    gt = {"name": "ravi kumar", "dob": "14/09/1990", "doc_number": "1234 5678 9012",
          "doc_type": "aadhaar card", "gender": None, "address": None}
    pred_correct = {"name": "ravi kumar", "dob": "14/09/1990", "doc_number": "1234 5678 9012",
                    "doc_type": "aadhaar card", "gender": None, "address": None}
    pred_wrong = {"name": "wrong name", "dob": "14/09/1990", "doc_number": "1234 5678 9012",
                  "doc_type": "aadhaar card", "gender": None, "address": None}

    m.record_sample("aadhaar", gt, pred_correct, json_valid=True, latency_ms=500.0)
    m.record_sample("aadhaar", gt, pred_wrong, json_valid=True, latency_ms=600.0)

    summary = m.compute_summary()
    assert summary["total_samples"] == 2
    assert summary["exact_match_rate"] == 50.0
    assert summary["field_accuracies"]["name"] == 50.0
    assert summary["field_accuracies"]["dob"] == 100.0
    assert summary["field_accuracies"]["gender"] == 100.0  # both None


def test_metrics_json_invalid():
    from verifylens_kaggle.metrics import BenchmarkMetrics

    m = BenchmarkMetrics()
    gt = {"name": "ravi kumar", "dob": "14/09/1990", "doc_number": "abc",
          "doc_type": "pan card", "gender": None, "address": None}
    pred = {"name": None, "dob": None, "doc_number": None,
            "doc_type": None, "gender": None, "address": None}

    m.record_sample("pan", gt, pred, json_valid=False, latency_ms=200.0, parse_error="No JSON")
    summary = m.compute_summary()
    assert summary["json_valid_rate"] == 0.0
    assert summary["exact_match_rate"] == 0.0
    assert summary["core_identity_exact_match_rate"] == 0.0


def test_metrics_latency():
    from verifylens_kaggle.metrics import BenchmarkMetrics

    m = BenchmarkMetrics()
    gt = {"name": "a", "dob": "b", "doc_number": "c", "doc_type": "d", "gender": None, "address": None}
    pred = {"name": "a", "dob": "b", "doc_number": "c", "doc_type": "d", "gender": None, "address": None}

    for lat in [100.0, 200.0, 300.0, 400.0, 500.0]:
        m.record_sample("pan", gt, pred, json_valid=True, latency_ms=lat)

    summary = m.compute_summary()
    lat = summary["latency_ms"]
    assert lat["mean"] == 300.0
    assert lat["min"] == 100.0
    assert lat["max"] == 500.0


# ──────────────────────────────────────────────────────────────────────────────
# 5. Checkpoint / Resume
# ──────────────────────────────────────────────────────────────────────────────

def test_checkpoint_fresh():
    from verifylens_kaggle.checkpoint import CheckpointManager

    with tempfile.TemporaryDirectory() as d:
        ckpt = CheckpointManager(str(Path(d) / "predictions.jsonl"))
        completed = ckpt.load_completed_ids()
        assert completed == set()


def test_checkpoint_write_and_resume():
    from verifylens_kaggle.checkpoint import CheckpointManager

    with tempfile.TemporaryDirectory() as d:
        ppath = str(Path(d) / "predictions.jsonl")
        ckpt = CheckpointManager(ppath)

        gt = {"name": "test", "dob": "01/01/2000", "doc_number": "123",
              "doc_type": "pan card", "gender": None, "address": None}
        pred = {"name": "test", "dob": "01/01/2000", "doc_number": "123",
                "doc_type": "pan card", "gender": None, "address": None}

        ckpt.append("00001", "pan", gt, pred, True, 500.0)
        ckpt.append("00002", "aadhaar", gt, None, False, 300.0, error="Test error")

        completed = ckpt.load_completed_ids()
        assert "00001" in completed
        assert "00002" in completed
        assert "00003" not in completed

        # Verify file contents
        records = ckpt.load_all_predictions()
        assert len(records) == 2
        assert records[0]["id"] == "00001"
        assert records[1]["error"] == "Test error"


def test_checkpoint_corrupted_line():
    """Corrupted JSONL lines should be skipped, not crash."""
    from verifylens_kaggle.checkpoint import CheckpointManager

    with tempfile.TemporaryDirectory() as d:
        ppath = Path(d) / "predictions.jsonl"
        with open(ppath, "w") as f:
            f.write('{"id": "00001", "document_type": "pan"}\n')
            f.write('THIS IS NOT JSON\n')
            f.write('{"id": "00003", "document_type": "aadhaar"}\n')

        ckpt = CheckpointManager(str(ppath))
        completed = ckpt.load_completed_ids()
        assert "00001" in completed
        assert "00003" in completed


# ──────────────────────────────────────────────────────────────────────────────
# 6. Hardware module (no CUDA check)
# ──────────────────────────────────────────────────────────────────────────────

def test_hardware_detect_no_fail():
    """detect_hardware with require_cuda=False should never crash."""
    from verifylens_kaggle.hardware import detect_hardware
    info = detect_hardware(require_cuda=False)
    assert "cuda_available" in info
    assert "cpu" in info
    assert "ram_gb" in info


def test_hardware_memory_stats_safe():
    """get_memory_stats should return safely even without CUDA."""
    from verifylens_kaggle.hardware import get_memory_stats
    stats = get_memory_stats()
    assert "peak_allocated_gb" in stats


# ──────────────────────────────────────────────────────────────────────────────
# 7. CLI args parsing
# ──────────────────────────────────────────────────────────────────────────────

def test_cli_args_vlm():
    """argparse should parse --model vlm --limit 5 correctly."""
    import importlib.util
    spec = importlib.util.spec_from_file_location("run_benchmark", HERE / "run_benchmark.py")
    mod = importlib.util.module_from_spec(spec)
    # We just test parse_args by monkey-patching sys.argv
    old_argv = sys.argv
    try:
        sys.argv = ["run_benchmark.py", "--model", "vlm", "--limit", "5"]
        spec.loader.exec_module(mod)
        args = mod.parse_args()
        assert args.model == "vlm"
        assert args.limit == 5
        assert args.batch_size == 1
    finally:
        sys.argv = old_argv


def test_cli_args_lora():
    import importlib.util
    spec = importlib.util.spec_from_file_location("run_benchmark", HERE / "run_benchmark.py")
    mod = importlib.util.module_from_spec(spec)
    old_argv = sys.argv
    try:
        sys.argv = ["run_benchmark.py", "--model", "lora"]
        spec.loader.exec_module(mod)
        args = mod.parse_args()
        assert args.model == "lora"
    finally:
        sys.argv = old_argv


def test_cli_args_base():
    import importlib.util
    spec = importlib.util.spec_from_file_location("run_benchmark", HERE / "run_benchmark.py")
    mod = importlib.util.module_from_spec(spec)
    old_argv = sys.argv
    try:
        sys.argv = ["run_benchmark.py", "--model", "base", "--limit", "200", "--output-dir", "my_results"]
        spec.loader.exec_module(mod)
        args = mod.parse_args()
        assert args.model == "base"
        assert args.limit == 200
        assert args.output_dir == "my_results"
    finally:
        sys.argv = old_argv


# ──────────────────────────────────────────────────────────────────────────────
# 8. JSON utils
# ──────────────────────────────────────────────────────────────────────────────

def test_json_utils_direct():
    from verifylens_kaggle.json_utils import parse_json, validate_fields, make_empty_fields

    raw = '{"name": "Ravi Kumar", "dob": "14/09/1990", "doc_number": "ABCDE1234F"}'
    parsed, valid, err = parse_json(raw)
    assert valid
    assert err is None
    assert parsed["name"] == "Ravi Kumar"

    fields = validate_fields(parsed)
    assert fields["name"] == "Ravi Kumar"
    assert fields["dob"] == "14/09/1990"
    assert fields["doc_number"] == "ABCDE1234F"
    assert fields["gender"] is None
    assert fields["address"] is None


def test_json_utils_markdown_fences():
    from verifylens_kaggle.json_utils import parse_json

    raw = '```json\n{"name": "Test", "dob": null}\n```'
    parsed, valid, err = parse_json(raw)
    assert valid
    assert parsed["name"] == "Test"
    assert parsed["dob"] is None


def test_json_utils_invalid():
    from verifylens_kaggle.json_utils import parse_json

    raw = "Sorry, I cannot extract anything from this image."
    parsed, valid, err = parse_json(raw)
    assert not valid
    assert parsed is None
    assert err is not None


def test_json_utils_embedded():
    from verifylens_kaggle.json_utils import parse_json

    raw = 'Here is the extracted JSON: {"name": "Ravi", "dob": "01/01/1990"} Hope this helps!'
    parsed, valid, err = parse_json(raw)
    assert valid
    assert parsed["name"] == "Ravi"


def test_validate_fields_empty_string():
    from verifylens_kaggle.json_utils import validate_fields

    parsed = {"name": "", "dob": "  ", "doc_number": "ABCDE1234F"}
    fields = validate_fields(parsed)
    assert fields["name"] is None      # empty string → None
    assert fields["dob"] is None       # whitespace → None
    assert fields["doc_number"] == "ABCDE1234F"


def test_validate_fields_unknown_keys():
    from verifylens_kaggle.json_utils import validate_fields

    parsed = {"name": "Ravi", "unknown_field": "should be ignored", "another_extra": 42}
    fields = validate_fields(parsed)
    assert "unknown_field" not in fields
    assert "another_extra" not in fields
    assert fields["name"] == "Ravi"


# ──────────────────────────────────────────────────────────────────────────────
# 9. Dataset loader with skip_ids
# ──────────────────────────────────────────────────────────────────────────────

def test_dataset_limit():
    from verifylens_kaggle.dataset import load_benchmark
    samples = list(load_benchmark(str(BENCHMARK_JSONL), limit=5))
    assert len(samples) == 5


def test_dataset_skip_ids():
    from verifylens_kaggle.dataset import load_benchmark
    all_samples = list(load_benchmark(str(BENCHMARK_JSONL), limit=10))
    first_id = all_samples[0]["id"]

    # limit=10 applies AFTER skip — skipped sample is excluded, we still get 10
    # (10 non-skipped samples starting from the second entry in the JSONL)
    remaining = list(load_benchmark(str(BENCHMARK_JSONL), limit=10, skip_ids={first_id}))
    assert len(remaining) == 10, f"Expected 10 non-skipped samples, got {len(remaining)}"
    assert all(s["id"] != first_id for s in remaining), "Skipped ID appeared in results"


def test_dataset_image_paths_resolve():
    from verifylens_kaggle.dataset import load_benchmark
    for sample in load_benchmark(str(BENCHMARK_JSONL), limit=3):
        assert "image_abs_path" in sample
        assert sample["image_abs_path"].exists(), f"Missing: {sample['image_abs_path']}"


# ──────────────────────────────────────────────────────────────────────────────
# 10. Reporter — metrics from predictions
# ──────────────────────────────────────────────────────────────────────────────

def test_reporter_build_metrics():
    from verifylens_kaggle.reporter import build_metrics_from_predictions

    gt = {"name": "ravi kumar", "dob": "14/09/1990", "doc_number": "abc",
          "doc_type": "pan card", "gender": None, "address": None}

    predictions = [
        {"id": "00001", "document_type": "pan", "ground_truth": gt,
         "prediction": {"name": "ravi kumar", "dob": "14/09/1990", "doc_number": "abc",
                        "doc_type": "pan card", "gender": None, "address": None},
         "json_valid": True, "latency_ms": 500.0, "parse_error": None, "error": None},
        {"id": "00002", "document_type": "pan", "ground_truth": gt,
         "prediction": {"name": "wrong", "dob": "14/09/1990", "doc_number": "abc",
                        "doc_type": "pan card", "gender": None, "address": None},
         "json_valid": True, "latency_ms": 600.0, "parse_error": None, "error": None},
    ]

    metrics = build_metrics_from_predictions(predictions)
    summary = metrics.compute_summary()
    assert summary["total_samples"] == 2
    assert summary["exact_match_rate"] == 50.0


def test_reporter_save_results():
    from verifylens_kaggle.reporter import save_results

    gt = {"name": "ravi kumar", "dob": "14/09/1990", "doc_number": "abc",
          "doc_type": "pan card", "gender": None, "address": None}
    predictions = [
        {"id": "00001", "document_type": "pan", "ground_truth": gt,
         "prediction": {"name": "ravi kumar", "dob": "14/09/1990", "doc_number": "abc",
                        "doc_type": "pan card", "gender": None, "address": None},
         "json_valid": True, "latency_ms": 500.0, "parse_error": None, "error": None},
    ]

    with tempfile.TemporaryDirectory() as d:
        save_results(
            output_dir=d,
            model_mode="vlm",
            model_info={"model_id": "test-model", "dtype": "bf16", "device": "cuda",
                        "production_model": "test-prod", "production_runtime": "mlx",
                        "kaggle_model": "test-model", "kaggle_runtime": "hf",
                        "comparability_note": "test"},
            hardware_info={"gpu_name": "T4", "gpu_memory_gb": 16.0,
                           "cuda_version": "12.1", "gpu_count": 1,
                           "cpu": "x86_64", "ram_gb": 29.0, "python_version": "3.10"},
            memory_stats={"peak_allocated_gb": 8.5, "peak_reserved_gb": 9.0,
                          "current_allocated_gb": 0.1},
            predictions=predictions,
            model_load_time_s=45.2,
            config={"model": "vlm", "limit": 1},
        )
        assert Path(d, "benchmark_results.json").exists()
        assert Path(d, "benchmark_results.csv").exists()
        assert Path(d, "benchmark_report.md").exists()


# ──────────────────────────────────────────────────────────────────────────────
# Runner
# ──────────────────────────────────────────────────────────────────────────────

def run_all_tests():
    print("\n" + "=" * 60)
    print("  VerifyLens Kaggle Benchmark — Lightweight Tests")
    print("=" * 60)
    print("  (No model loading — pure logic tests)\n")

    _test("benchmark_loads_200_samples", test_benchmark_loads)
    _test("ground_truth_gender_address_all_null", test_ground_truth_null_fields)
    _test("document_type_distribution", test_document_type_distribution)
    _test("all_200_images_exist", test_all_images_exist)
    _test("normalizer_basic", test_normalizer_basic)
    _test("normalizer_dates", test_normalizer_dates)
    _test("normalize_dict", test_normalize_dict)
    _test("metrics_basic", test_metrics_basic)
    _test("metrics_json_invalid", test_metrics_json_invalid)
    _test("metrics_latency", test_metrics_latency)
    _test("checkpoint_fresh", test_checkpoint_fresh)
    _test("checkpoint_write_and_resume", test_checkpoint_write_and_resume)
    _test("checkpoint_corrupted_line", test_checkpoint_corrupted_line)
    _test("hardware_detect_no_fail", test_hardware_detect_no_fail)
    _test("hardware_memory_stats_safe", test_hardware_memory_stats_safe)
    _test("cli_args_vlm", test_cli_args_vlm)
    _test("cli_args_lora", test_cli_args_lora)
    _test("cli_args_base", test_cli_args_base)
    _test("json_utils_direct_parse", test_json_utils_direct)
    _test("json_utils_markdown_fences", test_json_utils_markdown_fences)
    _test("json_utils_invalid", test_json_utils_invalid)
    _test("json_utils_embedded", test_json_utils_embedded)
    _test("validate_fields_empty_string", test_validate_fields_empty_string)
    _test("validate_fields_unknown_keys", test_validate_fields_unknown_keys)
    _test("dataset_limit", test_dataset_limit)
    _test("dataset_skip_ids", test_dataset_skip_ids)
    _test("dataset_image_paths_resolve", test_dataset_image_paths_resolve)
    _test("reporter_build_metrics", test_reporter_build_metrics)
    _test("reporter_save_results", test_reporter_save_results)

    print(f"\n{'=' * 60}")
    print(f"  RESULTS: {len(_PASS)} passed, {len(_FAIL)} failed")
    print(f"{'=' * 60}")
    if _FAIL:
        print("\nFailed tests:")
        for name, err in _FAIL:
            print(f"  ✗ {name}: {err}")
        return False
    else:
        print("\n  All tests passed ✓")
        return True


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)
