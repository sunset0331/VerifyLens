"""
verifylens_kaggle/checkpoint.py
---------------------------------
Checkpoint / resume logic for Kaggle benchmark.

After each successful (or failed) sample, the result is appended to
predictions.jsonl. If the benchmark process is interrupted and restarted,
completed sample IDs are detected and skipped.

Format of each line in predictions.jsonl:
{
    "id": "00042",
    "document_type": "pan",
    "ground_truth": {...},
    "prediction": {...},          # null if error
    "json_valid": true,
    "latency_ms": 1234.5,
    "ocr_latency_ms": 87.2,       # null for VLM mode
    "llm_latency_ms": 1147.3,     # null for VLM mode
    "error": null
}
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any, Dict, Optional, Set


class CheckpointManager:
    """
    Manages incremental prediction checkpointing.

    Usage
    -----
    ckpt = CheckpointManager("results/predictions.jsonl")
    completed = ckpt.load_completed_ids()  # set of already-done sample IDs
    # ... run inference ...
    ckpt.append(sample_id, doc_type, ground_truth, prediction, ...)
    """

    def __init__(self, predictions_path: str):
        self.path = Path(predictions_path)
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def load_completed_ids(self) -> Set[str]:
        """
        Read existing predictions.jsonl and return set of completed sample IDs.
        If the file does not exist, returns an empty set (fresh run).
        """
        completed: Set[str] = set()
        if not self.path.exists():
            return completed

        with open(self.path, "r", encoding="utf-8") as f:
            for lineno, line in enumerate(f, start=1):
                line = line.strip()
                if not line:
                    continue
                try:
                    rec = json.loads(line)
                    sid = rec.get("id")
                    if sid:
                        completed.add(sid)
                except json.JSONDecodeError:
                    # Corrupted line — skip silently, do not re-run completed work
                    print(
                        f"[checkpoint] Warning: corrupted line {lineno} in "
                        f"{self.path} — skipping"
                    )
        return completed

    def load_all_predictions(self) -> list:
        """Load all prediction records from the checkpoint file."""
        records = []
        if not self.path.exists():
            return records
        with open(self.path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    records.append(json.loads(line))
                except json.JSONDecodeError:
                    pass
        return records

    def append(
        self,
        sample_id: str,
        document_type: str,
        ground_truth: Dict[str, Any],
        prediction: Optional[Dict[str, Any]],
        json_valid: bool,
        latency_ms: Optional[float],
        ocr_latency_ms: Optional[float] = None,
        llm_latency_ms: Optional[float] = None,
        error: Optional[str] = None,
    ) -> None:
        """
        Append one prediction record to predictions.jsonl.

        This is called immediately after each sample is processed,
        so partial runs are always resumable.
        """
        record = {
            "id": sample_id,
            "document_type": document_type,
            "ground_truth": ground_truth,
            "prediction": prediction,
            "json_valid": json_valid,
            "latency_ms": latency_ms,
            "ocr_latency_ms": ocr_latency_ms,
            "llm_latency_ms": llm_latency_ms,
            "error": error,
        }

        with open(self.path, "a", encoding="utf-8") as f:
            f.write(json.dumps(record) + "\n")

    def handle_oom(
        self,
        sample_id: str,
        gpu_info: Dict[str, Any],
        model_config: Dict[str, Any],
        batch_size: int,
    ) -> None:
        """
        Called when a CUDA OOM is detected.

        Prints a diagnostic report and exits WITHOUT retrying.
        The checkpoint file is preserved so the run can be resumed
        after reducing batch_size or adjusting configuration.
        """
        print("\n" + "!" * 60)
        print("  GPU OUT OF MEMORY — BENCHMARK STOPPED")
        print("!" * 60)
        print(f"  Failed at sample : {sample_id}")
        print(f"  GPU name         : {gpu_info.get('gpu_name', 'unknown')}")
        print(f"  GPU total memory : {gpu_info.get('gpu_memory_gb', '?')} GB")
        print(f"  Model            : {model_config.get('model_id', '?')}")
        print(f"  Precision        : {model_config.get('dtype', '?')}")
        print(f"  Batch size       : {batch_size}")
        print()
        print("Completed predictions have been saved to:")
        print(f"  {self.path}")
        print()
        print("To resume, re-run the same command — completed samples")
        print("will be skipped automatically via checkpoint detection.")
        print()
        print("To reduce memory usage, try:")
        print("  --batch-size 1  (already minimum for sequential mode)")
        print("  Use a GPU with more VRAM (T4=16GB, P100=16GB, A100=40GB)")
        print("!" * 60 + "\n")
        sys.exit(2)
