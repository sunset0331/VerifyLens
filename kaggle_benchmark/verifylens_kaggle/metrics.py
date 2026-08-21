"""
verifylens_kaggle/metrics.py
------------------------------
COPIED FROM: src/evaluation/metrics.py
Keep synchronized with production metrics.

If the production metrics logic changes, update this file too.
"""

import statistics
from typing import Dict, Any, List


class BenchmarkMetrics:
    def __init__(self):
        # Counts
        self.total_samples = 0
        self.json_valid_count = 0
        self.exact_match_count = 0
        self.exact_match_no_doctype_count = 0
        self.core_identity_match_count = 0

        # Field level stats: { field_name: {"correct": 0, "total": 0} }
        self.fields = {}

        # Latency lists (to compute min/max/p50/p95/mean)
        self.latencies = []

        # Error categorization
        self.errors = []

        # Document type breakdown
        self.doc_types = {}

    def record_sample(
        self,
        doc_type: str,
        ground_truth: Dict[str, Any],
        predicted: Dict[str, Any],
        json_valid: bool,
        latency_ms: float,
        parse_error: str = None,
    ):
        self.total_samples += 1

        if json_valid:
            self.json_valid_count += 1

        if latency_ms is not None:
            self.latencies.append(latency_ms)

        if doc_type not in self.doc_types:
            self.doc_types[doc_type] = {
                "total": 0,
                "exact_match": 0,
                "exact_match_no_doctype": 0,
                "core_identity_match": 0,
                "fields": {},
            }
        self.doc_types[doc_type]["total"] += 1

        all_fields_match = True
        all_fields_no_doctype_match = True
        core_identity_match = True

        core_fields = {"name", "dob", "doc_number"}

        # Evaluate each field present in ground truth
        for field, gt_val in ground_truth.items():
            if field not in self.fields:
                self.fields[field] = {"correct": 0, "total": 0}
            if field not in self.doc_types[doc_type]["fields"]:
                self.doc_types[doc_type]["fields"][field] = {"correct": 0, "total": 0}

            self.fields[field]["total"] += 1
            self.doc_types[doc_type]["fields"][field]["total"] += 1

            pred_val = predicted.get(field, None)

            if pred_val == gt_val:
                self.fields[field]["correct"] += 1
                self.doc_types[doc_type]["fields"][field]["correct"] += 1
            else:
                all_fields_match = False

                if field != "doc_type":
                    all_fields_no_doctype_match = False

                if field in core_fields:
                    core_identity_match = False

                self.errors.append(
                    {
                        "type": "field_mismatch",
                        "doc_type": doc_type,
                        "field": field,
                        "expected": gt_val,
                        "actual": pred_val,
                    }
                )

        if not json_valid:
            all_fields_match = False
            all_fields_no_doctype_match = False
            core_identity_match = False
            self.errors.append(
                {
                    "type": "invalid_json",
                    "doc_type": doc_type,
                    "error": parse_error,
                }
            )

        if all_fields_match:
            self.exact_match_count += 1
            self.doc_types[doc_type]["exact_match"] += 1

        if all_fields_no_doctype_match:
            self.exact_match_no_doctype_count += 1
            self.doc_types[doc_type]["exact_match_no_doctype"] += 1

        if core_identity_match:
            self.core_identity_match_count += 1
            self.doc_types[doc_type]["core_identity_match"] += 1

    def compute_summary(self) -> Dict[str, Any]:
        if self.total_samples == 0:
            return {}

        # Latency calculations
        sorted_latencies = sorted(self.latencies)
        latency_stats = {}
        if sorted_latencies:
            latency_stats = {
                "mean": round(statistics.mean(sorted_latencies), 1),
                "median": round(statistics.median(sorted_latencies), 1),
                "min": round(sorted_latencies[0], 1),
                "max": round(sorted_latencies[-1], 1),
            }
            idx_p95 = int(len(sorted_latencies) * 0.95)
            latency_stats["p95"] = (
                round(sorted_latencies[idx_p95], 1)
                if idx_p95 < len(sorted_latencies)
                else latency_stats["max"]
            )

        field_accuracies = {}
        for f, stats in self.fields.items():
            field_accuracies[f] = (
                round((stats["correct"] / stats["total"]) * 100, 2)
                if stats["total"] > 0
                else 0
            )

        doc_type_stats = {}
        for dt, dt_stats in self.doc_types.items():
            dt_field_acc = {}
            for f, stats in dt_stats["fields"].items():
                dt_field_acc[f] = (
                    round((stats["correct"] / stats["total"]) * 100, 2)
                    if stats["total"] > 0
                    else 0
                )

            doc_type_stats[dt] = {
                "samples": dt_stats["total"],
                "exact_match_rate": round(
                    (dt_stats["exact_match"] / dt_stats["total"]) * 100, 2
                )
                if dt_stats["total"] > 0
                else 0,
                "exact_match_no_doctype_rate": round(
                    (dt_stats.get("exact_match_no_doctype", 0) / dt_stats["total"])
                    * 100,
                    2,
                )
                if dt_stats["total"] > 0
                else 0,
                "core_identity_exact_match_rate": round(
                    (dt_stats.get("core_identity_match", 0) / dt_stats["total"]) * 100,
                    2,
                )
                if dt_stats["total"] > 0
                else 0,
                "field_accuracies": dt_field_acc,
            }

        return {
            "total_samples": self.total_samples,
            "json_valid_rate": round(
                (self.json_valid_count / self.total_samples) * 100, 2
            ),
            "exact_match_rate": round(
                (self.exact_match_count / self.total_samples) * 100, 2
            ),
            "exact_match_no_doctype_rate": round(
                (self.exact_match_no_doctype_count / self.total_samples) * 100, 2
            )
            if self.total_samples > 0
            else 0,
            "core_identity_exact_match_rate": round(
                (self.core_identity_match_count / self.total_samples) * 100, 2
            )
            if self.total_samples > 0
            else 0,
            "field_accuracies": field_accuracies,
            "latency_ms": latency_stats,
            "doc_types": doc_type_stats,
            "error_sample_count": len(self.errors),
        }
