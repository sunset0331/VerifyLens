import pytest
from src.evaluation.normalizer import normalize_string, normalize_dict
from src.evaluation.metrics import BenchmarkMetrics

def test_normalize_string():
    assert normalize_string("UTKARSH GAUR") == "utkarsh gaur"
    assert normalize_string("  utkarsh   gaur  ") == "utkarsh gaur"
    assert normalize_string("15-08-1990") == "15/08/1990"
    assert normalize_string("15.08.1990") == "15/08/1990"
    assert normalize_string("15/08/1990") == "15/08/1990"
    assert normalize_string("") is None
    assert normalize_string("none") is None
    assert normalize_string("null") is None
    assert normalize_string(None) is None

def test_normalize_dict():
    raw = {
        "name": "Ravi   Sharma ",
        "dob": "12-05-1985",
        "gender": "none",
        "address": None
    }
    norm = normalize_dict(raw)
    assert norm["name"] == "ravi sharma"
    assert norm["dob"] == "12/05/1985"
    assert norm["gender"] is None
    assert norm["address"] is None

def test_benchmark_metrics_exact_match():
    metrics = BenchmarkMetrics()
    
    # Perfect match
    gt1 = {"name": "ravi sharma", "dob": "12/05/1985", "doc_type": None}
    pred1 = {"name": "ravi sharma", "dob": "12/05/1985", "doc_type": None, "extra": "ignored"}
    metrics.record_sample("pan", gt1, pred1, True, 100.0)
    
    # Partial match
    gt2 = {"name": "amit", "dob": "01/01/2000"}
    pred2 = {"name": "amit", "dob": "02/02/2000"}
    metrics.record_sample("pan", gt2, pred2, True, 120.0)
    
    # Missing field
    gt3 = {"name": "rahul"}
    pred3 = {"dob": "01/01/2000"}
    metrics.record_sample("aadhaar", gt3, pred3, True, 150.0)
    
    # Invalid JSON
    gt4 = {"name": "ajay"}
    pred4 = {"name": "ajay"}
    metrics.record_sample("passport", gt4, pred4, False, 50.0, "Parse error")
    
    summary = metrics.compute_summary()
    assert summary["total_samples"] == 4
    assert summary["json_valid_rate"] == 75.0
    assert summary["exact_match_rate"] == 25.0
    assert summary["field_accuracies"]["name"] == 75.0
    
    assert len(metrics.errors) == 3 # gt2 mismatch, gt3 missing, gt4 invalid json
    
def test_benchmark_metrics_extended_matches():
    metrics = BenchmarkMetrics()
    
    # 1. Exact match all
    gt1 = {"name": "ravi sharma", "dob": "12/05/1985", "doc_number": "1234", "doc_type": "pan"}
    pred1 = {"name": "ravi sharma", "dob": "12/05/1985", "doc_number": "1234", "doc_type": "pan"}
    metrics.record_sample("pan", gt1, pred1, True, 100.0)
    
    # 2. Match all EXCEPT doc_type (this is a Exact Match no doc_type, AND core match)
    gt2 = {"name": "amit", "dob": "01/01/2000", "doc_number": "5678", "doc_type": "aadhaar"}
    pred2 = {"name": "amit", "dob": "01/01/2000", "doc_number": "5678", "doc_type": None}
    metrics.record_sample("aadhaar", gt2, pred2, True, 100.0)
    
    # 3. Core identity matches, but address mismatches (Core identity match ONLY)
    gt3 = {"name": "rahul", "dob": "02/02/2000", "doc_number": "9999", "address": "delhi"}
    pred3 = {"name": "rahul", "dob": "02/02/2000", "doc_number": "9999", "address": "mumbai"}
    metrics.record_sample("passport", gt3, pred3, True, 100.0)
    
    # 4. Core identity mismatches on DOB
    gt4 = {"name": "ajay", "dob": "03/03/2000", "doc_number": "1111"}
    pred4 = {"name": "ajay", "dob": "04/04/2000", "doc_number": "1111"}
    metrics.record_sample("pan", gt4, pred4, True, 100.0)
    
    summary = metrics.compute_summary()
    assert summary["total_samples"] == 4
    
    # All exact: gt1 (1/4 = 25%)
    assert summary["exact_match_rate"] == 25.0
    
    # Exact no doc type: gt1, gt2 (2/4 = 50%)
    assert summary["exact_match_no_doctype_rate"] == 50.0
    
    # Core identity: gt1, gt2, gt3 (3/4 = 75%)
    assert summary["core_identity_exact_match_rate"] == 75.0
