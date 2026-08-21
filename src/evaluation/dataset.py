import json
from pathlib import Path
from typing import Iterator, Dict, Any

def load_kyc_benchmark(jsonl_path: str) -> Iterator[Dict[str, Any]]:
    """
    Load the synthetic KYC benchmark dataset.
    Returns an iterator of dictionaries:
    {
        "id": "...",
        "image_path": "...",
        "document_type": "...",
        "ground_truth": { "name": ..., "dob": ..., "doc_number": ..., "doc_type": ... }
    }
    """
    path = Path(jsonl_path)
    if not path.exists():
        raise FileNotFoundError(f"Benchmark file not found: {path}")
        
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            yield json.loads(line)

def load_sroie_benchmark(split: str = "test", limit: int = 50) -> Iterator[Dict[str, Any]]:
    """
    Load the SROIE receipt dataset from Hugging Face for the public benchmark track.
    Maps to the native SROIE flat fields.
    """
    try:
        from datasets import load_dataset
    except ImportError:
        raise ImportError("The 'datasets' package is required to load SROIE.")
        
    # jsdnrs/ICDAR2019-SROIE contains the receipt task
    ds = load_dataset("jsdnrs/ICDAR2019-SROIE", "key_information_extraction", split=split)
    
    count = 0
    for sample in ds:
        if limit is not None and count >= limit:
            break
            
        # SROIE gives PIL image in 'image' and entities in 'company', 'date', 'address', 'total'
        yield {
            "id": sample["id"],
            "image_obj": sample["image"], # Pass PIL Image directly for HF datasets
            "document_type": "receipt",
            "ground_truth": {
                "company": sample["company"],
                "date": sample["date"],
                "address": sample["address"],
                "total": sample["total"]
            }
        }
        count += 1
