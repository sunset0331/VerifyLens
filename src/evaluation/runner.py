import json
import time
from pathlib import Path
from PIL import Image

from src.document.extractor_router import get_extractor
from src.evaluation.dataset import load_kyc_benchmark
from src.evaluation.normalizer import normalize_dict
from src.evaluation.metrics import BenchmarkMetrics

def run_kyc_benchmark(dataset_path: str, output_dir: str):
    """
    Run Track A: VerifyLens KYC Benchmark
    """
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    print("=== Track A: VerifyLens KYC Benchmark ===")
    
    configs = [
        {"name": "OCR + Base", "mode": "ocr_llm", "adapter": None},
        {"name": "OCR + LoRA", "mode": "ocr_llm", "adapter": "checkpoints/verifylens-adapter"},
        {"name": "VLM", "mode": "vlm", "adapter": None},
    ]
    
    results = {}
    
    for cfg in configs:
        print(f"\n--- Evaluating: {cfg['name']} ---")
        
        # Load extractor
        t0 = time.time()
        try:
            if cfg["mode"] == "ocr_llm":
                from src.document.ocr_llm_extractor import OCRLLMExtractor
                # Fallback to None if adapter path doesn't exist
                if cfg["adapter"] and not Path(cfg["adapter"]).exists():
                    print(f"Skipping {cfg['name']} because adapter {cfg['adapter']} is missing.")
                    results[cfg['name']] = None
                    continue
                extractor = OCRLLMExtractor(adapter_path=cfg["adapter"])
            else:
                extractor = get_extractor(cfg["mode"])
        except Exception as e:
            print(f"Failed to load extractor for {cfg['name']}: {e}")
            results[cfg['name']] = None
            continue
            
        load_time = time.time() - t0
        print(f"Model loaded in {load_time:.1f}s")
        
        metrics = BenchmarkMetrics()
        
        # Warmup
        print("Warming up...")
        try:
            dummy_img = Image.new("RGB", (300, 300), color=(255, 255, 255))
            extractor.extract(dummy_img)
        except:
            pass
            
        print("Running inference...")
        
        dataset = load_kyc_benchmark(dataset_path)
        base_dir = Path(dataset_path).parent
        
        # Iteration
        for sample in dataset:
            img_path = base_dir / sample["image_path"]
            try:
                img = Image.open(img_path).convert("RGB")
            except Exception as e:
                print(f"Failed to load image {img_path}: {e}")
                continue
                
            gt_norm = normalize_dict(sample["ground_truth"])
            
            result = extractor.extract(img)
            
            # The result.fields is a Pydantic model DocumentFields
            pred_dict = result.fields.model_dump()
            pred_norm = normalize_dict(pred_dict)
            
            metrics.record_sample(
                doc_type=sample["document_type"],
                ground_truth=gt_norm,
                predicted=pred_norm,
                json_valid=result.json_valid,
                latency_ms=result.latency_ms,
                parse_error=result.parse_error
            )
            
        summary = metrics.compute_summary()
        summary["model_load_time_s"] = round(load_time, 1)
        
        # Save detailed errors
        error_file = Path(output_dir) / f"{cfg['name'].replace(' ', '_').lower()}_errors.json"
        with open(error_file, "w") as f:
            json.dump(metrics.errors, f, indent=2)
            
        results[cfg['name']] = summary
        print(f"Done. Exact Match Rate: {summary.get('exact_match_rate', 0)}%")
        
    # Save overall results
    with open(Path(output_dir) / "kyc_benchmark_results.json", "w") as f:
        json.dump(results, f, indent=2)
        
    return results
