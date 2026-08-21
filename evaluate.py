import argparse
from src.evaluation.runner import run_kyc_benchmark

def main():
    parser = argparse.ArgumentParser(description="Run VerifyLens Benchmarks")
    parser.add_argument("--track", choices=["kyc", "sroie", "all"], default="kyc", help="Which benchmark track to run")
    parser.add_argument("--dataset", type=str, default="data/synthetic/benchmark.jsonl", help="Path to KYC dataset")
    parser.add_argument("--output", type=str, default="results", help="Output directory for results")
    args = parser.parse_args()
    
    if args.track in ["kyc", "all"]:
        run_kyc_benchmark(args.dataset, args.output)
        
    if args.track in ["sroie", "all"]:
        print("SROIE benchmark track is not yet implemented.")

if __name__ == "__main__":
    main()
