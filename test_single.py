import sys
import asyncio
from PIL import Image
import json

# Adjust sys.path to run from the root directory
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from src.api.pipeline import pipeline

async def main():
    if len(sys.argv) < 3:
        print("Usage: python test_single.py <path_to_id_image> <path_to_selfie_image>")
        print("Example: python test_single.py data/synthetic/sample_aadhaar.jpg data/synthetic/sample_selfie.jpg")
        sys.exit(1)
        
    id_path = sys.argv[1]
    selfie_path = sys.argv[2]
    
    print(f"Loading ID image: {id_path}")
    print(f"Loading Selfie image: {selfie_path}")
    
    try:
        id_image = Image.open(id_path).convert("RGB")
        selfie_image = Image.open(selfie_path).convert("RGB")
    except Exception as e:
        print(f"Error loading images: {e}")
        sys.exit(1)
        
    print("\nRunning VerifyLens Pipeline...")
    print("(Note: First run will load all models including the fine-tuned VLM, which takes a few seconds)\n")
    
    result = await pipeline.run(id_image, selfie_image)
    
    print("="*50)
    print(" 🔎 VERIFYLENS KYC RESULTS")
    print("="*50)
    print(json.dumps(result, indent=2))
    
    print("\nVerdict:", result.get('verdict'))
    print("Face Match:", result.get('face_match'))

if __name__ == "__main__":
    asyncio.run(main())
