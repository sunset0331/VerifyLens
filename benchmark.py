import time
from PIL import Image, ImageDraw, ImageFont

from src.document.vlm.vlm_extractor import VLMExtractor

def create_synthetic_pan_card() -> Image.Image:
    """Creates a basic synthetic PAN card image for benchmarking."""
    # Create a simple white rectangle with a blue border
    img = Image.new('RGB', (600, 400), color=(255, 255, 255))
    draw = ImageDraw.Draw(img)
    draw.rectangle([10, 10, 590, 390], outline=(0, 0, 255), width=3)
    
    # We won't load a custom font to avoid dependency issues, just use default
    draw.text((20, 20), "INCOME TAX DEPARTMENT", fill=(0, 0, 0))
    draw.text((20, 50), "GOVT OF INDIA", fill=(0, 0, 0))
    
    draw.text((20, 120), "Name:", fill=(100, 100, 100))
    draw.text((20, 140), "UTKARSH GAUR", fill=(0, 0, 0))
    
    draw.text((20, 180), "Father's Name:", fill=(100, 100, 100))
    draw.text((20, 200), "ANIL GAUR", fill=(0, 0, 0))
    
    draw.text((20, 240), "Date of Birth:", fill=(100, 100, 100))
    draw.text((20, 260), "15/08/1990", fill=(0, 0, 0))
    
    draw.text((20, 300), "Permanent Account Number:", fill=(100, 100, 100))
    draw.text((20, 320), "ABCDE1234F", fill=(0, 0, 0))
    
    return img

def run_benchmark():
    print("Initializing VLMExtractor (this will download weights if not cached)...")
    t0 = time.time()
    extractor = VLMExtractor()
    t1 = time.time()
    print(f"Initialization took {t1 - t0:.2f} seconds.")
    
    print("\nGenerating synthetic PAN card image...")
    img = create_synthetic_pan_card()
    
    print("Running extraction...")
    t2 = time.time()
    result = extractor.extract(img)
    t3 = time.time()
    
    print(f"\nExtraction took {t3 - t2:.2f} seconds.")
    print(f"Model-reported Latency: {result.latency_ms} ms")
    
    print("\n--- JSON Result ---")
    print(f"Valid JSON: {result.json_valid}")
    if not result.json_valid:
        print(f"Error: {result.parse_error}")
        print(f"Raw Output: {result.raw_output}")
    else:
        print("Fields extracted:")
        for k, v in result.fields.to_dict().items():
            print(f"  {k}: {v}")

if __name__ == "__main__":
    run_benchmark()
