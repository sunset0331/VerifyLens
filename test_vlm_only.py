import sys
import argparse
from mlx_lm import load, generate

def main():
    parser = argparse.ArgumentParser(description="Test VerifyLens VLM directly on noisy text")
    parser.add_argument("--text", type=str, help="The noisy OCR text to process")
    args = parser.parse_args()
    
    print("Loading Fine-Tuned VerifyLens VLM Adapter...\n")
    # Load base model and our trained adapter
    model, tokenizer = load(
        "mlx-community/Qwen2.5-1.5B-Instruct-4bit",
        adapter_path="checkpoints/verifylens-adapter"
    )
    
    if args.text:
        noisy_text = args.text
    else:
        # Default fallback to a heavily degraded noisy sample
        noisy_text = "G0vernm ent of In dia | Unique Identific*tion Auth0rity of I ndia | AADHAAR | Ravi Sharma | DOB: 14/09/1992 | M a le | Flat 4A, Mumbai | 8 8 2 1 4 9 9 2 0 1 1 2 | D0wnload Dat e: 24/07/2025"
        
    print("==================================================")
    print("INPUT (Simulated Noisy OCR from Image):")
    print("==================================================")
    print(f"{noisy_text}\n")
    
    system_prompt = (
        "You are a document intelligence assistant specializing in Indian identity documents. "
        "You will receive the OCR-extracted text from an identity document and a question. "
        "Respond ONLY with a valid JSON object containing the requested field(s). "
        'Example: {"name": "Ravi Sharma"} or {"dob": "23/04/1990"}. '
        "If a field is not found in the text, use null as the value."
    )
    
    user_content = f"Document OCR text:\n{noisy_text}\n\nQuestion: Extract all key fields from this document as JSON."
    
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_content}
    ]
    
    prompt = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    
    print("==================================================")
    print("OUTPUT (VerifyLens Extracted JSON):")
    print("==================================================")
    
    response = generate(model, tokenizer, prompt=prompt, max_tokens=128, verbose=False)
    print(response.strip())
    print("\n✅ Extraction Complete!")

if __name__ == "__main__":
    main()
