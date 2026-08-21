import torch
import json
import os
from safetensors.torch import load_file
from peft import PeftConfig, PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

def test_weight_deltas():
    print("--- 1. Weight Delta Validation ---")
    mlx_weights = load_file("checkpoints/verifylens-adapter/adapters.safetensors")
    peft_weights = load_file("checkpoints/verifylens-adapter-peft/adapter_model.safetensors")
    
    with open("checkpoints/verifylens-adapter/adapter_config.json") as f:
        mlx_config = json.load(f)
    
    with open("checkpoints/verifylens-adapter-peft/adapter_config.json") as f:
        peft_config = json.load(f)
        
    mlx_scale = mlx_config["lora_parameters"]["scale"]
    peft_scale = peft_config["lora_alpha"] / peft_config["r"]
    
    print(f"MLX scale: {mlx_scale}, PEFT scale: {peft_scale}")
    assert mlx_scale == peft_scale, "Scaling mismatch"
    
    # Pick a sample layer: model.layers.20.self_attn.q_proj
    # MLX weights: lora_a [in, r], lora_b [r, out]
    # Delta = (lora_a @ lora_b).T * scale  [out, in]
    
    # PEFT weights: lora_A.weight [r, in], lora_B.weight [out, r]
    # Delta = (lora_B.weight @ lora_A.weight) * scale  [out, in]
    
    mlx_a = mlx_weights["model.layers.20.self_attn.q_proj.lora_a"] # [1536, 8]
    mlx_b = mlx_weights["model.layers.20.self_attn.q_proj.lora_b"] # [8, 1536]
    
    delta_mlx = (mlx_a @ mlx_b).t() * mlx_scale
    
    peft_a = peft_weights["base_model.model.model.layers.20.self_attn.q_proj.lora_A.weight"] # [8, 1536]
    peft_b = peft_weights["base_model.model.model.layers.20.self_attn.q_proj.lora_B.weight"] # [1536, 8]
    
    delta_peft = (peft_b @ peft_a) * peft_scale
    
    max_diff = torch.max(torch.abs(delta_mlx - delta_peft)).item()
    mean_diff = torch.mean(torch.abs(delta_mlx - delta_peft)).item()
    
    print(f"Delta shape: {delta_mlx.shape}")
    print(f"Max absolute difference: {max_diff:.8e}")
    print(f"Mean absolute difference: {mean_diff:.8e}")
    
    assert max_diff < 1e-6, "Delta mismatch exceeds tolerance"
    print("Weight delta validation PASSED.")
    return True

def test_peft_inference():
    print("\n--- 2. HF PEFT Load & Inference Test ---")
    base_model_id = "Qwen/Qwen2.5-1.5B-Instruct"
    adapter_path = "checkpoints/verifylens-adapter-peft"
    
    print("Loading base model...")
    # Use CPU to avoid issues if we don't have enough GPU on this specific node,
    # but try CUDA if available. Actually for a quick test, bf16 on CPU might fail on Mac,
    # let's just use float32 CPU for the test, or bfloat16 if Mac supports it.
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    tokenizer = AutoTokenizer.from_pretrained(base_model_id)
    base_model = AutoModelForCausalLM.from_pretrained(
        base_model_id, 
        torch_dtype=torch.float32,
        device_map=device
    )
    
    prompt = "<|im_start|>user\nExtract identity fields from this OCR: Name: JOHN DOE DOB: 01/01/1990<|im_end|>\n<|im_start|>assistant\n"
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    
    print("Generating with base model...")
    with torch.no_grad():
        out_base = base_model.generate(**inputs, max_new_tokens=20, do_sample=False)
    text_base = tokenizer.decode(out_base[0], skip_special_tokens=True)
    
    print("Loading PEFT adapter...")
    model_peft = PeftModel.from_pretrained(base_model, adapter_path)
    
    print("Generating with PEFT model...")
    with torch.no_grad():
        out_peft = model_peft.generate(**inputs, max_new_tokens=20, do_sample=False)
    text_peft = tokenizer.decode(out_peft[0], skip_special_tokens=True)
    
    print("\nBase Output:\n", text_base.replace(prompt, "").strip())
    print("\nPEFT Output:\n", text_peft.replace(prompt, "").strip())
    
    assert text_base != text_peft, "PEFT model output identical to base (adapter might not be active)"
    print("\nPEFT inference test PASSED.")

if __name__ == "__main__":
    test_weight_deltas()
    test_peft_inference()
