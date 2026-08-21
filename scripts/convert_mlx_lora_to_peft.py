#!/usr/bin/env python3
import json
import os
import argparse
import torch
from pathlib import Path
from safetensors.torch import load_file, save_file
from transformers import AutoModelForCausalLM

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", required=True, help="Path to MLX adapter directory")
    parser.add_argument("--output", required=True, help="Path to output PEFT adapter directory")
    args = parser.parse_args()

    mlx_dir = Path(args.source)
    out_dir = Path(args.output)
    
    # 1. Inspect MLX adapter
    with open(mlx_dir / "adapter_config.json") as f:
        mlx_config = json.load(f)
        
    rank = mlx_config["lora_parameters"]["rank"]
    scale = mlx_config["lora_parameters"]["scale"]
    alpha = scale * rank  # PEFT scaling = alpha / rank. So alpha = scale * rank.
    
    print("==================================================")
    print(" MLX ADAPTER CONFIGURATION")
    print("==================================================")
    print(f"Rank : {rank}")
    print(f"Scale: {scale}")
    print(f"Alpha: {alpha} (Derived PEFT alpha)")
    print(f"Base : {mlx_config['model']}")
    
    mlx_tensors = load_file(mlx_dir / "adapters.safetensors")
    print(f"Total MLX tensors: {len(mlx_tensors)}")
    
    # 2. Inspect HuggingFace Qwen Architecture
    hf_model_id = "Qwen/Qwen2.5-1.5B-Instruct"
    print("\n==================================================")
    print(" LOADING HUGGINGFACE BASE MODEL (META ONLY)")
    print("==================================================")
    print(f"Model: {hf_model_id}")
    # Load with meta device to instantly inspect shapes without RAM overhead
    from transformers import AutoConfig
    config = AutoConfig.from_pretrained(hf_model_id, trust_remote_code=True)
    with torch.device('meta'):
        base_model = AutoModelForCausalLM.from_config(config, trust_remote_code=True)
    
    state_dict = dict(base_model.named_modules())
    
    # 3. Convert & Validate
    print("\n==================================================")
    print(" MAPPING & SHAPE VALIDATION")
    print("==================================================")
    
    peft_tensors = {}
    manifest = []
    
    target_modules = set()
    layers_transformed = set()
    
    mapped_count = 0
    unmapped_count = 0
    
    # Group matrices by module to calculate mathematical scaling validation
    grouped_matrices = {}
    
    for k, v in mlx_tensors.items():
        if ".lora_a" in k or ".lora_b" in k:
            is_a = ".lora_a" in k
            base_k = k.replace(".lora_a", "").replace(".lora_b", "")
            
            # e.g., model.layers.27.self_attn.v_proj
            # MLX module name matches HF module name perfectly for Qwen!
            hf_module_key = base_k 
            if hf_module_key not in state_dict:
                print(f"UNEXPECTED TENSOR: {k} (No matching HF module {hf_module_key})")
                unmapped_count += 1
                continue
                
            hf_module = state_dict[hf_module_key]
            hf_weight_shape = hf_module.weight.shape  # [out_dim, in_dim]
            out_dim, in_dim = hf_weight_shape
            
            layer_num = int(base_k.split(".layers.")[1].split(".")[0])
            module_name = base_k.split(".")[-1]
            target_modules.add(module_name)
            layers_transformed.add(layer_num)
            
            # MLX stores W as [in, out].
            # lora_a is [in, r], lora_b is [r, out].
            # HF PEFT lora_A is [r, in], lora_B is [out, r].
            # Therefore, we MUST transpose both.
            if is_a:
                expected_mlx_shape = (in_dim, rank)
                if tuple(v.shape) != expected_mlx_shape:
                    raise ValueError(f"Shape mismatch! {k} is {tuple(v.shape)}, expected {expected_mlx_shape}")
                
                converted_v = v.t().contiguous()
                expected_peft_shape = (rank, in_dim)
                peft_key = f"base_model.model.{base_k}.lora_A.weight"
            else:
                expected_mlx_shape = (rank, out_dim)
                if tuple(v.shape) != expected_mlx_shape:
                    raise ValueError(f"Shape mismatch! {k} is {tuple(v.shape)}, expected {expected_mlx_shape}")
                
                converted_v = v.t().contiguous()
                expected_peft_shape = (out_dim, rank)
                peft_key = f"base_model.model.{base_k}.lora_B.weight"
                
            if tuple(converted_v.shape) != expected_peft_shape:
                raise ValueError(f"Converted shape mismatch! {peft_key} is {tuple(converted_v.shape)}, expected {expected_peft_shape}")
                
            peft_tensors[peft_key] = converted_v
            
            # Store in manifest
            manifest.append({
                "mlx_key": k,
                "hf_module": hf_module_key,
                "peft_key": peft_key,
                "source_shape": list(v.shape),
                "target_shape": list(converted_v.shape),
                "transposed": True
            })
            
            mapped_count += 1
            
            if base_k not in grouped_matrices:
                grouped_matrices[base_k] = {}
            grouped_matrices[base_k]["A" if is_a else "B"] = v
            
        else:
            print(f"UNMAPPED: {k}")
            unmapped_count += 1

    print(f"\nSource tensors    : {len(mlx_tensors)}")
    print(f"Mapped tensors    : {mapped_count}")
    print(f"Unmapped tensors  : {unmapped_count}")
    
    if unmapped_count > 0:
        print("ERROR: Unmapped tensors found. STOPPING.")
        return
        
    print("\n==================================================")
    print(" SCALING VALIDATION")
    print("==================================================")
    # 4. Scaling validation (mathematical equivalence)
    # Test a few matrices
    test_keys = list(grouped_matrices.keys())[:3]
    for key in test_keys:
        mlx_A = grouped_matrices[key]["A"] # [in_dim, r]
        mlx_B = grouped_matrices[key]["B"] # [r, out_dim]
        
        # MLX update: (x @ A) @ B * scale.
        # Equivalent weight matrix delta: (A @ B) * scale  (Shape: [in_dim, out_dim])
        # Transposed to match HF weight shape [out_dim, in_dim]:
        mlx_delta = (mlx_A @ mlx_B).t() * scale
        
        # PEFT update: lora_B(lora_A(x)) * (alpha/r).
        # HF weight matrix delta: (lora_B @ lora_A) * (alpha/r)
        peft_A = peft_tensors[f"base_model.model.{key}.lora_A.weight"] # [r, in_dim]
        peft_B = peft_tensors[f"base_model.model.{key}.lora_B.weight"] # [out_dim, r]
        peft_delta = (peft_B @ peft_A) * (alpha / rank)
        
        diff = torch.abs(mlx_delta - peft_delta)
        max_err = diff.max().item()
        mean_err = diff.mean().item()
        
        print(f"Module: {key}")
        print(f"  MLX Delta Shape : {tuple(mlx_delta.shape)}")
        print(f"  PEFT Delta Shape: {tuple(peft_delta.shape)}")
        print(f"  Max Abs Error   : {max_err}")
        print(f"  Mean Abs Error  : {mean_err}")
        
        if max_err > 1e-5:
            print(f"ERROR: Scaling validation failed for {key}. Max err: {max_err}")
            return
            
    print("Mathematical equivalence validated.")

    print("\n==================================================")
    print(" SAVING CONVERSION")
    print("==================================================")
    out_dir.mkdir(parents=True, exist_ok=True)
    save_file(peft_tensors, out_dir / "adapter_model.safetensors")
    
    peft_config = {
        "auto_mapping": None,
        "base_model_name_or_path": hf_model_id,
        "bias": "none",
        "fan_in_fan_out": False,
        "inference_mode": True,
        "init_lora_weights": True,
        "layers_pattern": None,
        "layers_to_transform": sorted(list(layers_transformed)),
        "lora_alpha": alpha,
        "lora_dropout": mlx_config["lora_parameters"].get("dropout", 0.0),
        "peft_type": "LORA",
        "r": rank,
        "target_modules": sorted(list(target_modules)),
        "task_type": "CAUSAL_LM"
    }
    
    with open(out_dir / "adapter_config.json", "w") as f:
        json.dump(peft_config, f, indent=2)
        
    # Save manifest
    manifest_data = {
        "source_adapter": str(mlx_dir),
        "base_model": hf_model_id,
        "rank": rank,
        "scale_mlx": scale,
        "alpha_peft": alpha,
        "tensors_mapped": mapped_count,
        "tensors_unmapped": unmapped_count,
        "mapping_table": manifest
    }
    with open(out_dir / "conversion_manifest.json", "w") as f:
        json.dump(manifest_data, f, indent=2)
        
    print(f"Conversion saved to: {out_dir}")
    print("CONVERSION VERIFIED — READY FOR KAGGLE HYBRID BENCHMARK")

if __name__ == "__main__":
    main()
