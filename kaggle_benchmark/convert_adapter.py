import json
import torch
import os
import argparse
from safetensors.torch import load_file, save_file

def convert(mlx_adapter_dir, output_dir):
    print(f"Loading MLX adapter from: {mlx_adapter_dir}")
    
    # 1. Load config
    with open(os.path.join(mlx_adapter_dir, "adapter_config.json")) as f:
        mlx_config = json.load(f)
        
    rank = mlx_config["lora_parameters"]["rank"]
    scale = mlx_config["lora_parameters"]["scale"]
    alpha = scale * rank
    print(f"MLX Config -> Rank: {rank}, Scale: {scale} (PEFT Alpha: {alpha})")
    
    # 2. Load Safetensors
    mlx_tensors = load_file(os.path.join(mlx_adapter_dir, "adapters.safetensors"))
    print(f"Loaded {len(mlx_tensors)} tensors from MLX.")
    
    # 3. Convert tensors
    peft_tensors = {}
    converted_count = 0
    target_modules = set()
    layers_transformed = set()
    
    for k, v in mlx_tensors.items():
        # Example key: model.layers.20.self_attn.q_proj.lora_a
        if ".lora_a" in k:
            base_k = k.replace(".lora_a", "")
            layer_num = int(base_k.split(".layers.")[1].split(".")[0])
            module_name = base_k.split(".")[-1]
            
            target_modules.add(module_name)
            layers_transformed.add(layer_num)
            
            # In MLX: x @ W. x is [..., in_dim], lora_a is [in_dim, rank].
            # PEFT lora_A is nn.Linear(in_dim, rank, bias=False), weight is [rank, in_dim]
            # So we must transpose MLX lora_a.
            peft_key_a = f"base_model.model.{base_k}.lora_A.weight"
            peft_tensors[peft_key_a] = v.t().contiguous()
            converted_count += 1
            
        elif ".lora_b" in k:
            base_k = k.replace(".lora_b", "")
            # In MLX: lora_b is [rank, out_dim].
            # PEFT lora_B is nn.Linear(rank, out_dim, bias=False), weight is [out_dim, rank]
            # So we must transpose MLX lora_b.
            peft_key_b = f"base_model.model.{base_k}.lora_B.weight"
            peft_tensors[peft_key_b] = v.t().contiguous()
            converted_count += 1
            
    print(f"Converted {converted_count} tensors.")
    print(f"Target Modules: {target_modules}")
    print(f"Layers Transformed: {sorted(list(layers_transformed))}")
    
    # 4. Save PEFT Safetensors
    os.makedirs(output_dir, exist_ok=True)
    save_file(peft_tensors, os.path.join(output_dir, "adapter_model.safetensors"))
    
    # 5. Save PEFT Config
    peft_config = {
        "auto_mapping": None,
        "base_model_name_or_path": "Qwen/Qwen2.5-1.5B-Instruct",
        "bias": "none",
        "fan_in_fan_out": False,
        "inference_mode": True,
        "init_lora_weights": True,
        "layers_pattern": None,
        "layers_to_transform": list(layers_transformed),
        "lora_alpha": alpha,
        "lora_dropout": mlx_config["lora_parameters"].get("dropout", 0.0),
        "megatron_config": None,
        "megatron_core": "megatron.core",
        "modules_to_save": None,
        "peft_type": "LORA",
        "r": rank,
        "revision": None,
        "target_modules": list(target_modules),
        "task_type": "CAUSAL_LM"
    }
    with open(os.path.join(output_dir, "adapter_config.json"), "w") as f:
        json.dump(peft_config, f, indent=2)
    print(f"Saved PEFT adapter to {output_dir}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--mlx_adapter", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    convert(args.mlx_adapter, args.output)
