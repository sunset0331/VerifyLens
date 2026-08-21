# VerifyLens Kaggle Benchmark — Models README

This directory does **not** contain model weights.
Models are downloaded automatically from HuggingFace when the benchmark runs.

---

## Models Used

### VLM (Approach 2) — `--model vlm`

| Property | Value |
|---|---|
| Model ID | `Qwen/Qwen2.5-VL-3B-Instruct` |
| Precision | BF16 (bfloat16) |
| Runtime | HuggingFace Transformers + CUDA |
| Download size | ~7 GB |
| GPU memory required | ~8–10 GB (BF16 inference) |
| Minimum GPU | NVIDIA T4 (16 GB) |
| HuggingFace page | https://huggingface.co/Qwen/Qwen2.5-VL-3B-Instruct |

**Production equivalent**: `mlx-community/Qwen2.5-VL-3B-Instruct-4bit` via `mlx-vlm`

> ⚠️ **These are the same model family and weights but are NOT bit-identical.**
> Production runs 4-bit MLX quantization on Apple Silicon.
> Kaggle runs BF16 full precision via HuggingFace Transformers on NVIDIA GPU.

---

### LLM Base (Approach 1, no LoRA) — `--model base`

| Property | Value |
|---|---|
| Model ID | `Qwen/Qwen2.5-1.5B-Instruct` |
| Precision | BF16 (bfloat16) |
| Runtime | HuggingFace Transformers + CUDA |
| Download size | ~3 GB |
| GPU memory required | ~4 GB (BF16 inference) |
| HuggingFace page | https://huggingface.co/Qwen/Qwen2.5-1.5B-Instruct |

**Production equivalent**: `mlx-community/Qwen2.5-1.5B-Instruct-4bit` + LoRA adapter via `mlx-lm`

> ⚠️ **Base model only — LoRA adapter not loaded.** See LoRA section below.

---

### LoRA Adapter — `--model lora`

The production LoRA adapter (`checkpoints/verifylens-adapter/`) **cannot be evaluated on Kaggle** without a conversion step.

**Reason**: The adapter was trained with `mlx-lm` against an MLX 4-bit quantized model.
HuggingFace PEFT expects a different layer naming convention and a full-precision base model.

**What would be needed for conversion**:

1. Load the MLX adapter weights (`adapters.safetensors`)
2. Map MLX LoRA layer names → HuggingFace PEFT naming convention
   - MLX: `model.layers.N.self_attn.q_proj.lora_a` (approximate)
   - PEFT: `base_model.model.model.layers.N.self_attn.q_proj.lora_A.weight`
3. Convert any quantized tensors to float32/BF16
4. Create a `adapter_config.json` in PEFT format
5. Save using `peft.PeftModel.save_pretrained()`

**Adapter details**:

| Property | Value |
|---|---|
| Adapter format | MLX LoRA (`mlx-lm`) |
| Base model | `mlx-community/Qwen2.5-1.5B-Instruct-4bit` |
| Rank | 8 |
| Scale | 20.0 |
| Trained iterations | 600 |
| Target modules | `q_proj`, `k_proj`, `v_proj`, `o_proj` |
| Adapter file | `checkpoints/verifylens-adapter/adapters.safetensors` |
| Adapter size | ~10 MB |

Running `python run_benchmark.py --model lora` will print this information and exit — it does **not** silently use the base model.

---

## Kaggle Cache

After the first run, HuggingFace models are cached in:

```
/root/.cache/huggingface/hub/
```

or the path specified by `HF_HOME` / `TRANSFORMERS_CACHE`.

Subsequent runs in the same Kaggle session will NOT re-download the models.

---

## Expected GPU Memory

| Mode | Peak allocated (approx.) | GPU recommendation |
|---|---|---|
| VLM (BF16) | 8–10 GB | T4 (16 GB), P100 (16 GB), or better |
| Base LLM (BF16) | 4–5 GB | T4 or better |
| Both simultaneously | NOT SUPPORTED | — |
