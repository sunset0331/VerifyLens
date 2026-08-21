"""
utils/config.py — Central config loader for VerifyLens.
Reads configs/training_config.yaml and exposes typed config objects.
"""

import yaml
from pathlib import Path
from dataclasses import dataclass, field
from typing import List


ROOT = Path(__file__).resolve().parents[2]
CONFIG_PATH = ROOT / "configs" / "training_config.yaml"


@dataclass
class LoraConfig:
    r: int = 16
    lora_alpha: int = 32
    target_modules: List[str] = field(default_factory=lambda: ["q_proj", "v_proj"])
    lora_dropout: float = 0.05
    bias: str = "none"
    task_type: str = "CAUSAL_LM"


@dataclass
class VLMConfig:
    name: str = "Qwen/Qwen2-VL-2B-Instruct"
    load_in_4bit: bool = True
    bnb_4bit_compute_dtype: str = "float16"
    bnb_4bit_quant_type: str = "nf4"
    use_nested_quant: bool = False


@dataclass
class FaceConfig:
    # Note: actual implementation uses MTCNN + InceptionResnetV1 (facenet-pytorch).
    # These fields are stored for documentation; the face code does not read them yet.
    detector: str = "mtcnn"
    embedder: str = "inceptionresnetv1"
    similarity_threshold: float = 0.65


@dataclass
class ExtractionConfig:
    """
    Controls which document extraction approach is active.

    mode : str
        'ocr_llm' — Approach 1: image → PaddleOCR → Qwen2.5-1.5B text LLM → JSON
        'vlm'     — Approach 2: image → Qwen2.5-VL-2B (MLX) → JSON
        'hybrid'  — Approach 3: VLM for doc_type + OCR+LoRA for identity fields
    """
    mode: str = "ocr_llm"
    # ── VLM (Approach 2) settings ────────────────────────────────────────────
    vlm_model: str = "mlx-community/Qwen2.5-VL-3B-Instruct-4bit"
    vlm_max_new_tokens: int = 256
    vlm_temperature: float = 0.0
    vlm_resize_max_px: int = 1120
    # ── OCR + LLM (Approach 1) settings ─────────────────────────────────────
    ocr_llm_model: str = "mlx-community/Qwen2.5-1.5B-Instruct-4bit"
    ocr_llm_adapter_path: str = "checkpoints/verifylens-adapter"
    ocr_llm_max_tokens: int = 128


@dataclass
class APIConfig:
    host: str = "0.0.0.0"
    port: int = 8000
    max_image_size_mb: int = 10
    confidence_threshold: float = 0.70


@dataclass
class AppConfig:
    vlm: VLMConfig = field(default_factory=VLMConfig)
    face: FaceConfig = field(default_factory=FaceConfig)
    lora: LoraConfig = field(default_factory=LoraConfig)
    api: APIConfig = field(default_factory=APIConfig)
    extraction: ExtractionConfig = field(default_factory=ExtractionConfig)


def _filter(dc_class, d: dict) -> dict:
    """Return only keys that are valid fields of a dataclass."""
    import dataclasses
    valid = {f.name for f in dataclasses.fields(dc_class)}
    return {k: v for k, v in d.items() if k in valid}


def load_config(path: Path = CONFIG_PATH) -> AppConfig:
    """Load and parse YAML config into typed dataclasses."""
    with open(path, "r") as f:
        raw = yaml.safe_load(f)

    cfg = AppConfig()
    if "model" in raw:
        m = raw["model"]
        if "vlm" in m:
            cfg.vlm = VLMConfig(**_filter(VLMConfig, m["vlm"]))
        if "face" in m:
            cfg.face = FaceConfig(**_filter(FaceConfig, m["face"]))
    if "training" in raw and "lora" in raw["training"]:
        cfg.lora = LoraConfig(**_filter(LoraConfig, raw["training"]["lora"]))
    if "api" in raw:
        cfg.api = APIConfig(**_filter(APIConfig, raw["api"]))
    if "extraction" in raw:
        cfg.extraction = ExtractionConfig(**_filter(ExtractionConfig, raw["extraction"]))

    return cfg



# Singleton for import convenience
config = load_config()
