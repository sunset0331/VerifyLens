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
    detector: str = "retinaface"
    embedder: str = "arcface"
    similarity_threshold: float = 0.65


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


def load_config(path: Path = CONFIG_PATH) -> AppConfig:
    """Load and parse YAML config into typed dataclasses."""
    with open(path, "r") as f:
        raw = yaml.safe_load(f)

    cfg = AppConfig()
    if "model" in raw:
        m = raw["model"]
        if "vlm" in m:
            cfg.vlm = VLMConfig(**{k: v for k, v in m["vlm"].items()})
        if "face" in m:
            cfg.face = FaceConfig(**{k: v for k, v in m["face"].items()})
    if "training" in raw and "lora" in raw["training"]:
        cfg.lora = LoraConfig(**{k: v for k, v in raw["training"]["lora"].items()})
    if "api" in raw:
        cfg.api = APIConfig(**{k: v for k, v in raw["api"].items()})

    return cfg


# Singleton for import convenience
config = load_config()
