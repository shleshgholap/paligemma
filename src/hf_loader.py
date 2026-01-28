import json
import torch
from typing import Tuple
from safetensors import safe_open
from transformers import AutoTokenizer
from huggingface_hub import hf_hub_download

from src.config import PaliGemmaConfig, SiglipVisionConfig, GemmaConfig
from src.model import PaliGemmaForConditionalGeneration
from src.processing import PaliGemmaProcessor


def load_hf_model(
    model_id: str,
    device: str = "cpu",
    dtype: torch.dtype = torch.bfloat16,
) -> Tuple[PaliGemmaForConditionalGeneration, PaliGemmaProcessor]:
    config_path = hf_hub_download(repo_id=model_id, filename="config.json")
    with open(config_path, "r") as f:
        config_dict = json.load(f)

    vision_config = SiglipVisionConfig(**config_dict.get("vision_config", {}))
    text_config = GemmaConfig(**config_dict.get("text_config", {}))
    
    config = PaliGemmaConfig(
        vision_config=config_dict.get("vision_config", {}),
        text_config=config_dict.get("text_config", {}),
        projection_dim=config_dict.get("projection_dim", text_config.hidden_size),
        hidden_size=config_dict.get("hidden_size", text_config.hidden_size),
        pad_token_id=config_dict.get("pad_token_id", 0),
        image_token_index=config_dict.get("image_token_index", 257152),
        vocab_size=config_dict.get("vocab_size", text_config.vocab_size),
    )

    model = PaliGemmaForConditionalGeneration(config).to(dtype=dtype)

    safetensors_files = _get_safetensors_files(model_id)
    state_dict = {}
    for sf_file in safetensors_files:
        with safe_open(sf_file, framework="pt", device="cpu") as f:
            for key in f.keys():
                state_dict[key] = f.get_tensor(key)

    state_dict = _remap_state_dict(state_dict)
    model.load_state_dict(state_dict, strict=False)
    model.tie_weights()
    model = model.to(device)
    model.eval()

    tokenizer = AutoTokenizer.from_pretrained(model_id)
    processor = PaliGemmaProcessor(
        tokenizer=tokenizer,
        num_image_tokens=config.num_image_tokens,
        image_size=config.vision_config.image_size,
    )

    return model, processor


def _get_safetensors_files(model_id: str):
    try:
        index_path = hf_hub_download(repo_id=model_id, filename="model.safetensors.index.json")
        with open(index_path, "r") as f:
            index = json.load(f)
        files = set(index["weight_map"].values())
        return [hf_hub_download(repo_id=model_id, filename=fn) for fn in files]
    except Exception:
        path = hf_hub_download(repo_id=model_id, filename="model.safetensors")
        return [path]


def _remap_state_dict(state_dict: dict) -> dict:
    new_state_dict = {}
    for key, value in state_dict.items():
        new_key = key
        if key.startswith("vision_tower.vision_model."):
            new_key = key.replace("vision_tower.vision_model.", "vision_tower.vision_model.")
        elif key.startswith("language_model."):
            new_key = key
        new_state_dict[new_key] = value
    return new_state_dict
