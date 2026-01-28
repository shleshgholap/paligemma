import numpy as np
import torch
from PIL import Image
from typing import List, Tuple, Union, Iterable, Optional

IMAGENET_MEAN = [0.5, 0.5, 0.5]
IMAGENET_STD = [0.5, 0.5, 0.5]


def rescale(image: np.ndarray, scale: float) -> np.ndarray:
    return (image * scale).astype(np.float32)


def resize(image: Image.Image, size: Tuple[int, int], resample: int = Image.Resampling.BICUBIC) -> Image.Image:
    return image.resize((size[1], size[0]), resample=resample)


def normalize(image: np.ndarray, mean: Iterable[float], std: Iterable[float]) -> np.ndarray:
    mean = np.array(mean, dtype=np.float32)
    std = np.array(std, dtype=np.float32)
    return (image - mean) / std


def process_images(
    images: List[Image.Image],
    size: Tuple[int, int],
    resample: int = Image.Resampling.BICUBIC,
    rescale_factor: float = 1 / 255.0,
    image_mean: Optional[List[float]] = None,
    image_std: Optional[List[float]] = None,
) -> np.ndarray:
    image_mean = image_mean if image_mean is not None else IMAGENET_MEAN
    image_std = image_std if image_std is not None else IMAGENET_STD
    processed = []
    for img in images:
        img = resize(img, size, resample)
        img = np.array(img)
        img = rescale(img, rescale_factor)
        img = normalize(img, image_mean, image_std)
        img = img.transpose(2, 0, 1)
        processed.append(img)
    return np.stack(processed, axis=0)


def add_image_tokens_to_prompt(prefix_prompt: str, bos_token: str, image_seq_len: int, image_token: str) -> str:
    return f"{image_token * image_seq_len}{bos_token}{prefix_prompt}\n"


class PaliGemmaProcessor:
    IMAGE_TOKEN = "<image>"

    def __init__(self, tokenizer, num_image_tokens: int, image_size: int):
        self.tokenizer = tokenizer
        self.image_seq_len = num_image_tokens
        self.image_size = image_size

        tokens_to_add = {"additional_special_tokens": [self.IMAGE_TOKEN]}
        tokenizer.add_special_tokens(tokens_to_add)

        extra_tokens = [f"<loc{i:04d}>" for i in range(1024)]
        extra_tokens += [f"<seg{i:03d}>" for i in range(128)]
        tokenizer.add_tokens(extra_tokens)

        self.image_token_id = tokenizer.convert_tokens_to_ids(self.IMAGE_TOKEN)
        tokenizer.add_bos_token = False
        tokenizer.add_eos_token = False

    def __call__(
        self,
        text: List[str],
        images: List[Image.Image],
        padding: str = "longest",
        truncation: bool = True,
    ) -> dict:
        pixel_values = process_images(images, size=(self.image_size, self.image_size))
        pixel_values = torch.tensor(pixel_values, dtype=torch.float32)

        input_strings = [
            add_image_tokens_to_prompt(prompt, self.tokenizer.bos_token, self.image_seq_len, self.IMAGE_TOKEN)
            for prompt in text
        ]

        inputs = self.tokenizer(input_strings, return_tensors="pt", padding=padding, truncation=truncation)

        return {"pixel_values": pixel_values, **inputs}
