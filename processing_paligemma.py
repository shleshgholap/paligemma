from typing import List, Dict, Optional, Union, Tuple, Iterable
import numpy as np
from PIL import Image
import torch

IMAGENET_STANDARD_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STANDARD_STD = [0.229, 0.224, 0.225]


def rescale(
    image: np.ndarray, scale: float, dtype: np.dtype = np.float32
) -> np.ndarray:
    rescaled_image = image * scale
    rescaled_image = rescaled_image.astype(dtype)
    return rescaled_image


def resize(
    image: Image,
    size: Tuple[int, int],
    resample: Image.Resampling = None,
    reducing_gap: Optional[int] = None,
) -> np.ndarray:
    height, width = size
    resized_image = image.resize(
        (width, height), resample=resample, reducing_gap=reducing_gap
    )
    return resized_image


def normalize(
    image: np.ndarray,
    mean: Union[float, Iterable[float]],
    std: Union[float, Iterable[float]],
) -> np.ndarray:
    mean = np.array(mean, dtype=image.dtype)
    std = np.array(std, dtype=image.dtype)
    image = (image - mean) / std
    return image


def process_images(
        images: List[Image.Image],
        size: Tuple[int, int],
        resample: Image.Resampling = None,
        rescale_factor: float = None,
        image_mean: Optional[Union[float, List[float]]] = None,
        image_std: Optional[Union[float, List[float]]] = None
    ) -> List[np.ndarray]:
    
    height, width = size
    images = [resize(image=image, size=size, resample=resample) for image in images]
    images = [np.array(image) for image in images]
    images = [rescale(image, scale=rescale_factor) for image in images]
    images = [normalize(image=image, mean=image_mean, std=image_std) for image in images]
    images = [image.transpose(2, 0, 1) for image in images] # Convert to CxHxW format
    return images


def add_image_tokens_to_prompt(
    prefix_prompt: str,
    bos_token: str,
    image_seq_len: int,
    image_token: str
) -> str:
    """
    Quoting from the blog (https://huggingface.co/blog/paligemma#detailed-inference-process):
    The input text is tokenized normally.
    A <bos> token is added at the beginning, and an additional newline token is appended.
    This newline token is an essential part of the input prompt the model was trained with, so adding it explicityle ensures it's always there.
    The tokenized text is also prefixed with a fixed number of <image> tokens.
    NOTE: from the paper it looks like the '\n' should be tokenized separately, but in the HF implementation this is not done.
    ref to HF implementation: https://github.com/huggingface/transformers/blob/7f79a97399bb52aad8460e1da2f36577d5dccfed/src/transformers/models/paligemma/processing_paligemma.py#L55-L73
    """

    return f"{image_token * image_seq_len}{bos_token}{prefix_prompt}\n"


class PaliGemmaProcessor:

    IMAGE_TOKEN = "<image>"
    
    def __init__(self, tokenizer, num_image_tokens: int, image_size: int):
        super().__init__()

        self.image_seq_len = num_image_tokens
        self.image_size = image_size

        tokens_to_add = {"additional_special_tokens": [self.IMAGE_TOKEN]}
        tokenizer.add_special_tokens(tokens_to_add)

        EXTRA_TOKENS = {
            f"<loc{i:04d}>" for i in range(1024) # Tokens used for object detection (bounding boxes)
        }

        EXTRA_TOKENS += {
            f"<seg{i:03d}>" for i in range(128) # Tokens used for segmentation
        }

        tokenizer.add_tokens(EXTRA_TOKENS)
        self.image_token_id = tokenizer.convert_tokens_to_ids(self.IMAGE_TOKEN)

        # These will be added manually later
        tokenizer.add_eos_token = False
        tokenizer.add_eos_token = False

        self.tokenizer = tokenizer

    def __call__(
            self,
            text: List[str],
            images: List[Image.Image],
            padding: str = "longest",
            truncation: bool = True,
    ) -> dict:
        
        assert len(text) == 1 and len(images) == 1, f"Only single text and image inputs are supported. Received {len(images)} images and {len(text)} texts prompts."

        pixel_values = process_images(
            images, 
            size = (self.image_size, self.image_size),
            resample = Image.Resampling.BICUBIC,
            rescale_factor = 1/255.0,
            image_mean = IMAGENET_STANDARD_MEAN,
            image_std = IMAGENET_STANDARD_STD
        )

        # Convert into a single batched torch tensor from a list of numpy images
        pixel_values = torch.tensor(np.stack(pixel_values, axis=0))

        # Prepend a 'self.image_seq_length' number of image tokens to the prompt
        input_strings = [
            add_image_tokens_to_prompt(
                prefix_prompt = prompt,
                bos_token = self.tokenizer.bos_token,
                image_seq_len = self.image_seq_len,
                image_token = self.IMAGE_TOKEN,
            )
            for prompt in text
        ]

        # Return the input_ids and attention_mask as Pytorch tensors
        inputs = self.tokenizer(
            input_strings,
            return_tensors = 'pt',
            padding = padding,
            truncation = truncation,
        )

        return  {
            "pixel_values" : pixel_values,
            **inputs
        }



        
