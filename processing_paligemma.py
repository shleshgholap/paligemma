from typing import List, Dict, Optional, Union, Tuple, Iterable
import numpy as np
from PIL import Image
import torch

IMAGENET_STANDARD_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STANDARD_STD = [0.229, 0.224, 0.225]

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
            resample=Image.Resampling.BICUBIC,
            rescale_factor = 1/255.0,
            image_mean = IMAGENET_STANDARD_MEAN,
            image_std = IMAGENET_STANDARD_STD
        )

        # Convert into a single batched torch tensor from a list of numpy images
        pixel_values = torch.tensor(np.stack(pixel_values, axis=0))

    def process_images():

        pass
