# paligemma

## Note: This repository is still under construction

**paligemma** is a Python library implementing a multi-modal transformer model for conditional generation, inspired by the PaliGemma architecture. The codebase features custom PyTorch modules for vision and language processing, including components for image embedding, attention, and multi-modal fusion.

## Features

- **Multi-modal Model**: Integrates vision and language models for tasks requiring both image and text understanding.
- **Vision Model (Siglip)**: Implements transformer-based vision encoders, patch embeddings, and attention mechanisms for image processing.
- **Language Model**: Utilizes a causal language model for text generation and understanding.
- **Custom Tokenizer**: Adds special tokens for object detection (`<loc####>`) and segmentation (`<seg###>`), as well as support for image tokens in prompts.
- **Processing Utilities**: Includes functions for image normalization, resizing, and prompt construction in alignment with PaliGemma/Huggingface standards.

## Installation

Clone the repository:
```bash
git clone https://github.com/shleshgholap/paligemma.git
```

## Usage

Example: Constructing a processor and passing text/image data:
```python
from processing_paligemma import PaliGemmaProcessor
from PIL import Image

# Initialize processor
processor = PaliGemmaProcessor(tokenizer, num_image_tokens=32, image_size=224)

# Prepare inputs
texts = ["Describe the scene."]
images = [Image.open("example.jpg")]

# Process inputs
inputs = processor(texts, images)
# inputs: dict with pixel_values, input_ids, attention_mask
```

## Architecture Overview

- `gemma.py`: Defines `PaliGemmaForConditionalGeneration`, integrating vision and language models.
- `siglip.py`: Contains the Siglip vision transformer and related modules (embedding, attention, encoder).
- `processing_paligemma.py`: Utilities for image processing and prompt construction.

## License

This project is licensed under the MIT License. See [LICENSE](LICENSE) for details.

## Author

Developed by [Shlesh Gholap](https://github.com/shleshgholap)
