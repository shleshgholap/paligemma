# PaliGemma

End-to-end implementation of a multimodal PaliGemma inference pipeline from scratch.

## Features

- **SigLIP Vision Encoder**: Transformer-based vision encoder with patch embeddings
- **Gemma Language Decoder**: Causal language model with RoPE, RMSNorm, and GQA
- **Multimodal Fusion**: Custom projection layer connecting vision and language
- **Fast Autoregressive Decoding**: KV caching for efficient token-by-token generation
- **Flexible Sampling**: Temperature and top-p sampling support
- **CLI Inference**: Ready-to-use command-line interface

## Installation

```bash
git clone https://github.com/shleshgholap/paligemma.git
cd paligemma
pip install -r requirements.txt
```

## Quick Start

### CLI Usage

```bash
python -m src.cli \
    --model google/paligemma-3b-pt-224 \
    --image path/to/image.jpg \
    --prompt "Describe this image" \
    --max-new-tokens 100 \
    --temperature 0.8 \
    --top-p 0.9
```

### Python API

```python
from src import load_hf_model
from PIL import Image

model, processor = load_hf_model("google/paligemma-3b-pt-224", device="cuda")

image = Image.open("example.jpg")
inputs = processor(text=["Describe this image"], images=[image])

input_ids = inputs["input_ids"].to("cuda")
attention_mask = inputs["attention_mask"].to("cuda")
pixel_values = inputs["pixel_values"].to("cuda", dtype=model.vision_tower.vision_model.embeddings.patch_embedding.weight.dtype)

generated_ids = model.generate(
    input_ids=input_ids,
    pixel_values=pixel_values,
    attention_mask=attention_mask,
    max_new_tokens=100,
)

output = processor.tokenizer.decode(generated_ids[0], skip_special_tokens=True)
print(output)
```

## Architecture

```
┌─────────────┐
│   Image     │
│  (224×224)  │
└──────┬──────┘
       │
       ▼
┌─────────────────────┐
│  SigLIP Encoder     │
│  • Patch Embed      │
│  • 27 Layers        │
│  • Output: 256×1152 │
└──────┬──────────────┘
       │
       ▼
┌─────────────────────┐      ┌─────────────┐
│  Projection Layer   │      │ Text Prompt │
│  1152 → 2048        │      └──────┬──────┘
└──────┬──────────────┘             │
       │                             ▼
       │                   ┌─────────────────┐
       │                   │   Tokenizer     │
       │                   │ + <image> token │
       │                   └────────┬────────┘
       │                            │
       │                            ▼
       │                   ┌─────────────────┐
       │                   │ Text Embeddings │
       │                   └────────┬────────┘
       │                            │
       └───────────► ┌──────────────▼─────────────┐
                     │   Multimodal Fusion        │
                     │ (Replace <image> positions │
                     │  with vision embeddings)   │
                     └──────────┬─────────────────┘
                                │
                                ▼
                     ┌────────────────────────────┐
                     │    Gemma Decoder           │
                     │  • 18 Layers + RoPE        │
                     │  • Grouped Query Attention │
                     │  • KV Cache for speed      │
                     └──────────┬─────────────────┘
                                │
                                ▼
                     ┌────────────────────────────┐
                     │  Sampling (Top-p/Temp)     │
                     └──────────┬─────────────────┘
                                │
                                ▼
                          Generated Text
```

**Flow:**
1. **Vision Path**: Image → Patch Embeddings (224×224 → 256 tokens) → SigLIP Transformer → Projection to text space
2. **Text Path**: Prompt → Tokenization → Embeddings with `<image>` token placeholders
3. **Fusion**: Replace `<image>` token embeddings with projected vision features
4. **Generation**: Autoregressive decoding with KV cache, RoPE, and nucleus sampling

### Components

| File | Description |
|------|-------------|
| `src/config.py` | Configuration classes for SigLIP, Gemma, and PaliGemma |
| `src/siglip.py` | SigLIP vision transformer implementation |
| `src/gemma.py` | Gemma decoder with RoPE and KV cache support |
| `src/model.py` | Multimodal wrapper with vision-language fusion |
| `src/processing.py` | Image preprocessing and tokenization |
| `src/sampling.py` | Top-p and temperature sampling |
| `src/kv_cache.py` | Key-value cache for fast autoregressive decoding |
| `src/hf_loader.py` | HuggingFace model loading utilities |
| `src/cli.py` | Command-line interface |

## CLI Options

| Option | Default | Description |
|--------|---------|-------------|
| `--model` | `google/paligemma-3b-pt-224` | HuggingFace model ID |
| `--image` | (required) | Path to input image |
| `--prompt` | (required) | Text prompt |
| `--max-new-tokens` | 100 | Maximum tokens to generate |
| `--temperature` | 0.8 | Sampling temperature |
| `--top-p` | 0.9 | Top-p (nucleus) sampling |
| `--seed` | None | Random seed for reproducibility |
| `--device` | auto | Device (cuda/cpu) |
| `--dtype` | bfloat16 | Model precision |
| `--no-sample` | False | Use greedy decoding |

## License

MIT License. See [LICENSE](LICENSE) for details.

