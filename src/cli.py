import argparse
import torch
from PIL import Image
from src.hf_loader import load_hf_model


def main():
    parser = argparse.ArgumentParser(description="PaliGemma inference CLI")
    parser.add_argument("--model", type=str, default="google/paligemma-3b-pt-224", help="HuggingFace model ID")
    parser.add_argument("--image", type=str, required=True, help="Path to input image")
    parser.add_argument("--prompt", type=str, required=True, help="Text prompt")
    parser.add_argument("--max-new-tokens", type=int, default=100, help="Maximum tokens to generate")
    parser.add_argument("--temperature", type=float, default=0.8, help="Sampling temperature")
    parser.add_argument("--top-p", type=float, default=0.9, help="Top-p sampling")
    parser.add_argument("--seed", type=int, default=None, help="Random seed")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu", help="Device")
    parser.add_argument("--dtype", type=str, default="bfloat16", choices=["float32", "float16", "bfloat16"], help="Model dtype")
    parser.add_argument("--no-sample", action="store_true", help="Use greedy decoding")
    args = parser.parse_args()

    dtype_map = {"float32": torch.float32, "float16": torch.float16, "bfloat16": torch.bfloat16}
    dtype = dtype_map[args.dtype]

    if args.seed is not None:
        torch.manual_seed(args.seed)

    print(f"Loading model {args.model}...")
    model, processor = load_hf_model(args.model, device=args.device, dtype=dtype)

    image = Image.open(args.image).convert("RGB")
    inputs = processor(text=[args.prompt], images=[image])

    input_ids = inputs["input_ids"].to(args.device)
    attention_mask = inputs["attention_mask"].to(args.device)
    pixel_values = inputs["pixel_values"].to(args.device, dtype=dtype)

    print("Generating...")
    generated_ids = model.generate(
        input_ids=input_ids,
        pixel_values=pixel_values,
        attention_mask=attention_mask,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        top_p=args.top_p,
        do_sample=not args.no_sample,
        eos_token_id=processor.tokenizer.eos_token_id,
    )

    generated_text = processor.tokenizer.decode(generated_ids[0], skip_special_tokens=True)
    print(f"\nOutput:\n{generated_text}")


if __name__ == "__main__":
    main()
