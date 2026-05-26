from __future__ import annotations

import argparse
import os
import random
from pathlib import Path

# Fix for duplicate OpenMP runtime on Windows (common with conda + CUDA envs)
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import torch

from gpt2_min import GPT2, GPTConfig


def set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def encode(prompt: str, stoi: dict[str, int]) -> list[int]:
    unk_id = stoi.get("<unk>", 0)
    return [stoi.get(ch, unk_id) for ch in prompt]


def decode(tokens: list[int], itos: list[str]) -> str:
    pieces: list[str] = []
    for token in tokens:
        if 0 <= token < len(itos):
            pieces.append(itos[token])
        else:
            pieces.append("<unk>")
    return "".join(pieces)


def load_checkpoint(model_path: Path, device: torch.device):
    checkpoint = torch.load(model_path, map_location=device)
    if "config" not in checkpoint or "model_state_dict" not in checkpoint:
        raise ValueError(f"Invalid checkpoint format: {model_path}")
    return checkpoint


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate text from a trained GPT-2 mini checkpoint")
    parser.add_argument("--model_path", type=Path, default=Path(__file__).with_name("gpt2_min.pt"))
    parser.add_argument("--prompt", type=str, default="")
    parser.add_argument("--max_new_tokens", type=int, default=100)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--top_k", type=int, default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    set_seed(args.seed)
    device = torch.device(args.device)
    checkpoint = load_checkpoint(args.model_path, device)

    config = GPTConfig(**checkpoint["config"])
    model = GPT2(config).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    stoi = checkpoint["stoi"]
    itos = checkpoint["itos"]

    prompt = args.prompt if args.prompt else input("Enter prompt: ").rstrip("\n")
    if not prompt:
        raise ValueError("Prompt cannot be empty.")

    idx = torch.tensor([encode(prompt, stoi)], dtype=torch.long, device=device)
    generated = model.generate(
        idx,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        top_k=args.top_k,
    )
    text = decode(generated[0].tolist(), itos)

    print("\n=== Generated Text ===")
    print(text)


if __name__ == "__main__":
    main()

