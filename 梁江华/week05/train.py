from __future__ import annotations

import argparse
import math
import os
import random
from dataclasses import asdict
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


def load_corpus(corpus_path: Path) -> str:
    text = corpus_path.read_text(encoding="utf-8")
    if not text.strip():
        raise ValueError(f"Corpus file is empty: {corpus_path}")
    return text


def build_vocab(text: str) -> tuple[dict[str, int], list[str]]:
    chars = sorted(set(text))
    itos = ["<unk>"] + chars
    stoi = {ch: i for i, ch in enumerate(itos)}
    return stoi, itos


def encode(text: str, stoi: dict[str, int]) -> list[int]:
    unk_id = stoi["<unk>"]
    return [stoi.get(ch, unk_id) for ch in text]


def get_batch(data: torch.Tensor, block_size: int, batch_size: int, device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
    if data.numel() <= block_size:
        raise ValueError(
            f"Corpus is too small for block_size={block_size}. Need more than {block_size + 1} tokens, got {data.numel()}."
        )
    ix = torch.randint(0, data.size(0) - block_size - 1, (batch_size,))
    x = torch.stack([data[i : i + block_size] for i in ix]).to(device)
    y = torch.stack([data[i + 1 : i + block_size + 1] for i in ix]).to(device)
    return x, y


@torch.no_grad()
def estimate_loss(model: GPT2, train_data: torch.Tensor, val_data: torch.Tensor, block_size: int, batch_size: int, eval_iters: int, device: torch.device) -> dict[str, float]:
    out: dict[str, float] = {}
    model.eval()
    for split, data in (("train", train_data), ("val", val_data)):
        losses = []
        for _ in range(eval_iters):
            xb, yb = get_batch(data, block_size, batch_size, device)
            _, loss = model(xb, yb)
            losses.append(loss.item())
        out[split] = sum(losses) / len(losses)
    model.train()
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description="Train a tiny GPT-2 model on corpus.txt")
    parser.add_argument("--corpus_path", type=Path, default=Path(__file__).with_name("corpus.txt"))
    parser.add_argument("--output_path", type=Path, default=Path(__file__).with_name("gpt2_min.pt"))
    parser.add_argument("--block_size", type=int, default=128)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--max_steps", type=int, default=2000)
    parser.add_argument("--eval_interval", type=int, default=200)
    parser.add_argument("--eval_iters", type=int, default=50)
    parser.add_argument("--learning_rate", type=float, default=3e-4)
    parser.add_argument("--weight_decay", type=float, default=0.1)
    parser.add_argument("--n_layer", type=int, default=4)
    parser.add_argument("--n_head", type=int, default=4)
    parser.add_argument("--n_embd", type=int, default=128)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    set_seed(args.seed)
    device = torch.device(args.device)

    text = load_corpus(args.corpus_path)
    stoi, itos = build_vocab(text)
    data = torch.tensor(encode(text, stoi), dtype=torch.long)

    split = int(0.9 * len(data))
    train_data = data[:split]
    val_data = data[split:]
    if len(val_data) <= args.block_size:
        raise ValueError("Validation split is too small for the chosen block_size. Lower block_size or use a larger corpus.")

    config = GPTConfig(
        vocab_size=len(itos),
        block_size=args.block_size,
        n_layer=args.n_layer,
        n_head=args.n_head,
        n_embd=args.n_embd,
        dropout=args.dropout,
    )
    model = GPT2(config).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)

    print(f"Loaded corpus from: {args.corpus_path}")
    print(f"Corpus characters: {len(text)}")
    print(f"Vocab size: {len(itos)}")
    print(f"Device: {device}")
    print(f"Model config: {config}")

    if args.max_steps == 0:
        checkpoint = {
            "model_state_dict": model.state_dict(),
            "config": asdict(config),
            "stoi": stoi,
            "itos": itos,
            "corpus_path": str(args.corpus_path),
            "seed": args.seed,
        }
        args.output_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(checkpoint, args.output_path)
        print(f"Saved checkpoint to {args.output_path}")
        return

    for step in range(1, args.max_steps + 1):
        xb, yb = get_batch(train_data, args.block_size, args.batch_size, device)
        logits, loss = model(xb, yb)

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        if step % args.eval_interval == 0 or step == 1 or step == args.max_steps:
            losses = estimate_loss(model, train_data, val_data, args.block_size, args.batch_size, args.eval_iters, device)
            print(
                f"step {step:5d}/{args.max_steps} | train loss {losses['train']:.4f} | val loss {losses['val']:.4f}"
            )

    checkpoint = {
        "model_state_dict": model.state_dict(),
        "config": asdict(config),
        "stoi": stoi,
        "itos": itos,
        "corpus_path": str(args.corpus_path),
        "seed": args.seed,
        "train_args": vars(args),
    }
    args.output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(checkpoint, args.output_path)
    print(f"Saved checkpoint to {args.output_path}")


if __name__ == "__main__":
    main()


