"""
字符级语言模型训练脚本，基于transformer实现。使用指定的文本文件作为语料，训练一个字符级语言模型，并完成文本生成任务。
"""

import math
import argparse
import glob
import os
import random
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader


# ─────────────────────────── 数据 ───────────────────────────

def load_corpus(pattern="*.txt"):
    """加载匹配模式的所有文本文件，并将它们拼接成一个字符串。"""
    texts = []
    for path in glob.glob(pattern):
        with open(path, encoding="utf-8", errors="ignore") as f:
            texts.append(f.read())
    return "".join(texts)


def build_vocab(text):
    """根据语料构建字符级词表：字符->索引和索引->字符。"""
    chars = sorted(set(text))
    char2idx = {c: i for i, c in enumerate(chars)}
    idx2char = {i: c for c, i in char2idx.items()}
    return char2idx, idx2char


class CharDataset(Dataset):
    """字符级数据集，用于训练语言模型。"""
    def __init__(self, text, char2idx, seq_len):
        self.seq_len = seq_len
        ids = [char2idx[c] for c in text if c in char2idx]
        self.data = torch.tensor(ids, dtype=torch.long)

    def __len__(self):
        # 每个样本包含 seq_len 个输入字符和 seq_len 个目标字符
        return max(0, len(self.data) - self.seq_len)

    def __getitem__(self, idx):
        x = self.data[idx: idx + self.seq_len]
        y = self.data[idx + 1: idx + self.seq_len + 1]
        return x, y


# ─────────────────────────── 模型 ───────────────────────────

class LM(nn.Module):
    """基于 Transformer 的字符级语言模型。"""
    def __init__(
        self,
        vocab_size,
        embed_dim,
        num_heads,
        ff_dim,
        num_layers,
        dropout,
        max_len=1024,
    ):
        super().__init__()
        # 词嵌入层，将字符索引映射到向量空间
        self.embed = nn.Embedding(vocab_size, embed_dim)
        # 位置嵌入，用于编码序列位置
        self.pos_embed = nn.Embedding(max_len, embed_dim)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=ff_dim,
            dropout=dropout,
            batch_first=True,
        )
        # Transformer 编码器由多个编码器层堆叠而成
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.drop = nn.Dropout(dropout)
        # 输出层，将 Transformer 输出映射到词表维度
        self.fc = nn.Linear(embed_dim, vocab_size)
        self.max_len = max_len

    def _generate_square_subsequent_mask(self, sz):
        """生成自回归的因果掩码，禁止模型看到未来位置。"""
        mask = torch.triu(torch.full((sz, sz), float("-inf"), device=self.embed.weight.device), diagonal=1)
        return mask

    def forward(self, x):
        seq_len = x.size(1)
        if seq_len > self.max_len:
            raise ValueError(f"Sequence length {seq_len} exceeds max_len {self.max_len}")

        # 位置编码加到词嵌入上，形成序列表示
        positions = torch.arange(seq_len, device=x.device).unsqueeze(0)
        e = self.embed(x) + self.pos_embed(positions)
        e = self.drop(e)
        # 使用因果掩码让每个位置只能看到前面的 token
        mask = self._generate_square_subsequent_mask(seq_len)
        out = self.transformer(e, mask=mask)
        logits = self.fc(self.drop(out))
        return logits


# ─────────────────────────── 训练 / 评估 ───────────────────────────

def run_epoch(model, loader, criterion, optimizer, device, train=True):
    """运行一个训练或验证 epoch，并返回平均损失与困惑度。"""
    model.train(train)
    total_loss = 0.0
    total_tokens = 0

    for x, y in loader:
        x, y = x.to(device), y.to(device)
        logits = model(x)
        loss = criterion(logits.reshape(-1, logits.size(-1)), y.reshape(-1))

        if train:
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        total_loss += loss.item() * y.numel()
        total_tokens += y.numel()

    avg_loss = total_loss / total_tokens
    ppl = math.exp(avg_loss)
    return avg_loss, ppl

def generate_text(model, char2idx, idx2char, seq_len, device, start_text="我是"):
    """使用训练好的模型生成文本，给定一个起始文本和生成长度。"""
    model.eval()
    input_ids = torch.tensor([char2idx[c] for c in start_text if c in char2idx], dtype=torch.long).unsqueeze(0).to(device)
    generated = start_text

    with torch.no_grad():
        for _ in range(seq_len):
            logits = model(input_ids)
            next_id = torch.argmax(logits[:, -1, :], dim=-1).item()
            next_char = idx2char[next_id]
            generated += next_char
            input_ids = torch.cat([input_ids, torch.tensor([[next_id]], device=device)], dim=1)

    return generated

# ─────────────────────────── 主函数 ───────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs",     type=int,   default=20)
    parser.add_argument("--seq_len",    type=int,   default=64)
    parser.add_argument("--batch_size", type=int,   default=128)
    parser.add_argument("--embed_dim",  type=int,   default=128)
    parser.add_argument("--num_heads",  type=int,   default=8)
    parser.add_argument("--ff_dim",     type=int,   default=512)
    parser.add_argument("--num_layers", type=int,   default=6)
    parser.add_argument("--dropout",    type=float, default=0.3)
    parser.add_argument("--lr",         type=float, default=1e-3)
    parser.add_argument("--val_ratio",  type=float, default=0.05)
    parser.add_argument("--save",       default="best_model.pt")
    args = parser.parse_args()

    # 选择运行设备，优先使用 GPU
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device: {device}  model: TRANSFORMER")

    # 数据准备：读取文本、构建词表、分割训练/验证集
    script_dir = os.path.dirname(os.path.abspath(__file__))
    corpus_path = os.path.join(script_dir, 'corpus.txt')
    text = load_corpus(corpus_path)
    if not text:
        raise FileNotFoundError("未找到任何 .txt 文件，请确认路径正确。")
    print(f"语料字符数: {len(text):,}")

    char2idx, idx2char = build_vocab(text)
    vocab_size = len(char2idx)
    print(f"词表大小: {vocab_size}")

    lines = text.splitlines()
    random.shuffle(lines)
    split = int(len(lines) * (1 - args.val_ratio))
    train_text = "\n".join(lines[:split])
    val_text   = "\n".join(lines[split:])

    train_ds = CharDataset(train_text, char2idx, args.seq_len)
    val_ds   = CharDataset(val_text,   char2idx, args.seq_len)

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, drop_last=True)
    val_loader   = DataLoader(val_ds,   batch_size=args.batch_size, shuffle=True, drop_last=True)

    # 模型
    model = LM(
        vocab_size=vocab_size,
        embed_dim=args.embed_dim,
        num_heads=args.num_heads,
        ff_dim=args.ff_dim,
        num_layers=args.num_layers,
        dropout=args.dropout,
    ).to(device)

    total_params = sum(p.numel() for p in model.parameters())
    print(f"模型参数量: {total_params:,}")

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    best_val_ppl = float("inf")

    print(f"\n{'Epoch':>6}  {'Train Loss':>10}  {'Train PPL':>10}  {'Val Loss':>10}  {'Val PPL':>10}")
    print("-" * 56)

    for epoch in range(1, args.epochs + 1):
        tr_loss, tr_ppl = run_epoch(model, train_loader, criterion, optimizer, device, train=True)
        with torch.no_grad():
            va_loss, va_ppl = run_epoch(model, val_loader, criterion, optimizer, device, train=False)

        marker = "  *" if va_ppl < best_val_ppl else ""
        if va_ppl < best_val_ppl:
            best_val_ppl = va_ppl
            torch.save({
                "model_state": model.state_dict(),
                "char2idx": char2idx,
                "idx2char": idx2char,
                "args": vars(args),
            }, args.save)

        print(f"{epoch:>6}  {tr_loss:>10.4f}  {tr_ppl:>10.2f}  {va_loss:>10.4f}  {va_ppl:>10.2f}{marker}")

    print(f"\n训练完成。最佳验证 PPL: {best_val_ppl:.2f}  已保存至 {args.save}")

    # 使用训练好的模型进行文本生成
    model.eval()
    with torch.no_grad():
        generated_text = generate_text(model, char2idx, idx2char, args.seq_len, device)

    print(f"\n生成的文本:\n{generated_text}")

if __name__ == "__main__":
    main()
