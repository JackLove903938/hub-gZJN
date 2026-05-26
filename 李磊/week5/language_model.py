"""
使用 decoder-only Transformer 构建字符级语言模型，
实现输入任意一个字符串后预测下一个字符。
"""

import math
import argparse
import glob
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader


# ─────────────────────────── 数据 ───────────────────────────
def load_corpus(pattern="*.txt"):
    texts = []
    for path in glob.glob(pattern):
        with open(path, encoding="utf-8", errors="ignore") as f:
            texts.append(f.read())
    return "".join(texts)


def build_vocab(text):
    chars = ["<UNK>"] + sorted(set(text))
    char2idx = {c: i for i, c in enumerate(chars)}
    idx2char = {i: c for c, i in char2idx.items()}
    return char2idx, idx2char


class CharDataset(Dataset):
    def __init__(self, text, char2idx, seq_len):
        self.seq_len = seq_len
        unk_id = char2idx["<UNK>"]
        ids = [char2idx.get(c, unk_id) for c in text]
        self.data = torch.tensor(ids, dtype=torch.long)

    def __len__(self):
        return max(0, len(self.data) - self.seq_len)

    def __getitem__(self, idx):
        x = self.data[idx: idx + self.seq_len]
        y = self.data[idx + 1: idx + self.seq_len + 1]
        return x, y


# ─────────────────────────── 模型 ───────────────────────────
class LM(nn.Module):
    def __init__(self, vocab_size, embed_dim, hidden_dim, num_layers, dropout, max_seq_len, nhead=8):
        super().__init__()
        # 字符编码
        self.embed = nn.Embedding(vocab_size, embed_dim)
        # 位置编码
        self.pos_embed = nn.Embedding(max_seq_len, embed_dim)

        """"transform-decoder使用案例"""
        # 声明decoder定义模版
        decoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=nhead,
            dim_feedforward=hidden_dim,
            dropout=dropout,
            batch_first=True,
            activation="gelu",
        )
        # 声明decoder
        # decoder_layer：定义模板
        # num_layers：堆叠层数
        self.decoder = nn.TransformerEncoder(decoder_layer, num_layers=num_layers)
        self.drop = nn.Dropout(dropout)
        self.fc = nn.Linear(embed_dim, vocab_size)

    def forward(self, x):
        # x:(B, seq_len)：表示B个样本，每个样本有seq_len个字符索引
        # logits:(B, seq_len, vocab_size)：每个位置输出下一个字符的预测分布
        batch_size, seq_len = x.size()
        positions = torch.arange(seq_len, device=x.device).unsqueeze(0).expand(batch_size, seq_len) # (B, seq_len)
        chars = self.embed(x)                   # (B, seq_len)-->(B, seq_len, embed_dim)
        positions = self.pos_embed(positions)   # (B, seq_len)-->(B, seq_len, embed_dim)
        h = chars + positions                   # 字符编码 + 位置编码：(B, seq_len, embed_dim)

        """"transform-decoder使用案例"""
        # mask模版：生成一个上三角矩阵，对角线及其以下位置为1，其余位置为0
        h = self.drop(h)
        causal_mask = torch.triu(
            torch.ones(seq_len, seq_len, device=x.device, dtype=torch.bool),
            diagonal=1,
        )
        out = self.decoder(h, mask=causal_mask)         # (B, seq_len, embed_dim)

        logits = self.fc(self.drop(out))                # (B, seq_len, embed_dim)-->(B, seq_len, vocab_size)
        return logits


# ─────────────────────────── 训练 / 评估 ───────────────────────────
def run_epoch(model, loader, criterion, optimizer, device, train=True):
    model.train(train)
    total_loss = 0.0
    total_tokens = 0

    for x, y in loader:
        x, y = x.to(device), y.to(device)
        logits = model(x)

        # 计算损失
        # logits.reshape(-1, logits.size(-1))：(B, seq_len, vocab_size)-->(B*seq_len, vocab_size)
        # y.reshape(-1)：(B, seq_len)-->(B*seq_len)
        loss = criterion(logits.reshape(-1, logits.size(-1)), y.reshape(-1))

        if train:
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        total_loss += loss.item() * y.numel()
        total_tokens += y.numel()

    avg_loss = total_loss / total_tokens
    ppl = math.exp(avg_loss)    # 困惑度
    return avg_loss, ppl


# ─────────────────────────── 主函数 ───────────────────────────
def train_model():
    """"参数准备"""
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs",      type=int,   default=20)
    parser.add_argument("--seq_len",     type=int,   default=64)
    parser.add_argument("--max_seq_len", type=int,   default=256)
    parser.add_argument("--batch_size",  type=int,   default=128)
    parser.add_argument("--embed_dim",   type=int,   default=128)
    parser.add_argument("--hidden_dim",  type=int,   default=256)
    parser.add_argument("--num_layers",  type=int,   default=2)
    parser.add_argument("--dropout",     type=float, default=0.3)
    parser.add_argument("--lr",          type=float, default=1e-3)
    parser.add_argument("--val_ratio",   type=float, default=0.05)
    parser.add_argument("--corpus",      default="corpus.txt")
    parser.add_argument("--save",        default="best_model.pt")
    args = parser.parse_args()

    if args.embed_dim % 8 != 0:
        raise ValueError(f"embed_dim={args.embed_dim} 必须能被 nhead=8 整除")
    if args.seq_len > args.max_seq_len:
        raise ValueError(
            f"seq_len={args.seq_len} 不能大于 max_seq_len={args.max_seq_len}"
        )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device: {device}  model: TRANSFORMER_DECODER")


    """"数据准备"""
    text = load_corpus(args.corpus)
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


    """模型/优化器/损失函数准备"""
    # to(device):移动到GPU/CPU
    model = LM(
        vocab_size=vocab_size,
        embed_dim=args.embed_dim,
        hidden_dim=args.hidden_dim,
        num_layers=args.num_layers,
        dropout=args.dropout,
        max_seq_len=args.max_seq_len,
        nhead=8,
    ).to(device)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"模型参数量: {total_params:,}")

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    best_val_ppl = float("inf")
    print(f"\n{'Epoch':>6}  {'Train Loss':>10}  {'Train PPL':>10}  {'Val Loss':>10}  {'Val PPL':>10}")
    print("-" * 56)


    """模型训练"""
    for epoch in range(1, args.epochs + 1):
        # 一轮训练
        # tr_loss：损失值
        # tr_ppl：困惑度
        tr_loss, tr_ppl = run_epoch(model, train_loader, criterion, optimizer, device, train=True)

        # 一轮验证
        # va_loss：损失值
        # va_ppl：困惑度
        with torch.no_grad():
            va_loss, va_ppl = run_epoch(model, val_loader, criterion, optimizer, device, train=False)

        # 保留va_ppl最小的epoch
        if va_ppl < best_val_ppl:
            best_val_ppl = va_ppl
            torch.save({
                "model_state": model.state_dict(),
                "char2idx": char2idx,
                "idx2char": idx2char,
                "args": vars(args),
            }, args.save)
        print(f"{epoch:>6}  {tr_loss:>10.4f}  {tr_ppl:>10.2f}  {va_loss:>10.4f}  {va_ppl:>10.2f}")

    print(f"\n训练完成。最佳验证 PPL: {best_val_ppl:.2f}  已保存至 {args.save}")


def predict_top_k(model_path="best_model.pt", top_k=5):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint = torch.load(model_path, map_location=device)

    saved_args = checkpoint["args"]
    char2idx = checkpoint["char2idx"]
    idx2char = checkpoint["idx2char"]

    model = LM(
        vocab_size=len(char2idx),
        embed_dim=saved_args["embed_dim"],
        hidden_dim=saved_args["hidden_dim"],
        num_layers=saved_args["num_layers"],
        dropout=saved_args["dropout"],
        max_seq_len=saved_args["max_seq_len"],
        nhead=8,
    ).to(device)
    model.load_state_dict(checkpoint["model_state"])
    model.eval()

    text = input("请输入一串字符串: ").strip()
    if not text:
        raise ValueError("输入字符串不能为空")

    unk_id = char2idx["<UNK>"]
    input_ids = [char2idx.get(c, unk_id) for c in text]
    max_seq_len = saved_args["max_seq_len"]
    if len(input_ids) > max_seq_len:
        input_ids = input_ids[-max_seq_len:]

    x = torch.tensor(input_ids, dtype=torch.long, device=device).unsqueeze(0)

    with torch.no_grad():
        logits = model(x)
        last_logits = logits[0, -1]
        probs = F.softmax(last_logits, dim=-1)
        k = min(top_k, probs.size(0))
        top_probs, top_indices = torch.topk(probs, k=k)

    print(f"\n输入文本: {text}")
    print(f"Top {k} 预测结果:")
    for rank, (prob, idx) in enumerate(zip(top_probs.tolist(), top_indices.tolist()), start=1):
        print(f"{rank}. 字符: {repr(idx2char[idx])}  概率: {prob:.6f}")


if __name__ == "__main__":
    train_model()
    predict_top_k()