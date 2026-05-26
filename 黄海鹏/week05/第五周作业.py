"""
字符级语言模型训练脚本，支持 Transformer架构，含 PPL 计算。
"""

import math
import argparse
import glob
import random
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm


# ─────────────────────────── 数据 ───────────────────────────

def load_corpus(pattern="*.txt"):
    texts = []
    for path in glob.glob(pattern):
        with open(path, encoding="utf-8", errors="ignore") as f:
            texts.append(f.read())
    return "".join(texts)


def build_vocab(text):
    chars = sorted(set(text))
    char2idx = {c: i for i, c in enumerate(chars)}
    idx2char = {i: c for c, i in char2idx.items()}
    return char2idx, idx2char


class CharDataset(Dataset):
    def __init__(self, text, char2idx, seq_len):
        self.seq_len = seq_len
        ids = [char2idx[c] for c in text if c in char2idx]
        self.data = torch.tensor(ids, dtype=torch.long)

    def __len__(self):
        return max(0, len(self.data) - self.seq_len)

    def __getitem__(self, idx):
        x = self.data[idx: idx + self.seq_len]
        y = self.data[idx + 1: idx + self.seq_len + 1]
        return x, y


# ─────────────────────────── 模型 ───────────────────────────

# 位置编码
class TransformerLM(nn.Module):
        def __init__(self, vocab_size, embed_dim=128, num_layers=2, nhead=8, dim_feedforward=None, dropout=0.1, max_len=5000):
            super().__init__()
            # 基础组件
            self.embed = nn.Embedding(vocab_size, embed_dim)
            self.dropout = nn.Dropout(dropout)

            # 位置编码
            pe = self._create_pos_encoding(max_len, embed_dim)
            self.register_buffer('pe', pe) # (1, max_len, embed_dim)

            # transformer 编码器
            if dim_feedforward is None:
                dim_feedforward = embed_dim * 4
            
            encoder_layer = nn.TransformerEncoderLayer(
                d_model=embed_dim, # 模型维度
                nhead=nhead, # 多头注意力的头数
                dim_feedforward=dim_feedforward, # FFN 的隐藏层维度
                dropout=dropout,   
                batch_first=True,  # 输入和输出的维度顺序
                activation='gelu', # 激活函数
            )
            self.transformer = nn.TransformerEncoder(encoder_layer, num_layers)
      
            # 输出层 
            self.fc_out = nn.Linear(embed_dim, vocab_size)

            # 初始化权重
            self._init_weights()

        def _create_pos_encoding(self, max_len, d_model):
            pe = torch.zeros(max_len, d_model)
            position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
            div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
            pe[:, 0::2] = torch.sin(position * div_term)
            pe[:, 1::2] = torch.cos(position * div_term)
            return pe.unsqueeze(0) # (1, max_len, d_model)
        
        def _init_weights(self):
            initrange = 0.1
            self.embed.weight.data.uniform_(-initrange, initrange)
            self.fc_out.bias.data.zero_()
            self.fc_out.weight.data.uniform_(-initrange, initrange)
    

        def genrate_causal_mask(self, seq_len, device):
            mask = (torch.triu(torch.ones(seq_len, seq_len,  device=device)) == 1).transpose(0, 1)
            mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))

            return mask
        
        def forward(self, x):
            # x (batch, seq_len)
            embed = self.embed(x) * math.sqrt(self.embed.embedding_dim) # 缩放嵌入值有助于稳定训练
            seq_len = x.size(1)
            embed = embed + self.pe[:, :seq_len, :]
            embed = self.dropout(embed)

            # causal mask
            mask = self.genrate_causal_mask(seq_len, x.device)

            # transformer forward
            out = self.transformer(embed, mask)

            # output
            logits = self.fc_out(out)
            return logits
# ─────────────────────────── 训练 / 评估 ───────────────────────────

def run_epoch(model, loader, criterion, optimizer, device, train=True):
    model.train(train)
    total_loss = 0.0
    total_tokens = 0

    for x, y in tqdm(loader, desc="Training" if train else "Evaluating"):
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


# ─────────────────────────── 主函数 ───────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model",      default="transformer", choices=["rnn", "lstm", "transformer"])
    parser.add_argument("--epochs",     type=int,   default=20) # 20
    parser.add_argument("--seq_len",    type=int,   default=16) # 64
    parser.add_argument("--batch_size", type=int,   default=16) #  128
    parser.add_argument("--embed_dim",  type=int,   default=64) # 128
    parser.add_argument("--hidden_dim", type=int,   default=128) # 256
    parser.add_argument("--num_layers", type=int,   default=2) # 2
    parser.add_argument("--dropout",    type=float, default=0.3)
    parser.add_argument("--lr",         type=float, default=1e-3)
    parser.add_argument("--val_ratio",  type=float, default=0.05)
    parser.add_argument("--corpus",     default="*.txt")
    parser.add_argument("--save",       default="best_tsfm_model.pt")
    args = parser.parse_args()

    # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    print(f"device: {device}  model: {args.model.upper()}")

    # 数据准备
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

    # 模型
    model = TransformerLM(
        vocab_size=vocab_size,
        embed_dim=args.embed_dim,
        num_layers=args.num_layers, 
        nhead=8, 
        dim_feedforward=args.hidden_dim, 
        dropout=args.dropout,
    ).to(device)

    print(f'模型结构:{vocab_size} -> {args.embed_dim} -> {args.hidden_dim} x {args.num_layers} -> {vocab_size}')
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


def textGeneration(prefix='你好', max_len=15, temperature=1.0, checkpoint_path="best_tsfm_model.pt"):
    # model.eval()
    # checkpoint_path = "best_tsfm_model.pt"

    # args = torch.load(checkpoint_path)
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    print("加载模型中...")

    try:
        checkpoint = torch.load("best_tsfm_model.pt", map_location=device)

        saved_args = checkpoint['args']
        char2idx = checkpoint['char2idx']
        idx2char = checkpoint['idx2char']
        model_state_dict = checkpoint['model_state']

        vocab_size = len(char2idx)
        model = TransformerLM(
            vocab_size=vocab_size,
            embed_dim=saved_args['embed_dim'],
            num_layers=saved_args['num_layers'],
            nhead=8,
            dim_feedforward=saved_args['hidden_dim'],
            dropout=saved_args['dropout'],
            max_len=5000
        ).to(device)

        # 模型权重
        model.load_state_dict(model_state_dict)
        model.eval()


        valid_prefix_chars = [c for c in prefix if c in char2idx]
        if not valid_prefix_chars:
            print("prefix contains invalid characters")
            return
        
        inputs_ids = [char2idx[c] for c in valid_prefix_chars]
        current_ids = inputs_ids[:]

        print("生成中...", end="", flush=True)

        with torch.no_grad():
            for _ in range(max_len):
                x = torch.tensor([current_ids], dtype=torch.long).to(device)
                # 前向传播
                logits = model(x)
                # 取最后一个时间步
                next_token_logits = logits[0, -1, :] / temperature
                # 转化为概率
                probs = torch.softmax(next_token_logits, dim=-1)

                # Top-p采样
                top_p = 0.9
                
                # 按概率降序
                sorted_probs, sorted_indices = torch.sort(probs, descending=True)

                # 计算累计概率
                cumulative_probs = torch.cumsum(sorted_probs, dim=-1)

                # 
                sorted_indices_to_remove = cumulative_probs > top_p

                sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
                sorted_indices_to_remove[..., 0] = False


                indices_to_remove = sorted_indices[sorted_indices_to_remove]
                probs[indices_to_remove] = 0.0

                probs = probs / probs.sum()

                next_idx = torch.multinomial(probs, num_samples=1).item()

                current_ids.append(next_idx)
        
        # 解码  打印
        generated_ids = current_ids[len(inputs_ids):]  # 只取生成的部分
        generated_text = "".join(idx2char[idx] for idx in generated_ids)
        
        print('\n' + '*' * 30)
        print(f"结果：{''.join(valid_prefix_chars)}{generated_text}")
    except Exception as e:
        print(f'错误{e}')


if __name__ == "__main__":
    # 训练 
    # main()

    # 测试生成文本
    textGeneration(prefix="你好", max_len=20)
    # 生成的文本: 你好可能经济政策的较去年其实现零售。”，20
    
    textGeneration(prefix="中国", max_len=20)
    # 生成的文本： 中国有所以来，无法行情较有大幅和定的都完善企
