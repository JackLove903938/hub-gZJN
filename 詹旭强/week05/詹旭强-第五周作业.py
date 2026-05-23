"""
字符级语言模型训练脚本，使用单向 Transformer (Decoder-only) 架构。
这个架构与 GPT 系列模型的核心原理一致。
用法:
    python language_model_transformer.py --epochs 20
"""
import os

os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
import math
import argparse
import glob
import random
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader


# ─────────────────────────── 数据准备模块 ───────────────────────────

def load_corpus(pattern="*.txt"):
    """加载指定路径下的所有文本文件并合并成一个大字符串"""
    texts = []
    # glob.glob 支持通配符，可以一次性读取目录下所有的 .txt 文件
    for path in glob.glob(pattern):
        with open(path, encoding="utf-8", errors="ignore") as f:
            texts.append(f.read())
    return "".join(texts)


def build_vocab(text):
    """构建字符到索引的映射表 (词汇表)"""
    # 提取文本中所有不重复的字符并排序，保证每次运行词汇表顺序一致
    chars = sorted(set(text))
    # 建立两个字典：字符->索引 (用于输入)，索引->字符 (用于输出/解码)
    char2idx = {c: i for i, c in enumerate(chars)}
    idx2char = {i: c for c, i in char2idx.items()}
    return char2idx, idx2char


class CharDataset(Dataset):
    """自定义 PyTorch 数据集类，负责将文本转换为模型可读取的张量"""
    def __init__(self, text, char2idx, seq_len):
        self.seq_len = seq_len
        # 将文本中的每个字符转换为对应的数字索引，过滤掉不在词汇表中的字符
        ids = [char2idx[c] for c in text if c in char2idx]
        self.data = torch.tensor(ids, dtype=torch.long)

    def __len__(self):
        # 数据集的总长度 = 总字符数 - 序列长度 (因为要留一个字符给目标值 y)
        return max(0, len(self.data) - self.seq_len)

    def __getitem__(self, idx):
        # 核心逻辑：滑动窗口采样
        # x 是当前序列 (例如: "我爱自然语言")
        x = self.data[idx: idx + self.seq_len]
        # y 是目标序列，相当于把 x 整体向后移动一位 (例如: "爱自然语言处")
        # 语言模型的任务就是：根据当前位置及之前的字符，预测下一个字符
        y = self.data[idx + 1: idx + self.seq_len + 1]
        return x, y


# ─────────────────────────── Transformer 模型定义 ───────────────────────────

class PositionalEncoding(nn.Module):
    """
    位置编码模块：为输入序列注入位置信息。
    因为 Transformer 是并行计算的（不像 RNN 那样按时间步顺序处理），
    它本身无法感知词语的先后顺序。所以必须显式地加上位置信息。
    """
    def __init__(self, d_model, max_len=5000):
        super().__init__()
        # 生成一个足够长的位置编码矩阵 (max_len, d_model)  d_model 就是emb_dim  没有彻底搞懂
        pe = torch.zeros(max_len, d_model)
        # 生成位置索引 (0, 1, 2, ..., max_len-1)，并增加一个维度以便广播  unsqueeze(1) 作用： shape (max_len) -> (max_len,1)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1) 
        # 计算除数项，用于控制正弦波的频率
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        
        # 偶数维度使用 sin 函数，奇数维度使用 cos 函数
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        # 注册为 buffer，意味着它会随模型保存，但不会作为可训练参数被梯度更新
        self.register_buffer('pe', pe)

    def forward(self, x):
        # x shape: (Batch, Seq_Len, D_model)
        # 将位置编码加到词嵌入上。pe[:x.size(1), :] 表示根据当前输入的实际长度截取位置编码
        return x + self.pe[:x.size(1), :]


class TransformerBlock(nn.Module):
    """
    标准的 Transformer Decoder 层 (采用 Pre-LN 结构)。
    Pre-LN (先归一化再计算) 相比 Post-LN 训练更稳定，是现代大模型（如 GPT-2/3, LLaMA）的标配。
    """
    def __init__(self, embed_dim, num_heads, ff_dim, dropout=0.1):
        super().__init__()
        # 多头自注意力机制，batch_first=True 表示输入张量维度为 (Batch, Seq_Len, Embed_Dim)
        self.self_attn = nn.MultiheadAttention(embed_dim, num_heads, dropout=dropout, batch_first=True)
        
        # 前馈神经网络 (FFN)：两层线性变换，中间加激活函数
        self.ffn = nn.Sequential(
            nn.Linear(embed_dim, ff_dim),
            nn.GELU(),  # GELU 激活函数比 ReLU 更平滑，在大模型中表现更好
            nn.Dropout(dropout),
            nn.Linear(ff_dim, embed_dim),
            nn.Dropout(dropout)
        )
        # 两个层归一化 (LayerNorm)，用于稳定深层网络的训练
        self.ln1 = nn.LayerNorm(embed_dim)
        self.ln2 = nn.LayerNorm(embed_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, attn_mask):
        # 1. 自注意力层 (Pre-LN 结构：先归一化，再进入子层)
        # attn_mask (因果掩码) 的作用是：在计算注意力时，把当前位置“未来”的字符全部屏蔽掉，
        # 保证模型在预测第 N 个字时，只能看到前 N-1 个字，不能“偷看”答案<websource>source_group_web_2</websource>。
        attn_output, _ = self.self_attn(x, x, x, attn_mask=attn_mask)
        x = x + self.dropout(attn_output)  # 残差连接 (Skip Connection)，防止梯度消失
        
        # 2. 前馈神经网络层 (同样采用 Pre-LN)
        ffn_output = self.ffn(self.ln2(x))
        x = x + ffn_output  # 残差连接
        return self.ln1(x)


class TransformerLM(nn.Module):
    """单向 Transformer 语言模型整体架构"""
    def __init__(self, vocab_size, embed_dim, num_layers, num_heads, ff_dim, dropout):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, embed_dim)  # 词嵌入层
        self.pos_encoder = PositionalEncoding(embed_dim)  # 位置编码层
        
        # 堆叠多个 Transformer Block，ModuleList 允许 PyTorch 识别并管理这些子模块的参数
        self.transformer_blocks = nn.ModuleList([
            TransformerBlock(embed_dim, num_heads, ff_dim, dropout)
            for _ in range(num_layers)
        ])
        
        self.fc = nn.Linear(embed_dim, vocab_size)  # 最终输出层，将隐藏状态映射回词汇表大小
        self.dropout = nn.Dropout(dropout)
        # 保存 embed_dim 以便在 forward 中使用
        self.embed_dim = embed_dim

    def forward(self, x):
        # x shape: (Batch, Seq_Len)
        seq_len = x.size(1)
        
        # 1. 词嵌入 + 缩放 + 位置编码
        # 乘以 sqrt(d_model) 是 Transformer 论文中的经典技巧，用于平衡词嵌入和位置编码的数值量级
        x = self.embed(x) * math.sqrt(self.embed_dim)
        x = self.pos_encoder(x)
        x = self.dropout(x)
        
        # 2. 生成因果掩码 (Causal Mask)
        # 生成一个 (Seq_Len, Seq_Len) 的上三角矩阵。
        # diagonal=1 表示主对角线及以下的元素为0，主对角线以上的元素为1。
        # .bool() 将其转换为布尔值，True 代表需要被屏蔽的位置（即当前位置之后的所有字符）<websource>source_group_web_3</websource>。
        causal_mask = torch.triu(torch.ones(seq_len, seq_len, device=x.device), diagonal=1).bool()
        
        # 3. 逐层经过 Transformer Blocks
        for block in self.transformer_blocks:
            x = block(x, causal_mask)
            
        # 4. 输出层：得到每个位置对应词汇表中每个字符的预测分数 (logits)
        logits = self.fc(x)
        return logits


# ─────────────────────────── 训练与评估模块 ───────────────────────────

def run_epoch(model, loader, criterion, optimizer, device, train=True):
    """
    运行一个完整的 epoch (遍历一次数据集)
    train=True 时为训练模式 (更新参数)，False 时为验证模式 (仅评估)
    """
    model.train(train)  # 切换模型模式 (影响 Dropout 和 LayerNorm 的行为)
    total_loss = 0.0
    total_tokens = 0

    for x, y in loader:
        x, y = x.to(device), y.to(device)  # 将数据搬运到 GPU 或 CPU
        
        logits = model(x)  # 模型前向预测，得到未归一化的分数
        # 计算交叉熵损失。需要将 (Batch, Seq_Len, Vocab) 展平为 (Batch*Seq_Len, Vocab)
        loss = criterion(logits.reshape(-1, logits.size(-1)), y.reshape(-1))

        if train:
            optimizer.zero_grad()  # 清空上一轮残留的梯度
            loss.backward()        # 反向传播，计算梯度
            # 梯度裁剪：防止 Transformer 训练初期梯度爆炸，保证训练稳定性
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()       # 优化器根据梯度更新模型参数<websource>source_group_web_4</websource>

        # 累加损失 (loss.item() 是平均损失，乘以 token 数量还原总损失)
        total_loss += loss.item() * y.numel()
        total_tokens += y.numel()  # 统计总共预测了多少个字符

    avg_loss = total_loss / total_tokens  # 计算平均每个字符的损失
    ppl = math.exp(avg_loss)              # 计算困惑度 (Perplexity)，越低代表模型越好
    return avg_loss, ppl


# ─────────────────────────── 主函数与配置 ───────────────────────────

def main():
    # 使用 argparse 解析命令行参数，方便在终端灵活调整超参数
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs",     type=int,   default=20)
    parser.add_argument("--seq_len",    type=int,   default=64)
    parser.add_argument("--batch_size", type=int,   default=64)   # Transformer 显存占用较大，适当调小 batch
    parser.add_argument("--embed_dim",  type=int,   default=128)  # 对应 d_model，词嵌入维度
    parser.add_argument("--num_layers", type=int,   default=2)    # Transformer 堆叠的层数
    parser.add_argument("--num_heads",  type=int,   default=4)    # 多头注意力的头数
    parser.add_argument("--ff_dim",     type=int,   default=256)  # 前馈网络隐藏层维度 (通常是 embed_dim 的 2-4 倍)
    parser.add_argument("--dropout",    type=float, default=0.1)
    parser.add_argument("--lr",         type=float, default=1e-3)
    parser.add_argument("--val_ratio",  type=float, default=0.05) # 验证集占比
    parser.add_argument("--corpus",     default="*.txt")
    parser.add_argument("--save",       default="best_transformer_model.pt")
    args = parser.parse_args()

    # 自动检测设备，优先使用 CUDA (NVIDIA 显卡)，否则使用 CPU
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device: {device}  model: TRANSFORMER")

    # --- 数据准备 ---
    text = load_corpus(args.corpus)
    if not text:
        raise FileNotFoundError("未找到任何 .txt 文件，请确认路径正确。")
    print(f"语料字符数: {len(text):,}")

    char2idx, idx2char = build_vocab(text)
    vocab_size = len(char2idx)
    print(f"词表大小: {vocab_size}")

    # 将文本按行分割并打乱，然后按比例划分训练集和验证集
    lines = text.splitlines()
    random.shuffle(lines)
    split = int(len(lines) * (1 - args.val_ratio))
    train_text = "\n".join(lines[:split])
    val_text   = "\n".join(lines[split:])

    # 实例化数据集和数据加载器 (DataLoader 负责分批和打乱数据)
    train_ds = CharDataset(train_text, char2idx, args.seq_len)
    val_ds   = CharDataset(val_text,   char2idx, args.seq_len)

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, drop_last=True)
    val_loader   = DataLoader(val_ds,   batch_size=args.batch_size, shuffle=True, drop_last=True)

    # --- 模型与优化器 ---
    model = TransformerLM(
        vocab_size=vocab_size,
        embed_dim=args.embed_dim,
        num_layers=args.num_layers,
        num_heads=args.num_heads,
        ff_dim=args.ff_dim,
        dropout=args.dropout,
    ).to(device)

    total_params = sum(p.numel() for p in model.parameters())
    print(f"模型参数量: {total_params:,}")

    criterion = nn.CrossEntropyLoss()  # 交叉熵损失函数 (多分类任务标配)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)  # AdamW 优化器 (带权重衰减的 Adam)

    best_val_ppl = float("inf")  # 记录验证集上最好的困惑度

    # 打印表头
    print(f"\n{'Epoch':>6}  {'Train Loss':>10}  {'Train PPL':>10}  {'Val Loss':>10}  {'Val PPL':>10}")
    print("-" * 56)

    # --- 训练循环 ---
    for epoch in range(1, args.epochs + 1):
        # 训练一个 epoch
        tr_loss, tr_ppl = run_epoch(model, train_loader, criterion, optimizer, device, train=True)
        
        # 验证一个 epoch (使用 torch.no_grad() 关闭梯度计算，节省显存并加速)
        with torch.no_grad():
            va_loss, va_ppl = run_epoch(model, val_loader, criterion, optimizer, device, train=False)

        # 如果当前模型的验证困惑度更低，则保存模型 (带 * 号表示该轮是历史最佳)
        marker = "  *" if va_ppl < best_val_ppl else ""
        if va_ppl < best_val_ppl:
            best_val_ppl = va_ppl
            torch.save({
                "model_state": model.state_dict(),
                "char2idx": char2idx,  # 保存词汇表，推理时必须用到
                "idx2char": idx2char,
                "args": vars(args),
            }, args.save)

        print(f"{epoch:>6}  {tr_loss:>10.4f}  {tr_ppl:>10.2f}  {va_loss:>10.4f}  {va_ppl:>10.2f}{marker}")

    print(f"\n训练完成。最佳验证 PPL: {best_val_ppl:.2f}  已保存至 {args.save}")


if __name__ == "__main__":
    main()
