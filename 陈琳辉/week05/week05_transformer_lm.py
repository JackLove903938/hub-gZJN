"""
Week05 作业：基于 Transformer 的单向语言模型与文本生成
=======================================================

本文件实现了一个完整的单向 Transformer 语言模型，包含：
1. 位置编码（Positional Encoding）
2. 单向多头自注意力（带因果掩码）
3. Transformer 解码器层（仅解码器，用于语言建模）
4. 训练循环
5. 文本生成函数
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
import math
import matplotlib.pyplot as plt
import json
import os


# ------------------------------
# 1. 位置编码模块
# ------------------------------
class PositionalEncoding(nn.Module):
    """
    正弦/余弦位置编码，用于给 Transformer 提供位置信息
    参考论文 "Attention Is All You Need"
    """
    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: shape [seq_len, batch_size, d_model]
        """
        x = x + self.pe[:x.size(0)]
        return self.dropout(x)


# ------------------------------
# 2. 单向多头自注意力
# ------------------------------
class MultiHeadAttention(nn.Module):
    """
    多头自注意力，支持因果掩码（单向）
    """
    def __init__(self, d_model: int, num_heads: int, dropout: float = 0.1):
        super().__init__()
        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"
        self.d_k = d_model // num_heads
        self.num_heads = num_heads
        
        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)
        self.W_o = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, query: torch.Tensor, key: torch.Tensor, value: torch.Tensor, 
                mask: torch.Tensor = None) -> tuple[torch.Tensor, torch.Tensor]:
        batch_size = query.size(1)
        
        # [batch_size, num_heads, seq_len, d_k]
        Q = self.W_q(query).view(-1, batch_size, self.num_heads, self.d_k).permute(1, 2, 0, 3)
        K = self.W_k(key).view(-1, batch_size, self.num_heads, self.d_k).permute(1, 2, 0, 3)
        V = self.W_v(value).view(-1, batch_size, self.num_heads, self.d_k).permute(1, 2, 0, 3)
        
        # Scaled dot-product attention
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_k)
        
        if mask is not None:
            scores = scores.masked_fill(mask == 0, float('-inf'))
        
        attn_weights = F.softmax(scores, dim=-1)
        attn_weights = self.dropout(attn_weights)
        
        context = torch.matmul(attn_weights, V)
        context = context.permute(2, 0, 1, 3).contiguous().view(-1, batch_size, self.num_heads * self.d_k)
        
        output = self.W_o(context)
        return output, attn_weights


# ------------------------------
# 3. Transformer 解码器层
# ------------------------------
class TransformerDecoderLayer(nn.Module):
    """
    Transformer 解码器层（仅用于语言建模，因此只有自注意力和前馈网络）
    """
    def __init__(self, d_model: int, num_heads: int, d_ff: int, dropout: float = 0.1):
        super().__init__()
        self.self_attn = MultiHeadAttention(d_model, num_heads, dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model)
        )
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, mask: torch.Tensor = None) -> torch.Tensor:
        # Self-attention sublayer
        attn_output, _ = self.self_attn(x, x, x, mask)
        x = self.norm1(x + self.dropout1(attn_output))
        
        # Feed-forward sublayer
        ffn_output = self.ffn(x)
        x = self.norm2(x + self.dropout2(ffn_output))
        return x


# ------------------------------
# 4. 完整的 Transformer 语言模型
# ------------------------------
class TransformerLanguageModel(nn.Module):
    def __init__(self, vocab_size: int, d_model: int = 512, num_heads: int = 8, 
                 num_layers: int = 6, d_ff: int = 2048, dropout: float = 0.1, 
                 max_len: int = 5000):
        super().__init__()
        self.d_model = d_model
        
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.pos_encoder = PositionalEncoding(d_model, dropout, max_len)
        
        self.layers = nn.ModuleList([
            TransformerDecoderLayer(d_model, num_heads, d_ff, dropout) 
            for _ in range(num_layers)
        ])
        
        self.fc_out = nn.Linear(d_model, vocab_size)
        self._init_weights()

    def _init_weights(self):
        initrange = 0.1
        self.embedding.weight.data.uniform_(-initrange, initrange)
        self.fc_out.bias.data.zero_()
        self.fc_out.weight.data.uniform_(-initrange, initrange)

    def forward(self, x: torch.Tensor, mask: torch.Tensor = None) -> torch.Tensor:
        """
        x: shape [seq_len, batch_size]
        """
        x = self.embedding(x) * math.sqrt(self.d_model)
        x = self.pos_encoder(x)
        
        for layer in self.layers:
            x = layer(x, mask)
        
        output = self.fc_out(x)
        return output


# ------------------------------
# 5. 因果掩码生成函数（用于单向语言模型）
# ------------------------------
def generate_causal_mask(sz: int) -> torch.Tensor:
    """
    生成下三角掩码，确保每个位置只能关注到前面的位置
    """
    mask = torch.triu(torch.ones(sz, sz), diagonal=1)
    mask = mask.masked_fill(mask == 1, False).masked_fill(mask == 0, True)
    return mask


# ------------------------------
# 6. 简单文本数据集（用于演示）
# ------------------------------
class TextDataset(Dataset):
    def __init__(self, text: str, seq_len: int = 32):
        self.seq_len = seq_len
        
        # 简单的字符级词典
        self.vocab = sorted(list(set(text)))
        self.vocab_size = len(self.vocab)
        self.char2idx = {ch: i for i, ch in enumerate(self.vocab)}
        self.idx2char = {i: ch for i, ch in enumerate(self.vocab)}
        
        # 将文本转换为索引
        self.data = [self.char2idx[ch] for ch in text]

    def __len__(self):
        return len(self.data) - self.seq_len

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        x = torch.tensor(self.data[idx:idx + self.seq_len], dtype=torch.long)
        y = torch.tensor(self.data[idx + 1:idx + self.seq_len + 1], dtype=torch.long)
        return x, y


# ------------------------------
# 7. 文本生成函数
# ------------------------------
def generate_text(model: TransformerLanguageModel, start_text: str, char2idx: dict, 
                  idx2char: dict, max_len: int = 100, temperature: float = 1.0, 
                  device: str = 'cpu') -> str:
    """
    使用训练好的模型生成文本
    """
    model.eval()
    input_ids = [char2idx[ch] for ch in start_text]
    input_tensor = torch.tensor(input_ids, dtype=torch.long).unsqueeze(1).to(device)
    
    with torch.no_grad():
        for _ in range(max_len):
            seq_len = input_tensor.size(0)
            mask = generate_causal_mask(seq_len).to(device)
            output = model(input_tensor, mask)
            logits = output[-1, 0, :] / temperature
            probs = F.softmax(logits, dim=-1)
            next_id = torch.multinomial(probs, num_samples=1).item()
            input_ids.append(next_id)
            input_tensor = torch.tensor(input_ids, dtype=torch.long).unsqueeze(1).to(device)
    
    return ''.join([idx2char[idx] for idx in input_ids])


# ------------------------------
# 8. 训练函数
# ------------------------------
def train():
    # --- 配置参数 ---
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"使用设备: {device}")
    
    # 文本数据
    sample_text = """
    深度学习是机器学习的一个分支，它基于表示学习。
    深度学习的概念源于人工神经网络的研究。
    含多隐层的多层感知器就是一种深度学习结构。
    深度学习通过组合低层特征形成更加抽象的高层表示属性类别或特征，以发现数据的分布式特征表示。
    Transformer 是一种基于自注意力机制的神经网络架构，最初用于自然语言处理任务。
    与传统的循环神经网络相比，Transformer 具有更好的并行计算能力。
    单向语言模型可以根据前面的文本预测下一个词，是文本生成任务的基础。
    通过训练大规模语料，语言模型可以学习到丰富的语言知识。
    """
    
    seq_len = 32
    batch_size = 8
    num_epochs = 200
    d_model = 128
    num_heads = 4
    num_layers = 4
    d_ff = 256
    dropout = 0.1
    learning_rate = 0.001
    
    # --- 准备数据 ---
    dataset = TextDataset(sample_text, seq_len)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    vocab_size = dataset.vocab_size
    print(f"词汇表大小: {vocab_size}")
    print(f"字符集: {dataset.vocab}")
    
    # --- 初始化模型 ---
    model = TransformerLanguageModel(
        vocab_size=vocab_size,
        d_model=d_model,
        num_heads=num_heads,
        num_layers=num_layers,
        d_ff=d_ff,
        dropout=dropout
    ).to(device)
    
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    
    # --- 训练循环 ---
    print("\n开始训练...")
    train_losses = []
    
    for epoch in range(num_epochs):
        model.train()
        epoch_loss = 0.0
        
        for batch in dataloader:
            x, y = batch
            x = x.permute(1, 0).to(device)  # [seq_len, batch_size]
            y = y.permute(1, 0).to(device)  # [seq_len, batch_size]
            
            optimizer.zero_grad()
            
            seq_len_curr = x.size(0)
            mask = generate_causal_mask(seq_len_curr).to(device)
            
            output = model(x, mask)
            loss = criterion(output.reshape(-1, vocab_size), y.reshape(-1))
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
        
        avg_loss = epoch_loss / len(dataloader)
        train_losses.append(avg_loss)
        
        if (epoch + 1) % 20 == 0:
            print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {avg_loss:.4f}")
            # 生成文本
            generated = generate_text(
                model, 
                start_text="深度", 
                char2idx=dataset.char2idx, 
                idx2char=dataset.idx2char, 
                max_len=50, 
                temperature=0.8, 
                device=device
            )
            print(f"生成: {generated}\n")
    
    # --- 保存训练结果 ---
    output_dir = "output"
    os.makedirs(output_dir, exist_ok=True)
    
    torch.save(model.state_dict(), os.path.join(output_dir, "best_transformer_lm.pt"))
    
    with open(os.path.join(output_dir, "training_history.json"), "w", encoding="utf-8") as f:
        json.dump({"train_loss": train_losses}, f, ensure_ascii=False)
    
    # 绘制损失曲线
    plt.figure(figsize=(10, 4))
    plt.plot(train_losses, label="训练损失")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("训练过程损失曲线")
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(output_dir, "training_loss.png"))
    plt.close()
    
    print("训练完成！模型和训练历史已保存到 output/ 目录")
    
    # --- 最终生成示例 ---
    print("\n=== 最终文本生成示例 ===")
    for temp in [0.5, 0.8, 1.0]:
        gen_text = generate_text(
            model, 
            start_text="Transformer", 
            char2idx=dataset.char2idx, 
            idx2char=dataset.idx2char, 
            max_len=100, 
            temperature=temp, 
            device=device
        )
        print(f"\n温度参数 {temp}:\n{gen_text}")


if __name__ == "__main__":
    train()

