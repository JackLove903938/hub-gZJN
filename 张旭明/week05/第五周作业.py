import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import numpy as np
import math
from collections import Counter
import random
from tqdm import tqdm
import json

# 设置随机种子
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed(SEED)

# 数据预处理
class TextDataset(Dataset):
    def __init__(self, text, tokenizer, max_length=512):
        """
        文本数据集
        Args:
            text: 原始文本
            tokenizer: 分词器
            max_length: 最大序列长度
        """
        self.tokenizer = tokenizer
        self.max_length = max_length
        
        # 分词
        self.tokens = tokenizer.encode(text)
        
        # 创建训练样本
        self.samples = []
        for i in range(0, len(self.tokens) - max_length, max_length // 2):
            sequence = self.tokens[i:i + max_length]
            if len(sequence) == max_length:
                self.samples.append(sequence)
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        sequence = self.samples[idx]
        # 输入是序列的前n-1个token，目标是序列的后n-1个token（偏移一位）
        x = torch.tensor(sequence[:-1], dtype=torch.long)
        y = torch.tensor(sequence[1:], dtype=torch.long)
        return x, y

# 分词器
class CharTokenizer:
    """字符级分词器"""
    def __init__(self, text=None, vocab_size=None):
        if text is not None:
            self.build_vocab(text, vocab_size)
    
    def build_vocab(self, text, vocab_size=None):
        """构建词汇表"""
        # 统计字符频率
        char_counts = Counter(text)
        
        # 选择最常见的字符
        if vocab_size is not None:
            most_common = char_counts.most_common(vocab_size - 3)  # 保留位置给特殊token
        else:
            most_common = char_counts.most_common()
        
        # 构建词汇表
        self.vocab = {}
        self.inverse_vocab = {}
        
        # 特殊token
        special_tokens = ['<pad>', '<sos>', '<eos>', '<unk>']
        for i, token in enumerate(special_tokens):
            self.vocab[token] = i
            self.inverse_vocab[i] = token
        
        # 普通字符
        for i, (char, _) in enumerate(most_common, start=len(special_tokens)):
            self.vocab[char] = i
            self.inverse_vocab[i] = char
        
        self.vocab_size = len(self.vocab)
        
    def encode(self, text):
        """将文本转换为token id序列"""
        tokens = []
        for char in text:
            if char in self.vocab:
                tokens.append(self.vocab[char])
            else:
                tokens.append(self.vocab['<unk>'])
        return tokens
    
    def decode(self, tokens):
        """将token id序列转换为文本"""
        chars = []
        for token_id in tokens:
            if token_id in self.inverse_vocab:
                chars.append(self.inverse_vocab[token_id])
            else:
                chars.append('<unk>')
        return ''.join(chars)
    
    def save(self, path):
        """保存分词器"""
        with open(path, 'w') as f:
            json.dump({
                'vocab': self.vocab,
                'inverse_vocab': self.inverse_vocab,
                'vocab_size': self.vocab_size
            }, f)
    
    def load(self, path):
        """加载分词器"""
        with open(path, 'r') as f:
            data = json.load(f)
        self.vocab = {k: int(v) for k, v in data['vocab'].items()}
        self.inverse_vocab = {int(k): v for k, v in data['inverse_vocab'].items()}
        self.vocab_size = data['vocab_size']

# 位置编码
class PositionalEncoding(nn.Module):
    """位置编码"""
    def __init__(self, d_model, max_len=5000):
        super().__init__()
        
        # 创建位置编码矩阵
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        
        pe = pe.unsqueeze(0)  # 增加batch维度
        self.register_buffer('pe', pe)
    
    def forward(self, x):
        # x形状: [batch_size, seq_len, d_model]
        return x + self.pe[:, :x.size(1)]

# Transformer解码器块
class TransformerBlock(nn.Module):
    """Transformer解码器块"""
    def __init__(self, d_model, n_head, d_ff, dropout=0.1):
        super().__init__()
        
        # 多头自注意力
        self.self_attn = nn.MultiheadAttention(d_model, n_head, dropout=dropout, batch_first=True)
        self.norm1 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        
        # 前馈网络
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model)
        )
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout2 = nn.Dropout(dropout)
    
    def forward(self, x, attn_mask=None, key_padding_mask=None):
        # 自注意力
        attn_output, _ = self.self_attn(
            x, x, x, 
            attn_mask=attn_mask,
            key_padding_mask=key_padding_mask,
            need_weights=False
        )
        x = x + self.dropout1(attn_output)
        x = self.norm1(x)
        
        # 前馈网络
        ffn_output = self.ffn(x)
        x = x + self.dropout2(ffn_output)
        x = self.norm2(x)
        
        return x

# 单向Transformer语言模型
class TransformerLM(nn.Module):
    """单向Transformer语言模型（类似GPT）"""
    def __init__(self, vocab_size, d_model=256, n_head=8, n_layer=6, d_ff=1024, max_len=512, dropout=0.1):
        super().__init__()
        
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.max_len = max_len
        
        # 词嵌入
        self.token_embedding = nn.Embedding(vocab_size, d_model)
        
        # 位置编码
        self.position_embedding = PositionalEncoding(d_model, max_len)
        
        # Dropout
        self.dropout = nn.Dropout(dropout)
        
        # Transformer块
        self.blocks = nn.ModuleList([
            TransformerBlock(d_model, n_head, d_ff, dropout)
            for _ in range(n_layer)
        ])
        
        # 层归一化
        self.final_norm = nn.LayerNorm(d_model)
        
        # 输出层
        self.output_layer = nn.Linear(d_model, vocab_size)
        
        # 初始化权重
        self._init_weights()
    
    def _init_weights(self):
        """初始化权重"""
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
    
    def create_causal_mask(self, sz):
        """创建因果注意力掩码（防止看到未来的token）"""
        mask = torch.triu(torch.ones(sz, sz) * float('-inf'), diagonal=1)
        return mask
    
    def forward(self, x, attention_mask=None):
        # x形状: [batch_size, seq_len]
        batch_size, seq_len = x.shape
        
        # 创建因果注意力掩码
        causal_mask = self.create_causal_mask(seq_len).to(x.device)
        
        # 词嵌入
        token_embeds = self.token_embedding(x)  # [batch_size, seq_len, d_model]
        
        # 位置编码
        x = self.position_embedding(token_embeds)
        x = self.dropout(x)
        
        # 通过Transformer块
        for block in self.blocks:
            x = block(x, attn_mask=causal_mask, key_padding_mask=attention_mask)
        
        # 最终层归一化
        x = self.final_norm(x)
        
        # 输出logits
        logits = self.output_layer(x)  # [batch_size, seq_len, vocab_size]
        
        return logits
    
    def generate(self, prompt, tokenizer, max_length=100, temperature=1.0, top_k=50, top_p=0.95):
        """生成文本"""
        self.eval()
        
        # 编码prompt
        input_ids = tokenizer.encode(prompt)
        input_tensor = torch.tensor([input_ids], dtype=torch.long).to(next(self.parameters()).device)
        
        generated = input_ids.copy()
        
        with torch.no_grad():
            for _ in range(max_length):
                # 获取模型输出
                logits = self(input_tensor)
                
                # 取最后一个位置的logits
                next_token_logits = logits[0, -1, :] / temperature
                
                # Top-k采样
                if top_k > 0:
                    indices_to_remove = next_token_logits < torch.topk(next_token_logits, top_k)[0][..., -1, None]
                    next_token_logits[indices_to_remove] = float('-inf')
                
                # Top-p（核）采样
                if top_p < 1.0:
                    sorted_logits, sorted_indices = torch.sort(next_token_logits, descending=True)
                    cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
                    
                    # 移除累积概率超过top_p的token
                    sorted_indices_to_remove = cumulative_probs > top_p
                    # 保留第一个超过阈值的token
                    sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
                    sorted_indices_to_remove[..., 0] = 0
                    
                    indices_to_remove = sorted_indices[sorted_indices_to_remove]
                    next_token_logits[indices_to_remove] = float('-inf')
                
                # 采样下一个token
                probs = F.softmax(next_token_logits, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1).item()
                
                # 添加到生成序列
                generated.append(next_token)
                
                # 准备下一个输入
                input_tensor = torch.tensor([generated[-self.max_len:]], dtype=torch.long).to(next(self.parameters()).device)
        
        # 解码生成的文本
        generated_text = tokenizer.decode(generated)
        
        return generated_text

# 训练函数
def train_model(model, dataloader, epochs=10, lr=1e-3, device='cuda'):
    """训练模型"""
    model = model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, epochs)
    criterion = nn.CrossEntropyLoss()
    
    train_losses = []
    
    for epoch in range(epochs):
        model.train()
        epoch_loss = 0
        
        progress_bar = tqdm(dataloader, desc=f'Epoch {epoch+1}/{epochs}')
        for batch_idx, (x, y) in enumerate(progress_bar):
            x, y = x.to(device), y.to(device)
            
            # 前向传播
            logits = model(x)
            
            # 计算损失
            # 将logits从[batch, seq, vocab]调整为[batch*vocab, seq]
            loss = criterion(logits.view(-1, logits.size(-1)), y.view(-1))
            
            # 反向传播
            optimizer.zero_grad()
            loss.backward()
            
            # 梯度裁剪
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            optimizer.step()
            
            epoch_loss += loss.item()
            
            # 更新进度条
            if batch_idx % 10 == 0:
                progress_bar.set_postfix({'loss': loss.item()})
        
        # 更新学习率
        scheduler.step()
        
        # 计算平均损失
        avg_loss = epoch_loss / len(dataloader)
        train_losses.append(avg_loss)
        
        print(f'Epoch {epoch+1}/{epochs}, Loss: {avg_loss:.4f}')
        
        # 每个epoch结束后生成示例文本
        if epoch % 2 == 0 or epoch == epochs - 1:
            print("\n生成示例:")
            generated = model.generate(
                "Once upon a time",
                tokenizer,
                max_length=50,
                temperature=0.8,
                top_k=30
            )
            print(generated)
            print("-" * 50)
    
    return train_losses

# 主程序
def main():
    """主函数"""
    # 读取数据
    print("读取数据...")
    with open('sample_text.txt', 'r', encoding='utf-8') as f:
        text = f.read()
    
    # 创建分词器
    print("创建分词器...")
    tokenizer = CharTokenizer(text, vocab_size=1000)
    print(f"词汇表大小: {tokenizer.vocab_size}")
    
    # 创建数据集和数据加载器
    print("创建数据集...")
    dataset = TextDataset(text, tokenizer, max_length=128)
    dataloader = DataLoader(dataset, batch_size=32, shuffle=True, num_workers=2)
    print(f"训练样本数: {len(dataset)}")
    
    # 创建模型
    print("创建模型...")
    model = TransformerLM(
        vocab_size=tokenizer.vocab_size,
        d_model=128,  # 可以调整
        n_head=4,     # 可以调整
        n_layer=4,    # 可以调整
        d_ff=512,     # 可以调整
        max_len=128,
        dropout=0.1
    )
    
    # 计算参数数量
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"总参数: {total_params:,}")
    print(f"可训练参数: {trainable_params:,}")
    
    # 训练模型
    print("\n开始训练...")
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"使用设备: {device}")
    
    train_losses = train_model(
        model,
        dataloader,
        epochs=20,  # 可以调整
        lr=1e-3,    # 可以调整
        device=device
    )
    
    # 保存模型
    print("保存模型...")
    torch.save({
        'model_state_dict': model.state_dict(),
        'vocab_size': tokenizer.vocab_size,
        'd_model': model.d_model,
        'n_head': model.n_head,
        'n_layer': len(model.blocks),
        'd_ff': model.blocks[0].ffn[0].out_features,
        'max_len': model.max_len
    }, 'transformer_lm.pth')
    
    # 保存分词器
    tokenizer.save('tokenizer.json')
    
    # 文本生成演示
    print("\n文本生成演示:")
    prompts = [
        "Once upon a time",
        "The future of artificial intelligence",
        "In a galaxy far, far away",
        "The secret to happiness is"
    ]
    
    for prompt in prompts:
        print(f"\nPrompt: {prompt}")
        generated = model.generate(
            prompt,
            tokenizer,
            max_length=100,
            temperature=0.7,
            top_k=40,
            top_p=0.9
        )
        print(f"Generated: {generated}")
        print("-" * 50)

# 简化版示例
def simple_example():
    """简化版示例，使用小数据快速演示"""
    # 示例文本
    text = """
    Once upon a time, in a small village, there lived a wise old man.
    He was known for his knowledge and kindness. Every day, children 
    would gather around him to listen to his stories. His stories were 
    always full of wisdom and taught important life lessons. The old 
    man believed that knowledge was the greatest treasure one could possess.
    He spent his entire life learning and sharing what he knew with others.
    """
    
    # 创建分词器
    tokenizer = CharTokenizer(text, vocab_size=200)
    
    # 创建数据集
    dataset = TextDataset(text, tokenizer, max_length=64)
    dataloader = DataLoader(dataset, batch_size=8, shuffle=True)
    
    # 创建小模型
    model = TransformerLM(
        vocab_size=tokenizer.vocab_size,
        d_model=64,
        n_head=2,
        n_layer=2,
        d_ff=128,
        max_len=64,
        dropout=0.1
    )
    
    # 快速训练
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = model.to(device)
    
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.CrossEntropyLoss()
    
    print("快速训练...")
    for epoch in range(10):
        model.train()
        total_loss = 0
        
        for x, y in dataloader:
            x, y = x.to(device), y.to(device)
            
            logits = model(x)
            loss = criterion(logits.view(-1, logits.size(-1)), y.view(-1))
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
        
        print(f"Epoch {epoch+1}, Loss: {total_loss/len(dataloader):.4f}")
    
    # 生成示例
    print("\n生成示例:")
    prompts = ["Once upon", "The wise", "He was"]
    
    for prompt in prompts:
        generated = model.generate(
            prompt,
            tokenizer,
            max_length=50,
            temperature=0.7,
            top_k=20
        )
        print(f"Prompt: {prompt}")
        print(f"Generated: {generated}")
        print()

# 加载和使用训练好的模型
def load_and_generate(model_path, tokenizer_path, prompt, max_length=100):
    """加载训练好的模型并生成文本"""
    # 加载分词器
    tokenizer = CharTokenizer()
    tokenizer.load(tokenizer_path)
    
    # 加载模型配置
    checkpoint = torch.load(model_path, map_location='cpu')
    
    # 创建模型
    model = TransformerLM(
        vocab_size=checkpoint['vocab_size'],
        d_model=checkpoint['d_model'],
        n_head=checkpoint['n_head'],
        n_layer=checkpoint['n_layer'],
        d_ff=checkpoint['d_ff'],
        max_len=checkpoint['max_len']
    )
    
    # 加载权重
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    # 生成文本
    generated = model.generate(
        prompt,
        tokenizer,
        max_length=max_length,
        temperature=0.7,
        top_k=30
    )
    
    return generated

if __name__ == "__main__":
    # 运行完整训练
    # main()
    
    # 或者运行简化示例
    simple_example()
    
