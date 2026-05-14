"""
用pytorch实现一个transformer层
"""
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import os

# 定义自实现的Transformer层，包括多头自注意力和前馈网络
class MyTransformerLayer(nn.Module):
    def __init__(self, embed_dim, num_heads, ff_dim):
        super(MyTransformerLayer, self).__init__()
        self.self_attn = nn.MultiheadAttention(embed_dim, num_heads)
        self.ffn = nn.Sequential(
            nn.Linear(embed_dim, ff_dim),
            nn.ReLU(),
            nn.Linear(ff_dim, embed_dim)
        )
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)

    def forward(self, x):
        # MultiheadAttention要求输入形状为 (seq_len, batch, embed_dim)
        attn_output, _ = self.self_attn(x, x, x)
        x = self.norm1(x + attn_output)
        ffn_output = self.ffn(x)
        x = self.norm2(x + ffn_output)
        return x

# 自定义Dataset，用于加载编码后的文本数据
class MyDataset(Dataset):
    def __init__(self, data):
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]

# 构建字符级词表，将每个字符映射为唯一索引
def build_vocab(lines):
    chars = sorted({c for line in lines for c in line})
    vocab = {'<pad>': 0, '<unk>': 1}
    vocab.update({char: idx + 2 for idx, char in enumerate(chars)})
    return vocab

# 将一行文本编码为固定长度的索引序列，不足部分补<pad>
def encode_line(line, vocab, max_len):
    ids = [vocab.get(char, vocab['<unk>']) for char in line[:max_len]]
    ids += [vocab['<pad>']] * (max_len - len(ids))
    return torch.tensor(ids, dtype=torch.long)

# 训练函数，使用重建损失来训练自编码器模型
def train_model(model, dataloader, criterion, optimizer, num_epochs, device):
    model.train()
    for epoch in range(num_epochs):
        total_loss = 0.0
        for batch in dataloader:
            batch = batch.to(device)
            optimizer.zero_grad()
            embedded = model.embedding(batch)
            outputs = model(batch)
            loss = criterion(outputs, embedded)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        print(f'Epoch {epoch+1}/{num_epochs}, Loss: {total_loss / len(dataloader):.4f}')

# 评估函数，计算验证集上的平均损失
def evaluate_model(model, dataloader, criterion, device):
    model.eval()
    total_loss = 0.0
    with torch.no_grad():
        for batch in dataloader:
            batch = batch.to(device)
            embedded = model.embedding(batch)
            outputs = model(batch)
            loss = criterion(outputs, embedded)
            total_loss += loss.item()
    return total_loss / len(dataloader)

# 读取语料，按行读取并去掉空行
script_dir = os.path.dirname(os.path.abspath(__file__))
corpus_path = os.path.join(script_dir, 'corpus.txt')
with open(corpus_path, 'r', encoding='utf-8') as f:
    corpus = [line.strip() for line in f if line.strip()]

# 构建词表并将文本编码为固定长度向量
vocab = build_vocab(corpus)
max_seq_len = 128
encoded_data = [encode_line(line, vocab, max_seq_len) for line in corpus]

dataset = MyDataset(encoded_data)
dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
embed_dim = 128
num_heads = 4
ff_dim = 256
num_epochs = 10

# 定义自实现的Transformer自编码器模型
class TransformerAutoencoder(nn.Module):
    def __init__(self, vocab_size, embed_dim, num_heads, ff_dim, pad_idx):
        super(TransformerAutoencoder, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=pad_idx)
        self.transformer = MyTransformerLayer(embed_dim, num_heads, ff_dim)

    def forward(self, x):
        x = self.embedding(x)
        x = x.transpose(0, 1)
        x = self.transformer(x)
        return x.transpose(0, 1)

# 定义PyTorch内置Transformer的自编码器模型
class PyTorchTransformerAutoencoder(nn.Module):
    def __init__(self, vocab_size, embed_dim, num_heads, ff_dim, pad_idx):
        super(PyTorchTransformerAutoencoder, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=pad_idx)
        self.transformer_layer = nn.TransformerEncoderLayer(d_model=embed_dim, nhead=num_heads, dim_feedforward=ff_dim)

    def forward(self, x):
        x = self.embedding(x)
        x = x.transpose(0, 1)
        x = self.transformer_layer(x)
        return x.transpose(0, 1)

# 训练自实现的Transformer模型
model = TransformerAutoencoder(len(vocab), embed_dim, num_heads, ff_dim, vocab['<pad>']).to(device)
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

train_model(model, dataloader, criterion, optimizer, num_epochs, device)

# 训练PyTorch内置Transformer模型
pytorch_model = PyTorchTransformerAutoencoder(len(vocab), embed_dim, num_heads, ff_dim, vocab['<pad>']).to(device)
pytorch_optimizer = optim.Adam(pytorch_model.parameters(), lr=0.001)

train_model(pytorch_model, dataloader, criterion, pytorch_optimizer, num_epochs, device)

# 评估两个模型的性能并打印结果
my_transformer_loss = evaluate_model(model, dataloader, criterion, device)
pytorch_transformer_loss = evaluate_model(pytorch_model, dataloader, criterion, device)

print(f'My Transformer Loss: {my_transformer_loss:.4f}')
print(f'PyTorch Transformer Loss: {pytorch_transformer_loss:.4f}')





