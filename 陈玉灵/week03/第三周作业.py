"""
设计一个以文本为输入的多分类任务，实验一下用RNN,LSTM等模型跑通训练
任务：对一任意包含“你”字的五个字文本，“你”在文本中的位置（0-4）进行分类
数据：随机生成包含“你”字的文本，文本长度为5，文本内容为随机汉字，标签为“你”字在文本中的位置
模型：Embedding → RNN → 取最后隐藏状态 → Linear → Sigmoid → (MSELoss)
"""
import random
import sys
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

# ─── 超参数 ────────────────────────────────────────────────
#SEED        = 42
#torch.manual_seed(SEED)
#random.seed(SEED)
N_SAMPLES   = 10000
MAXLEN      = 5
EMBED_DIM   = 64
HIDDEN_DIM  = 64
LR          = 1e-3
BATCH_SIZE  = 64
EPOCHS      = 20
VOCAB_SIZE  = 1001  # 词表大小，0 表示pad,1表示unk,2表示“你”，其余为随机中文字符

# 生成中文字符词表（不含“你”，因为“你”保留为索引 2）
CHINESE_CHAR_START = 0x4e00
CHINESE_CHARS = [chr(i) for i in range(CHINESE_CHAR_START, CHINESE_CHAR_START + VOCAB_SIZE)]
OTHER_CHINESE_CHARS = [char for char in CHINESE_CHARS if char != "你"]
CHAR2IDX = {char: idx + 1 for idx, char in enumerate(OTHER_CHINESE_CHARS)}
CHAR2IDX["你"] = 2  # “你”字的索引为2，0为pad，1为unk
IDX2CHAR = {idx: char for char, idx in CHAR2IDX.items()}

# ─── 1. 数据生成 ────────────────────────────────────────────
def generate_sample():
    # 随机生成一个包含“你”字的中文文本
    pos = random.randint(0, MAXLEN - 1)
    text_chars = [random.choice(OTHER_CHINESE_CHARS) for _ in range(MAXLEN)]
    text_chars[pos] = "你"
    text_indices = [CHAR2IDX[c] for c in text_chars]
    return text_indices, ''.join(text_chars), pos

# 定义数据集类
class MyDataset(Dataset):
    def __init__(self, n_samples):
        self.samples = [generate_sample() for _ in range(n_samples)]

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        text_indices, text_str, label = self.samples[idx]
        return torch.tensor(text_indices, dtype=torch.long), torch.tensor(label, dtype=torch.long), text_str

# ─── 2. 模型定义 ────────────────────────────────────────────
class MyModel(nn.Module):
    def __init__(self, vocab_size, embed_dim, hidden_dim):
        super(MyModel, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.rnn = nn.LSTM(embed_dim, hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, MAXLEN)  # 输出为MAXLEN维，表示位置分类

    def forward(self, x):
        x = self.embedding(x)  # (batch_size, seq_len, embed_dim)
        _, (h_n, _) = self.rnn(x)  # h_n: (1, batch_size, hidden_dim)
        h_n = h_n.squeeze(0)  # (batch_size, hidden_dim)
        out = self.fc(h_n)    # (batch_size, MAXLEN)
        return out
    
# ─── 3. 训练 ────────────────────────────────────────────────
def show_samples(dataset, n=5):
    print('示例数据：')
    for i in range(min(n, len(dataset))):
        _, _, text_str = dataset[i]
        print(f'  {i+1}: {text_str}')
    print()


def train():
    dataset = MyDataset(N_SAMPLES)
    show_samples(dataset, 5)
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)
    model = MyModel(VOCAB_SIZE, EMBED_DIM, HIDDEN_DIM)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LR)

    for epoch in range(EPOCHS):
        total_loss = 0
        for texts, labels, _ in dataloader:
            optimizer.zero_grad()
            outputs = model(texts)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        print(f'Epoch {epoch+1}/{EPOCHS}, Loss: {total_loss/len(dataloader):.4f}')

    return model

# ─── 4. 评估 ────────────────────────────────────────────────
def evaluate(model, dataloader):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for texts, labels, text_strs in dataloader:
            outputs = model(texts)
            print(f'输出 logits: {outputs}')
            _, predicted = torch.max(outputs.data, 1)
            for i in range(len(labels)):
                print(f'文本: {text_strs[i]}, 真实位置: {labels[i].item()}, 预测位置: {predicted[i].item()}')
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    print(f'Accuracy: {100 * correct / total:.2f}%')


# ─── 5. 主函数 ────────────────────────────────────────────────
if __name__ == "__main__":
    model = train()
    # 评估
    dataset = MyDataset(10)
    dataloader = DataLoader(dataset, batch_size=1)
    evaluate(model, dataloader)