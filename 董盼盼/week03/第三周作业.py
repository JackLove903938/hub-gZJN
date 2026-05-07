"""
第三周作业作业:
设计一个以文本为输入的多分类任务，
实验一下用RNN，LSTM等模型的跑通训练。
可以选择如下任务1:对一个任意包含“你”字的五个字的文本，“你”在第几位，就属于第几类。
"""
import random
import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt

# ─── 超参数 ────────────────────────────────────────────────
SEED        = 42
N_SAMPLES   = 5000
MAXLEN      = 5
EMBED_DIM   = 128
HIDDEN_DIM  = 128
LR          = 1e-3
BATCH_SIZE  = 64
EPOCHS      = 30
TRAIN_RATIO = 0.8

random.seed(SEED)#固定随机数生成器的种子
torch.manual_seed(SEED)

# ─── 1. 数据生成 ────────────────────────────────────────────

TEMPLATES = [
    '{}真的很好',
    '让{}体验好',
    '我让{}满意',
    '非常适合{}',
    '{}英文很棒',
    '{}的花拿着',
    '漂亮的{}呀',
    '{}真棒啊啊',
    '给{}的礼物', '和{}一起去','找{}有事呢', '把{}叫过来','喜欢{}的人', '看见{}了么','我找{}有事', '这是{}的书',
    '是不是{}的', '这就是{}的','我们和{}的', '这不是{}的','好朋友是{}', '这是不是{}','我好喜欢{}', '请转达给{}',
]

def make_text():
    tmpl = random.choice(TEMPLATES)
    sent = tmpl.format("你")#用于将参数按格式替换到字符串模板 tmpl 的占位符中。
    if random.random() < 0.3:
        chars = list(sent)
        other_chars = [
            # 数字类
            '一', '二', '三', '四', '五', '六', '七', '八', '九', '十',
            # 方位类
            '上', '下', '左', '右', '前', '后', '东', '西', '南', '北',
            # 简单汉字
            '人', '口', '手', '心', '水', '火', '土', '木', '金', '日'
        ]
        for i in range(len(chars)):
            if chars[i] != '你':
                    chars[i] = random.choice(other_chars)
        sent = ''.join(chars)#将字符列表组成字符串
    return sent

def build_dataset(n=N_SAMPLES):
    data = []
    #_：约定俗成的变量名，表示这个循环变量的值用不到，只是占位符
    for _ in range(n):
        sentence  = make_text()
        idx = sentence.find("你")
        data.append((sentence, idx))
    random.shuffle(data)#打乱顺序，避免模型学到顺序
    return data


# ─── 2. 词表构建与编码 ──────────────────────────────────────
def build_vocab(data):
    vocab = {'<PAD>': 0, '<UNK>': 1}
    for sent, _ in data:
        for ch in sent:
            if ch not in vocab:
                vocab[ch] = len(vocab)
    print(f"词表：{vocab}")
    return vocab


def encode(sent, vocab, maxlen=MAXLEN):
    ids  = [vocab.get(ch, 1) for ch in sent]#将句子中的每个字符转换为对应的ID
    ids  = ids[:maxlen]#截断（如果句子超过最大长度）
    ids += [0] * (maxlen - len(ids))#填充（如果句子不足最大长度，补0）
    return ids


# ─── 3. Dataset / DataLoader ────────────────────────────────
class TextDataset(Dataset):
    def __init__(self, data, vocab):
        self.X = [encode(s, vocab) for s, _ in data]
        self.y = [lb for _, lb in data]

    def __len__(self):
        return len(self.y)

    def __getitem__(self, i):
        return (
            torch.tensor(self.X[i], dtype=torch.long),
            torch.tensor(self.y[i], dtype=torch.long),
        )


# ─── 4. 模型定义 ────────────────────────────────────────────
class KeywordRNN(nn.Module):
    """
    中文关键词分类器（LSTM）
    架构：Embedding → LSTM → BN → Dropout → Linear → softmax
    """
    def __init__(self, vocab_size, embed_dim=EMBED_DIM, hidden_dim=HIDDEN_DIM, dropout=0.3):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.lstm = nn.LSTM(embed_dim, hidden_dim, batch_first=True)
        self.ln = nn.LayerNorm(hidden_dim)
        self.dropout   = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_dim, 5)

    def forward(self, x):
        e, _   = self.lstm(self.embedding(x))  # (B, L, hidden_dim)
        pooled = e[:, -1, :] # LSTM 具有记忆能力：最后一个时间步包含了整个序列的信息
        pooled = self.dropout(self.ln(pooled))
        out = self.fc(pooled)
        return out 


# ─── 5. 训练与评估 ──────────────────────────────────────────
def evaluate(model, loader):
    model.eval()
    correct, wrong = 0, 0
    with torch.no_grad():
        for X, y in loader:
            y_pred = model(X)  # 模型预测 model.forward(x)
            for y_p, y_t in zip(y_pred, y):  # 与真实标签进行对比
                y_p_maxindx = torch.argmax(y_p)
                if y_p_maxindx == int(y_t):
                    correct += 1  # 分类判断正确
                else:
                    wrong += 1
    print("正确预测个数：%d, 正确率：%f" % (correct, correct / (correct + wrong)))
    return correct / (correct + wrong)  


def train():
    print("生成数据集...")
    data  = build_dataset(N_SAMPLES)
    vocab = build_vocab(data)
    print(f"  样本数：{len(data)}，词表大小：{len(vocab)}")

    split      = int(len(data) * TRAIN_RATIO)
    train_data = data[:split]
    val_data   = data[split:]

    train_loader = DataLoader(TextDataset(train_data, vocab), batch_size=BATCH_SIZE, shuffle=True)
    val_loader   = DataLoader(TextDataset(val_data,   vocab), batch_size=BATCH_SIZE)

    model     = KeywordRNN(vocab_size=len(vocab))
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    criterion = nn.CrossEntropyLoss()

    total_params = sum(p.numel() for p in model.parameters())
    print(f"  模型参数量：{total_params:,}\n")
    log = []
    for epoch in range(1, EPOCHS + 1):
        model.train()
        total_loss = 0.0
        for X, y in train_loader:
            optimizer.zero_grad()#梯度归零
            pred = model(X) 
            loss = criterion(pred, y)#计算loss
            loss.backward()#计算梯度
            optimizer.step()#更新权重
            total_loss += loss.item()

        avg_loss = total_loss / len(train_loader)
        val_acc  = evaluate(model, val_loader)
        print(f"Epoch {epoch:2d}/{EPOCHS}  loss={avg_loss:.4f}  val_acc={val_acc:.4f}")
        log.append([val_acc, avg_loss])

    print(f"\n最终验证准确率：{evaluate(model, val_loader):.4f}")
    # 画图
    print(log)
    plt.plot(range(len(log)), [l[0] for l in log], label="acc")  # 画acc曲线
    plt.plot(range(len(log)), [l[1] for l in log], label="loss")  # 画loss曲线
    plt.legend()
    plt.show()

    print("\n--- 推理示例 ---")
    model.eval()#测试模式
    test_sents = [
        '你看见我了',
        '礼物适合你',
        '和你一起去',
        '让你满意的',
        '这是你的花',
        '非常喜欢你'
    ]
    with torch.no_grad():
        for sent in test_sents:
            ids   = torch.tensor([encode(sent, vocab)], dtype=torch.long)
            prob  = model(ids)
            prob = torch.softmax(prob, dim=1)
            max_prob = torch.max(prob).item()
            max_prob_index = torch.argmax(prob).item()
            print("输入：%s, 预测类别：%d, 概率值：%f" % (sent, max_prob_index, max_prob))  # 打印结果


if __name__ == '__main__':
    train()
