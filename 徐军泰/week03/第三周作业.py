'''
任务：5字文本 "你" 字位置多分类任务
输入：任意 5 个汉字的文本
规则："你" 在第几位，就属于第几类（位置从 0 开始计数）
模型：LSTM（RNN类）+ 全连接层 实现多分类
功能：自动生成数据集 + 训练 + 早停 + 模型保存 + 绘图 + 预测
'''

import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import random

# ======================
# 1. 自动生成数据集（5字文本 + 你字位置标签）
# ======================
def generate_data(num_samples=500):   
    # 常用汉字（用来随机组合，避免只有你字）
    chars = ['我', '他', '爱', '好', '坏', '高', '矮', '大', '小', '跑', '跳', '吃', '喝', '睡', '醒']
    data = []
    labels = []

    for _ in range(num_samples):  # 执行500次，生成500条样本数据用于训练模型
        # 随机生成 5 个汉字
        text = random.choices(chars, k=5)  # k=5 表示生成5个汉字，形成一个列表
        # 随机选一个位置，把这个字换成 "你"
        pos = random.randint(0, 4)   # 生成一个0到4之间的随机整数，表示 "你" 字在文本中的位置
        text[pos] = '你'  # 将选定位置的汉字替换为 "你"
        # 转成字符串
        text_str = ''.join(text)  # text 是一个包含5个汉字的列表，使用 ''.join(text) 可以将这些汉字连接成一个连续的字符串，例如如果 text 是 ['我', '你', '爱', '好', '大']，那么 ''.join(text) 就会得到 "我你爱好大" 这样的字符串形式，这样更符合我们训练模型时需要的文本输入格式。
        # 标签 = "你" 的位置
        label = pos

        data.append(text_str)   # 将生成的文本字符串添加到数据列表中，这些文本将作为模型的输入
        labels.append(label)    # 将 "你" 字的位置标签添加到标签列表中，这些标签将作为模型的目标输出，用于训练模型进行位置分类

    return data, labels

# 调用函数生成 500 条数据
texts, labels = generate_data(500)

# ======================
# 2. 构建词表（所有出现的汉字 → 编号）
# ======================
vocab = {
    "<PAD>": 0,   # 填充字符（固定为0）
    "<UNK>": 1    # ✅ 未知字符（遇到词表外的字，都用它表示）
}

# 把数据中所有字加入词表（第2种方法）
for text in texts:   # 遍历每条文本数据
    for char in text:    # 遍历文本中的每个汉字
        if char not in vocab:
            vocab[char] = len(vocab)  # 按顺序编号

# 上面的词表构建也可以直接使用enumerate对chars列表进行处理：
# for idx, char in enumerate(chars, start=2):  # 从 2 开始编号，保留 0 和 1
    # vocab[char] = idx

vocab_size = len(vocab)

# ======================
# ✅ 文本转序列：遇到新词自动用 <UNK> 代替，不会报错
# ======================
def text2seq(text):
    # 词表里有就用编号，没有就用 <UNK> 的编号 1
    return [vocab.get(char, vocab["<UNK>"]) for char in text]

# 构建数据集
sequences = [torch.tensor(text2seq(t)) for t in texts]  
labels = torch.tensor(labels)

# ======================
# 3. LSTM 模型（RNN类）
# ======================
class LSTMTextModel(nn.Module):
    def __init__(self, vocab_size, embed_dim=16, hidden_size=32, num_classes=5):
        super().__init__()
        # 词嵌入层：把汉字编号 → 向量
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        # LSTM层（标准RNN类模型）
        self.lstm = nn.LSTM(
            input_size=embed_dim,
            hidden_size=hidden_size,
            batch_first=True,
            num_layers=1  # 1层时只学表面规律（如字的顺序），2层可以再学深层规律（如语义、逻辑），
        )
        # 分类层：LSTM输出 → 5分类
        self.fc = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        # x shape: (batch_size, seq_len=5)
        x = self.embedding(x)    # (B, 5, embed_dim)
        # LSTM 前向传播
        out, (hn, cn) = self.lstm(x)  
        # 取最后一步的隐藏状态作为句子特征，文本分类只用hn，不用cn、不用out
        feat = hn[-1]
        # 分类
        logits = self.fc(feat)
        return logits

# ======================
# 4. 训练配置
# ======================
embed_dim = 32
hidden_size = 64
batch_size = 30
epochs = 100
lr = 0.001
num_classes = 5
total_num = len(sequences)

# 早停
patience = 10
best_loss = float('inf')
stop_count = 0

# 绘图数据
loss_list = []
acc_list = []

# 模型、优化器、损失
model = LSTMTextModel(vocab_size, embed_dim, hidden_size, num_classes)
optimizer = optim.Adam(model.parameters(), lr=lr)
criterion = nn.CrossEntropyLoss()

# ======================
# 5. 训练循环（含早停 + 模型保存）
# ======================
print("="*60)
print("🚀 开始训练：5字文本'你'字位置分类")
print("="*60)

for epoch in range(epochs):
    model.train()
    # 随机取 batch
    idx = torch.randperm(total_num)[:batch_size]  
    batch_x = torch.stack([sequences[i] for i in idx])
    batch_y = labels[idx]

    optimizer.zero_grad()
    outputs = model(batch_x)
    loss = criterion(outputs, batch_y)
    loss.backward()
    optimizer.step()

    # 计算准确率
    acc = (outputs.argmax(1) == batch_y).float().mean().item()
    loss_list.append(loss.item())
    acc_list.append(acc)

    print(f"第{epoch+1:2d}轮 | Loss: {loss.item():.3f} | Acc: {acc:.3f}")

    # 早停
    if loss.item() < best_loss:
        best_loss = loss.item()
        stop_count = 0
        torch.save(model.state_dict(), "best_lstm_model.pth")  
    else:
        stop_count += 1
        if stop_count >= patience:
            print("🛑 早停触发")   
            break

# ======================
# 6. 绘制训练曲线
# ======================
plt.figure(figsize=(12,5))
plt.subplot(121)
plt.plot(loss_list, label='Train Loss')
plt.title('Loss Curve')
plt.legend()

plt.subplot(122)
plt.plot(acc_list, label='Train Acc', color='orange')
plt.title('Accuracy Curve')
plt.legend()
plt.show()
# plt.show(block=False)
# plt.pause(1) 

# ======================
# 7. 加载最优模型 + 测试预测
# ======================
print("\n" + "="*60)
print("✅ 模型加载完成，开始预测")
print("="*60)

best_model = LSTMTextModel(vocab_size, embed_dim, hidden_size, num_classes)
best_model.load_state_dict(torch.load("best_lstm_model.pth"))
best_model.eval()

def predict(text):
    # 必须输入 5 个字
    assert len(text) == 5, "必须输入5个汉字！"
    with torch.no_grad():
        seq = torch.tensor(text2seq(text)).unsqueeze(0)
        out = best_model(seq)
        pred_pos = out.argmax(1).item()
    return f"输入：{text} → 预测'你'在第【{pred_pos+1}】位（类别：{pred_pos}）"

# 测试
test_texts = [
    "你好世界呀",
    "我你他她它",
    "爱你一万年",
    "今天你开心",
    "最好看的你"
]

for t in test_texts:
    print(predict(t))
