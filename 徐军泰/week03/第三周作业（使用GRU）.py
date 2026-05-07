'''
任务：5字文本 "你" 字位置多分类任务
输入：任意 5 个汉字的文本
规则："你" 在第几位，就属于第几类（位置从 0 开始计数）
模型：GRU（RNN类）+ 全连接层 实现多分类
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

    for _ in range(num_samples):
        text = random.choices(chars, k=5)
        pos = random.randint(0, 4)
        text[pos] = '你'
        text_str = ''.join(text)
        label = pos

        data.append(text_str)
        labels.append(label)

    return data, labels

# 调用函数生成 500 条数据
texts, labels = generate_data(500)

# ======================
# 2. 构建词表（所有出现的汉字 → 编号）
# ======================
vocab = {
    "<PAD>": 0,
    "<UNK>": 1
}

for text in texts:
    for char in text:
        if char not in vocab:
            vocab[char] = len(vocab)

vocab_size = len(vocab)

# ======================
# 文本转序列
# ======================
def text2seq(text):
    return [vocab.get(char, vocab["<UNK>"]) for char in text]

sequences = [torch.tensor(text2seq(t)) for t in texts]
labels = torch.tensor(labels)

# ======================
# 3. GRU 模型（已修复！）
# ======================
class GRUTextModel(nn.Module):
    def __init__(self, vocab_size, embed_dim=16, hidden_size=32, num_classes=5):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        
        # ✅ 正确定义 GRU 层
        self.gru = nn.GRU(
            input_size=embed_dim,
            hidden_size=hidden_size,
            batch_first=True,
            num_layers=1
        )
        
        self.fc = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        x = self.embedding(x)
        out, hn = self.gru(x)  # ✅ GRU 只有 hn，没有 cell
        feat = hn[-1]  # 取最后一层隐藏状态
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

patience = 10
best_loss = float('inf')
stop_count = 0

loss_list = []
acc_list = []

# ✅ 模型实例化正确
model = GRUTextModel(vocab_size, embed_dim, hidden_size, num_classes)
optimizer = optim.Adam(model.parameters(), lr=lr)
criterion = nn.CrossEntropyLoss()

# ======================
# 5. 训练循环
# ======================
print("="*60)
print("🚀 开始训练：5字文本'你'字位置分类 (GRU版)")
print("="*60)

for epoch in range(epochs):
    model.train()
    idx = torch.randperm(total_num)[:batch_size]
    batch_x = torch.stack([sequences[i] for i in idx])
    batch_y = labels[idx]

    optimizer.zero_grad()
    outputs = model(batch_x)
    loss = criterion(outputs, batch_y)
    loss.backward()
    optimizer.step()

    acc = (outputs.argmax(1) == batch_y).float().mean().item()
    loss_list.append(loss.item())
    acc_list.append(acc)

    print(f"第{epoch+1:2d}轮 | Loss: {loss.item():.3f} | Acc: {acc:.3f}")

    if loss.item() < best_loss:
        best_loss = loss.item()
        stop_count = 0
        torch.save(model.state_dict(), "best_gru_model.pth")  # ✅ 改名
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

# ======================
# 7. 加载最优模型 + 预测
# ======================
print("\n" + "="*60)
print("✅ 模型加载完成，开始预测")
print("="*60)

# ✅ 加载正确
best_model = GRUTextModel(vocab_size, embed_dim, hidden_size, num_classes)
best_model.load_state_dict(torch.load("best_gru_model.pth"))
best_model.eval()

def predict(text):
    assert len(text) == 5, "必须输入5个汉字！"
    with torch.no_grad():
        seq = torch.tensor(text2seq(text)).unsqueeze(0)
        out = best_model(seq)
        pred_pos = out.argmax(1).item()
    return f"输入：{text} → 预测'你'在第【{pred_pos+1}】位（类别：{pred_pos}）"

test_texts = [
    "你好世界呀",
    "我你他她它",
    "爱你一万年",
    "今天你开心",
    "最好看的你"
]

for t in test_texts:
    print(predict(t))
