"""
train_chinese_cls_rnn.py
中文句子关键词分类 —— 简单 RNN 版本

任务：句子中含有关键字（好/棒/赞/喜欢/满意）→ 正样本(1)，否则 → 负样本(0)
模型：Embedding → RNN → 取最后隐藏状态 → Linear → Sigmoid
优化：Adam (lr=1e-3)   损失：MSELoss   无需 GPU，CPU 即可运行

依赖：torch >= 2.0   (pip install torch)
"""

import random
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

# ─── 超参数 ────────────────────────────────────────────────

SEED        = 42   #种子 每次用这个数字作为起点，每次生成的随机数都一样
N_SAMPLES   = 4000 #样本数
MAXLEN      = 5   #一个输入样本中，通过词表转化后的最大长度，超过就切断，不足就补0
EMBED_DIM   = 64   #embedding的维度，单个汉字由64维的向量表示
HIDDEN_DIM  = 64   #RNN中隐状态的大小
LR          = 1e-3  #学习率
BATCH_SIZE  = 64   #每次训练的样本数量
EPOCHS      = 20   #训练轮数
TRAIN_RATIO = 0.8  # 80%训练集

random.seed(SEED)  # 固定训练的数据
torch.manual_seed(SEED)  # 固定模型的随机初始化，保证结果一致。如，权重初始化、Dropout随机置0.

# ─── 1. 数据生成 ────────────────────────────────────────────
POS_KEYS = ['好', '棒', '赞', '喜欢', '满意']

TEMPLATES_POS = [
    '这家{}真的很{}，下次还来',
    '这款{}设计让我{}',
    '{}的服务态度让我感到{}',
    '{}体验非常{}',
    '这次购物感觉{}极了',
]

TEMPLATES_NEG = [
    '今天天气阴沉，出门忘带雨伞',
    '这部电影情节比较平淡',
    '下午开了三个小时的会议',
    '路上堵车耽误了不少时间',
    '这道题做了很久还没解出来',
    '最近工作任务比较繁重',
    '超市里人很多，排队结账',
    '这个季节换季容易感冒',
    '今天作业布置得有点多',
    '公交车又晚点了十分钟',
]

OBJ_WORDS = ['店铺', '餐厅', '产品', '服务', '环境', '系统', '设计', '课程']
ADJ_WORDS = ['方便', '简洁', '独特', '舒适', '高效']


def make_positive():
    kw   = random.choice(POS_KEYS)
    tmpl = random.choice(TEMPLATES_POS)
    obj  = random.choice(OBJ_WORDS)
    try:
        sent = tmpl.format(obj, kw)
    except Exception:
        sent = obj + kw + random.choice(ADJ_WORDS)
    if random.random() < 0.3:
        extra = random.choice(POS_KEYS)
        pos   = random.randint(0, len(sent))
        sent  = sent[:pos] + extra + sent[pos:]
    return sent


def make_negative():
    base = random.choice(TEMPLATES_NEG)
    if random.random() < 0.4:
        base += random.choice(TEMPLATES_NEG)
    return base


def build_dataset(n=N_SAMPLES):
    data = []
    for _ in range(n // 2):
        data.append((make_positive(), 1))
        data.append((make_negative(), 0))
    random.shuffle(data)
    return data


#  ─── 1. 数据生成 ──────────────────────────────────────────── new
def generate_chinese_samples(N_SAMPLES):
    """
    生成包含特定字符'你'的随机中文样本数据
    
    参数:
        N_SAMPLES (int): 生成的样本数量
        
    返回:
        list: 包含元组的列表，元组结构为 (样本字符串, 标志位列表)
    """
    samples = []
    
    # 常用汉字 Unicode 范围: \u4e00 到 \u9fa5
    # 这里定义一个辅助函数来生成随机汉字
    def get_random_chinese():
        val = random.randint(0x4e00, 0x9fa5)
        return chr(val)
    
    for _ in range(N_SAMPLES):
        # 1. 随机确定'你'的位置 (0 到 4)
        ni_index = random.randint(0, 4)
        
        # 2. 初始化一个长度为5的列表，用于存放字符
        temp_chars = [''] * 5
        
        # 3. 在确定位置放入'你'
        temp_chars[ni_index] = '你'
        
        # 4. 在其他位置填充随机中文字符
        for i in range(5):
            if i != ni_index:
                temp_chars[i] = get_random_chinese()
        
        # 5. 拼接成字符串
        sample_str = "".join(temp_chars)
        
        # 6. 生成标志位列表 (0,0,0,0,0 -> 对应位置变1)
        flag_list = [0] * 5
        flag_list[ni_index] = 1
        
        # 7. 添加到结果列表
        samples.append((sample_str, flag_list))
        
    return samples



# ─── 2. 词表构建与编码 ──────────────────────────────────────
def build_vocab(data):
    vocab = {'<PAD>': 0, '<UNK>': 1}
    for sent, _ in data:
        for ch in sent:
            if ch not in vocab:
                vocab[ch] = len(vocab)
    return vocab

# 将一段话根据词表进行编码，这段话最多不超过32个字，超过就进行截取，没超过就补0，最后返回一个数组[1,3,4,34,35,63,23]
def encode(sent, vocab, maxlen=MAXLEN):
    ids  = [vocab.get(ch, 1) for ch in sent]
    ids  = ids[:maxlen] 
    ids += [0] * (maxlen - len(ids)) 
    return ids


# ─── 3. 自定义一个继承Dataset的类 ───────────────────────────────
class TextDataset(Dataset):
    def __init__(self, data, vocab):
        self.X = [encode(s, vocab) for s, _ in data] #返回一个二维数组，也就是一个4000行，32列的二维数组
        self.y = [lb for _, lb in data] #返回一个二维数组 长度为4000

    def __len__(self):
        return len(self.y)

    def __getitem__(self, i):
        return (
            torch.tensor(self.X[i], dtype=torch.long),
            torch.tensor(self.y[i], dtype=torch.float),
        )
    


# ─── 4. 模型定义 ────────────────────────────────────────────
class KeywordLSTM(nn.Module):
    """
    中文关键词分类器（LSTM + MaxPooling 版）
    架构：Embedding → LSTM → MaxPool → BN → Dropout → Linear → ReLU → (CrossEntropyLoss)
    """
    def __init__(self, vocab_size, embed_dim=EMBED_DIM, hidden_dim=HIDDEN_DIM, dropout=0.3):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        #self.rnn       = nn.RNN(embed_dim, hidden_dim, batch_first=True)
        self.LSTM       = nn.LSTM(embed_dim, hidden_dim, bias=True,batch_first=True)
        self.bn        = nn.BatchNorm1d(hidden_dim)
        self.dropout   = nn.Dropout(dropout) 
        self.fc        = nn.Linear(hidden_dim, 5)

    def forward(self, x):
        # x: (batch, seq_len)
        e, _ = self.LSTM(self.embedding(x))  # (B, L, hidden_dim)  传入x到embedding层，再将输出传入LSTM x形状：1x5 embedding后输出形状：5x64 ，rnn后输出形状：5x64 因为hidden_size为64,如果为128，就是5*128
        pooled = e.max(dim=1)[0]            # (B, hidden_dim)  对序列做 max pooling 最大池化 输出64维向量。
        pooled = self.dropout(self.bn(pooled))  #过归一化层 不改变形状，再过dropout将部分数据置0 也不改变形状
        
        tensorfc = self.fc(pooled) # (B,) 最后再过线性层，输出一个64x5d 矩阵，因为是一个多分类问题，损失函数用的交叉熵。
        #print(tensorfc.squeeze(1).shape)

        out = torch.relu(tensorfc.squeeze(1))
        

        return out


# ─── 5. 训练与评估 ──────────────────────────────────────────
def evaluate(model, loader):
    model.eval() # 进入评估模式
    correct = total = 0
    with torch.no_grad():
        for X, y in loader:
            prob    = model(X)
            prob_index =  prob.argmax(dim=1)

            #pred    = (prob > 0.5).long() # 预测值prob 大于0.5就置1 [1,0,1,0]

            correct += (prob_index == y.argmax(dim=1)).sum().item() # 预测值pred 如果等于 真实值为true否则false，[True, False, True] sum() 把 True 当作 1，False 当作 0 进行求和， item()获取原生int数
            total   += len(y)
    return correct / total




def train():
    print("生成数据集...")
    #data  = build_dataset(N_SAMPLES)
    data = generate_chinese_samples(N_SAMPLES)
    print(f"data:{data[:10]}")
    vocab = build_vocab(data)
    print(f"vocab:{vocab}")

    print(f"  样本数：{len(data)}，词表大小：{len(vocab)}")

    split      = int(len(data) * TRAIN_RATIO)
    train_data = data[:split] # 80% 训练集
    val_data   = data[split:] # 20% 验证集

    train_loader = DataLoader(TextDataset(train_data, vocab), batch_size=BATCH_SIZE, shuffle=True) #数据加载器，并且打乱，
    val_loader   = DataLoader(TextDataset(val_data,   vocab), batch_size=BATCH_SIZE) #验证集，不需要打乱


    #创建模型
    model     = KeywordLSTM(vocab_size=len(vocab))
    #创建损失函数
    criterion = nn.CrossEntropyLoss()
    #创建Adam优化器，model.parameters() 是指模型的所有权重参数，如RNN的Wh Wx等，需要将这些可优化的参数给到优化器，优化器才知道要去优化那些参数。
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    #可优化的参数数量
    total_params = sum(p.numel() for p in model.parameters())
    

    print(f"  模型参数量：{total_params:,}\n")

    for epoch in range(1, EPOCHS + 1):
        model.train() # 进入训练模式
        total_loss = 0.0

        # 每次循环，X, y  里都自动包含 32 条（batch_size）打乱过的数据
        # X为包含32条样本的训练数据 格式如：[1,3,5,64,23,42,23] 内容为表示某个汉字的编码，y为[1,0,1,0,1,1,1] 内容为表示某个汉字的编码，y为
        for X, y in train_loader:
            #预测值 这个预测值 pred 是32个样本的预测值
            pred = model(X)
            #计算损失值 这个loss 是 32个样本的 损失值 
            loss = criterion(pred, y)
            #梯度归0 
            optimizer.zero_grad()
            #反向传播，计算梯度
            loss.backward()
            #更新参数权重
            optimizer.step()
            #累加总的损失值
            total_loss += loss.item()
        #平均损失值
        avg_loss = total_loss / len(train_loader)

        #每轮训练进行一次评估，拿到正确率
        val_acc  = evaluate(model, val_loader)
        print(f"Epoch {epoch:2d}/{EPOCHS}  loss={avg_loss:.4f}  val_acc={val_acc:.4f}")

    print(f"\n最终验证准确率：{evaluate(model, val_loader):.4f}")

    print("\n--- 推理示例 ---")



    
    model.eval() #进入评估模式
    test_sents = [
        '今天你好美',
        '你今天好美',
        '怎么了你呀',
        '没你我不行',
        '还好碰上你'
    ]
    with torch.no_grad(): #在这个代码块内部，不要追踪任何计算过程，不要构建计算图，也不要计算梯度
        for sent in test_sents:
            ids   = torch.tensor([encode(sent, vocab)], dtype=torch.long)
            prob  = model(ids).argmax(dim=1) #这里将列表张量最大值索引
            label = '正样本' if int(prob) == sent.index('你') else '负样本'
            print(f"  [{label}({int(prob)})]  {sent}")


if __name__ == '__main__':
    train()
