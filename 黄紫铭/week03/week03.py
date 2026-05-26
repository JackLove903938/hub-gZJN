
#练习RNN
#做一个中文的五分类任务
import random
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader



#定义随机种子，保证随机数是固定的
SEED = 42
#设置random的随机数种子
random.seed(SEED)
#设置torch框架的CPU随机数种子
torch.manual_seed(SEED)
BATCH_SIZE  = 64
EPOCHS      = 20
TRAIN_RATIO = 0.8


#1、定义字符池——————————————————————————————————————————————————————————————
Raw = (
    '的一是了我不人在他有这个上们来到时大地为子中说生国年着'
    '就那和要她出也得里后自以会家可下而过天去能对小多然于心'
    '学么之都好看起发当没成只如事把还用第样道想作种开美总从'
    '无情己面最女但现前些所同日手又行意别信走定回爱进此'
)
char_pool = list(set(char for char in Raw if char != '你'))

#2、数据生产————————————————————————————————————————————————————————————————
#样本控制为5个字
char_num=5
#随机从字符池中选择字符，和输入的你，组成五个字的样本
def make_sample(pos:int):
    #在char_pool中获取四个随机汉字
    chars=random.choices(char_pool,char_num-1)
    #插入到随机位置pos
    chars.insert(pos,'你')
    #空串依次添加chars中的字符
    return ''.join(chars),pos

#样本数据集
#样本数据量
N_SAMPLES=10000
# 字符‘你’的位置，1-5对应下标0-4
NUM_CLASSES=5
def build_dataset(N_SAMPLES):
    per_class =N_SAMPLES//NUM_CLASSES
    data=[]
    for i in range(NUM_CLASSES):
        for j in range(per_class):
            #添加‘你’在第i个位置2000个数据
            data.append(make_sample(i))
    #随机打乱数据集
    random.shuffle(data)
    return data

#3、词表与编码————————————————————————————————————————————————————
#词表
def build_vocab(data):
    #PAD填充字符，可以把不同长度的字符填充到一样长
    #UNK，未知的字符用UNK表示
    vocab={'<PAD>':0,'<UNK>':1}
    for sent,_ in data:
        for ch in sent:
            if ch not in vocab:
                vocab[ch]=len(vocab)
    return vocab

#把字符转换为编码
def encode(sent,vocab):
    #ch在sent中返回get（ch），即key值，不存在返回1（UNK）
    #ids样例，[2, 3, 4, 5, 6]
    ids=[vocab.get(ch,1) for ch in sent[:char_num]]
    #如果ids的位数不足5，在末尾用0（PAD）补充
    ids+=[0]*(char_num-len(ids))
    return ids

#4、创建类————————————————————————————————————————————
class PositionDataset(Dataset):
    #data是文本集，vocab是词表
    def __init__(self, data, vocab):
        #长度为5的串
        self.X=[encode(s,vocab) for s,_ in data]
        #‘你’的坐标
        self.Y=[lb for _,lb in data]

    def __len__(self):
        return len(self.Y)

    def __getitem__(self, idx):
        #把测试集输出为模型张量，用于后续训练
        return (torch.tensor(self.X[idx]),torch.tensor(self.Y[idx]))

#5、模型定义————————————————————————————————————————
#自定义RNN
EMBED_DIM=64
HIDDEN_DIM=64
class RNNModel(nn.Module):
    #Embedding ->RNN-> MaxPool->Linear
    # (5,)
    # 数字序列
    # ↓
    # (5, 64)
    # 每个字64维
    # ↓
    # (5, H)
    # RNN每个时间步输出H维
    # ↓
    # (H,)
    # 池化后一个特征
    # ↓
    # (5,)
    # 5
    # 分类结果
    def __init__(self, vocab_size: int):
        super().__init__()
        #embedding层，
        #5个64维的向量
        self.embedding = (nn.Embedding(vocab_size, #需要转换字符集大小vocab_size个字符
                          EMBED_DIM,#每个字变成维度EMBED_DIM为64的向量，你->[1,2,3,...64]
                          padding_idx=0))#padding_idx代表PAD的数字为0，0向量不参与计算
        #rnn层
        #两个返回，一个是5*64的特征值（每个字的特征值），一个是最后一步的1*64的特征值
        self.rnn       = nn.RNN(EMBED_DIM, #输入向量维度
                                HIDDEN_DIM, #输出的特征维度
                                batch_first=True)#输入形状顺序，不加的话，会错误，
        #线性层
        #5维的概率，“你在”哪个位置
        self.fc        = nn.Linear(HIDDEN_DIM, #rnn的输出维度
                                   NUM_CLASSES)#输出维度，5分类
    #上方构造是定义参数，下方使用只需要输入即可
    def forward(self, x):
        #x: (batch_size, seq_len),（一批多少个，5），seq_len，5个汉字，int常量
        emdedding = self.embedding(x)#输出（batch_size, seq_len，EMBED_DIM），（一批多少个，5，64维向量）
        rnn_out, _ = self.rnn(emdedding)#输出（batch_size, seq_len，HIDDEN_DIM），（一批多少个，5，64维向量特征值）
        pooled = rnn_out.max(dim=-1)[0]#输出（batch_size,HIDDEN_DIM），（一批多少个，1，64维向量），5*64，每个维度取最大值，池化成1*64
        #64 维 × 权重 [64,5] → 压缩成 5 维分数
        return self.fc(pooled)#输出（batch_size,NUM_CLASSES）（一批多少个，1，5）靠权重矩阵把64维特征值转为5分类

class BagModel(nn.Module):
    #不适用RNN
    def __init__(self, vocab_size: int):
        super().__init__()
        self.embedding = (nn.Embedding(vocab_size, EMBED_DIM,padding_idx=0))
        self.fc = nn.Linear(EMBED_DIM, NUM_CLASSES)

    def forward(self, x):
        emdedding = self.embedding(x)
        pooled = emdedding.max(dim=-1)
        return self.fc(pooled)

#6、训练
def evaluate(model,loader):
    model.eval()
    correct= total=0
    with torch.no_grad():
        for x,y in loader:
            pred = model(x).argmax(dim=-1)
            correct+=(pred==y).sum().item()
            total+=len(y)
    return correct/total

def evaluate_per_class(model, loader):
    model.eval()
    correct = [0] * NUM_CLASSES
    total   = [0] * NUM_CLASSES
    with torch.no_grad():
        for X, y in loader:
            pred = model(X).argmax(dim=1)
            for cls in range(NUM_CLASSES):
                mask = (y == cls)
                correct[cls] += (pred[mask] == cls).sum().item()
                total[cls]   += mask.sum().item()
    return [correct[c] / total[c] if total[c] else 0.0 for c in range(NUM_CLASSES)]


def train_model(model, train_loader, val_loader, name):
    print(f"\n{'='*56}")
    print(f"  训练：{name}")
    print(f"  参数量：{sum(p.numel() for p in model.parameters()):,}")
    print(f"{'='*56}")

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)

    for epoch in range(1, EPOCHS + 1):
        model.train()
        total_loss = 0.0
        for X, y in train_loader:
            logits = model(X)
            loss   = criterion(logits, y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        if epoch % 4 == 0 or epoch == 1:
            avg_loss = total_loss / len(train_loader)
            val_acc  = evaluate(model, val_loader)
            print(f"  Epoch {epoch:2d}/{EPOCHS}  loss={avg_loss:.4f}  val_acc={val_acc:.4f}")

    final_acc = evaluate(model, val_loader)
    print(f"  最终验证准确率：{final_acc:.4f}")
    return final_acc, model


# ─── 7. 主流程 ─────────────────────────────────────────────────────────
def main():
    print("─── 数据集准备 ────────────────────────────────────────────────")
    data  = build_dataset(N_SAMPLES)
    vocab = build_vocab(data)
    print(f"  总样本：{len(data)}，每类 {len(data)//NUM_CLASSES} 条")
    print(f"  词表大小：{len(vocab)}")
    print(f'  任务：预测「你」在 5 字句子中的位置（第 1~5 位，{NUM_CLASSES} 分类）')
    print(f"  随机猜测基准准确率：{1/NUM_CLASSES:.0%}")

    split        = int(len(data) * TRAIN_RATIO)
    train_loader = DataLoader(
        PositionDataset(data[:split], vocab), batch_size=BATCH_SIZE, shuffle=True)
    val_loader   = DataLoader(
        PositionDataset(data[split:], vocab),  batch_size=BATCH_SIZE)

    rnn_acc, rnn_model = train_model(
        RNNModel(len(vocab)), train_loader, val_loader,
        "模型A — RNN + MaxPool（保留语序）"
    )
    bow_acc, bow_model = train_model(
        BagModel(len(vocab)), train_loader, val_loader,
        "模型B — 直接 Embedding MaxPool（无 RNN，丢失语序）"
    )

    # ─── 总体对比 ──────────────────────────────────────────────────────
    print(f"\n{'='*56}")
    print(f"  结果对比（{NUM_CLASSES} 分类，随机基准 {1/NUM_CLASSES:.0%}）")
    print(f"{'='*56}")
    print(f"  模型A（RNN + MaxPool）          val_acc = {rnn_acc:.4f}")
    print(f"  模型B（直接 Embedding MaxPool）  val_acc = {bow_acc:.4f}")

    # ─── 逐类准确率 ────────────────────────────────────────────────────
    rnn_per_cls = evaluate_per_class(rnn_model, val_loader)
    bow_per_cls = evaluate_per_class(bow_model, val_loader)
    print('\n  各位置准确率（「你」在第 X 位）：')
    print(f"  {'位置':>6}  {'模型A(RNN)':>12}  {'模型B(BoW)':>12}")
    for i in range(NUM_CLASSES):
        print(f"  第 {i+1} 位   {rnn_per_cls[i]:>10.4f}    {bow_per_cls[i]:>10.4f}")

    # ─── 结论 ──────────────────────────────────────────────────────────
    print(f"""
【结论】
  模型A 使用 RNN 逐字处理，第 t 步的隐藏状态编码了"前 t 字的上下文"，
  因此即使最终对时序做 MaxPool，池化结果仍携带位置信息，能准确判断"你"在哪位。

  模型B 直接对 Embedding 做 MaxPool，等价于词袋(BoW)：
  把句子中所有字符的向量混在一起取最大值，输出与字符顺序完全无关。
  5 个类别对它而言"看起来一模一样"，准确率约等于随机基准 20%。

  → 语序信息对于位置感知类任务至关重要；RNN 通过顺序递推有效保留了这一信息。
""")

    # ─── 推理示例 ──────────────────────────────────────────────────────
    print('─── 推理示例（各模型预测「你」所在位置）──────────────────────')
    rnn_model.eval()
    bow_model.eval()
    random.seed(0)
    print(f"  {'句子':>6}  真实位置  模型A(RNN)  模型B(BoW)")
    with torch.no_grad():
        for pos in range(NUM_CLASSES):
            sent, label = make_sample(pos)
            ids         = torch.tensor([encode(sent, vocab)], dtype=torch.long)
            rnn_pred    = rnn_model(ids).argmax(dim=1).item()
            bow_pred    = bow_model(ids).argmax(dim=1).item()
            rnn_mark    = '✓' if rnn_pred == label else '✗'
            bow_mark    = '✓' if bow_pred == label else '✗'
            print(f"  「{sent}」  第{label+1}位     "
                  f"{rnn_mark} 第{rnn_pred+1}位      "
                  f"{bow_mark} 第{bow_pred+1}位")


if __name__ == '__main__':
    main()
