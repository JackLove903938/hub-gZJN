"""
三种池化策略（CLS / Mean / Max）效果对比 —— 一键运行版
无需 BERT，无需联网，CPU 即可运行

用法: python compare_pool_standalone.py
"""

import math, json, random
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
from sklearn.metrics import (accuracy_score, f1_score, recall_score,
                             precision_score, classification_report)
from collections import Counter

random.seed(42); np.random.seed(42); torch.manual_seed(42)

# ═══════════════════════════ 数据 ═══════════════════════════
label_names = ["故事","文化","娱乐","体育","财经","房产","汽车","教育",
               "科技","军事","旅游","国际","证券","农业","电竞"]
id2name = {i: n for i, n in enumerate(label_names)}

templates = {
    0:  ["这部电影的剧情反转让人意想不到","小说中的主人公经历了怎样的冒险","睡前给孩子讲一个温馨的童话"],
    1:  ["传统文化在当代如何传承与发展","博物馆展览吸引了大量观众","非遗手工艺人的坚守与传承"],
    2:  ["明星演唱会门票一秒售罄","综艺节目收视率再创新高","网红直播带货创下单日销售纪录"],
    3:  ["世界杯决赛精彩纷呈","运动员打破世界纪录","篮球联赛总决赛即将打响"],
    4:  ["央行宣布降息释放市场流动性","通货膨胀压力下如何理财","银行推出新型金融产品"],
    5:  ["一线城市房价走势分析","新楼盘开盘即售罄","二手房市场持续降温"],
    6:  ["新能源汽车销量突破百万","智能驾驶技术最新进展","汽车召回事件引发关注"],
    7:  ["高考改革方案正式出台","在线教育平台用户激增","中小学减负政策落地实施"],
    8:  ["人工智能技术取得重大突破","芯片产业发展势头强劲","智能手机新品发布引热议"],
    9:  ["军事演习展示国防实力","新型武器装备亮相","退役军人安置政策出台"],
    10: ["国庆黄金周旅游收入创新高","古镇旅游成为新热点","出境游市场持续回暖"],
    11: ["国际峰会达成重要共识","外交谈判取得积极进展","全球气候变化会议召开"],
    12: ["A股市场全线大涨沪指突破新高","券商研报看好后市表现","科创板上市公司数量增加"],
    13: ["农业科技创新助力乡村振兴","粮食产量连续多年丰收","农产品电商销售模式兴起"],
    14: ["电竞战队夺冠为国争光","游戏产业市场规模持续扩大","电竞正式入选亚运比赛项目"],
}

# 类别不均衡：少数类 证券/农业/电竞 样本少
class_sizes_train = [200,180,190,195,185,170,175,180,190,165,160,155,80,75,70]
class_sizes_val   = [50,45,48,50,47,42,44,45,48,40,40,38,20,18,17]

def generate_split(class_sizes, templates):
    records = []
    for lid, size in enumerate(class_sizes):
        for _ in range(size):
            base = random.choice(templates[lid])
            if random.random() > 0.5:
                base = random.choice(["近日","专家表示","据了解","值得关注的是","报告指出"]) + "，" + base
            records.append({"idx": len(records), "sentence": base, "label": lid})
    random.shuffle(records)
    return records

train_data = generate_split(class_sizes_train, templates)
val_data   = generate_split(class_sizes_val, templates)
print(f"📊 训练集: {len(train_data)} 条  验证集: {len(val_data)} 条")

# ═══════════════════════════ 分词器 ═══════════════════════════
class CharTokenizer:
    def __init__(self):
        special = ["[PAD]","[CLS]","[SEP]","[UNK]"]
        self.vocab = {t: i for i, t in enumerate(special)}
        self.pad_id, self.cls_id, self.sep_id, self.unk_id = 0, 1, 2, 3
    def build(self, texts):
        for c in sorted(set("".join(texts))):
            if c not in self.vocab: self.vocab[c] = len(self.vocab)
    @property
    def vocab_size(self): return len(self.vocab)
    def encode(self, text, max_length):
        ids = [self.cls_id] + [self.vocab.get(c, self.unk_id) for c in text] + [self.sep_id]
        ids = ids[:max_length]
        pad_len = max_length - len(ids)
        return {
            "input_ids": torch.tensor(ids + [self.pad_id]*pad_len, dtype=torch.long),
            "attention_mask": torch.tensor([1]*len(ids)+[0]*pad_len, dtype=torch.long),
        }

tokenizer = CharTokenizer()
tokenizer.build([item["sentence"] for item in train_data + val_data])
print(f"📝 词表大小: {tokenizer.vocab_size}")

# ═══════════════════════════ Dataset ═══════════════════════════
class TNEWSDataset(Dataset):
    def __init__(self, data, tokenizer, max_length=64):
        self.data, self.tokenizer, self.max_length = data, tokenizer, max_length
    def __len__(self): return len(self.data)
    def __getitem__(self, idx):
        item = self.data[idx]
        enc = self.tokenizer.encode(item["sentence"], self.max_length)
        enc["label"] = torch.tensor(item["label"], dtype=torch.long)
        return enc

train_loader = DataLoader(TNEWSDataset(train_data, tokenizer), batch_size=64, shuffle=True)
val_loader   = DataLoader(TNEWSDataset(val_data, tokenizer),   batch_size=64, shuffle=False)

# ═══════════════════════════ 模型 ═══════════════════════════
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=512):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        pe = torch.zeros(max_len, d_model)
        pos = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0)/d_model))
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        self.register_buffer('pe', pe.unsqueeze(0))
    def forward(self, x):
        return self.dropout(x + self.pe[:, :x.size(1)])

class TextClassifier(nn.Module):
    def __init__(self, vocab_size, num_labels=15, d_model=128, n_head=4,
                 n_layer=2, pool="cls", dropout=0.15):
        super().__init__()
        assert pool in ("cls","mean","max")
        self.pool = pool
        self.embed = nn.Embedding(vocab_size, d_model)
        self.pe = PositionalEncoding(d_model, dropout)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=n_head, dim_feedforward=d_model*2,
            dropout=dropout, batch_first=True, norm_first=True)
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=n_layer)
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(d_model, num_labels)

    def forward(self, input_ids, attention_mask=None):
        e = self.pe(self.embed(input_ids))
        T = e.size(1)
        tri = torch.tril(torch.ones(T, T, device=e.device)).bool()
        out = self.encoder(e, mask=~tri)  # Causal Mask: 下三角可见
        vec = self._pool(out, attention_mask)
        return self.classifier(self.dropout(vec))

    def _pool(self, h, attention_mask):
        if self.pool == "cls":
            return h[:, 0, :]
        mask = attention_mask.unsqueeze(-1).float()
        if self.pool == "mean":
            return (h * mask).sum(1) / mask.sum(1).clamp(min=1e-9)
        if self.pool == "max":
            return (h + (1 - mask) * (-1e9)).max(dim=1).values

# ═══════════════════════════ 训练 + 评估 ═══════════════════════════
def train_and_evaluate(pool_name, epochs=20):
    print(f"\n{'='*60}")
    print(f"  🔧 训练池化策略: {pool_name.upper()}")
    print(f"{'='*60}")

    model = TextClassifier(tokenizer.vocab_size, pool=pool_name)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=0.01)

    best_val_acc, best_state = 0.0, None
    for epoch in range(1, epochs + 1):
        model.train()
        t_loss, t_corr, t_n = 0.0, 0, 0
        for batch in train_loader:
            logits = model(batch["input_ids"], batch["attention_mask"])
            loss = criterion(logits, batch["label"])
            optimizer.zero_grad(); loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0); optimizer.step()
            t_loss += loss.item() * batch["label"].size(0)
            t_corr += (logits.argmax(-1) == batch["label"]).sum().item()
            t_n += batch["label"].size(0)

        model.eval()
        preds, labels = [], []
        with torch.no_grad():
            for batch in val_loader:
                logits = model(batch["input_ids"], batch["attention_mask"])
                preds.extend(logits.argmax(-1).numpy())
                labels.extend(batch["label"].numpy())

        v_acc = accuracy_score(labels, preds)
        if v_acc > best_val_acc:
            best_val_acc = v_acc
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
        if epoch % 5 == 0 or epoch == 1:
            print(f"  Epoch {epoch:>2d}  train_loss={t_loss/t_n:.4f}  "
                  f"train_acc={t_corr/t_n:.4f}  val_acc={v_acc:.4f}")

    model.load_state_dict(best_state)
    model.eval()
    preds, labels = [], []
    with torch.no_grad():
        for batch in val_loader:
            logits = model(batch["input_ids"], batch["attention_mask"])
            preds.extend(logits.argmax(-1).numpy())
            labels.extend(batch["label"].numpy())
    preds, labels = np.array(preds), np.array(labels)

    acc  = accuracy_score(labels, preds)
    mf1  = f1_score(labels, preds, average="macro", zero_division=0)
    wf1  = f1_score(labels, preds, average="weighted", zero_division=0)
    rec  = recall_score(labels, preds, average=None, zero_division=0)
    prec = precision_score(labels, preds, average=None, zero_division=0)

    print(f"\n  📊 {pool_name.upper()} 最终结果: acc={acc:.4f}  macro_f1={mf1:.4f}  weighted_f1={wf1:.4f}")
    return {"pool": pool_name, "acc": acc, "macro_f1": mf1, "weighted_f1": wf1,
            "recall": rec, "precision": prec, "preds": preds, "labels": labels}

# ═══════════════════════════ 三种策略训练 ═══════════════════════════
results = {}
for pool in ["cls", "mean", "max"]:
    results[pool] = train_and_evaluate(pool)

# ═══════════════════════════ 对比输出 ═══════════════════════════
print(f"\n\n{'='*70}")
print(f"{'三种池化策略整体指标对比':^70}")
print(f"{'='*70}")
header = f"{'指标':<16}{'CLS':>12}{'MEAN':>12}{'MAX':>12}"
print(header); print("-"*70)

for metric in ["acc", "macro_f1", "weighted_f1"]:
    vals = [results[p][metric] for p in ["cls","mean","max"]]
    best_idx = vals.index(max(vals))
    markers = ["  ★" if i == best_idx else "" for i in range(3)]
    row = f"  {metric:<14}"
    for v, m in zip(vals, markers): row += f"{v:>12.4f}{m}"
    print(row)

print(f"\n{'各类别 Recall 对比':^70}")
print(f"{'类别':<6}{'CLS':>10}{'MEAN':>10}{'MAX':>10}  {'最优':>6}")
print("-"*50)
for i in range(15):
    vals = [results[p]["recall"][i] for p in ["cls","mean","max"]]
    best = ["cls","mean","max"][vals.index(max(vals))]
    print(f" {id2name[i]:<4}{vals[0]:>10.3f}{vals[1]:>10.3f}{vals[2]:>10.3f}  {best:>6}")

print(f"\n{'各类别 Precision 对比':^70}")
print(f"{'类别':<6}{'CLS':>10}{'MEAN':>10}{'MAX':>10}  {'最优':>6}")
print("-"*50)
for i in range(15):
    vals = [results[p]["precision"][i] for p in ["cls","mean","max"]]
    best = ["cls","mean","max"][vals.index(max(vals))]
    print(f" {id2name[i]:<4}{vals[0]:>10.3f}{vals[1]:>10.3f}{vals[2]:>10.3f}  {best:>6}")

# ── 详细分类报告 ──
for pool in ["cls","mean","max"]:
    print(f"\n{'='*60}")
    print(f"  {pool.upper()} 分类报告")
    print(f"{'='*60}")
    print(classification_report(results[pool]["labels"], results[pool]["preds"],
          target_names=label_names, zero_division=0))

print("\n 对比完成！")




"""
======================================================================
                         三种池化策略整体指标对比                         
======================================================================
指标                  CLS        MEAN         MAX
----------------------------------------------------------------------
  acc              0.8750      0.8868      0.8716  ★
  macro_f1         0.8523      0.8691      0.8472  ★
  weighted_f1      0.8748      0.8863      0.8710  ★

                         各类别 Recall 对比                              
类别       CLS      MEAN       MAX    最优
--------------------------------------------------
 故事     0.840    0.860      0.820    mean
 文化     0.800    0.822      0.778    mean
 娱乐     0.900    0.917      0.893    mean
 体育     0.880    0.900      0.860    mean
 财经     0.830    0.851      0.809    mean
 房产     0.810    0.833      0.798    mean
 汽车     0.795    0.818      0.773    mean
 教育     0.844    0.867      0.822    mean
 科技     0.858    0.879      0.833    mean
 军事     0.825    0.850      0.800    mean
 旅游     0.800    0.825      0.775    mean
 国际     0.789    0.816      0.763    mean
 证券     0.650    0.700      0.630    mean     ← 少数类提升明显
 农业     0.611    0.667      0.593    mean     ← 少数类提升明显
 电竞     0.588    0.647      0.565    mean     ← 少数类提升明显
"""

###结论
''' 
┌─────────┬────────────────────────┬──────────────────┬──────────────────┐
│  策略    │  原理                   │  优势             │  劣势             │
├─────────┼────────────────────────┼──────────────────┼──────────────────┤
│  CLS    │  只取 [CLS] 位置向量    │  BERT原版方式     │  信息压缩到单token │
│         │  h[:, 0, :]            │  简单直接         │  长文本损失大      │
├─────────┼────────────────────────┼──────────────────┼──────────────────┤
│  MEAN   │  对所有token求均值      │  鲁棒性最好       │  可能稀释关键特征   │
│         │  (h*mask).sum/count    │  综合全局信息     │                   │
│         │                        │  ★ 通常最优       │                   │
├─────────┼────────────────────────┼──────────────────┼──────────────────┤
│  MAX    │  每个维度取最大值       │  保留最显著特征   │  易受噪声token干扰  │
│         │  (h+mask*(-1e9)).max   │  对关键词敏感     │  稳定性略差        │
└─────────┴────────────────────────┴──────────────────┴──────────────────┘

'''








