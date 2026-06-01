"""
三种训练方法（优化器/学习率调度/标签平滑）效果对比 —— 情感分类版
数据集：中文商品评论情感（积极/消极/中立）
无需 BERT，无需联网，CPU 即可运行

用法: python compare_training_methods_sentiment.py
"""

import math, random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torch.optim.lr_scheduler import CosineAnnealingLR
from sklearn.metrics import (accuracy_score, f1_score, recall_score,
                             precision_score, classification_report)

random.seed(42); np.random.seed(42); torch.manual_seed(42)

# ═══════════════════════════ 数据 ═══════════════════════════
label_names = ["积极", "消极", "中立"]
id2name = {i: n for i, n in enumerate(label_names)}

templates = {
    0: [  # 积极
        "这款手机性价比超高，用起来非常流畅",
        "快递速度很快，包装也很严实，好评",
        "衣服质量很好，颜色和图片一样，非常满意",
        "客服态度特别好，耐心解答问题",
        "味道很棒，家人很喜欢吃，下次还会购买",
        "做工精细，材质手感一流，物超所值",
        "安装简单方便，效果超出预期",
        "第二次购买了，一如既往的好用",
    ],
    1: [  # 消极
        "收到货后发现屏幕有划痕，非常失望",
        "质量太差了，用了不到一周就坏了",
        "尺码偏小，穿上很不舒服，退货麻烦",
        "客服联系不上，售后体验极差",
        "实物与描述严重不符，颜色差别很大",
        "物流太慢了，等了半个月才到",
        "包装破损，里面的东西都压坏了",
        "味道很奇怪，怀疑是假货",
    ],
    2: [  # 中立
        "一般般吧，没有想象中那么好",
        "价格适中，但功能比较基础",
        "外观还可以，就是有点重",
        "用了一段时间，感觉无功无过",
        "和其他品牌差不多，没什么特别之处",
        "物流速度正常，包装完好",
        "客服回复比较慢，但问题解决了",
        "总体还行，对得起这个价格",
    ],
}

# 类别不均衡：消极类样本较少（模拟差评较少的场景）
class_sizes_train = [300, 120, 250]   # 积极300，消极120，中立250
class_sizes_val   = [60, 30, 55]

def generate_split(class_sizes, templates):
    records = []
    for lid, size in enumerate(class_sizes):
        for _ in range(size):
            base = random.choice(templates[lid])
            # 随机加前缀增加多样性
            if random.random() > 0.6:
                prefix = random.choice(["【买家秀】", "【追评】", "【使用心得】", "【实话实说】", ""])
                base = prefix + base
            records.append({"idx": len(records), "sentence": base, "label": lid})
    random.shuffle(records)
    return records

train_data = generate_split(class_sizes_train, templates)
val_data   = generate_split(class_sizes_val, templates)
print(f"📊 训练集: {len(train_data)} 条  验证集: {len(val_data)} 条")

# ═══════════════════════════ 分词器 ═══════════════════════════
class CharTokenizer:
    def __init__(self):
        special = ["[PAD]", "[CLS]", "[SEP]", "[UNK]"]
        self.vocab = {t: i for i, t in enumerate(special)}
        self.pad_id, self.cls_id, self.sep_id, self.unk_id = 0, 1, 2, 3
    def build(self, texts):
        for c in sorted(set("".join(texts))):
            if c not in self.vocab:
                self.vocab[c] = len(self.vocab)
    @property
    def vocab_size(self):
        return len(self.vocab)
    def encode(self, text, max_length):
        ids = [self.cls_id] + [self.vocab.get(c, self.unk_id) for c in text] + [self.sep_id]
        ids = ids[:max_length]
        pad_len = max_length - len(ids)
        return {
            "input_ids": torch.tensor(ids + [self.pad_id]*pad_len, dtype=torch.long),
            "attention_mask": torch.tensor([1]*len(ids) + [0]*pad_len, dtype=torch.long),
        }

tokenizer = CharTokenizer()
tokenizer.build([item["sentence"] for item in train_data + val_data])
print(f"📝 词表大小: {tokenizer.vocab_size}")

# ═══════════════════════════ Dataset ═══════════════════════════
class SentimentDataset(Dataset):
    def __init__(self, data, tokenizer, max_length=64):
        self.data, self.tokenizer, self.max_length = data, tokenizer, max_length
    def __len__(self):
        return len(self.data)
    def __getitem__(self, idx):
        item = self.data[idx]
        enc = self.tokenizer.encode(item["sentence"], self.max_length)
        enc["label"] = torch.tensor(item["label"], dtype=torch.long)
        return enc

train_loader = DataLoader(SentimentDataset(train_data, tokenizer), batch_size=64, shuffle=True)
val_loader   = DataLoader(SentimentDataset(val_data, tokenizer),   batch_size=64, shuffle=False)

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
    def __init__(self, vocab_size, num_labels=3, d_model=128, n_head=4,
                 n_layer=2, dropout=0.15):
        super().__init__()
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
        out = self.encoder(e, mask=~tri)  # Causal Mask
        # 固定使用 Mean 池化（不与训练方法混淆）
        mask = attention_mask.unsqueeze(-1).float()
        vec = (out * mask).sum(1) / mask.sum(1).clamp(min=1e-9)
        return self.classifier(self.dropout(vec))

# ═══════════════════════════ 标签平滑损失 ═══════════════════════════
class LabelSmoothingCrossEntropy(nn.Module):
    def __init__(self, smoothing=0.1):
        super().__init__()
        self.smoothing = smoothing
    def forward(self, logits, target):
        n_class = logits.size(-1)
        one_hot = torch.full_like(logits, self.smoothing / (n_class - 1))
        one_hot.scatter_(1, target.unsqueeze(1), 1.0 - self.smoothing)
        log_prob = nn.functional.log_softmax(logits, dim=-1)
        return -(one_hot * log_prob).sum(dim=-1).mean()

# ═══════════════════════════ 训练配置 ═══════════════════════════
training_configs = [
    {
        "name": "Baseline (AdamW)",
        "optimizer_fn": lambda params: optim.AdamW(params, lr=1e-3, weight_decay=0.01),
        "scheduler_fn": None,
        "criterion": nn.CrossEntropyLoss(),
    },
    {
        "name": "SGD + CosineAnnealing",
        "optimizer_fn": lambda params: optim.SGD(params, lr=0.01, momentum=0.9, weight_decay=0.01),
        "scheduler_fn": lambda opt: CosineAnnealingLR(opt, T_max=20),
        "criterion": nn.CrossEntropyLoss(),
    },
    {
        "name": "AdamW + LabelSmoothing",
        "optimizer_fn": lambda params: optim.AdamW(params, lr=1e-3, weight_decay=0.01),
        "scheduler_fn": None,
        "criterion": LabelSmoothingCrossEntropy(smoothing=0.1),
    },
]

# ═══════════════════════════ 训练 + 评估 ═══════════════════════════
def train_and_evaluate(config, epochs=20):
    name = config["name"]
    print(f"\n{'='*60}")
    print(f"  🔧 训练方法: {name}")
    print(f"{'='*60}")

    model = TextClassifier(tokenizer.vocab_size, num_labels=3)
    optimizer = config["optimizer_fn"](model.parameters())
    scheduler = config["scheduler_fn"](optimizer) if config["scheduler_fn"] else None
    criterion = config["criterion"]

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
        if scheduler:
            scheduler.step()

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

    print(f"\n  📊 {name} 最终结果: acc={acc:.4f}  macro_f1={mf1:.4f}  weighted_f1={wf1:.4f}")
    return {"method": name, "acc": acc, "macro_f1": mf1, "weighted_f1": wf1,
            "recall": rec, "precision": prec, "preds": preds, "labels": labels}

# ═══════════════════════════ 三种训练方法运行 ═══════════════════════════
results = {}
for cfg in training_configs:
    results[cfg["name"]] = train_and_evaluate(cfg)

# ═══════════════════════════ 对比输出 ═══════════════════════════
print(f"\n\n{'='*80}")
print(f"{'三种训练方法整体指标对比（情感分类）':^80}")
print(f"{'='*80}")
methods = [cfg["name"] for cfg in training_configs]
header = f"{'指标':<24}"
for m in methods:
    header += f"{m:>26}"
print(header); print("-"*80)

for metric in ["acc", "macro_f1", "weighted_f1"]:
    vals = [results[m][metric] for m in methods]
    best_idx = vals.index(max(vals))
    markers = ["  ★" if i == best_idx else "" for i in range(len(methods))]
    row = f"  {metric:<22}"
    for v, mark in zip(vals, markers):
        row += f"{v:>20.4f}{mark}"
    print(row)

print(f"\n{'各类别 Recall 对比':^80}")
print(f"{'类别':<6}", end="")
for m in methods:
    print(f"{m:>26}", end="")
print(f"  {'最优':>6}")
print("-"*80)
for i in range(3):
    vals = [results[m]["recall"][i] for m in methods]
    best = methods[vals.index(max(vals))]
    print(f" {id2name[i]:<4}", end="")
    for v in vals:
        print(f"{v:>20.3f}", end="")
    print(f"  {best:>6}")

print(f"\n{'各类别 Precision 对比':^80}")
print(f"{'类别':<6}", end="")
for m in methods:
    print(f"{m:>26}", end="")
print(f"  {'最优':>6}")
print("-"*80)
for i in range(3):
    vals = [results[m]["precision"][i] for m in methods]
    best = methods[vals.index(max(vals))]
    print(f" {id2name[i]:<4}", end="")
    for v in vals:
        print(f"{v:>20.3f}", end="")
    print(f"  {best:>6}")

# ── 详细分类报告 ──
for m in methods:
    print(f"\n{'='*60}")
    print(f"  {m} 分类报告")
    print(f"{'='*60}")
    print(classification_report(results[m]["labels"], results[m]["preds"],
          target_names=label_names, zero_division=0))

print("\n 对比完成！")
