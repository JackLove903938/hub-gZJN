"""
bert_pipeline.py —— BERT 文本分类微调完整流程

教学重点：
  1. BertModel + 自定义分类头（cls/mean/max 三种池化）
  2. 分层学习率：BERT 层 2e-5，分类头 1e-4
  3. Warmup + Linear Decay 学习率调度
  4. 加权 CrossEntropyLoss 处理类别不均衡
  5. 只保存验证集最优模型

使用方式：
  from bert_pipeline import BertClassifier, train_bert, evaluate_bert
  model = train_bert()
  metrics = evaluate_bert(model)
"""

import json
import time
from pathlib import Path

import torch
import torch.nn as nn
from torch.optim import AdamW
from transformers import BertModel, get_linear_schedule_with_warmup
from sklearn.utils.class_weight import compute_class_weight
import numpy as np
from tqdm import tqdm

from config import cfg
from data_utils import load_split, load_label_map


# ──────────────────────────────────────────────────────────────────────────────
# 1. 模型定义：BertModel + 自定义分类头
# ──────────────────────────────────────────────────────────────────────────────

class BertClassifier(nn.Module):
    """
    结构：BertModel → Pooling → Dropout → Linear → logits

    为什么不用 BertForSequenceClassification？
      - 官方实现是黑盒，看不到向量提取逻辑
      - 手写分类头只有 3 行核心代码，方便替换 pooling 策略
      - 便于教学理解和后续扩展（多任务头、换 RoBERTa 等）
    """
    def __init__(self, model_path, num_labels=15, pool="cls", dropout=0.1):
        super().__init__()
        assert pool in ("cls", "mean", "max")
        self.pool = pool

        # 加载预训练 BERT，获取隐藏层维度（bert-base = 768）
        self.bert = BertModel.from_pretrained(model_path)
        hidden_size = self.bert.config.hidden_size

        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(hidden_size, num_labels)

    def forward(self, input_ids, attention_mask, token_type_ids):
        # return_dict=True：确保返回命名对象（transformers 5.x 兼容）
        outputs = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            return_dict=True,
        )
        last_hidden = outputs.last_hidden_state  # [B, L, H]
        vec = self._pool(last_hidden, attention_mask)  # [B, H]
        vec = self.dropout(vec)
        logits = self.classifier(vec)  # [B, num_labels]
        return logits

    def _pool(self, last_hidden, attention_mask):
        if self.pool == "cls":
            # [CLS] 在位置 0，直接切片
            return last_hidden[:, 0, :]

        # mask: [B, L, 1]，padding 位置为 0
        mask = attention_mask.unsqueeze(-1).float()

        if self.pool == "mean":
            # 有效 token 的均值（排除 padding）
            sum_hidden = (last_hidden * mask).sum(dim=1)
            count = mask.sum(dim=1).clamp(min=1e-9)
            return sum_hidden / count

        if self.pool == "max":
            # padding 位置设为 -inf，取 max 时不会被选中
            masked = last_hidden + (1 - mask) * (-1e9)
            return masked.max(dim=1).values


# ──────────────────────────────────────────────────────────────────────────────
# 2. 训练流程
# ──────────────────────────────────────────────────────────────────────────────

def compute_class_weights(device):
    """根据训练集类别频次计算 balanced weight。"""
    train_data = load_split("train")
    labels = np.array([item["label"] for item in train_data])
    classes = np.arange(cfg.num_labels)
    weights = compute_class_weight("balanced", classes=classes, y=labels)
    return torch.tensor(weights, dtype=torch.float).to(device)


def train_bert(train_loader, val_loader, device="cpu"):
    """
    BERT 微调训练主函数。

    返回:
      model: 训练好的模型（已加载最优 checkpoint）
      history: list, 每 epoch 的 train_loss, train_acc, val_acc, val_f1
    """
    print("\n" + "=" * 50)
    print("【BERT Fine-tune】开始训练")
    print("=" * 50)

    # ── 模型 ──
    model = BertClassifier(cfg.bert_model_path, cfg.num_labels, pool=cfg.bert_pool)
    model = model.to(device)

    # 打印参数量
    n_total = sum(p.numel() for p in model.parameters()) / 1e6
    n_bert = sum(p.numel() for p in model.bert.parameters()) / 1e6
    n_head = sum(p.numel() for p in model.classifier.parameters()) / 1e3
    print(f"[模型] 总参数量: {n_total:.1f}M (BERT: {n_bert:.1f}M, 分类头: {n_head:.1f}K)")
    print(f"[模型] Pooling: {cfg.bert_pool}")

    # ── Loss ──
    weights = compute_class_weights(device)
    criterion = nn.CrossEntropyLoss(weight=weights)
    print(f"[Loss] 使用类别加权 CrossEntropyLoss")

    # ── 优化器：分层学习率 ──
    # BERT 层用较小 lr（保护预训练知识），分类头用较大 lr（快速收敛）
    bert_params = list(model.bert.parameters())
    head_params = list(model.classifier.parameters()) + list(model.dropout.parameters())
    optimizer = AdamW([
        {"params": bert_params, "lr": cfg.bert_lr},
        {"params": head_params, "lr": cfg.bert_lr * cfg.bert_head_lr_mult},
    ], weight_decay=0.01)

    # ── 学习率调度：Warmup + Linear Decay ──
    total_steps = len(train_loader) * cfg.bert_epochs
    warmup_steps = int(total_steps * cfg.bert_warmup_ratio)
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=warmup_steps, num_training_steps=total_steps
    )
    print(f"[训练] epochs={cfg.bert_epochs}, total_steps={total_steps}, warmup={warmup_steps}")

    # ── 训练循环 ──
    best_val_f1 = 0.0
    history = []
    ckpt_path = cfg.checkpoint_dir / f"best_bert_{cfg.bert_pool}.pt"

    for epoch in range(1, cfg.bert_epochs + 1):
        model.train()
        total_loss, total_correct, total_samples = 0.0, 0, 0

        pbar = tqdm(train_loader, desc=f"Epoch {epoch}/{cfg.bert_epochs}", leave=False)
        for batch in pbar:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            token_type_ids = batch["token_type_ids"].to(device)
            labels = batch["label"].to(device)

            logits = model(input_ids, attention_mask, token_type_ids)
            loss = criterion(logits, labels)

            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)  # 梯度裁剪防爆炸
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()

            preds = logits.argmax(dim=-1)
            total_loss += loss.item() * labels.size(0)
            total_correct += (preds == labels).sum().item()
            total_samples += labels.size(0)

            pbar.set_postfix(loss=f"{total_loss / total_samples:.4f}", acc=f"{total_correct / total_samples:.4f}")

        train_loss = total_loss / total_samples
        train_acc = total_correct / total_samples

        # 验证
        val_metrics = evaluate_bert(model, val_loader, device, verbose=False)
        val_acc = val_metrics["accuracy"]
        val_f1 = val_metrics["macro_f1"]

        print(f"[Epoch {epoch}] train_loss={train_loss:.4f} train_acc={train_acc:.4f} | val_acc={val_acc:.4f} val_f1={val_f1:.4f}")

        history.append({
            "epoch": epoch, "train_loss": train_loss, "train_acc": train_acc,
            "val_acc": val_acc, "val_macro_f1": val_f1,
        })

        # 只保存验证集 Macro-F1 最优的模型（比 Accuracy 更稳健）
        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            torch.save({
                "epoch": epoch,
                "state_dict": model.state_dict(),
                "val_acc": val_acc,
                "val_macro_f1": val_f1,
                "pool": cfg.bert_pool,
            }, ckpt_path)
            print(f"  [OK] 最优模型已保存 -> {ckpt_path} (val_f1={val_f1:.4f})")

    # 加载最优模型返回
    print(f"[BERT] 训练完成。最优 val_f1={best_val_f1:.4f}")
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["state_dict"])
    return model, history


# ──────────────────────────────────────────────────────────────────────────────
# 3. 评估流程
# ──────────────────────────────────────────────────────────────────────────────

def evaluate_bert(model, val_loader, device="cpu", verbose=True):
    """
    评估 BERT 模型，返回 accuracy 和 macro_f1。
    """
    model.eval()
    all_preds, all_labels = [], []

    with torch.no_grad():
        for batch in val_loader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            token_type_ids = batch["token_type_ids"].to(device)
            labels = batch["label"]

            logits = model(input_ids, attention_mask, token_type_ids)
            preds = logits.argmax(dim=-1).cpu().numpy()
            all_preds.extend(preds)
            all_labels.extend(labels.numpy())

    from sklearn.metrics import accuracy_score, f1_score
    acc = accuracy_score(all_labels, all_preds)
    macro_f1 = f1_score(all_labels, all_preds, average="macro", zero_division=0)

    if verbose:
        print(f"[BERT 评估] accuracy={acc:.4f}, macro_f1={macro_f1:.4f}")

    return {"accuracy": acc, "macro_f1": macro_f1, "preds": all_preds, "labels": all_labels}
