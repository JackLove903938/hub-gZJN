"""
对比 BERT 文本分类中三种向量提取策略的效果：
  - cls   : 使用 [CLS] 位置的输出
  - mean  : 所有有效 token 的隐状态取平均
  - max   : 所有有效 token 的隐状态取最大值

实验流程：
  1. 加载 SST-2 数据集
  2. 以相同的超参数分别训练三个模型（仅 pool 参数不同）
  3. 记录每个模型在验证集上的最佳准确率与 F1 分数
  4. 绘制训练过程中验证准确率的变化曲线及最终结果对比柱状图
  5. 输出对比表格与结论
"""

import os
import random
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from transformers import BertTokenizer, AdamW, get_linear_schedule_with_warmup
from datasets import load_dataset
from sklearn.metrics import accuracy_score, f1_score

# ------------------------------------------------------------
# 1. BertClassifier模型
# ------------------------------------------------------------
import transformers

POOL_OPTIONS = ("cls", "mean", "max")

class BertClassifier(nn.Module):
    def __init__(self, bert_path: str, num_labels: int, pool: str = "cls", dropout: float = 0.1):
        super().__init__()
        assert pool in POOL_OPTIONS, f"pool 必须是 {POOL_OPTIONS} 之一，收到: {pool}"
        self.pool = pool
        _prev_verbosity = transformers.logging.get_verbosity()
        transformers.logging.set_verbosity_error()
        self.bert = BertModel.from_pretrained(bert_path)
        transformers.logging.set_verbosity(_prev_verbosity)
        hidden_size = self.bert.config.hidden_size
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(hidden_size, num_labels)

    def forward(self, input_ids, attention_mask, token_type_ids):
        outputs = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            return_dict=True,
        )
        last_hidden = outputs.last_hidden_state
        vec = self._pool(last_hidden, attention_mask)
        vec = self.dropout(vec)
        logits = self.classifier(vec)
        return logits

    def _pool(self, last_hidden, attention_mask):
        if self.pool == "cls":
            return last_hidden[:, 0, :]
        mask = attention_mask.unsqueeze(-1).float()
        if self.pool == "mean":
            sum_hidden = (last_hidden * mask).sum(dim=1)
            count = mask.sum(dim=1).clamp(min=1e-9)
            return sum_hidden / count
        if self.pool == "max":
            masked = last_hidden + (1 - mask) * (-1e9)
            return masked.max(dim=1).values
        raise ValueError(f"未知池化策略: {self.pool}")


# ------------------------------------------------------------
# 2. 实验配置
# ------------------------------------------------------------
CONFIG = {
    "bert_path": "bert-base-uncased",
    "max_length": 128,
    "batch_size": 16,
    "epochs": 3,
    "learning_rate": 2e-5,
    "warmup_ratio": 0.1,
    "weight_decay": 0.01,
    "dropout": 0.1,
    "seed": 42,
    "device": "cuda" if torch.cuda.is_available() else "cpu",
    "num_workers": 2,
}

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True

# ------------------------------------------------------------
# 3. 数据准备
# ------------------------------------------------------------
def load_and_prepare_data(config):
    """加载 SST-2 数据集，并返回 tokenizer 与 DataLoader"""
    print("加载数据集 SST-2 ...")
    dataset = load_dataset("glue", "sst2")
    tokenizer = BertTokenizer.from_pretrained(config["bert_path"])

    def tokenize_function(examples):
        return tokenizer(
            examples["sentence"],
            truncation=True,
            padding="max_length",
            max_length=config["max_length"],
            return_token_type_ids=True,
        )

    
    
    encoded = dataset.map(tokenize_function, batched=True)
    encoded.set_format(type="torch", columns=["input_ids", "attention_mask", "token_type_ids", "label"])
    
    train_loader = DataLoader(
        encoded["train"], batch_size=config["batch_size"], shuffle=True,
        num_workers=config["num_workers"], pin_memory=True
    )
    val_loader = DataLoader(
        encoded["validation"], batch_size=config["batch_size"], shuffle=False,
        num_workers=config["num_workers"], pin_memory=True
    )
    num_labels = len(set(encoded["train"]["label"].numpy()))
    return train_loader, val_loader, num_labels, tokenizer

# ------------------------------------------------------------
# 4. 训练与评估函数
# ------------------------------------------------------------
def train_epoch(model, loader, optimizer, scheduler, device):
    model.train()
    total_loss = 0
    for batch in tqdm(loader, desc="Training", leave=False):
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        token_type_ids = batch["token_type_ids"].to(device)
        labels = batch["label"].to(device)

        optimizer.zero_grad()
        logits = model(input_ids, attention_mask, token_type_ids)
        loss = nn.CrossEntropyLoss()(logits, labels)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        scheduler.step()
        total_loss += loss.item()
    return total_loss / len(loader)

def evaluate(model, loader, device):
    model.eval()
    preds, true_labels = [], []
    with torch.no_grad():
        for batch in tqdm(loader, desc="Evaluating", leave=False):
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            token_type_ids = batch["token_type_ids"].to(device)
            labels = batch["label"].to(device)
            logits = model(input_ids, attention_mask, token_type_ids)
            pred = torch.argmax(logits, dim=-1).cpu().numpy()
            preds.extend(pred)
            true_labels.extend(labels.cpu().numpy())
    acc = accuracy_score(true_labels, preds)
    f1 = f1_score(true_labels, preds, average="binary")
    return acc, f1

# ------------------------------------------------------------
# 5. 训练单个模型并记录历史
# ------------------------------------------------------------
def train_model(pool_strategy, train_loader, val_loader, num_labels, config):
    print(f"\n{'='*50}\n开始训练 pool = {pool_strategy}\n{'='*50}")
    model = BertClassifier(
        bert_path=config["bert_path"],
        num_labels=num_labels,
        pool=pool_strategy,
        dropout=config["dropout"],
    ).to(config["device"])
    
    total_steps = len(train_loader) * config["epochs"]
    warmup_steps = int(total_steps * config["warmup_ratio"])
    optimizer = AdamW(model.parameters(), lr=config["learning_rate"], weight_decay=config["weight_decay"])
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=warmup_steps, num_training_steps=total_steps)
    
    history = {"val_acc": [], "val_f1": []}
    best_val_acc = 0.0
    best_state_dict = None
    
    for epoch in range(1, config["epochs"] + 1):
        print(f"Epoch {epoch}/{config['epochs']}")
        train_loss = train_epoch(model, train_loader, optimizer, scheduler, config["device"])
        val_acc, val_f1 = evaluate(model, val_loader, config["device"])
        history["val_acc"].append(val_acc)
        history["val_f1"].append(val_f1)
        print(f"Epoch {epoch} | Train Loss: {train_loss:.4f} | Val Acc: {val_acc:.4f} | Val F1: {val_f1:.4f}")
        
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_state_dict = model.state_dict().copy()
    
    # 加载最佳模型并返回指标
    model.load_state_dict(best_state_dict)
    final_acc, final_f1 = evaluate(model, val_loader, config["device"])
    print(f"Best results for {pool_strategy}: Acc = {final_acc:.4f}, F1 = {final_f1:.4f}")
    return final_acc, final_f1, history

# ------------------------------------------------------------
# 6. 可视化对比结果
# ------------------------------------------------------------
def plot_results(results_dict, histories_dict):
    """results_dict: {策略: (acc, f1)}; histories_dict: {策略: history}"""
    # 柱状图：最终准确率与 F1 对比
    strategies = list(results_dict.keys())
    accs = [results_dict[s][0] for s in strategies]
    f1s = [results_dict[s][1] for s in strategies]
    
    x = np.arange(len(strategies))
    width = 0.35
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.bar(x - width/2, accs, width, label='Accuracy', color='skyblue')
    ax.bar(x + width/2, f1s, width, label='F1 Score', color='lightcoral')
    ax.set_ylabel('Score')
    ax.set_title('Pooling Strategy Comparison on Validation Set')
    ax.set_xticks(x)
    ax.set_xticklabels(strategies)
    ax.legend()
    for i, (a, f) in enumerate(zip(accs, f1s)):
        ax.text(i - width/2, a + 0.01, f"{a:.3f}", ha='center', va='bottom', fontsize=9)
        ax.text(i + width/2, f + 0.01, f"{f:.3f}", ha='center', va='bottom', fontsize=9)
    plt.tight_layout()
    plt.savefig("pooling_comparison_bar.png", dpi=150)
    plt.show()
    
    # 学习曲线：验证准确率随 epoch 的变化
    plt.figure(figsize=(8, 5))
    for strat, hist in histories_dict.items():
        plt.plot(range(1, len(hist["val_acc"])+1), hist["val_acc"], marker='o', label=strat)
    plt.xlabel("Epoch")
    plt.ylabel("Validation Accuracy")
    plt.title("Training Dynamics of Different Pooling Strategies")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig("pooling_comparison_curves.png", dpi=150)
    plt.show()

# ------------------------------------------------------------
# 7. 主程序入口
# ------------------------------------------------------------
def main():
    set_seed(CONFIG["seed"])
    print(f"使用设备: {CONFIG['device']}")
    
    # 加载数据
    train_loader, val_loader, num_labels, _ = load_and_prepare_data(CONFIG)
    print(f"类别数: {num_labels}, 训练批次: {len(train_loader)}, 验证批次: {len(val_loader)}")
    
    results = {}
    histories = {}
    for pool in POOL_OPTIONS:
        acc, f1, hist = train_model(pool, train_loader, val_loader, num_labels, CONFIG)
        results[pool] = (acc, f1)
        histories[pool] = hist
    
    # 打印最终对比表格
    print("\n" + "="*60)
    print("最终对比结果 (验证集)")
    print("="*60)
    print(f"{'Strategy':<8} | {'Accuracy':<10} | {'F1 Score':<10}")
    print("-"*40)
    for strat, (acc, f1) in results.items():
        print(f"{strat:<8} | {acc:.4f}      | {f1:.4f}")
    
    # 可视化
    plot_results(results, histories)
    
    # 保存结果到文本文件
    with open("pooling_comparison.txt", "w") as f:
        f.write("Pooling Strategy Comparison\n")
        f.write(f"{'Strategy':<8}\t{'Accuracy':<10}\t{'F1 Score':<10}\n")
        for strat, (acc, f1) in results.items():
            f.write(f"{strat:<8}\t{acc:.4f}\t\t{f1:.4f}\n")
    print("\n对比图表已保存为 pooling_comparison_bar.png / curves.png")
    print("数值结果保存为 pooling_comparison.txt")

if __name__ == "__main__":
    main()
