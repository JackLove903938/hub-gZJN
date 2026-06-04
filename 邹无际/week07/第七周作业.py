

import os
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import json
import time
import argparse
from pathlib import Path
from collections import Counter
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
import transformers
from transformers import BertModel, BertTokenizer, get_linear_schedule_with_warmup
from tqdm import tqdm
import matplotlib.pyplot as plt
import matplotlib
matplotlib.rcParams["font.sans-serif"] = ["SimHei", "Microsoft YaHei", "Arial Unicode MS"]
matplotlib.rcParams["axes.unicode_minus"] = False

# ==================== 配置区域 ====================
ROOT = Path(__file__).parent.parent
DATA_DIR = ROOT / "data" / "peoples_daily"
BERT_PATH = ROOT.parent / "pretrain_models" / "bert-base-chinese"
OUTPUT_DIR = ROOT / "outputs_work"
CKPT_DIR = OUTPUT_DIR / "checkpoints"
LOG_DIR = OUTPUT_DIR / "logs"
FIG_DIR = OUTPUT_DIR / "figures"

# BIO 标签体系（3类实体：PER/ORG/LOC）
LABELS = ["O", "B-PER", "I-PER", "B-ORG", "I-ORG", "B-LOC", "I-LOC"]
LABEL2ID = {lbl: i for i, lbl in enumerate(LABELS)}
ID2LABEL = {i: lbl for lbl, i in LABEL2ID.items()}
NUM_LABELS = len(LABELS)


# ==================== 数据处理 ====================
class PeoplesDailyDataset(Dataset):
    """人民日报 NER 数据集。"""

    def __init__(self, records: list, tokenizer: BertTokenizer, max_length: int = 128):
        self.records = records
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.records)

    def __getitem__(self, idx: int) -> dict:
        row = self.records[idx]
        tokens = row["tokens"]
        ner_tags = row["ner_tags"]

        # BIO 标签 → id
        char_labels = [LABEL2ID.get(tag, 0) for tag in ner_tags]

        # Tokenizer
        encoding = self.tokenizer(
            tokens,
            is_split_into_words=True,
            max_length=self.max_length,
            truncation=True,
            padding="max_length",
            return_tensors="pt",
        )

        # 子词对齐
        word_ids = encoding.word_ids(batch_index=0)
        aligned_labels = []
        prev_word_id = None
        for wid in word_ids:
            if wid is None:
                aligned_labels.append(-100)
            elif wid != prev_word_id:
                if wid < len(char_labels):
                    aligned_labels.append(char_labels[wid])
                else:
                    aligned_labels.append(-100)
                prev_word_id = wid
            else:
                aligned_labels.append(-100)

        return {
            "input_ids": encoding["input_ids"].squeeze(0),
            "attention_mask": encoding["attention_mask"].squeeze(0),
            "token_type_ids": encoding["token_type_ids"].squeeze(0),
            "labels": torch.tensor(aligned_labels, dtype=torch.long),
        }


def load_records(split: str) -> list:
    """加载数据集。"""
    with open(DATA_DIR / f"{split}.json", "r", encoding="utf-8") as f:
        return json.load(f)


def build_dataloaders(tokenizer: BertTokenizer, batch_size: int = 32, max_length: int = 128):
    """构建 DataLoader。"""
    train_records = load_records("train")
    val_records = load_records("validation")
    test_records = load_records("test")

    train_ds = PeoplesDailyDataset(train_records, tokenizer, max_length)
    val_ds = PeoplesDailyDataset(val_records, tokenizer, max_length)
    test_ds = PeoplesDailyDataset(test_records, tokenizer, max_length)

    print(f"📊 数据集规模：训练={len(train_ds)}，验证={len(val_ds)}，测试={len(test_ds)}")

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=0)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=0)

    return train_loader, val_loader, test_loader


# ==================== 模型定义 ====================
def _load_bert(bert_path: str) -> BertModel:
    """加载 BERT 模型。"""
    prev = transformers.logging.get_verbosity()
    transformers.logging.set_verbosity_error()
    bert = BertModel.from_pretrained(bert_path)
    transformers.logging.set_verbosity(prev)
    return bert


class BertNER(nn.Module):
    """BERT + Linear 分类头。"""

    def __init__(self, bert_path: str, dropout: float = 0.1):
        super().__init__()
        self.bert = _load_bert(bert_path)
        hidden_size = self.bert.config.hidden_size
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(hidden_size, NUM_LABELS)

    def forward(self, input_ids, attention_mask, token_type_ids, labels=None):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask,
                           token_type_ids=token_type_ids, return_dict=True)
        seq_output = outputs.last_hidden_state
        logits = self.classifier(self.dropout(seq_output))

        loss = None
        if labels is not None:
            loss = F.cross_entropy(logits.view(-1, NUM_LABELS), labels.view(-1), ignore_index=-100)
        return logits, loss


class BertCRFNER(nn.Module):
    """BERT + CRF 层。"""

    def __init__(self, bert_path: str, dropout: float = 0.1):
        super().__init__()
        from torchcrf import CRF

        self.bert = _load_bert(bert_path)
        hidden_size = self.bert.config.hidden_size
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(hidden_size, NUM_LABELS)
        self.crf = CRF(NUM_LABELS, batch_first=True)

    def _get_emissions(self, input_ids, attention_mask, token_type_ids):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask,
                           token_type_ids=token_type_ids, return_dict=True)
        seq_output = outputs.last_hidden_state
        return self.classifier(self.dropout(seq_output))

    def forward(self, input_ids, attention_mask, token_type_ids, labels=None):
        emissions = self._get_emissions(input_ids, attention_mask, token_type_ids)
        mask = attention_mask.bool()

        loss = None
        if labels is not None:
            labels_crf = labels.clone()
            labels_crf[labels_crf == -100] = 0
            loss = -self.crf(emissions, labels_crf, mask=mask, reduction="mean")

        return emissions, loss

    def decode(self, input_ids, attention_mask, token_type_ids):
        """Viterbi 解码。"""
        emissions = self._get_emissions(input_ids, attention_mask, token_type_ids)
        mask = attention_mask.bool()
        return self.crf.decode(emissions, mask=mask)


def build_model(use_crf: bool, bert_path: str, dropout: float = 0.1) -> nn.Module:
    """构建模型。"""
    model_cls = BertCRFNER if use_crf else BertNER
    model = model_cls(bert_path=bert_path, dropout=dropout)

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    model_name = "BERT + CRF" if use_crf else "BERT + Linear"
    print(f"🤖 模型：{model_name}")
    print(f"   标签数：{NUM_LABELS}")
    print(f"   参数总量：{total_params / 1e6:.1f}M")
    print(f"   可训练参数：{trainable_params / 1e6:.1f}M")
    return model


# ==================== 训练与评估 ====================
def evaluate_epoch(model, loader, device, use_crf):
    """评估一个 epoch。"""
    from seqeval.metrics import f1_score as seqeval_f1

    model.eval()
    total_loss = 0.0
    all_preds = []
    all_golds = []

    with torch.no_grad():
        for batch in loader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            token_type_ids = batch["token_type_ids"].to(device)
            labels = batch["labels"].to(device)

            if use_crf:
                _, loss = model(input_ids, attention_mask, token_type_ids, labels)
                pred_ids_list = model.decode(input_ids, attention_mask, token_type_ids)
            else:
                _, loss = model(input_ids, attention_mask, token_type_ids, labels)
                logits, _ = model(input_ids, attention_mask, token_type_ids)
                pred_ids_list = logits.argmax(dim=-1).tolist()

            total_loss += loss.item()

            labels_np = labels.cpu().tolist()
            for i in range(len(input_ids)):
                gold_seq = []
                pred_seq = []
                token_labels = labels_np[i]

                for j, gold_id in enumerate(token_labels):
                    if gold_id == -100:
                        continue
                    gold_seq.append(ID2LABEL[gold_id])
                    if use_crf:
                        pred_seq.append(ID2LABEL.get(pred_ids_list[i][j] if j < len(pred_ids_list[i]) else 0, "O"))
                    else:
                        pred_seq.append(ID2LABEL.get(pred_ids_list[i][j], "O"))

                all_golds.append(gold_seq)
                all_preds.append(pred_seq)

    avg_loss = total_loss / len(loader)
    entity_f1 = seqeval_f1(all_golds, all_preds)
    return avg_loss, entity_f1, all_preds, all_golds


def train_one_epoch(model, loader, optimizer, scheduler, device, epoch, total_epochs, grad_accum):
    """训练一个 epoch。"""
    model.train()
    total_loss = 0.0
    optimizer.zero_grad()

    pbar = tqdm(loader, desc=f"Epoch {epoch}/{total_epochs} [Train]", leave=False)
    for step, batch in enumerate(pbar):
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        token_type_ids = batch["token_type_ids"].to(device)
        labels = batch["labels"].to(device)

        _, loss = model(input_ids, attention_mask, token_type_ids, labels)
        (loss / grad_accum).backward()
        total_loss += loss.item()

        if (step + 1) % grad_accum == 0:
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()

        pbar.set_postfix(loss=f"{loss.item():.4f}")

    remainder = len(loader) % grad_accum
    if remainder != 0:
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        scheduler.step()
        optimizer.zero_grad()

    return total_loss / len(loader)


def count_illegal_sequences(pred_seqs):
    """统计非法 BIO 序列。"""
    stats = {"illegal_start": 0, "illegal_transition": 0, "total_seqs": len(pred_seqs)}
    for seq in pred_seqs:
        if not seq:
            continue
        if seq[0].startswith("I-"):
            stats["illegal_start"] += 1
        for i in range(1, len(seq)):
            prev, curr = seq[i - 1], seq[i]
            if curr.startswith("I-"):
                curr_type = curr[2:]
                if prev == "O":
                    stats["illegal_transition"] += 1
                elif prev.startswith("B-") or prev.startswith("I-"):
                    prev_type = prev[2:]
                    if prev_type != curr_type:
                        stats["illegal_transition"] += 1
    stats["total_illegal"] = stats["illegal_start"] + stats["illegal_transition"]
    return stats


# ==================== 可视化 ====================
def plot_training_curve(log_records, run_tag):
    """绘制训练曲线。"""
    epochs = [r["epoch"] for r in log_records]
    train_losses = [r["train_loss"] for r in log_records]
    val_f1s = [r["val_entity_f1"] for r in log_records]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

    ax1.plot(epochs, train_losses, 'b-o', label='Train Loss')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.set_title('Training Loss Curve')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    ax2.plot(epochs, val_f1s, 'r-o', label='Val F1')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('F1 Score')
    ax2.set_title('Validation F1 Curve')
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIG_DIR / f"training_curve_{run_tag}.png", dpi=120)
    print(f"  📊 训练曲线已保存 → {FIG_DIR / f'training_curve_{run_tag}.png'}")
    plt.close()


# ==================== 主流程 ====================
def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"⚡ 设备：{device}")
    print(f"🏷️  BIO 标签数：{NUM_LABELS}（{LABELS}）")

    # 创建目录
    CKPT_DIR.mkdir(parents=True, exist_ok=True)
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    FIG_DIR.mkdir(parents=True, exist_ok=True)

    # Tokenizer
    print("\n📝 加载 Tokenizer...")
    tokenizer = BertTokenizer.from_pretrained(str(args.bert_path))

    # DataLoader
    print("\n📊 加载数据...")
    train_loader, val_loader, test_loader = build_dataloaders(
        tokenizer=tokenizer,
        batch_size=args.batch_size,
        max_length=args.max_length,
    )

    # 模型
    print("\n🔧 构建模型...")
    model = build_model(
        use_crf=args.use_crf,
        bert_path=str(args.bert_path),
        dropout=args.dropout,
    ).to(device)

    # 优化器（分层学习率）
    bert_params = list(model.bert.parameters())
    head_params = (
        list(model.classifier.parameters()) +
        list(model.dropout.parameters()) +
        (list(model.crf.parameters()) if args.use_crf else [])
    )
    optimizer = AdamW([
        {"params": bert_params, "lr": args.lr},
        {"params": head_params, "lr": args.lr * args.head_lr_mult},
    ], weight_decay=0.01)

    total_steps = len(train_loader) * args.epochs // args.grad_accum
    warmup_steps = int(total_steps * args.warmup_ratio)
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=warmup_steps, num_training_steps=total_steps
    )
    print(f"📈 训练步数：{total_steps}，预热步数：{warmup_steps}")

    # 训练
    run_tag = "crf" if args.use_crf else "linear"
    ckpt_path = CKPT_DIR / f"best_{run_tag}.pt"
    log_path = LOG_DIR / f"train_{run_tag}.json"

    best_f1 = 0.0
    log_records = []

    print(f"\n{'='*70}")
    print(f"开始训练（{'BERT+CRF' if args.use_crf else 'BERT+Linear'}）...")
    print(f"{'='*70}")

    for epoch in range(1, args.epochs + 1):
        t0 = time.time()
        train_loss = train_one_epoch(
            model, train_loader, optimizer, scheduler, device,
            epoch, args.epochs, args.grad_accum
        )
        val_loss, val_f1, _, _ = evaluate_epoch(model, val_loader, device, args.use_crf)
        elapsed = time.time() - t0

        print(
            f"Epoch {epoch}/{args.epochs} | "
            f"train_loss={train_loss:.4f} | "
            f"val_loss={val_loss:.4f} | "
            f"val_entity_f1={val_f1:.4f} | "
            f"time={elapsed:.0f}s"
        )

        log_records.append({
            "epoch": epoch,
            "train_loss": round(train_loss, 6),
            "val_loss": round(val_loss, 6),
            "val_entity_f1": round(val_f1, 6),
            "elapsed_s": round(elapsed, 1),
        })

        if val_f1 > best_f1:
            best_f1 = val_f1
            torch.save({
                "epoch": epoch,
                "use_crf": args.use_crf,
                "state_dict": model.state_dict(),
                "val_entity_f1": val_f1,
                "label2id": LABEL2ID,
                "id2label": ID2LABEL,
                "args": vars(args),
            }, ckpt_path)
            print(f"  ⭐ 新最优 F1={val_f1:.4f}，已保存 → {ckpt_path}")

    # 保存训练日志
    with open(log_path, "w", encoding="utf-8") as f:
        json.dump(log_records, f, ensure_ascii=False, indent=2)

    # 绘制训练曲线
    plot_training_curve(log_records, run_tag)

    print(f"\n✅ 训练完成！最优 val_entity_f1={best_f1:.4f}")
    print(f"   Checkpoint: {ckpt_path}")
    print(f"   训练日志:   {log_path}")

    # 测试集评估
    print(f"\n{'='*70}")
    print("在测试集上评估...")
    print(f"{'='*70}")

    # 重新加载最佳模型
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["state_dict"])
    print(f"加载最佳模型（epoch={ckpt['epoch']}，val_f1={ckpt['val_entity_f1']:.4f}）")

    test_loss, test_f1, all_preds, all_golds = evaluate_epoch(
        model, test_loader, device, args.use_crf
    )

    from seqeval.metrics import precision_score, recall_score, classification_report as seqeval_report

    p = precision_score(all_golds, all_preds)
    r = recall_score(all_golds, all_preds)
    f1 = test_f1

    print("\n" + "=" * 70)
    print(f"测试结果 | 模型：{'BERT + CRF' if args.use_crf else 'BERT + Linear'}")
    print("=" * 70)
    print(f"Entity-level Precision: {p:.4f}")
    print(f"Entity-level Recall:    {r:.4f}")
    print(f"Entity-level F1:        {f1:.4f}")

    print("\n【逐类型 F1】")
    print(seqeval_report(all_golds, all_preds, digits=4))

    # 非法序列统计
    illegal_stats = count_illegal_sequences(all_preds)
    print("【非法 BIO 序列统计】")
    print(f"  总序列数：{illegal_stats['total_seqs']}")
    print(f"  非法开头（I-X 开头）：{illegal_stats['illegal_start']} 条")
    print(f"  非法转移（B-X/I-X → I-Y, X≠Y）：{illegal_stats['illegal_transition']} 条")
    print(f"  合计非法序列：{illegal_stats['total_illegal']} 条")
    pct = illegal_stats["total_illegal"] / max(illegal_stats["total_seqs"], 1) * 100
    if args.use_crf:
        if illegal_stats["total_illegal"] == 0:
            print("  ✅ CRF Viterbi 解码：非法序列 0 条（转移矩阵已充分学习约束）")
        else:
            print(f"  ⚠️  CRF 非法序列 {illegal_stats['total_illegal']} 条（{pct:.1f}%）")
    else:
        print(f"  ℹ️  线性头约 {pct:.1f}% 的序列含非法转移，充分训练的 CRF 可完全消除")

    # 保存评估结果
    result = {
        "model": "BERT+CRF" if args.use_crf else "BERT+Linear",
        "split": "test",
        "precision": round(p, 6),
        "recall": round(r, 6),
        "f1": round(f1, 6),
        "illegal_stats": illegal_stats,
    }
    eval_path = LOG_DIR / f"eval_{run_tag}_test.json"
    with open(eval_path, "w", encoding="utf-8") as fout:
        json.dump(result, fout, ensure_ascii=False, indent=2)
    print(f"\n💾 评估结果已保存 → {eval_path}")

    print(f"\n{'='*70}")
    print("🎉 全部完成！")
    print(f"{'='*70}")
    print(f"\n📊 查看结果：")
    print(f"  - 模型检查点：{ckpt_path}")
    print(f"  - 训练日志：{log_path}")
    print(f"  - 评估结果：{eval_path}")
    print(f"  - 训练曲线：{FIG_DIR / f'training_curve_{run_tag}.png'}")


def parse_args():
    parser = argparse.ArgumentParser(description="人民日报 NER 一体化训练脚本")
    parser.add_argument("--use_crf", action="store_true", help="使用 CRF 层（否则使用线性头）")
    parser.add_argument("--bert_path", type=Path, default=BERT_PATH)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--max_length", type=int, default=128)
    parser.add_argument("--lr", type=float, default=2e-5, help="BERT 层学习率")
    parser.add_argument("--head_lr_mult", type=float, default=5.0, help="分类头学习率倍数")
    parser.add_argument("--warmup_ratio", type=float, default=0.1)
    parser.add_argument("--grad_accum", type=int, default=1)
    parser.add_argument("--dropout", type=float, default=0.1)
    return parser.parse_args()


if __name__ == "__main__":
    main()
