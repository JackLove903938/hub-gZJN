import os
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import sys
import io
import json
import time
import argparse
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
from transformers import BertTokenizer, get_linear_schedule_with_warmup
from tqdm import tqdm
from seqeval.metrics import (
    f1_score as seqeval_f1,
    precision_score as seqeval_precision,
    recall_score as seqeval_recall,
    classification_report as seqeval_report,
)

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")

# ── 路径配置 ─────────────────────────────────────────────────────────────────
ROOT = Path(__file__).parent
DATA_DIR = ROOT / "data" / "peoples_daily"
BERT_PATH = ROOT.parent.parent / "pretrain_models" / "bert-base-chinese"
CKPT_DIR = ROOT / "outputs" / "checkpoints"
LOG_DIR = ROOT / "outputs" / "logs"

# 添加 src 目录到路径，以便复用 model.py
sys.path.insert(0, str(ROOT / "src"))
from model import build_model


# ══════════════════════════════════════════════════════════════════════════════
# 一、标签体系
# ══════════════════════════════════════════════════════════════════════════════

LABEL_NAMES = ["O", "B-PER", "I-PER", "B-ORG", "I-ORG", "B-LOC", "I-LOC"]


def build_label_schema():
    label2id = {lbl: i for i, lbl in enumerate(LABEL_NAMES)}
    id2label = {i: lbl for i, lbl in enumerate(LABEL_NAMES)}
    return LABEL_NAMES, label2id, id2label


# ══════════════════════════════════════════════════════════════════════════════
# 二、数据集类
# ══════════════════════════════════════════════════════════════════════════════

class PeoplesDailyDataset(Dataset):
    """人民日报 NER 数据集。

    数据格式：{"tokens": ["海","钓",...], "ner_tags": ["O","B-LOC",...]}
    已是字符级 BIO 标注，无需 span→BIO 转换。

    处理流程：
      tokens → BertTokenizer(is_split_into_words=True)
             → word_ids() 对齐子词标签（非首子词设 -100）
    """

    def __init__(self, records, tokenizer, label2id, max_length=128):
        self.records = records
        self.tokenizer = tokenizer
        self.label2id = label2id
        self.max_length = max_length

    def __len__(self):
        return len(self.records)

    def __getitem__(self, idx):
        item = self.records[idx]
        tokens = item["tokens"]
        ner_tags = item["ner_tags"]

        char_label_ids = [self.label2id.get(tag, 0) for tag in ner_tags]

        encoding = self.tokenizer(
            tokens,
            is_split_into_words=True,
            max_length=self.max_length,
            truncation=True,
            padding="max_length",
            return_tensors="pt",
        )

        word_ids = encoding.word_ids(batch_index=0)
        aligned_labels = []
        prev_word_id = None
        for wid in word_ids:
            if wid is None:
                aligned_labels.append(-100)
            elif wid != prev_word_id:
                if wid < len(char_label_ids):
                    aligned_labels.append(char_label_ids[wid])
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


def build_dataloaders(tokenizer, label2id, batch_size=32, max_length=128):
    splits = {}
    for split in ["train", "validation", "test"]:
        path = DATA_DIR / f"{split}.json"
        with open(path, "r", encoding="utf-8") as f:
            splits[split] = json.load(f)

    train_ds = PeoplesDailyDataset(splits["train"], tokenizer, label2id, max_length)
    val_ds = PeoplesDailyDataset(splits["validation"], tokenizer, label2id, max_length)
    test_ds = PeoplesDailyDataset(splits["test"], tokenizer, label2id, max_length)

    print(f"数据集规模：训练={len(train_ds)}，验证={len(val_ds)}，测试={len(test_ds)}")

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=0)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=0)
    return train_loader, val_loader, test_loader


# ══════════════════════════════════════════════════════════════════════════════
# 三、训练与评估
# ══════════════════════════════════════════════════════════════════════════════

def evaluate_epoch(model, loader, id2label, device, use_crf):
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
                logits, loss = model(input_ids, attention_mask, token_type_ids, labels)
                pred_ids_list = logits.argmax(dim=-1).tolist()

            total_loss += loss.item()
            labels_np = labels.cpu().tolist()

            for i in range(len(input_ids)):
                gold_seq = []
                pred_seq = []
                token_labels = labels_np[i]
                pred_ids = pred_ids_list[i]

                for j, gold_id in enumerate(token_labels):
                    if gold_id == -100:
                        continue
                    gold_seq.append(id2label[gold_id])
                    if use_crf:
                        pred_seq.append(id2label.get(pred_ids[j] if j < len(pred_ids) else 0, "O"))
                    else:
                        pred_seq.append(id2label.get(pred_ids[j], "O"))

                all_golds.append(gold_seq)
                all_preds.append(pred_seq)

    avg_loss = total_loss / len(loader)
    entity_f1 = seqeval_f1(all_golds, all_preds)
    return avg_loss, entity_f1, all_preds, all_golds


def train_one_epoch(model, loader, optimizer, scheduler, device, epoch, total_epochs, grad_accum):
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


# ══════════════════════════════════════════════════════════════════════════════
# 四、主程序
# ══════════════════════════════════════════════════════════════════════════════

def parse_args():
    parser = argparse.ArgumentParser(description="人民日报 NER 序列标注训练")
    parser.add_argument("--use_crf", action="store_true", help="使用 CRF 层")
    parser.add_argument("--bert_path", type=Path, default=BERT_PATH)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--max_length", type=int, default=128)
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--head_lr_mult", type=float, default=5.0)
    parser.add_argument("--warmup_ratio", type=float, default=0.1)
    parser.add_argument("--grad_accum", type=int, default=1)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--eval_only", action="store_true", help="仅评估已有 checkpoint")
    return parser.parse_args()


def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"设备：{device}")

    labels, label2id, id2label = build_label_schema()
    num_labels = len(labels)
    print(f"BIO 标签数：{num_labels}（{labels}）")

    tokenizer = BertTokenizer.from_pretrained(str(args.bert_path))

    train_loader, val_loader, test_loader = build_dataloaders(
        tokenizer=tokenizer,
        label2id=label2id,
        batch_size=args.batch_size,
        max_length=args.max_length,
    )

    model = build_model(
        use_crf=args.use_crf,
        bert_path=str(args.bert_path),
        num_labels=num_labels,
        dropout=args.dropout,
    ).to(device)

    run_tag = "pd_crf" if args.use_crf else "pd_linear"
    CKPT_DIR.mkdir(parents=True, exist_ok=True)
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    ckpt_path = CKPT_DIR / f"best_{run_tag}.pt"

    # ── 仅评估模式 ──────────────────────────────────────────────────────────
    if args.eval_only:
        if not ckpt_path.exists():
            print(f"[错误] checkpoint 不存在：{ckpt_path}")
            print(f"请先运行训练：python 作业_peoples_daily_ner.py {'--use_crf' if args.use_crf else ''}")
            return
        ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
        model.load_state_dict(ckpt["state_dict"])
        print(f"加载 checkpoint（epoch={ckpt['epoch']}，val_f1={ckpt['val_entity_f1']:.4f}）")

        print("\n正在评估验证集...")
        _, val_f1, val_preds, val_golds = evaluate_epoch(model, val_loader, id2label, device, args.use_crf)
        print_evaluation_results(val_preds, val_golds, args.use_crf, "validation")
        return

    # ── 训练模式 ─────────────────────────────────────────────────────────────
    bert_params = list(model.bert.parameters())
    head_params = (
        list(model.classifier.parameters()) +
        list(model.dropout.parameters()) +
        (list(model.crf.parameters()) if args.use_crf else [])
    )
    optimizer = AdamW(
        [
            {"params": bert_params, "lr": args.lr},
            {"params": head_params, "lr": args.lr * args.head_lr_mult},
        ],
        weight_decay=0.01,
    )

    total_steps = len(train_loader) * args.epochs // args.grad_accum
    warmup_steps = int(total_steps * args.warmup_ratio)
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=warmup_steps, num_training_steps=total_steps
    )
    model_name = "BERT+CRF" if args.use_crf else "BERT+Linear"
    print(f"\n训练步数：{total_steps}，预热步数：{warmup_steps}")
    print(f"开始训练（{model_name}）...\n")

    best_f1 = 0.0
    log_records = []

    for epoch in range(1, args.epochs + 1):
        t0 = time.time()
        train_loss = train_one_epoch(
            model, train_loader, optimizer, scheduler, device,
            epoch, args.epochs, args.grad_accum
        )
        val_loss, val_f1, _, _ = evaluate_epoch(model, val_loader, id2label, device, args.use_crf)
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
                "label2id": label2id,
                "id2label": id2label,
                "args": vars(args),
            }, ckpt_path)
            print(f"  -> 新最优 F1={val_f1:.4f}，已保存 -> {ckpt_path}")

    log_path = LOG_DIR / f"train_{run_tag}.json"
    with open(log_path, "w", encoding="utf-8") as f:
        json.dump(log_records, f, ensure_ascii=False, indent=2)

    print(f"\n训练完成！最优 val_entity_f1={best_f1:.4f}")
    print(f"  Checkpoint: {ckpt_path}")
    print(f"  训练日志:   {log_path}")

    # ── 最终评估 ─────────────────────────────────────────────────────────────
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["state_dict"])
    print(f"\n{'='*70}")
    print(f"  最终评估（加载最优 checkpoint, epoch={ckpt['epoch']}）")
    print(f"{'='*70}")

    print("\n[验证集评估]")
    _, _, val_preds, val_golds = evaluate_epoch(model, val_loader, id2label, device, args.use_crf)
    print_evaluation_results(val_preds, val_golds, args.use_crf, "validation")


def print_evaluation_results(all_preds, all_golds, use_crf, split_name):
    p = seqeval_precision(all_golds, all_preds)
    r = seqeval_recall(all_golds, all_preds)
    f1 = seqeval_f1(all_golds, all_preds)

    model_name = "BERT+CRF" if use_crf else "BERT+Linear"
    print(f"\n  模型：{model_name}  |  数据集：peoples_daily  |  评估集：{split_name}")
    print(f"  {'─'*50}")
    print(f"  Entity Precision: {p:.4f}")
    print(f"  Entity Recall:    {r:.4f}")
    print(f"  Entity F1:        {f1:.4f}")

    print(f"\n  【逐类型 F1】")
    print(seqeval_report(all_golds, all_preds, digits=4))

    illegal_stats = count_illegal_sequences(all_preds)
    print(f"  【非法 BIO 序列统计】")
    print(f"    总序列数：{illegal_stats['total_seqs']}")
    print(f"    非法开头（I-X 开头）：{illegal_stats['illegal_start']} 条")
    print(f"    非法转移（类型不一致）：{illegal_stats['illegal_transition']} 条")
    print(f"    合计非法序列：{illegal_stats['total_illegal']} 条")

    if use_crf:
        if illegal_stats["total_illegal"] == 0:
            print(f"    -> CRF Viterbi 解码保证合法序列（非法 = 0）")
        else:
            print(f"    -> CRF 仍有 {illegal_stats['total_illegal']} 条非法（训练轮数不足）")
    else:
        pct = illegal_stats["total_illegal"] / max(illegal_stats["total_seqs"], 1) * 100
        print(f"    -> 线性头 {pct:.1f}% 含非法转移，CRF 可消除")


if __name__ == "__main__":
    main()
