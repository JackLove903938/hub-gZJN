"""
训练与评估流水线

教学重点：
  1. 分层学习率：BERT 层用 2e-5，分类头用 1e-4（加速头部收敛）
  2. Linear Warmup：防止训练初期大梯度破坏预训练参数
  3. seqeval 评估：entity-level F1（不是 token-level accuracy）
  4. 非法序列统计：量化 CRF vs 线性头的关键差异

使用方式：
  from train_eval import train_model, evaluate_model
"""

import os
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import json
import time
from pathlib import Path

import torch
import torch.nn as nn
from torch.optim import AdamW
from transformers import get_linear_schedule_with_warmup
from tqdm import tqdm
from seqeval.metrics import (
    f1_score as seqeval_f1,
    precision_score,
    recall_score,
    classification_report as seqeval_report,
)


def train_one_epoch(
    model: nn.Module,
    loader,
    optimizer,
    scheduler,
    device: torch.device,
    epoch: int,
    total_epochs: int,
    grad_accum: int,
) -> float:
    """训练一个 epoch，返回平均 loss。"""
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

    # 处理最后不足 grad_accum 的批次
    remainder = len(loader) % grad_accum
    if remainder != 0:
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        scheduler.step()
        optimizer.zero_grad()

    return total_loss / len(loader)


def evaluate_epoch(
    model: nn.Module,
    loader,
    id2label: dict,
    device: torch.device,
    use_crf: bool,
) -> tuple[float, float]:
    """在 loader 上评估，返回 (avg_loss, entity_f1)。"""
    model.eval()
    total_loss = 0.0
    all_preds: list[list[str]] = []
    all_golds: list[list[str]] = []

    with torch.no_grad():
        for batch in loader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            token_type_ids = batch["token_type_ids"].to(device)
            labels = batch["labels"].to(device)

            if use_crf:
                emissions, loss = model(input_ids, attention_mask, token_type_ids, labels)
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

                for j, gold_id in enumerate(token_labels):
                    if gold_id == -100:
                        continue
                    gold_seq.append(id2label[gold_id])
                    if use_crf:
                        if j < len(pred_ids_list[i]):
                            pred_seq.append(id2label.get(pred_ids_list[i][j], "O"))
                        else:
                            pred_seq.append("O")
                    else:
                        pred_seq.append(id2label.get(pred_ids_list[i][j], "O"))

                all_golds.append(gold_seq)
                all_preds.append(pred_seq)

    avg_loss = total_loss / len(loader)
    entity_f1 = seqeval_f1(all_golds, all_preds)
    return avg_loss, entity_f1


def train_model(
    model: nn.Module,
    train_loader,
    val_loader,
    device: torch.device,
    id2label: dict,
    use_crf: bool = False,
    epochs: int = 3,
    lr: float = 2e-5,
    head_lr_mult: float = 5.0,
    warmup_ratio: float = 0.1,
    grad_accum: int = 1,
    ckpt_dir: Path = None,
    log_dir: Path = None,
) -> tuple[nn.Module, dict]:
    """
    完整训练流程。

    返回：
      model: 加载了最优权重的模型
      best_info: {"best_f1": float, "ckpt_path": Path, "log_path": Path}
    """
    # 分层学习率
    bert_params = list(model.bert.parameters())
    head_params = (
        list(model.classifier.parameters()) +
        list(model.dropout.parameters()) +
        (list(model.crf.parameters()) if use_crf else [])
    )
    optimizer = AdamW(
        [
            {"params": bert_params, "lr": lr},
            {"params": head_params, "lr": lr * head_lr_mult},
        ],
        weight_decay=0.01,
    )

    total_steps = len(train_loader) * epochs // grad_accum
    warmup_steps = int(total_steps * warmup_ratio)
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=warmup_steps, num_training_steps=total_steps
    )
    print(f"\n训练步数：{total_steps}，预热步数：{warmup_steps}")

    run_tag = "crf" if use_crf else "linear"
    if ckpt_dir:
        ckpt_dir.mkdir(parents=True, exist_ok=True)
    if log_dir:
        log_dir.mkdir(parents=True, exist_ok=True)
    ckpt_path = (ckpt_dir or Path(".")) / f"best_{run_tag}.pt"
    log_path = (log_dir or Path(".")) / f"train_{run_tag}.json"

    best_f1 = 0.0
    log_records = []

    print(f"\n开始训练（{'BERT+CRF' if use_crf else 'BERT+Linear'}）...")
    for epoch in range(1, epochs + 1):
        t0 = time.time()
        train_loss = train_one_epoch(
            model, train_loader, optimizer, scheduler, device,
            epoch, epochs, grad_accum
        )
        val_loss, val_f1 = evaluate_epoch(model, val_loader, id2label, device, use_crf)
        elapsed = time.time() - t0

        print(
            f"Epoch {epoch}/{epochs} | "
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
            torch.save(
                {
                    "epoch": epoch,
                    "use_crf": use_crf,
                    "state_dict": model.state_dict(),
                    "val_entity_f1": val_f1,
                    "args": {
                        "epochs": epochs,
                        "lr": lr,
                        "head_lr_mult": head_lr_mult,
                    },
                },
                ckpt_path,
            )
            print(f"  ★ 新最优 F1={val_f1:.4f}，已保存 → {ckpt_path}")

    with open(log_path, "w", encoding="utf-8") as f:
        json.dump(log_records, f, ensure_ascii=False, indent=2)

    print(f"\n训练完成！最优 val_entity_f1={best_f1:.4f}")

    # 加载最优权重
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["state_dict"])
    return model, {"best_f1": best_f1, "ckpt_path": ckpt_path, "log_path": log_path}


def count_illegal_sequences(pred_seqs: list[list[str]]) -> dict:
    """统计非法 BIO 序列数量。"""
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

    total_illegal = stats["illegal_start"] + stats["illegal_transition"]
    stats["total_illegal"] = total_illegal
    return stats


def run_inference(
    model,
    loader,
    id2label: dict,
    device: torch.device,
    use_crf: bool,
) -> tuple[list[list[str]], list[list[str]]]:
    """在 loader 上推理，返回 (all_preds, all_golds)。"""
    model.eval()
    all_preds = []
    all_golds = []

    with torch.no_grad():
        for batch in loader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            token_type_ids = batch["token_type_ids"].to(device)
            labels = batch["labels"].to(device)

            if use_crf:
                pred_ids_list = model.decode(input_ids, attention_mask, token_type_ids)
            else:
                logits, _ = model(input_ids, attention_mask, token_type_ids)
                pred_ids_list = logits.argmax(dim=-1).tolist()

            labels_list = labels.cpu().tolist()

            for i in range(len(input_ids)):
                gold_seq = []
                pred_seq = []
                token_labels = labels_list[i]

                for j, gold_id in enumerate(token_labels):
                    if gold_id == -100:
                        continue
                    gold_seq.append(id2label[gold_id])
                    if use_crf:
                        pred_seq.append(id2label.get(pred_ids_list[i][j] if j < len(pred_ids_list[i]) else 0, "O"))
                    else:
                        pred_seq.append(id2label.get(pred_ids_list[i][j], "O"))

                all_golds.append(gold_seq)
                all_preds.append(pred_seq)

    return all_preds, all_golds


def evaluate_model(
    model: nn.Module,
    loader,
    id2label: dict,
    device: torch.device,
    use_crf: bool = False,
    split_name: str = "validation",
    log_dir: Path = None,
) -> dict:
    """
    完整评估流程，返回结果字典。
    """
    print(f"\n正在在 [{split_name}] 集上推理...")
    all_preds, all_golds = run_inference(model, loader, id2label, device, use_crf)

    # seqeval entity-level 指标
    p = precision_score(all_golds, all_preds)
    r = recall_score(all_golds, all_preds)
    f1 = seqeval_f1(all_golds, all_preds)

    print("\n" + "=" * 70)
    print(f"模型：{'BERT + CRF' if use_crf else 'BERT + Linear'}  |  评估集：{split_name}")
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
    if use_crf:
        if illegal_stats["total_illegal"] == 0:
            print("  → CRF Viterbi 解码：非法序列 0 条 ✓（转移矩阵已充分学习约束）")
        else:
            print(f"  → CRF 非法序列 {illegal_stats['total_illegal']} 条（{pct:.1f}%）")
            print(f"  → 提示：训练 epoch 不足时转移矩阵尚未收敛；充分训练（3+ epochs）后可降至 0")
    else:
        print(f"  → 线性头约 {pct:.1f}% 的序列含非法转移，充分训练的 CRF 可完全消除")

    # 保存结果
    if log_dir:
        log_dir.mkdir(parents=True, exist_ok=True)
    run_tag = "crf" if use_crf else "linear"
    result = {
        "model": "BERT+CRF" if use_crf else "BERT+Linear",
        "split": split_name,
        "precision": round(p, 6),
        "recall": round(r, 6),
        "f1": round(f1, 6),
        "illegal_stats": illegal_stats,
    }
    out_path = (log_dir or Path(".")) / f"eval_{run_tag}_{split_name}.json"
    with open(out_path, "w", encoding="utf-8") as fout:
        json.dump(result, fout, ensure_ascii=False, indent=2)
    print(f"\n评估结果已保存 → {out_path}")

    return result
