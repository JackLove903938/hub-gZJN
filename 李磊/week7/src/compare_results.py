"""
加载 BERT Linear / CRF 模型，在测试集上推理并对比

使用方式：
  python compare_results.py

前提：
  - output/checkpoints/best_linear.pt   （已运行 python train.py）
  - output/checkpoints/best_crf.pt      （已运行 python train.py --use_crf）
  - output/logs/eval_llm.json           （可选，已运行 llm_ner.py）
"""

import os
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import json
from pathlib import Path

import torch
from transformers import BertTokenizer
from seqeval.metrics import f1_score, precision_score, recall_score

from dataset import build_label_schema, build_dataloaders
from model import build_model

ROOT = Path(__file__).parent.parent
BERT_PATH = ROOT / "model" / "bert-base-chinese"
DATA_DIR = ROOT / "data"
CKPT_DIR = ROOT / "output" / "checkpoints"
LOG_DIR = ROOT / "output" / "logs"


def count_illegal_sequences(pred_seqs: list[list[str]]) -> dict:
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
                    if prev[2:] != curr_type:
                        stats["illegal_transition"] += 1
    stats["total_illegal"] = stats["illegal_start"] + stats["illegal_transition"]
    return stats


def run_inference(model, loader, id2label, device, use_crf):
    model.eval()
    all_preds, all_golds = [], []
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
                gold_seq, pred_seq = [], []
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


def load_json(path: Path) -> dict | None:
    if not path.exists():
        return None
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"设备：{device}")

    labels, label2id, id2label = build_label_schema()
    num_labels = len(labels)
    print(f"BIO 标签数：{num_labels}")

    tokenizer = BertTokenizer.from_pretrained(str(BERT_PATH))
    _, _, test_loader = build_dataloaders(
        tokenizer=tokenizer, label2id=label2id, data_dir=DATA_DIR,
    )

    results = {}
    model_names = {"linear": "BERT + Linear", "crf": "BERT + CRF"}

    for run_tag, use_crf in [("linear", False), ("crf", True)]:
        # 扫描 run_tag 开头的子目录，取最新的
        run_dirs = sorted([
            d for d in CKPT_DIR.iterdir()
            if d.is_dir() and d.name.startswith(run_tag) and (d / "best.pt").exists()
        ], key=lambda d: d.stat().st_mtime, reverse=True)
        ckpt_path = run_dirs[0] / "best.pt" if run_dirs else CKPT_DIR / f"best_{run_tag}.pt"
        if not ckpt_path.exists():
            print(f"\n未找到 checkpoint：{ckpt_path}，跳过")
            continue

        ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
        model = build_model(
            use_crf=use_crf, bert_path=str(BERT_PATH), num_labels=num_labels,
        ).to(device)
        model.load_state_dict(ckpt["state_dict"])
        print(f"加载 {model_names[run_tag]}（epoch={ckpt['epoch']}，val_f1={ckpt['val_entity_f1']:.4f}）")

        print(f"  正在在 [test] 集上推理...")
        all_preds, all_golds = run_inference(model, test_loader, id2label, device, use_crf)

        p = precision_score(all_golds, all_preds)
        r = recall_score(all_golds, all_preds)
        f1 = f1_score(all_golds, all_preds)
        illegal = count_illegal_sequences(all_preds)

        results[run_tag] = {
            "model": model_names[run_tag],
            "precision": round(p, 6),
            "recall": round(r, 6),
            "f1": round(f1, 6),
            "illegal_stats": illegal,
        }

    # 加载 LLM 结果（可选）
    llm_res = load_json(LOG_DIR / "eval_llm.json")

    # 打印对比表
    print("\n" + "=" * 80)
    print("BERT NER 项目 — 方案汇总对比（测试集）")
    print("=" * 80)
    header = f"{'方案':<25} {'Precision':>10} {'Recall':>10} {'F1':>10} {'非法序列':>10}"
    print(header)
    print("-" * 72)

    linear_res = results.get("linear")
    crf_res = results.get("crf")

    if linear_res:
        ill = linear_res["illegal_stats"]["total_illegal"]
        print(f"{'BERT + Linear':<25} {linear_res['precision']:>10.4f} {linear_res['recall']:>10.4f} {linear_res['f1']:>10.4f} {ill:>10d}")
    else:
        print(f"{'BERT + Linear':<25} {'（未找到 checkpoint，请运行 python train.py）':>50}")

    if crf_res:
        ill = crf_res["illegal_stats"]["total_illegal"]
        print(f"{'BERT + CRF':<25} {crf_res['precision']:>10.4f} {crf_res['recall']:>10.4f} {crf_res['f1']:>10.4f} {ill:>10d}")
    else:
        print(f"{'BERT + CRF':<25} {'（未找到 checkpoint，请运行 python train.py --use_crf）':>50}")

    if llm_res:
        zs = llm_res["zero_shot"]
        fs = llm_res["few_shot"]
        model_name = llm_res.get("model", "qwen-plus")
        n = llm_res.get("n_samples", "?")
        print(f"{f'LLM zero-shot ({model_name})':<25} {zs['precision']:>10.4f} {zs['recall']:>10.4f} {zs['f1']:>10.4f} {'N/A':>10}")
        print(f"{f'LLM few-shot ({model_name})':<25} {fs['precision']:>10.4f} {fs['recall']:>10.4f} {fs['f1']:>10.4f} {'N/A':>10}")
        print(f"\n  注：LLM 结果基于 {n} 条采样，非完整测试集")

    # 教学结论
    print("\n" + "=" * 80)
    print("关键教学结论：")
    if linear_res and crf_res:
        f1_diff = crf_res["f1"] - linear_res["f1"]
        ill_linear = linear_res["illegal_stats"]["total_illegal"]
        print(f"  1. CRF vs Linear：F1 {'↑' if f1_diff >= 0 else '↓'}{abs(f1_diff):.4f}")
        print(f"  2. 线性头非法序列：{ill_linear} 条；CRF 非法序列：{crf_res['illegal_stats']['total_illegal']} 条")
        print(f"     → CRF 通过 Viterbi 解码在数学上保证序列合法性")
    if llm_res and linear_res:
        fs_f1 = llm_res["few_shot"]["f1"]
        gap = linear_res["f1"] - fs_f1
        print(f"  3. 微调 BERT vs LLM few-shot：F1 差距 {gap:.4f}")
        print(f"     → 特定领域NER任务中，小模型微调通常显著优于大模型zero/few-shot")
    print("=" * 80)

    # 保存结果
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    out_path = LOG_DIR / "compare_test.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print(f"\n对比结果已保存 → {out_path}")


if __name__ == "__main__":
    main()
