"""
运行示例：
python3 作业7/作业7.py --epochs 1 --train_limit 1000 --val_limit 500
python3 作业7/作业7.py --epochs 3 --train_limit -1 --val_limit -1
"""

import argparse
import json
import os
import random
import sys
import time
from pathlib import Path


ROOT_DIR = Path(__file__).resolve().parents[1]
WEEK7_PROJECT_DIR = (
    ROOT_DIR
    / "week7序列标注问题"
    / "week7 序列标注问题"
    / "序列标注项目"
)
DATA_DIR = WEEK7_PROJECT_DIR / "data" / "peoples_daily"
OUTPUT_DIR = Path(__file__).resolve().parent / "outputs"
DEFAULT_REPORT = Path(__file__).resolve().with_name("人民日报NER序列标注训练结果.md")


def import_dependencies():
    try:
        import torch
        import torch.nn as nn
        import torch.nn.functional as F
        if not hasattr(torch, "float8_e8m0fnu") and hasattr(torch, "float8_e5m2"):
            torch.float8_e8m0fnu = torch.float8_e5m2
        import transformers
        from torch.optim import AdamW
        from torch.utils.data import DataLoader, Dataset
        from tqdm import tqdm
        from transformers import BertModel, BertTokenizer, get_linear_schedule_with_warmup
    except ImportError as e:
        raise SystemExit(
            "缺少课程依赖，无法训练模型。\n"
            "请先安装 week7 项目依赖，例如：\n"
            f"  {sys.executable} -m pip install -r "
            f"'{WEEK7_PROJECT_DIR / 'requirements.txt'}'\n"
            f"原始错误：{type(e).__name__}: {e}"
        ) from e

    return {
        "torch": torch,
        "nn": nn,
        "F": F,
        "transformers": transformers,
        "AdamW": AdamW,
        "DataLoader": DataLoader,
        "Dataset": Dataset,
        "tqdm": tqdm,
        "BertModel": BertModel,
        "BertTokenizer": BertTokenizer,
        "get_linear_schedule_with_warmup": get_linear_schedule_with_warmup,
    }


def load_json(path):
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"文件不存在：{path}")
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def auto_bert_path():
    candidates = [
        ROOT_DIR / "week4语言模型" / "bert-base-chinese",
        ROOT_DIR
        / "week7序列标注问题"
        / "pretrain_models"
        / "bert-base-chinese",
    ]
    for path in candidates:
        if (path / "config.json").exists() and (path / "vocab.txt").exists():
            return path
    return "bert-base-chinese"


def set_seed(seed, torch):
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        torch.mps.manual_seed(seed)


def choose_device(device_name, torch):
    if device_name != "auto":
        return torch.device(device_name)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def limit_data(data, limit, seed):
    if limit is None or limit < 0 or limit >= len(data):
        return list(data)
    rng = random.Random(seed)
    indices = list(range(len(data)))
    rng.shuffle(indices)
    return [data[i] for i in indices[:limit]]


def build_label_schema(label_names):
    label2id = {label: idx for idx, label in enumerate(label_names)}
    id2label = {idx: label for label, idx in label2id.items()}
    return label2id, id2label


def align_bio_tags(word_ids, tag_ids):
    """按 tokenizer.word_ids() 把字符级 BIO 标签对齐到 BERT token。"""
    aligned = []
    previous_word_id = None
    for word_id in word_ids:
        if word_id is None:
            aligned.append(-100)
        elif word_id != previous_word_id:
            aligned.append(tag_ids[word_id] if word_id < len(tag_ids) else -100)
            previous_word_id = word_id
        else:
            aligned.append(-100)
    return aligned


def bio_entities(tags):
    """从 BIO 标签中抽取实体，返回 (type, start, end)；end 为闭区间。"""
    entities = []
    start = None
    ent_type = None

    for index, tag in enumerate(tags + ["O"]):
        if tag == "O" or tag.startswith("B-"):
            if ent_type is not None:
                entities.append((ent_type, start, index - 1))
                start = None
                ent_type = None
            if tag.startswith("B-"):
                ent_type = tag[2:]
                start = index
        elif tag.startswith("I-"):
            current_type = tag[2:]
            if ent_type is None or current_type != ent_type:
                if ent_type is not None:
                    entities.append((ent_type, start, index - 1))
                ent_type = current_type
                start = index
    return entities


def entity_prf(gold_sequences, pred_sequences):
    gold_total = 0
    pred_total = 0
    correct = 0

    for gold_tags, pred_tags in zip(gold_sequences, pred_sequences):
        gold_set = set(bio_entities(gold_tags))
        pred_set = set(bio_entities(pred_tags))
        gold_total += len(gold_set)
        pred_total += len(pred_set)
        correct += len(gold_set & pred_set)

    precision = correct / pred_total if pred_total else 0.0
    recall = correct / gold_total if gold_total else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return precision, recall, f1


def build_runtime_classes(deps):
    torch = deps["torch"]
    nn = deps["nn"]
    F = deps["F"]
    Dataset = deps["Dataset"]
    BertModel = deps["BertModel"]
    transformers = deps["transformers"]

    class PeoplesDailyDataset(Dataset):
        def __init__(self, records, tokenizer, label2id, max_length):
            self.records = records
            self.tokenizer = tokenizer
            self.label2id = label2id
            self.max_length = max_length

        def __len__(self):
            return len(self.records)

        def __getitem__(self, index):
            item = self.records[index]
            tokens = item["tokens"]
            tag_ids = [self.label2id[tag] for tag in item["ner_tags"]]

            encoded = self.tokenizer(
                tokens,
                is_split_into_words=True,
                max_length=self.max_length,
                truncation=True,
                padding="max_length",
                return_tensors="pt",
            )
            labels = align_bio_tags(encoded.word_ids(batch_index=0), tag_ids)

            return {
                "input_ids": encoded["input_ids"].squeeze(0),
                "attention_mask": encoded["attention_mask"].squeeze(0),
                "token_type_ids": encoded["token_type_ids"].squeeze(0),
                "labels": torch.tensor(labels, dtype=torch.long),
            }

    class BertTokenClassifier(nn.Module):
        def __init__(self, bert_path, num_labels, dropout=0.1):
            super().__init__()
            previous_level = transformers.logging.get_verbosity()
            transformers.logging.set_verbosity_error()
            self.bert = BertModel.from_pretrained(str(bert_path))
            transformers.logging.set_verbosity(previous_level)

            hidden_size = self.bert.config.hidden_size
            self.dropout = nn.Dropout(dropout)
            self.classifier = nn.Linear(hidden_size, num_labels)
            self.num_labels = num_labels

        def forward(self, input_ids, attention_mask, token_type_ids, labels=None):
            outputs = self.bert(
                input_ids=input_ids,
                attention_mask=attention_mask,
                token_type_ids=token_type_ids,
                return_dict=True,
            )
            logits = self.classifier(self.dropout(outputs.last_hidden_state))
            loss = None
            if labels is not None:
                loss = F.cross_entropy(
                    logits.view(-1, self.num_labels),
                    labels.view(-1),
                    ignore_index=-100,
                )
            return logits, loss

    return PeoplesDailyDataset, BertTokenClassifier


def build_dataloaders(train_data, val_data, tokenizer, label2id, args, deps):
    DataLoader = deps["DataLoader"]
    PeoplesDailyDataset, _ = build_runtime_classes(deps)
    train_dataset = PeoplesDailyDataset(train_data, tokenizer, label2id, args.max_length)
    val_dataset = PeoplesDailyDataset(val_data, tokenizer, label2id, args.max_length)
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=0,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0,
    )
    return train_loader, val_loader


def train_one_epoch(model, loader, optimizer, scheduler, device, deps, desc):
    torch = deps["torch"]
    nn = deps["nn"]
    tqdm = deps["tqdm"]
    model.train()
    total_loss = 0.0
    total_tokens = 0

    for batch in tqdm(loader, desc=desc, leave=False):
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        token_type_ids = batch["token_type_ids"].to(device)
        labels = batch["labels"].to(device)

        optimizer.zero_grad()
        _, loss = model(input_ids, attention_mask, token_type_ids, labels)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        scheduler.step()

        valid_tokens = (labels != -100).sum().item()
        total_loss += loss.item() * valid_tokens
        total_tokens += valid_tokens

    return total_loss / max(total_tokens, 1)


def evaluate(model, loader, id2label, device, deps):
    torch = deps["torch"]
    model.eval()
    total_loss = 0.0
    total_tokens = 0
    gold_sequences = []
    pred_sequences = []

    with torch.no_grad():
        for batch in loader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            token_type_ids = batch["token_type_ids"].to(device)
            labels = batch["labels"].to(device)

            logits, loss = model(input_ids, attention_mask, token_type_ids, labels)
            pred_ids = logits.argmax(dim=-1).cpu().tolist()
            label_ids = labels.cpu().tolist()
            valid_tokens = (labels != -100).sum().item()
            total_loss += loss.item() * valid_tokens
            total_tokens += valid_tokens

            for sample_labels, sample_preds in zip(label_ids, pred_ids):
                gold_tags = []
                pred_tags = []
                for gold_id, pred_id in zip(sample_labels, sample_preds):
                    if gold_id == -100:
                        continue
                    gold_tags.append(id2label[gold_id])
                    pred_tags.append(id2label[pred_id])
                gold_sequences.append(gold_tags)
                pred_sequences.append(pred_tags)

    precision, recall, f1 = entity_prf(gold_sequences, pred_sequences)
    return {
        "val_loss": total_loss / max(total_tokens, 1),
        "precision": precision,
        "recall": recall,
        "entity_f1": f1,
    }


def train_model(train_data, val_data, label2id, id2label, args, deps, device):
    torch = deps["torch"]
    AdamW = deps["AdamW"]
    BertTokenizer = deps["BertTokenizer"]
    get_linear_schedule_with_warmup = deps["get_linear_schedule_with_warmup"]
    _, BertTokenClassifier = build_runtime_classes(deps)

    tokenizer = BertTokenizer.from_pretrained(str(args.bert_path))
    train_loader, val_loader = build_dataloaders(
        train_data,
        val_data,
        tokenizer,
        label2id,
        args,
        deps,
    )
    model = BertTokenClassifier(args.bert_path, len(label2id), dropout=args.dropout).to(device)

    bert_params = list(model.bert.parameters())
    head_params = list(model.classifier.parameters()) + list(model.dropout.parameters())
    optimizer = AdamW(
        [
            {"params": bert_params, "lr": args.lr},
            {"params": head_params, "lr": args.lr * args.head_lr_mult},
        ],
        weight_decay=args.weight_decay,
    )
    total_steps = max(1, len(train_loader) * args.epochs)
    warmup_steps = int(total_steps * args.warmup_ratio)
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=total_steps,
    )

    print(f"模型：BERT + Linear token classifier")
    print(f"标签数：{len(label2id)}")
    print(f"训练步数：{total_steps}，预热步数：{warmup_steps}")

    best_metrics = None
    logs = []
    checkpoint_path = args.output_dir / "best_peoples_daily_ner.pt"
    start = time.time()

    for epoch in range(1, args.epochs + 1):
        epoch_start = time.time()
        train_loss = train_one_epoch(
            model,
            train_loader,
            optimizer,
            scheduler,
            device,
            deps,
            desc=f"epoch {epoch}/{args.epochs}",
        )
        metrics = evaluate(model, val_loader, id2label, device, deps)
        elapsed = time.time() - epoch_start
        row = {
            "epoch": epoch,
            "train_loss": round(train_loss, 6),
            "val_loss": round(metrics["val_loss"], 6),
            "precision": round(metrics["precision"], 6),
            "recall": round(metrics["recall"], 6),
            "entity_f1": round(metrics["entity_f1"], 6),
            "elapsed_s": round(elapsed, 1),
        }
        logs.append(row)

        print(
            f"epoch {epoch}: train_loss={train_loss:.4f} "
            f"val_loss={metrics['val_loss']:.4f} "
            f"P={metrics['precision']:.4f} R={metrics['recall']:.4f} "
            f"F1={metrics['entity_f1']:.4f} time={elapsed:.1f}s"
        )

        if best_metrics is None or metrics["entity_f1"] > best_metrics["entity_f1"]:
            best_metrics = dict(metrics)
            best_metrics["best_epoch"] = epoch
            torch.save(
                {
                    "epoch": epoch,
                    "state_dict": model.state_dict(),
                    "label2id": label2id,
                    "id2label": id2label,
                    "metrics": metrics,
                    "args": vars(args),
                },
                checkpoint_path,
            )
            print(f"  保存最优模型：{checkpoint_path}")

    return {
        "logs": logs,
        "best": best_metrics,
        "checkpoint_path": str(checkpoint_path),
        "elapsed_s": time.time() - start,
    }


def write_report(result, args, label_names, report_path):
    rows = []
    for item in result["logs"]:
        rows.append(
            "| {epoch} | {train_loss:.4f} | {val_loss:.4f} | {precision:.4f} | "
            "{recall:.4f} | {entity_f1:.4f} | {elapsed_s:.1f}s |".format(**item)
        )

    best = result["best"]
    content = f"""# 人民日报 NER 序列标注训练结果

## 实验设置

- 数据集：`{args.data_dir}`
- 训练样本数：{args.train_limit if args.train_limit >= 0 else "全部"}
- 验证样本数：{args.val_limit if args.val_limit >= 0 else "全部"}
- 标签体系：{", ".join(label_names)}
- BERT 路径：`{args.bert_path}`
- epoch：{args.epochs}
- batch size：{args.batch_size}
- max length：{args.max_length}

## 训练结果

| Epoch | Train Loss | Val Loss | Precision | Recall | Entity F1 | 耗时 |
|---:|---:|---:|---:|---:|---:|---:|
{chr(10).join(rows)}

## 结论

- 最优 epoch：{best["best_epoch"]}
- 最优 Entity F1：{best["entity_f1"]:.4f}
- 最优模型文件：`{result["checkpoint_path"]}`
"""
    report_path.write_text(content, encoding="utf-8")


def parse_args():
    parser = argparse.ArgumentParser(description="在人民日报 NER 数据集上训练序列标注模型")
    parser.add_argument("--data_dir", type=Path, default=DATA_DIR)
    parser.add_argument("--bert_path", default=str(auto_bert_path()))
    parser.add_argument("--output_dir", type=Path, default=OUTPUT_DIR)
    parser.add_argument("--report_path", type=Path, default=DEFAULT_REPORT)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--max_length", type=int, default=128)
    parser.add_argument("--train_limit", type=int, default=1000, help="-1 表示使用全部训练集")
    parser.add_argument("--val_limit", type=int, default=500, help="-1 表示使用全部验证集")
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--head_lr_mult", type=float, default=5.0)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--warmup_ratio", type=float, default=0.1)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="auto", help="auto/cpu/cuda/mps")
    return parser.parse_args()


def main():
    args = parse_args()
    os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
    deps = import_dependencies()
    torch = deps["torch"]
    set_seed(args.seed, torch)
    device = choose_device(args.device, torch)
    args.bert_path = str(Path(args.bert_path)) if Path(args.bert_path).exists() else args.bert_path

    label_names = load_json(args.data_dir / "label_names.json")
    label2id, id2label = build_label_schema(label_names)
    train_raw = load_json(args.data_dir / "train.json")
    val_raw = load_json(args.data_dir / "validation.json")
    train_data = limit_data(train_raw, args.train_limit, args.seed)
    val_data = limit_data(val_raw, args.val_limit, args.seed + 1)

    print(f"使用设备：{device}")
    print(f"BERT 路径：{args.bert_path}")
    print(f"数据集：人民日报 NER")
    print(f"训练样本：{len(train_data):,}，验证样本：{len(val_data):,}")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    result = train_model(train_data, val_data, label2id, id2label, args, deps, device)

    result_path = args.output_dir / "peoples_daily_ner_result.json"
    with open(result_path, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)
    write_report(result, args, label_names, args.report_path)

    print("\n训练完成")
    print(f"最优 Entity F1：{result['best']['entity_f1']:.4f}")
    print(f"结果 JSON：{result_path}")
    print(f"报告文件：{args.report_path}")


if __name__ == "__main__":
    main()
