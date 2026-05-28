
import argparse
import json
import os
import random
import sys
import time
from collections import Counter
from pathlib import Path


ROOT_DIR = Path(__file__).resolve().parents[1]
PROJECT_DIR = (
    ROOT_DIR
    / "week6文本分类问题"
    / "week6 文本分类问题"
    / "text_classification项目"
)
DATA_DIR = PROJECT_DIR / "data"
OUTPUT_DIR = Path(__file__).resolve().parent / "outputs"
DEFAULT_REPORT = Path(__file__).resolve().with_name("文本分类不同训练方法效果对比.md")

METHODS = ("standard", "weighted", "freeze")


def import_dependencies():
    try:
        import numpy as np
        import torch
        import torch.nn as nn
        from sklearn.metrics import accuracy_score, f1_score
        from torch.optim import AdamW
        from torch.utils.data import DataLoader, Dataset
        from tqdm import tqdm
        from transformers import BertModel, BertTokenizer, get_linear_schedule_with_warmup
    except ImportError as e:
        raise SystemExit(
            "缺少课程依赖，无法训练模型。\n"
            "请先安装 week6 项目依赖，例如：\n"
            f"  {sys.executable} -m pip install -r "
            f"'{PROJECT_DIR / 'requirements.txt'}'\n"
            f"原始错误：{type(e).__name__}: {e}"
        ) from e

    return {
        "np": np,
        "torch": torch,
        "nn": nn,
        "accuracy_score": accuracy_score,
        "f1_score": f1_score,
        "AdamW": AdamW,
        "DataLoader": DataLoader,
        "Dataset": Dataset,
        "tqdm": tqdm,
        "BertModel": BertModel,
        "BertTokenizer": BertTokenizer,
        "get_linear_schedule_with_warmup": get_linear_schedule_with_warmup,
    }


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


def load_json(path):
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"文件不存在：{path}")
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def auto_bert_path():
    candidates = [
        PROJECT_DIR.parent.parent / "pretrain_models" / "bert-base-chinese",
        ROOT_DIR / "week4语言模型" / "bert-base-chinese",
    ]
    for path in candidates:
        if (path / "config.json").exists() and (path / "vocab.txt").exists():
            return path
    return "bert-base-chinese"


def limit_data(data, limit, seed):
    if limit is None or limit < 0 or limit >= len(data):
        return list(data)
    rng = random.Random(seed)
    indices = list(range(len(data)))
    rng.shuffle(indices)
    return [data[i] for i in indices[:limit]]


def build_label_info(data_dir):
    label_map = load_json(data_dir / "label_map.json")
    id2name = {int(k): v for k, v in label_map["id2name"].items()}
    return label_map["num_labels"], id2name


def class_weights(labels, num_labels, torch, device):
    counts = Counter(labels)
    total = len(labels)
    weights = []
    for label_id in range(num_labels):
        count = counts.get(label_id, 0)
        weight = 0.0 if count == 0 else total / (num_labels * count)
        weights.append(weight)
    return torch.tensor(weights, dtype=torch.float32, device=device)


def freeze_bert_parameters(model):
    for param in model.bert.parameters():
        param.requires_grad = False


def trainable_param_count(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def total_param_count(model):
    return sum(p.numel() for p in model.parameters())


def format_percent(value):
    return f"{value * 100:.2f}%"


def format_time(seconds):
    return f"{seconds:.1f}s"


def write_report(results, args, id2name, report_path):
    report_path = Path(report_path)
    report_path.parent.mkdir(parents=True, exist_ok=True)

    rows = []
    for item in results:
        rows.append(
            "| {method} | {train_loss} | {train_acc} | {val_acc} | {macro_f1} | "
            "{weighted_f1} | {params} | {elapsed} |".format(
                method=item["method"],
                train_loss=f"{item['train_loss']:.4f}",
                train_acc=format_percent(item["train_acc"]),
                val_acc=format_percent(item["val_acc"]),
                macro_f1=format_percent(item["macro_f1"]),
                weighted_f1=format_percent(item["weighted_f1"]),
                params=f"{item['trainable_params']:,}",
                elapsed=format_time(item["elapsed_s"]),
            )
        )

    best = max(results, key=lambda x: x["macro_f1"])
    content = f"""# 对比文本分类不同训练方法效果

## 实验设置

- 数据目录：`{args.data_dir}`
- BERT 路径：`{args.bert_path}`
- 类别数：{len(id2name)}
- 对比方法：{", ".join(args.methods)}
- 训练样本数：{args.train_limit if args.train_limit >= 0 else "全部"}
- 验证样本数：{args.val_limit if args.val_limit >= 0 else "全部"}
- epoch：{args.epochs}
- batch size：{args.batch_size}
- max length：{args.max_length}
- 池化方式：{args.pool}

## 结果对比

| 方法 | Train Loss | Train Acc | Val Acc | Macro F1 | Weighted F1 | 可训练参数 | 耗时 |
|---|---:|---:|---:|---:|---:|---:|---:|
{chr(10).join(rows)}

## 结论

- 本次实验 Macro F1 最好的方法是 `{best["method"]}`，Macro F1 = {format_percent(best["macro_f1"])}。
- `standard` 是最直接的 BERT 全量微调，适合有足够训练数据、追求稳定输出的场景。
- `weighted` 在类别不均衡时更值得关注，不能只看 Accuracy，还要看 Macro F1。
- `freeze` 只训练分类头，训练成本最低，但通常上限低于全量微调。
"""
    report_path.write_text(content, encoding="utf-8")


def build_runtime_classes(deps):
    torch = deps["torch"]
    nn = deps["nn"]
    Dataset = deps["Dataset"]
    BertModel = deps["BertModel"]

    class TNewsDataset(Dataset):
        def __init__(self, data, tokenizer, max_length):
            self.data = data
            self.tokenizer = tokenizer
            self.max_length = max_length

        def __len__(self):
            return len(self.data)

        def __getitem__(self, index):
            item = self.data[index]
            encoded = self.tokenizer(
                item["sentence"],
                max_length=self.max_length,
                truncation=True,
                padding="max_length",
                return_tensors="pt",
            )
            return {
                "input_ids": encoded["input_ids"].squeeze(0),
                "attention_mask": encoded["attention_mask"].squeeze(0),
                "token_type_ids": encoded["token_type_ids"].squeeze(0),
                "label": torch.tensor(item["label"], dtype=torch.long),
            }

    class BertTextClassifier(nn.Module):
        def __init__(self, bert_path, num_labels, pool="cls", dropout=0.1):
            super().__init__()
            if pool not in {"cls", "mean", "max"}:
                raise ValueError("pool 只能是 cls / mean / max")
            self.pool = pool
            self.bert = BertModel.from_pretrained(str(bert_path))
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
            vector = self.pool_hidden(last_hidden, attention_mask)
            return self.classifier(self.dropout(vector))

        def pool_hidden(self, last_hidden, attention_mask):
            if self.pool == "cls":
                return last_hidden[:, 0, :]

            mask = attention_mask.unsqueeze(-1).float()
            if self.pool == "mean":
                summed = (last_hidden * mask).sum(dim=1)
                count = mask.sum(dim=1).clamp(min=1e-9)
                return summed / count

            masked = last_hidden + (1 - mask) * (-1e9)
            return masked.max(dim=1).values

    return TNewsDataset, BertTextClassifier


def build_dataloaders(train_data, val_data, tokenizer, args, deps):
    DataLoader = deps["DataLoader"]
    TNewsDataset, _ = build_runtime_classes(deps)
    train_dataset = TNewsDataset(train_data, tokenizer, args.max_length)
    val_dataset = TNewsDataset(val_data, tokenizer, args.max_length)
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


def run_epoch(model, loader, criterion, optimizer, scheduler, device, deps, desc):
    torch = deps["torch"]
    tqdm = deps["tqdm"]
    is_train = optimizer is not None
    model.train(is_train)
    total_loss = 0.0
    total_correct = 0
    total = 0

    for batch in tqdm(loader, desc=desc, leave=False):
        labels = batch["label"].to(device)
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        token_type_ids = batch["token_type_ids"].to(device)

        with torch.set_grad_enabled(is_train):
            logits = model(input_ids, attention_mask, token_type_ids)
            loss = criterion(logits, labels)
            if is_train:
                optimizer.zero_grad()
                loss.backward()
                deps["nn"].utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
                scheduler.step()

        preds = logits.argmax(dim=-1)
        total_loss += loss.item() * labels.size(0)
        total_correct += (preds == labels).sum().item()
        total += labels.size(0)

    return total_loss / total, total_correct / total


def evaluate(model, loader, device, deps):
    torch = deps["torch"]
    np = deps["np"]
    model.eval()
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for batch in loader:
            logits = model(
                batch["input_ids"].to(device),
                batch["attention_mask"].to(device),
                batch["token_type_ids"].to(device),
            )
            all_preds.extend(logits.argmax(dim=-1).cpu().numpy().tolist())
            all_labels.extend(batch["label"].numpy().tolist())

    labels = np.array(all_labels)
    preds = np.array(all_preds)
    return {
        "val_acc": deps["accuracy_score"](labels, preds),
        "macro_f1": deps["f1_score"](labels, preds, average="macro", zero_division=0),
        "weighted_f1": deps["f1_score"](labels, preds, average="weighted", zero_division=0),
    }


def train_method(method, train_data, val_data, num_labels, args, deps, device):
    torch = deps["torch"]
    nn = deps["nn"]
    AdamW = deps["AdamW"]
    BertTokenizer = deps["BertTokenizer"]
    get_linear_schedule_with_warmup = deps["get_linear_schedule_with_warmup"]
    _, BertTextClassifier = build_runtime_classes(deps)

    tokenizer = BertTokenizer.from_pretrained(str(args.bert_path))
    train_loader, val_loader = build_dataloaders(train_data, val_data, tokenizer, args, deps)
    model = BertTextClassifier(args.bert_path, num_labels, pool=args.pool, dropout=args.dropout)

    if method == "freeze":
        freeze_bert_parameters(model)

    model.to(device)

    if method == "weighted":
        weights = class_weights([item["label"] for item in train_data], num_labels, torch, device)
        criterion = nn.CrossEntropyLoss(weight=weights)
    else:
        criterion = nn.CrossEntropyLoss()

    bert_params = [p for p in model.bert.parameters() if p.requires_grad]
    head_params = list(model.classifier.parameters()) + list(model.dropout.parameters())
    param_groups = [{"params": head_params, "lr": args.lr * args.head_lr_mult}]
    if bert_params:
        param_groups.insert(0, {"params": bert_params, "lr": args.lr})

    optimizer = AdamW(param_groups, weight_decay=args.weight_decay)
    total_steps = max(1, len(train_loader) * args.epochs)
    warmup_steps = int(total_steps * args.warmup_ratio)
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=total_steps,
    )

    start = time.time()
    last_train_loss = 0.0
    last_train_acc = 0.0
    best_metrics = None
    best_epoch = 0

    print(f"\n开始训练方法：{method}")
    print(f"  可训练参数：{trainable_param_count(model):,}/{total_param_count(model):,}")

    for epoch in range(1, args.epochs + 1):
        last_train_loss, last_train_acc = run_epoch(
            model,
            train_loader,
            criterion,
            optimizer,
            scheduler,
            device,
            deps,
            desc=f"{method} epoch {epoch}/{args.epochs}",
        )
        metrics = evaluate(model, val_loader, device, deps)
        print(
            f"  epoch {epoch}: train_loss={last_train_loss:.4f} "
            f"train_acc={last_train_acc:.4f} val_acc={metrics['val_acc']:.4f} "
            f"macro_f1={metrics['macro_f1']:.4f}"
        )
        if best_metrics is None or metrics["macro_f1"] > best_metrics["macro_f1"]:
            best_metrics = metrics
            best_epoch = epoch

    elapsed = time.time() - start
    return {
        "method": method,
        "best_epoch": best_epoch,
        "train_loss": last_train_loss,
        "train_acc": last_train_acc,
        "val_acc": best_metrics["val_acc"],
        "macro_f1": best_metrics["macro_f1"],
        "weighted_f1": best_metrics["weighted_f1"],
        "trainable_params": trainable_param_count(model),
        "total_params": total_param_count(model),
        "elapsed_s": elapsed,
    }


def parse_args():
    parser = argparse.ArgumentParser(description="对比文本分类不同训练方法效果")
    parser.add_argument("--data_dir", type=Path, default=DATA_DIR)
    parser.add_argument("--bert_path", default=str(auto_bert_path()))
    parser.add_argument("--output_dir", type=Path, default=OUTPUT_DIR)
    parser.add_argument("--report_path", type=Path, default=DEFAULT_REPORT)
    parser.add_argument("--methods", nargs="+", choices=METHODS, default=list(METHODS))
    parser.add_argument("--pool", choices=["cls", "mean", "max"], default="cls")
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--max_length", type=int, default=64)
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
    train_raw = load_json(args.data_dir / "train.json")
    val_raw = load_json(args.data_dir / "val.json")
    num_labels, id2name = build_label_info(args.data_dir)
    train_data = limit_data(train_raw, args.train_limit, args.seed)
    val_data = limit_data(val_raw, args.val_limit, args.seed + 1)

    print(f"使用设备：{device}")
    print(f"BERT 路径：{args.bert_path}")
    print(f"训练样本：{len(train_data):,}，验证样本：{len(val_data):,}，类别数：{num_labels}")

    results = []
    for method in args.methods:
        results.append(train_method(method, train_data, val_data, num_labels, args, deps, device))

    args.output_dir.mkdir(parents=True, exist_ok=True)
    result_path = args.output_dir / "compare_training_methods.json"
    with open(result_path, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    write_report(results, args, id2name, args.report_path)

    print("\n对比结果：")
    print(f"{'方法':<10} {'Val Acc':>10} {'Macro F1':>10} {'耗时':>10}")
    print("-" * 45)
    for item in results:
        print(
            f"{item['method']:<10} "
            f"{item['val_acc']:>10.4f} "
            f"{item['macro_f1']:>10.4f} "
            f"{item['elapsed_s']:>9.1f}s"
        )
    print(f"\n结果 JSON：{result_path}")
    print(f"报告文件：{args.report_path}")


if __name__ == "__main__":
    main()

