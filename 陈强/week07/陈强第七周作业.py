import json
from pathlib import Path
from typing import Optional

import torch
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizer

ROOT = Path(__file__).parent.parent
DATA_DIR = ROOT / "data" / "peoples_daily"


def load_label_names(data_dir: Optional[Path] = None) -> list[str]:
    d = data_dir or DATA_DIR
    labels_path = d / "label_names.json"
    if labels_path.exists():
        with open(labels_path, "r", encoding="utf-8") as f:
            return json.load(f)

    # 如果没有 label_names.json，则从 train split 中推断标签。
    records = load_records("train", d)
    label_set = set()
    for row in records:
        if "ner_tags" in row:
            label_set.update(row["ner_tags"])
        elif "label" in row:
            label_set.add("O")
            for etype in row["label"].keys():
                label_set.add(f"B-{etype}")
                label_set.add(f"I-{etype}")

    if not label_set:
        return ["O"]

    entity_types = sorted({lbl[2:] for lbl in label_set if lbl != "O"})
    labels = ["O"]
    for etype in entity_types:
        labels.append(f"B-{etype}")
    for etype in entity_types:
        labels.append(f"I-{etype}")
    return labels


def build_label_schema(data_dir: Optional[Path] = None) -> tuple[list[str], dict[str, int], dict[int, str]]:
    """构建 BIO 标签体系，返回 (labels, label2id, id2label)。"""
    labels = load_label_names(data_dir)
    label2id = {lbl: i for i, lbl in enumerate(labels)}
    id2label = {i: lbl for lbl, i in label2id.items()}
    return labels, label2id, id2label


def span_to_bio(text: str, label_dict: dict, label2id: dict) -> list[int]:
    n = len(text)
    bio = ["O"] * n

    if not label_dict:
        return [label2id[t] for t in bio]

    for etype, spans in label_dict.items():
        b_tag = f"B-{etype}"
        i_tag = f"I-{etype}"
        for surface, positions in spans.items():
            for start, end in positions:
                if start >= n or end >= n:
                    continue
                bio[start] = b_tag
                for idx in range(start + 1, end + 1):
                    bio[idx] = i_tag

    return [label2id.get(t, 0) for t in bio]


class NERDataset(Dataset):
    def __init__(
        self,
        records: list,
        tokenizer: BertTokenizer,
        label2id: dict,
        max_length: int = 128,
    ):
        self.records = records
        self.tokenizer = tokenizer
        self.label2id = label2id
        self.max_length = max_length

    def __len__(self):
        return len(self.records)

    def __getitem__(self, idx: int) -> dict:
        row = self.records[idx]

        if "tokens" in row and "ner_tags" in row:
            chars = row["tokens"]
            char_labels = [self.label2id.get(tag, self.label2id.get("O", 0)) for tag in row["ner_tags"]]
        else:
            text: str = row["text"]
            label_dict: dict = row.get("label") or {}
            char_labels = span_to_bio(text, label_dict, self.label2id)
            chars = list(text)

        encoding = self.tokenizer(
            chars,
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
                if wid < len(char_labels):
                    aligned_labels.append(char_labels[wid])
                else:
                    aligned_labels.append(-100)
                prev_word_id = wid
            else:
                aligned_labels.append(-100)

        labels_tensor = torch.tensor(aligned_labels, dtype=torch.long)

        return {
            "input_ids": encoding["input_ids"].squeeze(0),
            "attention_mask": encoding["attention_mask"].squeeze(0),
            "token_type_ids": encoding["token_type_ids"].squeeze(0),
            "labels": labels_tensor,
        }


def load_records(split: str, data_dir: Optional[Path] = None) -> list:
    d = data_dir or DATA_DIR
    with open(d / f"{split}.json", "r", encoding="utf-8") as f:
        return json.load(f)


def build_dataloaders(
    tokenizer: BertTokenizer,
    label2id: dict,
    batch_size: int = 32,
    max_length: int = 128,
    data_dir: Optional[Path] = None,
) -> tuple[DataLoader, DataLoader, DataLoader]:
    """构建训练/验证/测试 DataLoader，返回 (train_loader, val_loader, test_loader)。"""
    train_records = load_records("train", data_dir)
    val_records = load_records("validation", data_dir)
    test_records = load_records("test", data_dir)

    train_ds = NERDataset(train_records, tokenizer, label2id, max_length)
    val_ds = NERDataset(val_records, tokenizer, label2id, max_length)
    test_ds = NERDataset(test_records, tokenizer, label2id, max_length)

    print(f"数据集规模：训练={len(train_ds)}，验证={len(val_ds)}，测试={len(test_ds)}")

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=0)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=0)

    return train_loader, val_loader, test_loader
