"""
NER 数据集类：适配 peoples_daily 的 token+BI O 格式

教学重点：
  1. peoples_daily 已经是 BIO 格式（tokens + ner_tags 列表）
  2. BERT 子词对齐（word_ids 策略）：非首子词设为 -100
  3. DataLoader 工厂函数统一封装

与 cluener 的区别：
  - cluener 是 span 格式，需要 span_to_bio 转换
  - peoples_daily 直接提供 tokens 和 ner_tags，无需转换
"""

import json
from pathlib import Path
from typing import Optional, Union

import torch
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizerFast


ROOT = Path(__file__).parent.parent
DATA_DIR = ROOT / "data" / "peoples_daily"


def build_label_schema(entity_types: Optional[list[str]] = None) -> tuple[list[str], dict[str, int], dict[int, str]]:
    """构建 BIO 标签体系，返回 (labels, label2id, id2label)。"""
    labels = ["O"]
    for etype in (entity_types or ["PER", "ORG", "LOC"]):
        labels.append(f"B-{etype}")
        labels.append(f"I-{etype}")

    label2id = {lbl: i for i, lbl in enumerate(labels)}
    id2label = {i: lbl for lbl, i in label2id.items()}
    return labels, label2id, id2label


def load_label_names(data_dir: Path) -> list[str]:
    """从 label_names.json 加载标签列表。"""
    path = data_dir / "label_names.json"
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def build_label_schema_from_names(label_names: list[str]) -> tuple[list[str], dict[str, int], dict[int, str]]:
    """根据数据集中的标签名构建映射（peoples_daily 已含 O/B-PER/I-PER...）。"""
    label2id = {lbl: i for i, lbl in enumerate(label_names)}
    id2label = {i: lbl for lbl, i in label2id.items()}
    return label_names, label2id, id2label


class PeoplesDailyDataset(Dataset):
    """peoples_daily 的 PyTorch Dataset。

    数据格式：
      {
        "tokens": ["海", "钓", "比", "赛", ...],
        "ner_tags": ["O", "O", "O", "O", ...]
      }

    处理流程：
      tokens → BertTokenizerFast (is_split_into_words=True)
           → 用 word_ids() 对齐子词标签（非首子词设为 -100）
           → 返回 input_ids / attention_mask / token_type_ids / labels
    """

    def __init__(
        self,
        records: list,
        tokenizer: BertTokenizerFast,
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
        tokens: list[str] = row["tokens"]
        ner_tags: list[str] = row["ner_tags"]

        # 1. 将 ner_tags 转为 id 列表
        char_labels = [self.label2id.get(tag, 0) for tag in ner_tags]

        # 2. tokens 已经是分词后的列表，直接传入 tokenizer
        encoding = self.tokenizer(
            tokens,
            is_split_into_words=True,
            max_length=self.max_length,
            truncation=True,
            padding="max_length",
            return_tensors="pt",
        )

        # 3. 子词对齐：取每个 token 对应的原始词索引
        word_ids = encoding.word_ids(batch_index=0)
        aligned_labels = []
        prev_word_id = None
        for wid in word_ids:
            if wid is None:
                # [CLS] / [SEP] / [PAD]
                aligned_labels.append(-100)
            elif wid != prev_word_id:
                # 首次出现这个词：保留标签
                if wid < len(char_labels):
                    aligned_labels.append(char_labels[wid])
                else:
                    aligned_labels.append(-100)
                prev_word_id = wid
            else:
                # 同一词的后续子词（中文通常不会出现，但保留正确处理）
                aligned_labels.append(-100)

        labels_tensor = torch.tensor(aligned_labels, dtype=torch.long)

        return {
            "input_ids": encoding["input_ids"].squeeze(0),
            "attention_mask": encoding["attention_mask"].squeeze(0),
            "token_type_ids": encoding["token_type_ids"].squeeze(0),
            "labels": labels_tensor,
        }


def load_records(split: str, data_dir: Optional[Path] = None) -> list:
    """加载数据集 split（train / validation / test）。"""
    d = data_dir or DATA_DIR
    with open(d / f"{split}.json", "r", encoding="utf-8") as f:
        return json.load(f)


def build_dataloaders(
    tokenizer: BertTokenizerFast,
    label2id: dict,
    batch_size: int = 32,
    max_length: int = 128,
    data_dir: Optional[Path] = None,
) -> tuple[DataLoader, DataLoader, DataLoader]:
    """构建训练/验证/测试 DataLoader，返回 (train_loader, val_loader, test_loader)。"""
    train_records = load_records("train", data_dir)
    val_records = load_records("validation", data_dir)
    test_records = load_records("test", data_dir)

    train_ds = PeoplesDailyDataset(train_records, tokenizer, label2id, max_length)
    val_ds = PeoplesDailyDataset(val_records, tokenizer, label2id, max_length)
    test_ds = PeoplesDailyDataset(test_records, tokenizer, label2id, max_length)

    print(f"数据集规模：训练={len(train_ds)}，验证={len(val_ds)}，测试={len(test_ds)}")

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=0)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=0)

    return train_loader, val_loader, test_loader
