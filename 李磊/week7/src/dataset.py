"""
NER 数据集类：直接加载 BIO 标注 + BERT 子词对齐

教学重点：
  1. 直接使用 tokens + ner_tags（BIO 格式）的数据
     - 相比 cluener 的 span 格式，省去 span→BIO 转换
  2. BERT 子词对齐（word_ids 策略）
     - 中文字符通常一字一token，但 [UNK] 和特殊字符可能例外
     - 非首子词标记为 -100，在 loss 计算中被忽略
  3. DataLoader 工厂函数统一封装

使用方式：
  from dataset import build_label_schema, build_dataloaders
"""

import json
from pathlib import Path
from typing import Optional

import torch
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizer

ROOT = Path(__file__).parent.parent
DATA_DIR = ROOT / "data"


def build_label_schema(label_names_path: Optional[Path] = None) -> tuple[list[str], dict[str, int], dict[int, str]]:
    """从 label_names.json 构建 BIO 标签体系，返回 (labels, label2id, id2label)。"""
    if label_names_path is None:
        label_names_path = DATA_DIR / "label_names.json"
    with open(label_names_path, "r", encoding="utf-8") as f:
        labels = json.load(f)

    label2id = {lbl: i for i, lbl in enumerate(labels)}
    id2label = {i: lbl for lbl, i in label2id.items()}
    return labels, label2id, id2label


class PeopleDailyDataset(Dataset):
    """人民日报 NER 的 PyTorch Dataset。

    数据格式：
      {"tokens": ["海","钓",...], "ner_tags": ["O","O",...]}

    处理流程：
      tokens → 字符级 BIO ids（直接从 ner_tags 映射）
             → BertTokenizer (is_split_into_words=True)
             → 用 word_ids() 对齐子词标签（非首子词设为 -100）
             → 返回 input_ids / attention_mask / token_type_ids / labels
    """

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
        tokens: list[str] = row["tokens"]
        ner_tags: list[str] = row["ner_tags"]

        # 1. ner_tags → 字符级 BIO id 列表
        char_labels = [self.label2id.get(tag, 0) for tag in ner_tags]

        # 2. 将字符列表传入 tokenizer
        encoding = self.tokenizer(
            tokens,
            is_split_into_words=True,
            max_length=self.max_length,
            truncation=True,
            padding="max_length",
            return_tensors="pt",
        )

        # 3. 子词对齐：取每个 token 对应的字符索引
        #    - word_ids() 返回 [None, 0, 0, 1, 2, 2, ..., None]
        #      None 对应 [CLS]/[SEP]/[PAD]
        #    - 非首子词、特殊token 标记为 -100，cross_entropy 的 ignore_index
        word_ids = encoding.word_ids(batch_index=0)
        aligned_labels = []
        prev_word_id = None
        for wid in word_ids:
            if wid is None:
                aligned_labels.append(-100)
            elif wid != prev_word_id:
                # 首次出现这个字符索引：使用 BIO 标签
                if wid < len(char_labels):
                    aligned_labels.append(char_labels[wid])
                else:
                    aligned_labels.append(-100)
                prev_word_id = wid
            else:
                # 同一字符的后续子词（中文通常不会出现，但保留正确处理）
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

    train_ds = PeopleDailyDataset(train_records, tokenizer, label2id, max_length)
    val_ds = PeopleDailyDataset(val_records, tokenizer, label2id, max_length)
    test_ds = PeopleDailyDataset(test_records, tokenizer, label2id, max_length)

    print(f"数据集规模：训练={len(train_ds)}，验证={len(val_ds)}，测试={len(test_ds)}")

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=0)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=0)

    return train_loader, val_loader, test_loader
