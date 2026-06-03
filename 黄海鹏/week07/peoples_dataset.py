# 1.数据处理

import json
from pathlib import Path
from typing import Optional

import torch
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizer

ROOT = Path(__file__).parent.parent
DATA_DIR = ROOT / "data" / "peoples_daily"

ENTITY_TYPES = ["PER", "ORG", "LOC"]

# 构建标签体系
def build_label_schema() -> tuple[list[str], dict[str, int], dict[int, str]]:
    """构建 BIO 标签体系，返回 (labels, label2id, id2label)。"""
    labels = ["O"]
    for etype in ENTITY_TYPES:
        labels.append(f"B-{etype}")
        labels.append(f"I-{etype}")

    label2id = {lbl: i for i, lbl in enumerate(labels)}
    id2label = {i: lbl for lbl, i in label2id.items()}
    
    return labels, label2id, id2label

# 数据集封装与对齐
class PeoplesDailyDataset(Dataset):
    def __init__(self, data_path, tokenizer, label2id, max_length = 128):
        self.records = json.load(open(data_path))
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.label2id = label2id
    
    def __getitem__(self, idx):
        # 例 tokens: ["任", "正", "非"]
        #     tags:  ["B-PER", "I-PER", "I-PER"]
        tokens = self.records[idx]["tokens"]
        tags = self.records[idx]["ner_tags"]

        # char_labels = [1, 2, 2]  # 假设B-PER=1, I-PER=2 
        char_labels = [self.label2id[t] for t in tags]

        # 子词对齐：BERT
        encoding = self.tokenizer(
            tokens,
            is_split_into_words=True,
            max_length=self.max_length,
            truncation=True,
            padding="max_length",
            return_tensors="pt",
        )
        
        # 标签对齐
        word_ids = encoding.word_ids(batch_index=0)
        aligned_labels = [
            -100 if wid is None or wid >= len(char_labels) else char_labels[wid] 
            for wid in word_ids
        ]
        
        # 🔧 修复：使用 .squeeze(0) 去除 pt 返回的 batch 维度
        return {
            "input_ids": encoding["input_ids"].squeeze(0),
            "attention_mask": encoding["attention_mask"].squeeze(0),
            "token_type_ids": encoding["token_type_ids"].squeeze(0),
            "labels": torch.tensor(aligned_labels, dtype=torch.long)
        }

    def __len__(self):
        return len(self.records)

# 构建数据加载器
def build_dataloaders(
    tokenizer: BertTokenizer,
    label2id: dict,
    batch_size: int = 32,
    max_length: int = 128,
    data_dir: Optional[Path] = None,
) -> tuple[DataLoader, DataLoader, DataLoader]:
    '''构建训练、验证、测试 DataLoader。 返回（train_loader, val_loader, test_loader)。）'''
    d = data_dir or DATA_DIR
    train_ds = PeoplesDailyDataset(d / "train.json", tokenizer, label2id, max_length)
    val_ds = PeoplesDailyDataset(d / "validation.json", tokenizer, label2id, max_length)
    test_ds = PeoplesDailyDataset(d / "test.json", tokenizer, label2id, max_length)

    print(f'数据集规模： 训练={len(train_ds)}, 测试={len(test_ds)}, 验证={len(val_ds)}')
    # 
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=0)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=0)

    # 打印一条数据
    sample_batch = next(iter(train_loader))
    print("{")
    for key, value in sample_batch.items():
        print(f'    "{key}": Tensor(shape={list(value.shape)}),  # {value.dtype}')
    print("}")


    return train_loader, val_loader, test_loader
