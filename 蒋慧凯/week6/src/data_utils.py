"""
data_utils.py —— 数据下载、处理与加载

教学重点：
  1. 标准数据集（TNEWS）的自动下载与本地缓存
  2. 类别映射（id ↔ code ↔ name）的三层转换
  3. PyTorch Dataset 的标准写法与 DataLoader 构建
  4. SFT 格式的 chat-template 数据转换

使用方式：
  from data_utils import download_tnews, build_bert_loaders, build_sft_loaders
  download_tnews()
  train_loader, val_loader = build_bert_loaders(cfg.bert_model_path)
"""

import json
import random
from pathlib import Path

import torch
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizer, AutoTokenizer
from config import cfg


# ──────────────────────────────────────────────────────────────────────────────
# 1. 数据下载与本地化
# ──────────────────────────────────────────────────────────────────────────────

LABEL_CODE_TO_NAME = {
    "100": "故事", "101": "文化", "102": "娱乐", "103": "体育", "104": "财经",
    "106": "房产", "107": "汽车", "108": "教育", "109": "科技", "110": "军事",
    "112": "旅游", "113": "国际", "114": "证券", "115": "农业", "116": "电竞",
}


def download_tnews(force=False):
    """
    从 HuggingFace 下载 CLUE/TNEWS 数据集，保存为本地 JSON。
    若本地已存在且 force=False，则跳过下载。
    
    返回:
      label_map: dict, 包含 id2code, id2name, num_labels 等
    """
    from datasets import load_dataset  # 延迟导入：避免 pyarrow 在 Windows 上触发 segfault

    label_map_path = cfg.data_dir / "label_map.json"
    train_path = cfg.data_dir / "train.json"

    # 若本地已有数据，直接加载
    if not force and label_map_path.exists() and train_path.exists():
        print(f"[数据] 本地数据已存在，跳过下载。路径: {cfg.data_dir}")
        with open(label_map_path, encoding="utf-8") as f:
            return json.load(f)

    print("[数据] 正在从 HuggingFace 下载 TNEWS ...")
    ds = load_dataset("clue", "tnews")

    # 构建类别映射
    label_names = ds["train"].features["label"].names
    id2code = {i: code for i, code in enumerate(label_names)}
    id2name = {i: LABEL_CODE_TO_NAME[code] for i, code in id2code.items()}
    label_map = {
        "id2code": id2code,
        "id2name": id2name,
        "code2id": {v: k for k, v in id2code.items()},
        "name2id": {v: k for k, v in id2name.items()},
        "num_labels": len(label_names),
    }

    # 保存 label_map
    with open(label_map_path, "w", encoding="utf-8") as f:
        json.dump(label_map, f, ensure_ascii=False, indent=2)

    # 保存各数据集分割
    for split, filename in [("train", "train.json"), ("validation", "val.json"), ("test", "test.json")]:
        records = [{"idx": item["idx"], "sentence": item["sentence"], "label": item["label"]}
                   for item in ds[split]]
        path = cfg.data_dir / filename
        with open(path, "w", encoding="utf-8") as f:
            json.dump(records, f, ensure_ascii=False, indent=2)
        print(f"[数据] {split}: {len(records)} 条 → {path}")

    print("[数据] 下载完成。")
    return label_map


def load_split(split_name="train"):
    """加载本地 JSON 数据。"""
    path = cfg.data_dir / f"{split_name}.json"
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def load_label_map():
    """加载类别映射。"""
    path = cfg.data_dir / "label_map.json"
    with open(path, encoding="utf-8") as f:
        return json.load(f)


# ──────────────────────────────────────────────────────────────────────────────
# 2. BERT 数据集（判别式）
# ──────────────────────────────────────────────────────────────────────────────

class BERTDataset(Dataset):
    """
    BERT 微调用的 Dataset。
    每条样本返回: input_ids, attention_mask, token_type_ids, label
    """
    def __init__(self, data, tokenizer, max_length=64):
        self.data = data
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        # tokenizer 返回的默认是 [1, L]，需 squeeze 成 [L]
        encoding = self.tokenizer(
            item["sentence"],
            max_length=self.max_length,
            truncation=True,
            padding="max_length",
            return_tensors="pt",
        )
        return {
            "input_ids": encoding["input_ids"].squeeze(0),
            "attention_mask": encoding["attention_mask"].squeeze(0),
            "token_type_ids": encoding["token_type_ids"].squeeze(0),
            "label": torch.tensor(item["label"], dtype=torch.long),
        }


def build_bert_loaders(model_path, batch_size=None, max_length=None, num_workers=0, num_train=None):
    """
    构建 BERT 的 train / val DataLoader。
    
    参数:
      model_path: BERT 模型路径（用于加载对应 tokenizer）
      batch_size: 默认使用 cfg.bert_batch_size
      max_length: 默认使用 cfg.bert_max_length
      num_workers: Windows 建议 0，Linux 可设 2/4
      num_train: 训练样本数上限，默认使用 cfg.bert_num_train（-1 表示全部）
    """
    batch_size = batch_size or cfg.bert_batch_size
    max_length = max_length or cfg.bert_max_length
    num_train = num_train if num_train is not None else getattr(cfg, "bert_num_train", -1)

    tokenizer = BertTokenizer.from_pretrained(model_path)
    train_data = load_split("train")
    val_data = load_split("val")

    if num_train > 0:
        train_data = train_data[:num_train]

    train_ds = BERTDataset(train_data, tokenizer, max_length)
    val_ds = BERTDataset(val_data, tokenizer, max_length)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    val_loader = DataLoader(val_ds, batch_size=batch_size * 2, shuffle=False, num_workers=num_workers)

    print(f"[BERT DataLoader] train={len(train_ds)}, val={len(val_ds)}, batch={batch_size}")
    return train_loader, val_loader


# ──────────────────────────────────────────────────────────────────────────────
# 3. SFT 数据集（生成式，chat 格式）
# ──────────────────────────────────────────────────────────────────────────────

SYSTEM_PROMPT = (
    "你是一个新闻标题分类助手。请将给定的新闻标题分类到以下类别之一，"
    "只输出类别名称，不要输出任何其他内容。\n"
    "可选类别：" + "、".join(cfg.label_names)
)


class SFTDataset(Dataset):
    """
    LLM SFT 用的 Dataset。
    将分类任务转化为 chat 格式，并构造 Loss Masking（prompt 部分 label=-100）。
    """
    def __init__(self, data, tokenizer, max_length=128):
        self.data = data
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        label_name = cfg.label_names[item["label"]]

        # Step 1: 构建 prompt（system + user）
        prompt_text = self.tokenizer.apply_chat_template(
            [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": f"新闻标题：{item['sentence']}\n类别："},
            ],
            tokenize=False,
            add_generation_prompt=True,
        )
        prompt_ids = self.tokenizer.encode(prompt_text, add_special_tokens=False)
        prompt_len = len(prompt_ids)

        # Step 2: response = 类别名 + EOS
        response_ids = self.tokenizer.encode(label_name, add_special_tokens=False)
        response_ids += [self.tokenizer.eos_token_id]

        # Step 3: 拼接并截断
        input_ids = (prompt_ids + response_ids)[: self.max_length]

        # Step 4: Loss Masking —— prompt 部分设为 -100（不计算 loss）
        labels = ([-100] * prompt_len + response_ids)[: self.max_length]

        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "labels": torch.tensor(labels, dtype=torch.long),
        }


def collate_fn(batch, pad_token_id):
    """右填充（padding）使同批次序列等长。"""
    max_len = max(item["input_ids"].size(0) for item in batch)

    input_ids_list, labels_list, mask_list = [], [], []
    for item in batch:
        n = item["input_ids"].size(0)
        pad = max_len - n
        input_ids_list.append(torch.cat([item["input_ids"], torch.full((pad,), pad_token_id, dtype=torch.long)]))
        labels_list.append(torch.cat([item["labels"], torch.full((pad,), -100, dtype=torch.long)]))
        mask_list.append(torch.cat([torch.ones(n, dtype=torch.long), torch.zeros(pad, dtype=torch.long)]))

    return {
        "input_ids": torch.stack(input_ids_list),
        "labels": torch.stack(labels_list),
        "attention_mask": torch.stack(mask_list),
    }


def build_sft_loaders(model_path, num_train=None, batch_size=None, max_length=None):
    """
    构建 SFT 的 train / val DataLoader。
    
    参数:
      num_train: 训练样本数，默认 cfg.sft_num_train（-1 表示全部）
    """
    batch_size = batch_size or cfg.sft_batch_size
    max_length = max_length or cfg.sft_max_length
    num_train = num_train if num_train is not None else cfg.sft_num_train

    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    train_raw = load_split("train")
    val_raw = load_split("val")

    if num_train > 0:
        train_raw = random.sample(train_raw, min(num_train, len(train_raw)))

    train_ds = SFTDataset(train_raw, tokenizer, max_length)
    val_ds = SFTDataset(val_raw[:500], tokenizer, max_length)

    _collate = lambda b: collate_fn(b, tokenizer.pad_token_id)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, collate_fn=_collate)
    val_loader = DataLoader(val_ds, batch_size=batch_size * 2, shuffle=False, collate_fn=_collate)

    print(f"[SFT DataLoader] train={len(train_ds)}, val={len(val_ds)}, batch={batch_size}")
    return train_loader, val_loader, tokenizer
