"""
NER 数据集类：支持 Peoples Daily (Token-Tag 格式)

教学重点：
  1. 处理 Token-Tag 格式数据 (Peoples Daily)
     - 输入: {"tokens": ["海", "钓"...], "ner_tags": ["O", "O"...]}
     - 流程: Tokens -> BERT Tokenizer (is_split_into_words=True) -> Label Alignment
  2. BERT 子词对齐（word_ids 策略）
     - 非首子词标记为 -100，在 loss 计算中被忽略
  3. DataLoader 工厂函数统一封装
"""

import json
from pathlib import Path
from typing import Optional, List

import torch
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizer

# 定义项目根目录和数据目录
# 假设文件结构为: work7/src/dataset.py, work7/data/peoples_daily/...
ROOT = Path(__file__).parent.parent.parent
DATA_DIR = ROOT / "data" / "peoples_daily"

ENTITY_TYPES = [
    "PER", "LOC", "ORG"
]


def build_label_schema() -> tuple[list[str], dict[str, int], dict[int, str]]:
    """
    构建 BIO 标签体系。
    
    根据 ENTITY_TYPES 生成标准的 BIO 标签列表，并创建标签到ID的双向映射字典。
    
    Returns:
        tuple: 
            - labels (list[str]): 所有标签名的列表，如 ['O', 'B-PER', 'I-PER', ...]
            - label2id (dict[str, int]): 标签名到整数ID的映射
            - id2label (dict[int, str]): 整数ID到标签名的映射
    """
    labels = ["O"]
    # 为每种实体类型添加 B- (Begin) 和 I- (Inside) 标签
    for etype in ENTITY_TYPES:
        labels.append(f"B-{etype}")
        labels.append(f"I-{etype}")
    label2id = {lbl: i for i, lbl in enumerate(labels)}
    id2label = {i: lbl for lbl, i in label2id.items()}
    
    return labels, label2id, id2label


def tags_to_bio_ids(ner_tags: List[str], label2id: dict) -> List[int]:
    """
    将原始的 NER 标签字符串列表转换为对应的 ID 列表。
    
    Peoples Daily 数据中的 ner_tags 已经是 BIO 格式（如 'B-LOC'），
    此函数仅负责通过 label2id 字典将其映射为模型可处理的整数 ID。
    
    Args:
        ner_tags (List[str]): 原始标签列表，长度与 tokens 一致
        label2id (dict): 标签到ID的映射字典
        
    Returns:
        List[int]: 转换后的标签 ID 列表。如果标签不在字典中，默认映射为 0 ('O')
    """
    # 使用 get 方法提供默认值 0，防止出现未知标签导致报错
    return [label2id.get(tag, 0) for tag in ner_tags]


class NerDataset(Dataset):
    """
    通用的 NER PyTorch Dataset，专门用于处理 Token-Tag 格式的数据（如 Peoples Daily）。
    
    核心逻辑：
    1. 读取 tokens 和 ner_tags。
    2. 将 ner_tags 转换为初始的字符级/Token级 ID。
    3. 使用 BertTokenizer 对 tokens 进行编码，启用 is_split_into_words=True。
    4. 利用 tokenizer 输出的 word_ids 将初始标签对齐到 BERT 的子词级别。
    5. 对特殊 token ([CLS], [SEP], [PAD]) 和非首子词标记为 -100 (ignore_index)。
    """

    def __init__(
        self,
        records: list,
        tokenizer: BertTokenizer,
        label2id: dict,
        max_length: int = 128,
    ):
        """
        初始化数据集。
        
        Args:
            records (list): 从 JSON 文件加载的数据记录列表
            tokenizer (BertTokenizer): BERT 分词器实例
            label2id (dict): 标签到ID的映射
            max_length (int): 序列最大长度，超出部分截断，不足部分填充
        """
        self.records = records
        self.tokenizer = tokenizer
        self.label2id = label2id
        self.max_length = max_length

    def __len__(self):
        """返回数据集的大小（样本数量）。"""
        return len(self.records)

    def __getitem__(self, idx: int) -> dict:
        """
        获取单个样本，并进行预处理和张量转换。
        
        Args:
            idx (int): 样本索引
            
        Returns:
            dict: 包含 input_ids, attention_mask, token_type_ids, labels 的字典
        """
        # 1. 获取原始数据
        row = self.records[idx]
        tokens = row["tokens"]      # 字符列表，如 ["海", "钓", ...]
        ner_tags = row["ner_tags"]  # 标签列表，如 ["O", "O", ...]
        
        # 2. 将标签字符串转换为 ID 列表
        # 此时标签长度与 tokens 长度一致
        char_labels = tags_to_bio_ids(ner_tags, self.label2id)

        # 3. 使用 Tokenizer 编码
        # is_split_into_words=True: 告诉 tokenizer 输入已经是分好词的列表，
        # 这样 tokenizer 内部会记录每个子词对应原始哪个 word 的索引
        encoding = self.tokenizer(
            tokens,
            is_split_into_words=True,
            max_length=self.max_length,
            truncation=True,       # 超过 max_length 时截断
            padding="max_length",  # 不足 max_length 时填充
            return_tensors="pt",   # 直接返回 PyTorch 张量
        )

        # 4. 子词标签对齐 (Label Alignment)
        # word_ids() 返回一个列表，表示每个 token 对应原始 tokens 中的哪个索引
        # 例如: [None, 0, 1, 2, None] 对应 [CLS, token0, token1, token2, SEP]
        # None 表示特殊 token 或填充 token
        word_ids = encoding.word_ids(batch_index=0)
        
        aligned_labels = []
        prev_word_id = None
        
        for wid in word_ids:
            if wid is None:
                # 情况 A: 特殊 token ([CLS], [SEP]) 或填充 token ([PAD])
                # 设置为 -100，CrossEntropyLoss 会自动忽略这些位置的损失计算
                aligned_labels.append(-100)
            elif wid != prev_word_id:
                # 情况 B: 当前 token 是某个原始 word 的第一个子词
                # 需要赋予该 word 对应的真实标签
                if wid < len(char_labels):
                    aligned_labels.append(char_labels[wid])
                else:
                    # 防止因截断导致的索引越界（虽然通常不会发生，因为 truncation 会同步处理）
                    aligned_labels.append(-100)
                prev_word_id = wid
            else:
                # 情况 C: 当前 token 是某个原始 word 的后续子词 (例如英文中的 ##ing)
                # 中文通常一字一token，较少出现此情况，但为了通用性保留
                # 设置为 -100，避免对同一个实体重复计算损失
                aligned_labels.append(-100)

        # 将对齐后的标签列表转换为 PyTorch 长整型张量
        labels_tensor = torch.tensor(aligned_labels, dtype=torch.long)

        # 5. 返回处理好的数据字典
        # squeeze(0) 去掉 batch 维度，因为 DataLoader 会自动添加 batch 维度
        return {
            "input_ids": encoding["input_ids"].squeeze(0),
            "attention_mask": encoding["attention_mask"].squeeze(0),
            "token_type_ids": encoding["token_type_ids"].squeeze(0),
            "labels": labels_tensor,
        }


def load_records(split: str, data_dir: Optional[Path] = None) -> list:
    """
    从 JSON 文件中加载数据记录。
    
    Args:
        split (str): 数据集划分名称，如 'train', 'validation', 'test'
        data_dir (Optional[Path]): 数据目录路径，默认为 DATA_DIR
        
    Returns:
        list: 解析后的 JSON 数据列表
        
    Raises:
        FileNotFoundError: 如果文件不存在
    """
    d = data_dir or DATA_DIR
    file_path = d / f"{split}.json"
    
    # 检查文件是否存在，提供更友好的错误提示
    if not file_path.exists():
        raise FileNotFoundError(f"Data file not found: {file_path}")
        
    with open(file_path, "r", encoding="utf-8") as f:
        return json.load(f)


def build_dataloaders(
    tokenizer: BertTokenizer,
    label2id: dict,
    batch_size: int = 32,
    max_length: int = 128,
    data_dir: Optional[Path] = None,
) -> tuple[DataLoader, DataLoader, DataLoader]:
    """
    构建训练、验证和测试的 DataLoader。
    
    自动加载对应 split 的数据，创建 Dataset 实例，并封装为 DataLoader。
    针对 Peoples Daily 数据集，尝试加载 'validation' 或 'dev' 作为验证集。
    
    Args:
        tokenizer (BertTokenizer): BERT 分词器
        label2id (dict): 标签映射字典
        batch_size (int): 批大小
        max_length (int): 最大序列长度
        data_dir (Optional[Path]): 数据目录
        
    Returns:
        tuple: (train_loader, val_loader, test_loader)
    """
    # 1. 加载训练集
    train_records = load_records("train", data_dir)
    # 2. 加载验证集 (尝试多种常见命名)
    val_records = load_records("validation", data_dir)           
    # 3. 加载测试集
    test_records = load_records("test", data_dir)


    # 4. 创建 Dataset 实例
    train_ds = NerDataset(train_records, tokenizer, label2id, max_length)
    val_ds = NerDataset(val_records, tokenizer, label2id, max_length)
    test_ds = NerDataset(test_records, tokenizer, label2id, max_length)

    # 打印数据集规模信息
    print(f"数据集规模：训练={len(train_ds)}，验证={len(val_ds)}，测试={len(test_ds)}")


    # 5. 创建 DataLoader
    # train_loader 设置 shuffle=True 以打乱数据
    # num_workers=0 表示在主进程中加载数据，适合调试或小数据集
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=0)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=0)

    return train_loader, val_loader, test_loader



if __name__ == "__main__":
    build_label_schema()
