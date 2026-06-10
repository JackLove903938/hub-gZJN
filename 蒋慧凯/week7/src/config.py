"""
全局配置（路径、超参数、设备）

教学重点：
  1. 用类封装配置，避免全局变量污染命名空间
  2. 自动检测本地模型路径，不存在时回退到 HuggingFace 模型名
  3. 所有路径统一用 Path 对象，跨平台兼容
"""

from pathlib import Path
import torch


class Config:
    """ peoples_daily NER 作业配置 """

    def __init__(self):
        # 项目根目录（homework/）
        self.root = Path(__file__).parent.parent

        # 数据路径
        self.data_dir = self.root / "data" / "peoples_daily"

        # 输出路径
        self.output_dir = self.root / "outputs"
        self.ckpt_dir = self.output_dir / "checkpoints"
        self.log_dir = self.output_dir / "logs"
        self.figure_dir = self.output_dir / "figures"

        # 模型路径（优先本地，否则 HuggingFace 自动下载）
        candidates = [
            self.root.parent.parent / "pretrain_models" / "bert-base-chinese",
            self.root.parent.parent / "预习" / "第六周 语言模型" / "bert-base-chinese",
        ]
        self.bert_path = "bert-base-chinese"
        for c in candidates:
            if c.exists():
                self.bert_path = str(c)
                break

        # 设备
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        # 标签体系（peoples_daily：PER / ORG / LOC）
        self.entity_types = ["PER", "ORG", "LOC"]

        # BERT 训练超参
        self.epochs = 3
        self.batch_size = 32
        self.max_length = 128
        self.lr = 2e-5
        self.head_lr_mult = 5.0
        self.warmup_ratio = 0.1
        self.grad_accum = 1
        self.dropout = 0.1

    def print_config(self):
        print("=" * 50)
        print("【配置信息】")
        print(f"  数据目录: {self.data_dir}")
        print(f"  BERT 路径: {self.bert_path}")
        print(f"  设备: {self.device}")
        print(f"  Epochs: {self.epochs}")
        print(f"  Batch size: {self.batch_size}")
        print(f"  Max length: {self.max_length}")
        print(f"  LR (BERT): {self.lr}")
        print(f"  LR (head): {self.lr * self.head_lr_mult}")
        print("=" * 50)


# 全局单例
cfg = Config()
