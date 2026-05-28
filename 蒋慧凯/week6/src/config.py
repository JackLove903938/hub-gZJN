"""
config.py —— 全局配置管理

设计原则：
  1. 所有路径、超参集中管理，一处修改全局生效
  2. 支持本地模型路径（优先）和 HuggingFace 自动下载（降级）
  3. 输出目录自动创建，确保拷贝到新环境也能运行

使用方式：
  from config import Config
  cfg = Config()
  print(cfg.bert_model_path)
"""

import os
from pathlib import Path


class Config:
    """
    统一管理作业所需的所有路径和超参数。
    
    路径规则：
      - 优先查找本地预训练模型（../../pretrain_models/）
      - 本地不存在时，使用 HuggingFace 模型名，由 transformers 自动下载
    """

    def __init__(self):
        # ── 项目根目录（homework/ 文件夹所在位置）─────────────────────────────
        # 注意：Windows UNC 网络路径上 .resolve() 可能报错，用 absolute() 替代
        self.project_root = Path(__file__).parent.parent.absolute()

        # ── 数据目录 ──────────────────────────────────────────────────────────
        self.data_dir = self.project_root / "data"
        self.data_dir.mkdir(parents=True, exist_ok=True)

        # ── 输出目录 ──────────────────────────────────────────────────────────
        self.output_dir = self.project_root / "outputs"
        self.checkpoint_dir = self.output_dir / "checkpoints"
        self.figure_dir = self.output_dir / "figures"
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.figure_dir.mkdir(parents=True, exist_ok=True)

        # ── 模型路径配置 ──────────────────────────────────────────────────────
        # 本地模型根目录（与 text_classification项目 共用同一套预训练模型）
        local_model_root = self.project_root.parent / "pretrain_models"

        # BERT 模型：优先本地，否则从 HuggingFace 自动下载
        local_bert = local_model_root / "bert-base-chinese"
        self.bert_model_path = str(local_bert) if local_bert.exists() else "bert-base-chinese"

        # Qwen2 模型：优先本地，否则从 HuggingFace 自动下载
        local_qwen = local_model_root / "Qwen2-0.5B-Instruct"
        self.qwen_model_path = str(local_qwen) if local_qwen.exists() else "Qwen/Qwen2-0.5B-Instruct"

        # ── 标签配置（TNEWS 15 类新闻）─────────────────────────────────────────
        self.num_labels = 15
        self.label_names = [
            "故事", "文化", "娱乐", "体育", "财经",
            "房产", "汽车", "教育", "科技", "军事",
            "旅游", "国际", "证券", "农业", "电竞",
        ]

        # ── BERT 微调超参 ─────────────────────────────────────────────────────
        self.bert_epochs = 1          # CPU 快速模式
        self.bert_batch_size = 32     # CPU 保持较大 batch 减少迭代次数
        self.bert_max_length = 64
        self.bert_lr = 2e-5
        self.bert_head_lr_mult = 5.0
        self.bert_warmup_ratio = 0.1
        self.bert_pool = "cls"
        self.bert_num_train = 2000    # 只用 2000 条快速训练（原 53360）

        # ── LLM SFT (LoRA) 超参 ───────────────────────────────────────────────
        self.sft_epochs = 1           # CPU 快速模式（原 2）
        self.sft_batch_size = 2       # CPU 减小显存占用
        self.sft_grad_accum = 8       # 等效 batch = 2*8 = 16
        self.sft_max_length = 128
        self.sft_lr = 2e-4            # LoRA 可用较大学习率（原始权重冻结）
        self.sft_lora_r = 8
        self.sft_lora_alpha = 16
        self.sft_num_train = 1000     # CPU 快速模式（原 3000）

        # ── LLM Zero-shot 超参 ────────────────────────────────────────────────
        self.zeroshot_num_samples = 50   # CPU 快速模式（原 200）

        # ── 设备配置 ──────────────────────────────────────────────────────────
        # 自动检测 CUDA，无 GPU 则回退到 CPU（BERT 可在 CPU 运行，LLM 建议 GPU）
        import torch as _torch
        self.device = "cuda" if _torch.cuda.is_available() else "cpu"

    def print_config(self):
        """打印当前配置，方便调试和确认。"""
        print("=" * 50)
        print("【作业配置】")
        print(f"  项目根目录 : {self.project_root}")
        print(f"  数据目录   : {self.data_dir}")
        print(f"  输出目录   : {self.output_dir}")
        print(f"  BERT 模型  : {self.bert_model_path}")
        print(f"  Qwen 模型  : {self.qwen_model_path}")
        print(f"  设备       : {self.device}")
        print("=" * 50)


# 全局单例，各模块直接导入使用
cfg = Config()
