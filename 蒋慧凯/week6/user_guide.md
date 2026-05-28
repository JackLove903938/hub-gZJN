# 文本分类方法对比

对比 BERT Fine-tune、LLM Zero-Shot、LLM SFT (LoRA) 三种文本分类方法在 TNEWS 数据集上的效果。

## 文件结构

```
.
├── src/
│   ├── config.py              # 全局配置（路径、超参、设备自动检测）
│   ├── data_utils.py          # 数据下载、Dataset、DataLoader
│   ├── bert_pipeline.py       # BERT 微调（模型/训练/评估）
│   ├── llm_zeroshot_pipeline.py   # LLM Zero-Shot 分类
│   ├── llm_sft_pipeline.py        # LLM SFT + LoRA 微调
│   └── compare_report.py      # 对比报告生成（表格+图表）
├── data/                      # TNEWS 数据集（JSON 格式）
├── outputs/                   # 结果输出
│   ├── checkpoints/           # 模型检查点
│   ├── figures/               # 可视化图表
│   ├── bert_results.json
│   ├── zeroshot_results.json
│   ├── sft_results.json
│   └── comparison_table.md
├── run_all.py                 # 统一入口
├── requirements.txt           # Python 依赖
└── README.md                  # 本文件
```

## 环境要求

- Python >= 3.10
- PyTorch >= 2.6.0
- Transformers >= 5.5.0
- 可选：CUDA（GPU 可加速，CPU 亦可运行）

## 安装依赖

```bash
pip install -r requirements.txt
```

## 运行方式

### 一键运行全部对比

```bash
python run_all.py --all
```

### 单独运行某种方法

```bash
python run_all.py --bert       # BERT 微调
python run_all.py --zeroshot   # LLM Zero-Shot
python run_all.py --sft        # LLM SFT (LoRA)
python run_all.py --report     # 只生成对比报告
```

### 数据准备

首次运行会自动从 HuggingFace 下载 TNEWS 数据集并保存到 `data/` 目录。若下载失败，可从其他来源获取 `train.json`、`val.json`、`test.json`、`label_map.json` 并放入 `data/` 目录。

## 输出结果

运行完成后，`outputs/` 目录下会生成：

- `comparison_table.md` —— Markdown 格式的对比报告
- `figures/comparison_chart.png` —— 准确率与推理速度的可视化对比图
- `*_results.json` —— 各方法的原始评估结果

## 模型路径

默认优先查找本地预训练模型（`../../pretrain_models/`），不存在时自动从 HuggingFace 下载：

- BERT：`bert-base-chinese`
- LLM：`Qwen/Qwen2-0.5B-Instruct`

可通过修改 `src/config.py` 中的路径配置使用其他模型。

## 硬件说明

- BERT 微调：CPU 可运行，GPU 可加速
- LLM Zero-Shot / SFT：建议 GPU，CPU 运行较慢
- 显存不足时可调整 `src/config.py` 中的 batch_size 和 grad_accum
