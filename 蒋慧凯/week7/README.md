# Week7 作业：序列标注模型训练（peoples_daily）

> 在 peoples_daily 数据集上实现序列标注模型训练，对比 BERT+Linear、BERT+CRF、LLM API、LLM SFT (LoRA) 四种方案。

---

## 目录结构

```
homework/
├── README.md                  # 本文件
├── requirements.txt           # Python 依赖
├── run_all.py                 # 一键运行入口
├── src/
│   ├── config.py              # 全局配置（路径、超参）
│   ├── dataset.py             # 数据集处理（peoples_daily BIO 格式）
│   ├── model.py               # BertNER（线性）+ BertCRFNER（CRF）
│   ├── train_eval.py          # 训练与评估流水线
│   └── compare_report.py      # 对比报告生成（表格+图表）
├── src_llm/
│   ├── llm_ner.py             # LLM API zero-shot / few-shot NER
│   ├── train_sft.py           # LoRA SFT 训练
│   └── evaluate_sft.py        # SFT 评估
├── data/
│   └── peoples_daily/         # 人民日报 NER 数据集
│       ├── train.json
│       ├── validation.json
│       ├── test.json
│       └── label_names.json
└── outputs/
    ├── checkpoints/           # 模型检查点
    ├── figures/               # 可视化图表
    ├── sft_adapter/           # LoRA adapter
    └── logs/                  # 训练和评估日志
```

---

## 快速开始

### 1. 安装依赖

```bash
cd "week7序列标注问题/homework"
pip install -r requirements.txt
```

### 2. 运行全部流程

```bash
# 运行 BERT + LLM API + SFT + 报告（需 GPU 和 API Key）
python run_all.py --all

# 只运行 BERT 两种模型（默认）
python run_all.py

# 只运行 BERT+Linear
python run_all.py --linear

# 只运行 BERT+CRF
python run_all.py --crf

# 只运行 LLM API（需 DASHSCOPE_API_KEY）
python run_all.py --llm

# 只运行 LLM SFT（LoRA）
python run_all.py --sft

# 只生成对比报告
python run_all.py --report
```

---

## 四种方法对比

| 维度 | BERT+Linear | BERT+CRF | LLM API | LLM SFT (LoRA) |
|------|-------------|----------|---------|----------------|
| **预测方式** | token 独立 softmax | Viterbi 全局解码 | 生成式 JSON | 生成式 JSON |
| **标签约束** | ❌ 无 | ✅ 转移矩阵保证 | ❌ 无 | ❌ 无 |
| **训练数据** | 20K 全量 | 20K 全量 | 0 | 20K 全量 |
| **可训练参数** | 102M | 102M | 0 | ~1M (0.22%) |
| **推理延迟** | ~5ms/条 | ~7ms/条 | ~2s/条 | ~0.5s/条 |
| **是否需要 API Key** | 否 | 否 | 是 | 否 |

---

## 核心代码模块说明

### config.py
- 统一管理所有路径、超参数
- 自动检测本地模型路径，不存在时回退到 HuggingFace 下载
- 支持 CPU/GPU 自动切换

### dataset.py
- 适配 peoples_daily 的 `tokens` + `ner_tags` BIO 格式
- `BertTokenizerFast` 子词对齐（`word_ids()` 策略），非首子词设为 `-100`
- `build_dataloaders()` 统一封装训练/验证/测试 DataLoader

### model.py
- `BertNER`：BERT + Linear，逐 token 独立预测
- `BertCRFNER`：BERT + CRF，全局最优序列解码
- `build_model()` 工厂函数统一创建

### train_eval.py
- `train_model()`：完整训练循环（分层学习率、warmup、梯度裁剪、最优模型保存）
- `evaluate_model()`：seqeval entity-level F1 + 非法序列统计
- `count_illegal_sequences()`：量化线性头的非法 BIO 序列数

### src_llm/llm_ner.py
- `build_client()`：封装 DashScope API 调用
- `zero_shot_prompt()` / `few_shot_prompt()`：两种 prompt 策略
- `compute_span_f1()`：span-level F1 计算（与 BERT 可比）

### src_llm/train_sft.py
- `SFTDataset`：chat 格式 + Loss Masking（只在 JSON 输出部分算 loss）
- LoRA 配置：r=8, alpha=16, 目标层 q/k/v/o_proj
- 支持 `--full_ft` 全量微调开关

### src_llm/evaluate_sft.py
- 自动识别 LoRA / 全量 checkpoint
- `merge_and_unload()` 合并 adapter 后评估
- span F1 与 BERT+CRF、LLM API 横向对比

### compare_report.py
- `generate_report()`：读取四种方法的 JSON 结果，生成 Markdown 表格 + 可视化柱状图

---

## 输出结果

运行完成后，`outputs/` 目录下会生成：

- `checkpoints/best_linear.pt` —— BERT+Linear 最优模型
- `checkpoints/best_crf.pt` —— BERT+CRF 最优模型
- `logs/train_linear.json` / `train_crf.json` —— 训练日志
- `logs/eval_linear_validation.json` / `eval_crf_validation.json` —— BERT 评估结果
- `logs/eval_llm.json` —— LLM API 评估结果
- `logs/eval_sft.json` —— SFT 评估结果
- `sft_adapter/` —— LoRA adapter 文件
- `comparison_table.md` —— 对比报告（Markdown 表格）
- `figures/comparison_chart.png` —— 可视化对比图

---

## 注意事项

1. **模型路径**：默认优先查找本地模型（`pretrain_models/bert-base-chinese` 或 `预习/第六周 语言模型/bert-base-chinese`），不存在时自动从 HuggingFace 下载。
2. **显存要求**：
   - BERT+Linear / BERT+CRF：~4GB，CPU 也可运行
   - LLM SFT (LoRA)：~6GB，建议 GPU
   - LLM API：无需本地显存
3. **训练时间**（RTX 4060 Ti 参考）：
   - BERT+Linear 3 epochs：~5 分钟
   - BERT+CRF 3 epochs：~7 分钟
   - LLM SFT LoRA 1 epoch：~30 分钟
4. **API Key**：LLM API 需要设置环境变量 `DASHSCOPE_API_KEY="sk-xxx"`，未设置时会自动跳过。
5. **Windows 用户**：若遇到多进程报错，DataLoader 的 `num_workers` 已默认设为 0。

---

## 第三方使用指南

本仓库提供一套完整的序列标注（NER）实验流程，支持 BERT 微调、LLM API 调用、LLM LoRA 微调三种技术路线。拿到本项目后，按以下步骤即可独立运行。

### 环境要求

- Python >= 3.9
- PyTorch >= 2.0（带 CUDA 支持可加速）
- Transformers >= 4.35
- 可选：peft（LoRA 微调必需）、torchcrf（CRF 模型必需）

### 安装依赖

```bash
cd "week7序列标注问题/homework"
pip install -r requirements.txt
```

### 运行方式

#### 一键运行全部对比

```bash
python run_all.py --all
```

#### 单独运行某种方法

```bash
python run_all.py --linear       # BERT + Linear
python run_all.py --crf          # BERT + CRF
python run_all.py --llm          # DeepSeek API Zero/Few-shot
python run_all.py --sft          # Qwen2 LoRA SFT 训练
python run_all.py --report       # 只生成对比报告
```

#### 直接运行独立脚本

```bash
# BERT 训练
cd src && python train_eval.py

# LLM API 评估
cd src_llm && python llm_ner.py

# LLM SFT 训练
cd src_llm && python train_sft.py --num_train 500 --epochs 1

# SFT 模型评估
cd src_llm && python evaluate_sft.py --n_samples 100
```

### 数据准备

数据集已内置在 `data/peoples_daily/` 目录（`train.json`、`validation.json`、`test.json`、`label_names.json`）。如需使用其他 NER 数据，保持相同 JSON 格式即可：

```json
{
  "tokens": ["北", "京", "是", "中", "国", "首", "都", "。"],
  "ner_tags": ["B-LOC", "I-LOC", "O", "B-LOC", "I-LOC", "O", "O", "O"]
}
```

### 输出结果

运行完成后，`outputs/` 目录下会生成：

| 文件 | 说明 |
|------|------|
| `comparison_table.md` | Markdown 格式对比报告 |
| `figures/comparison_chart.png` | 可视化柱状图 |
| `checkpoints/*.pt` | BERT 模型权重 |
| `sft_adapter/` | LoRA adapter 文件 |
| `logs/*.json` | 各方法原始评估结果 |

### 模型路径配置

默认优先查找本地预训练模型（`../../pretrain_models/`），不存在时自动从 HuggingFace 下载：

- BERT：`bert-base-chinese`
- LLM：`Qwen/Qwen2-0.5B-Instruct`

可通过修改 `src/config.py` 中的路径配置使用其他模型。

### 硬件与配置说明

| 方法 | 显存需求 | 是否必需 GPU | 备注 |
|------|---------|-------------|------|
| BERT+Linear | ~4 GB | 否，GPU 可加速 | CPU 亦可运行 |
| BERT+CRF | ~4 GB | 否，GPU 可加速 | CPU 亦可运行 |
| LLM API | 0 | 否 | 需配置 `DEEPSEEK_API_KEY` |
| LLM SFT (LoRA) | ~6 GB | 建议 GPU | CPU 运行极慢 |

显存不足时，可在 `src_llm/train_sft.py` 中调小 `batch_size` 或增大 `grad_accum`。
