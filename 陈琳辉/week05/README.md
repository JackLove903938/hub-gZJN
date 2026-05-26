# Week05 作业：基于 Transformer 的单向语言模型与文本生成
 
本项目实现了一个完整的单向 Transformer 语言模型，包含训练和文本生成功能。
 
## 📁 文件结构
 
```
week05/
├── week05_transformer_lm.py  # 主代码文件
└── README.md                  # 本说明文件
```
 
## 🚀 快速开始
 
### 环境要求
- Python 3.8+
- PyTorch 2.0+
- NumPy
- Matplotlib
 
### 运行代码
 
```bash
# 进入目录
cd week05
 
# 运行训练和生成
python week05_transformer_lm.py
```
 
## 🏗️ 核心组件
 
### 1. `PositionalEncoding`
位置编码模块，使用正弦/余弦函数为 Transformer 提供位置信息。
 
### 2. `MultiHeadAttention`
多头自注意力模块，支持因果掩码实现单向注意力。
 
### 3. `TransformerDecoderLayer`
Transformer 解码器层，包含自注意力子层和前馈网络子层。
 
### 4. `TransformerLanguageModel`
完整的语言模型，由嵌入层、位置编码、多个解码器层和输出层组成。
 
### 5. `generate_causal_mask`
生成下三角因果掩码，确保每个位置只能关注到前面的位置。
 
### 6. `TextDataset`
简单的字符级文本数据集。
 
### 7. `generate_text`
文本生成函数，支持温度参数控制生成多样性。
 
## 📊 代码说明
 
### 单向语言模型原理
单向语言模型的任务是：给定前面的文本序列，预测下一个字符（或词）。在训练时，我们使用因果掩码（Causal Mask）确保每个位置只能看到前面的位置，而看不到后面的位置。
 
### 文本生成
使用自回归方式生成文本：每次生成一个字符，然后将该字符加入输入序列，继续生成下一个字符。温度参数 `temperature` 控制生成的随机性：
- 温度越低，生成越保守确定
- 温度越高，生成越随机多样
 
## 🔧 可调参数
 
| 参数 | 默认值 | 含义 |
|------|--------|------|
| `d_model` | 128 | 模型维度 |
| `num_heads` | 4 | 注意力头数 |
| `num_layers` | 4 | Transformer 层数 |
| `d_ff` | 256 | 前馈网络中间层维度 |
| `dropout` | 0.1 | Dropout 比率 |
| `num_epochs` | 200 | 训练轮数 |
| `seq_len` | 32 | 序列长度 |
| `batch_size` | 8 | 批次大小 |
 
## 📝 输出结果
训练完成后，会在 `output/` 目录下保存：
- `best_transformer_lm.pt` — 训练好的模型权重
- `training_history.json` — 训练损失历史
- `training_loss.png` — 训练损失曲线图
 
 
