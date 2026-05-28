"""
config_fast.py —— CPU 快速模式配置（用于紧急出结果）

覆盖 cfg 的关键参数，减少训练时间和数据量。
使用方式：把 config.py 里的配置替换为这些值，或运行时动态修改。
"""

# BERT 快速模式
BERT_EPOCHS = 1              # 从 2 降到 1
BERT_BATCH_SIZE = 16         # 可适当减小以加速
BERT_NUM_TRAIN = 5000        # 只用 5000 条训练（原 53360）

# Zero-Shot 快速模式
ZEROSHOT_NUM_SAMPLES = 50    # 从 200 降到 50（快速验证）

# SFT 快速模式
SFT_EPOCHS = 1
SFT_NUM_TRAIN = 1000         # 只用 1000 条
SFT_BATCH_SIZE = 2
SFT_GRAD_ACCUM = 8
