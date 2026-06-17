# -*- coding: utf-8 -*-

"""
配置参数信息
"""

Config = {
    "model_path": "model_output",
    "schema_path": "D:/AI大模型课程/第八周 文本匹配/week8_homework/data/schema.json",
    "train_data_path": "D:/AI大模型课程/第八周 文本匹配/week8_homework/data/train.json",
    "valid_data_path": "D:/AI大模型课程/第八周 文本匹配/week8_homework/data/valid.json",
    "vocab_path":"D:/AI大模型课程/第八周 文本匹配/week8_homework/chars.txt",
    "max_length": 20,
    "hidden_size": 128,
    "epoch": 10,
    "batch_size": 32,
    "epoch_data_size": 200,     #每轮训练中采样数量
    "positive_sample_rate":0.5,  #正样本比例
    "optimizer": "adam",
    "learning_rate": 1e-3,
}