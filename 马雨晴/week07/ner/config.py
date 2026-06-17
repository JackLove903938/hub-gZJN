# -*- coding: utf-8 -*-

"""
配置参数信息
"""

Config = {
    "model_path": "model_output",
    "schema_path": "D:/AI大模型课程/week9 序列标注问题/ner/ner_data/schema.json",
    "train_data_path": "D:/AI大模型课程/week9 序列标注问题/ner/ner_data/train",
    "valid_data_path": "D:/AI大模型课程/week9 序列标注问题/ner/ner_data/test",
    "vocab_path":"D:/AI大模型课程/week9 序列标注问题/ner/ner_data/bert_vocab.txt",
    "max_length": 100,
    "hidden_size": 768,
    "num_layers": 2,
    "epoch": 20,
    "batch_size": 16,
    "optimizer": "adam",
    "learning_rate": 1e-5,
    "use_crf": True,
    "class_num": 9,
    "bert_path": "C:/Users/11513/Desktop/bert-base-chinese"
}

