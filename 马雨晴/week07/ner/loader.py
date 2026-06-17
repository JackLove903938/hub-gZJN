# -*- coding: utf-8 -*-

import json
import torch
from torch.utils.data import DataLoader

class DataGenerator:
    def __init__(self, data_path, config):
        self.config = config
        self.path = data_path
        self.vocab = load_vocab(config["vocab_path"])
        self.config["vocab_size"] = len(self.vocab)
        self.sentences = []
        self.schema = self.load_schema(config["schema_path"])
        self.max_length = config["max_length"]
        self.load()

    def load(self):
        self.data = []
        with open(self.path, encoding="utf8") as f:
            segments = f.read().split("\n\n")
            for segment in segments:
                chars = []
                labels = []
                for line in segment.split("\n"):
                    if line.strip() == "":
                        continue
                    char, label = line.split()
                    chars.append(char)
                    labels.append(self.schema[label])
                self.sentences.append("".join(chars))   # 原始句子，不含特殊标记
                # 直接编码字符序列（不加 [CLS]/[SEP]）
                input_ids = [self.vocab.get(c, self.vocab.get('[UNK]', 0)) for c in chars]
                label_ids = labels
                # 文本对齐，统一到最大长度
                input_ids = input_ids[:self.max_length]
                label_ids = label_ids[:self.max_length]
                pad_len = self.max_length - len(input_ids)
                input_ids += [0] * pad_len          # 0 是 [PAD]
                label_ids += [-1] * pad_len
                self.data.append([
                    torch.LongTensor(input_ids),
                    torch.LongTensor(label_ids)
                ])

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        return self.data[index]

    def load_schema(self, path):
        with open(path, encoding="utf8") as f:
            return json.load(f)

def load_vocab(vocab_path):
    token_dict = {}
    with open(vocab_path, encoding="utf8") as f:
        for index, line in enumerate(f):
            token = line.strip()
            token_dict[token] = index   # BERT 词表直接使用原始索引，0 是 [PAD]
    return token_dict

def load_data(data_path, config, shuffle=True):
    dg = DataGenerator(data_path, config)
    dl = DataLoader(dg, batch_size=config["batch_size"], shuffle=shuffle)
    return dl

if __name__ == "__main__":
    from config import Config
    dg = DataGenerator("../ner_data/train.txt", Config)