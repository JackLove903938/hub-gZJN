import json
import re
import os
import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader,random_split
from transformers import BertTokenizer
import csv
"""
数据加载
"""


class DataGenerator:
    def __init__(self, data_path, config):
        self.config = config
        self.path = data_path
        # self.index_to_label = {'差评': '0', '好评': '1'}
        # self.label_to_index = dict((y, x) for x, y in self.index_to_label.items())
        # self.config["class_num"] = len(self.index_to_label)
        if self.config["model_type"] == "bert":
            self.tokenizer = BertTokenizer.from_pretrained(config["pretrain_model_path"])
        self.vocab = load_vocab(config["vocab_path"])
        self.config["vocab_size"] = len(self.vocab)
        self.load()
        


    def load(self):
        self.data = []
        if self.path.endswith('.csv'):
            with open(self.path, encoding="utf8") as f:
                reader = csv.reader(f)
                next(reader)
                for row in reader:
                    label = int(row[0])
                    review = row[1]
                    if self.config["model_type"] == "bert":
                        input_id = self.tokenizer.encode(review,max_length=self.config["max_length"],
                        padding='max_length',truncation=True)
                    else:
                        input_id = self.encode_sentence(review)
                    input_id = torch.LongTensor(input_id)
                    label = torch.LongTensor([label])
                    self.data.append([input_id,label])
                # print(f"加载了 {len(self.data)} 条数据")
        else:
            with open(self.path, encoding="utf8") as f:
                for line in f:
                    line = json.loads(line)
                    label = line["label"]
                    title = line["input_id"]
                    if self.config["model_type"] == "bert":
                        input_id = self.tokenizer.encode(title, max_length=self.config["max_length"], pad_to_max_length=True)
                    else:
                        input_id = self.encode_sentence(title)
                    input_id = torch.LongTensor(input_id)
                    label_index = torch.LongTensor([label])
                    self.data.append([input_id, label_index])
        return

    def encode_sentence(self, text):
        input_id = []
        for char in text:
            input_id.append(self.vocab.get(char, self.vocab["[UNK]"]))
        input_id = self.padding(input_id)
        return input_id

    #补齐或截断输入的序列，使其可以在一个batch内运算
    def padding(self, input_id):
        input_id = input_id[:self.config["max_length"]]
        input_id += [0] * (self.config["max_length"] - len(input_id))
        return input_id

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        return self.data[index]

def load_vocab(vocab_path):
    token_dict = {}
    with open(vocab_path, encoding="utf8") as f:
        for index, line in enumerate(f):
            token = line.strip()
            token_dict[token] = index + 1  #0留给padding位置，所以从1开始
    return token_dict

def split_data(data,train_ratio=0.8,random=42):
    train_len = int(len(data) * train_ratio)
    valid_len = len(data) - train_len
    torch.manual_seed(random)
    train_data,valid_data = random_split(data,[train_len,valid_len])
    return train_data,valid_data

def save_file(data,file_path):
    data_list = []
    for line in data:
        input_id = line[0].tolist()
        label = line[1].item()
        data_list.append({
            "input_id": input_id,
            "label": label
        })
    print(len(data_list))
    with open(file_path,'w',encoding='utf-8') as f:
        for data in data_list:
            f.write(json.dumps(data, ensure_ascii=False) + '\n')
    
#用torch自带的DataLoader类封装数据
def load_data(data_path, config, shuffle=True):
    dg = DataGenerator(data_path, config)
    dl = DataLoader(dg, batch_size=config["batch_size"], shuffle=shuffle)
    return dl

if __name__ == "__main__":
    from config import Config
    print(Config["pretrain_model_path"])
    dg = DataGenerator(r"D:\AI大模型课程\课程- 第七周 文本分类\week07_evaluate\文本分类练习.csv", Config)
    train_data,valid_data = split_data(dg)
    save_file(train_data,file_path=Config["train_data_path"])
    save_file(valid_data,file_path=Config["valid_data_path"])
    
