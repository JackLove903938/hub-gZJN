import re

import torch
from torch.utils.data import Dataset

"""
语料使用title训练 content太长
"""
class OffsetDataSet(Dataset):
    """
    用于错位微调
    """
    def __init__(self, train_path:str, vocab_path:str):
        super(OffsetDataSet, self).__init__()
        self.res = []
        self.res_offset = []
        self.load_vocab(vocab_path)
        self.load_data(train_path)

    def load_vocab(self, vocab_path):
        with open(vocab_path, encoding='utf-8', mode='r') as f:
            self.vocab = {}
            for index, line in enumerate(f):
                self.vocab[line.strip()] = index

    def load_data(self, path:str):
        """
        加载词表，加载数据集
        """
        texts = []
        with open(path, encoding='utf-8', mode='r') as f:
            for index, line in enumerate(f):
                result = re.split(r'[，。！？；]', line.strip())
                texts.extend(result)
        max = 0
        for text in texts:
            if text == '':
                continue
            data = [self.vocab.get(char, self.vocab['[UNK]']) for char in text]
            data_offset = data[1:]
            data_offset.append(self.vocab['[CLS]'])
            self.res.append(data)
            self.res_offset.append(data_offset)
            if max < len(data_offset):
                max = len(data_offset)

        for idx, data in enumerate(self.res_offset):
            dff = max - len(data)
            for i in range(dff):
                data.append(self.vocab['[CLS]'])
                self.res[idx].append(self.vocab['[CLS]'])
            self.res[idx] = torch.LongTensor(self.res[idx])
            self.res_offset[idx] = torch.LongTensor(data)

    def __len__(self):
        return len(self.res)

    def __getitem__(self, idx):
        return self.res[idx], self.res_offset[idx]