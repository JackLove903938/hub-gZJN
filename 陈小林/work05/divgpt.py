"""
定义一个简单版本的gpt模型
训练阶段：
1、使用语料库进行错位预测
2、使用带[CLS]的语料进行微调，使其学会合适结束预测
预测阶段：
1，根据前文获取一下个词的概率
2、top-N 取满满足的词再做概率抽取
3、如果预测词='[CLS]'结束预测，输出结果；否则继续从1开始
"""
import math

import torch.nn as nn
import torch

class DivGPT(nn.Module):
    def __init__(self,
                 vocab_size,
                 pos_size,
                 head_size,
                 hidden_size,
                 n_layers,
                 ignore_index):
        super(DivGPT, self).__init__()
        self.vocab_size = vocab_size
        self.head_size = head_size
        self.embedding_vocab = nn.Embedding(vocab_size, hidden_size)
        self.embedding_position = nn.Embedding(pos_size, hidden_size)
        self.attention_layers = []
        for layer in range(n_layers):
            self.attention_layers.append(
                {
                    'q_linear': nn.Linear(hidden_size, hidden_size),
                    'k_linear': nn.Linear(hidden_size, hidden_size),
                    'v_linear': nn.Linear(hidden_size, hidden_size),
                    'layer1': nn.Linear(hidden_size, hidden_size),
                    'layer2': nn.Linear(hidden_size, hidden_size),
                    'feed1': nn.Linear(hidden_size, hidden_size * 2),
                    'feed2': nn.Linear(hidden_size * 2, hidden_size)
                }
            )
        self.ignore_index = ignore_index
        self.dk = hidden_size // hidden_size
        self.gelu = nn.GELU()
        self.classifier = nn.Linear(hidden_size, vocab_size)
        self.loss = nn.CrossEntropyLoss(ignore_index=ignore_index)

    def forward(self, x, y = None):
        #词向量处理

        x = self.embedding(x)

        for layer in self.attention_layers:
            x = self.attention(x, layer)

        x = self.classifier(x)
        x = x.view(-1, self.vocab_size)
        if y is not None:
            y = y.view(-1)
            return self.loss(x, y)

        return x

    def attention(self, x:torch.Tensor, layer_info:dict[str, nn.Linear])-> torch.Tensor:
        batch_size, length, hidde_size = x.shape
        q = layer_info['q_linear'](x)
        k = layer_info['k_linear'](x)
        v = layer_info['v_linear'](x)
        q = torch.reshape(q, (batch_size, length, self.head_size, -1)).transpose(1,2)
        k = torch.reshape(k, (batch_size, length, self.head_size, -1)).transpose(1,2)
        v = torch.reshape(v, (batch_size, length, self.head_size, -1)).transpose(1,2)
        score = q @ k.transpose(-1, -2) / math.sqrt(self.dk)

        #取下三角 只能看到前面的信息
        mask = torch.triu(torch.ones(length, length), diagonal=1).bool()
        score.masked_fill_(mask, float('-inf'))
        x_new = torch.softmax(score, dim=-1) @ v
        x_new = x_new.transpose(1,2).reshape(batch_size, length, -1)

        #残差
        x = self.layer_normal(x + x_new, layer_info['layer1'])

        #前馈
        x_new = self.feed_ward(x, layer_info['feed1'], layer_info['feed2'])

        #残差
        x = self.layer_normal(x + x_new, layer_info['layer2'])
        return x

    def feed_ward(self, x:torch.Tensor, linear1:nn.Linear, linear2:nn.Linear)-> torch.Tensor:
        x = linear1(x)
        x = self.gelu(x)
        return linear2(x)


    def layer_normal(self, x:torch.Tensor, linear:nn.Linear)-> torch.Tensor:
        x_mean = torch.mean(x, dim=-1, keepdim=True)
        std = torch.std(x, dim=-1, keepdim=True, unbiased=False)
        x = (x - x_mean) / (std + 1e-12)
        return linear(x)


    def embedding(self, x):
        _, length = x.shape
        x = self.embedding_vocab(x)
        posi_ = torch.arange(length, dtype=torch.long, device=x.device)
        return x + self.embedding_position(posi_)


