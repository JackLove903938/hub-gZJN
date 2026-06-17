# -*- coding: utf-8 -*-

import torch
import torch.nn as nn
from torch.optim import Adam, SGD
from torchcrf import CRF
from transformers import BertModel

class TorchModel(nn.Module):
    def __init__(self, config):
        super(TorchModel, self).__init__()
        class_num = config["class_num"]
        self.bert = BertModel.from_pretrained(config["bert_path"])
        hidden_size = self.bert.config.hidden_size   # 例如 768
        self.classify = nn.Linear(hidden_size, class_num)
        self.crf_layer = CRF(class_num, batch_first=True)
        self.use_crf = config["use_crf"]
        self.loss = torch.nn.CrossEntropyLoss(ignore_index=-1)

    def forward(self, x, target=None):
        attention_mask = (x != 0).bool()   # 布尔掩码
        outputs = self.bert(input_ids=x, attention_mask=attention_mask)
        sequence_output = outputs[0]       # (batch, seq_len, hidden_size)
        predict = self.classify(sequence_output)  # (batch, seq_len, num_tags)

        if target is not None:
            if self.use_crf:
                # CRF 返回负对数似然，直接作为 loss
                return -self.crf_layer(predict, target, attention_mask, reduction="mean")
            else:
                return self.loss(predict.view(-1, predict.shape[-1]), target.view(-1))
        else:
            if self.use_crf:
                decoded = self.crf_layer.decode(predict, mask=attention_mask)
                batch_pred = []
                max_len = x.size(1)
                for dec in decoded:
                    dec_tensor = torch.tensor(dec, dtype=torch.long, device=x.device)
                    padded = torch.full((max_len,), -1, dtype=torch.long, device=x.device)
                    valid_len = dec_tensor.size(0)
                    padded[:valid_len] = dec_tensor
                    batch_pred.append(padded)
                return torch.stack(batch_pred)
            else:
                pred_ids = predict.argmax(dim=-1)          # (batch, seq_len)
                # attention_mask 中 False 表示 padding，将这些位置设为 -1
                pred_ids = torch.where(attention_mask, pred_ids, torch.tensor(-1, device=x.device))
                return pred_ids

def choose_optimizer(config, model):
    optimizer = config["optimizer"]
    learning_rate = config["learning_rate"]
    if optimizer == "adam":
        return Adam(model.parameters(), lr=learning_rate)
    elif optimizer == "sgd":
        return SGD(model.parameters(), lr=learning_rate)

if __name__ == "__main__":
    from config import Config
    model = TorchModel(Config)