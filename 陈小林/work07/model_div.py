from torchcrf import CRF
import torch.nn as nn
from torch import Tensor
from transformers import BertModel

class ModelDiv(nn.Module):
    def __init__(self,
                 label_size:int
                 ) -> None:
        super(ModelDiv, self).__init__()
        self.bert = BertModel.from_pretrained(r'/Users/cxl/hub-gZJN/陈小林/work06/bert-base-chinese')
        self.dropout = nn.Dropout(0.1)
        self.classifier = nn.Linear(768, label_size)
        self.crf = CRF(num_tags=label_size, batch_first=True)

    def forward(self,
                input_ids:Tensor,           #shape is (B,L)
                attention_mask:Tensor,      #shape is (B,L)
                token_type_ids:Tensor = None,        #shape is (B,L)
                labels:Tensor=None):

        #output shape is (B,L,768)
        output, _ = self.bert(
                    input_ids = input_ids,
                    attention_mask = attention_mask,
                    token_type_ids = token_type_ids)
        output = self.dropout(output)
        output = self.classifier(output) #shape is (B,L,label_size)

        if labels is not None:
            labels_crf = labels.clone()
            labels_crf.masked_fill_(labels_crf == -100,0)
            return output, -self.crf(output, labels_crf, attention_mask.bool(), reduction='mean')

        return output, None