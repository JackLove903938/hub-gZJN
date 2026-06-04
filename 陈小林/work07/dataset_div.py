import json
from pathlib import Path

from torch import Tensor,LongTensor
from torch.nn.functional import embedding
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizer

"""
1、定义数据集入参
    a、bert解码器
    b、解析数据集文件名
    c、固定文件集根路径
    d、固定标签数据集路径
2、类初始化
    a、初始化数据集，包含bert解码结果
    b、初始化label2id    
"""
ROOT = Path(__file__).parent.parent
DATA_DIR = ROOT / 'data' / 'peoples_daily'

class PeoplesDailyDataset(Dataset):
    def __init__(self,
                 data_name: str,
                 max_length: int = 128,
                 tokenizer: BertTokenizer = None):
        super(Dataset, self).__init__()
        self.data_name = data_name
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.tokens = []
        self.embeddings = []
        self.tags = []
        self.init()

    def init(self):
        with open(DATA_DIR / 'label_names.json') as f:
            labels = json.load(f)
            self.label2id = {label: label_id for label_id, label in enumerate(labels)}
            self.id2label = {label_id: label for label_id, label in enumerate(labels)}
        limit = 100
        with open(DATA_DIR / self.data_name) as f:
            datas = json.load(f)
            for data in datas:
                if limit <= 0:
                    break
                limit -= 1
                tokens = data['tokens']
                self.tokens.append(tokens)
                self.tags = data['ner_tags']

                res = self.tokenizer(
                    text=tokens,
                    is_split_into_words = True,
                    padding= 'max_length',
                    truncation = True,
                    max_length=self.max_length
                )
                #bert解码
                work_ids = res.encodings[0].word_ids
                labels = []
                label_ids = []
                for work_id in work_ids:
                    if work_id is None:
                        labels.append(-100)
                        label_ids.append(-100)
                    else:
                        labels.append(self.tags[work_id])
                        label_ids.append(self.label2id[self.tags[work_id]])
                self.embeddings.append({
                    'input_ids': LongTensor(res['input_ids']),
                    'attention_mask': LongTensor(res['attention_mask']),
                    'token_type_ids': LongTensor(res['token_type_ids']),
                    'labels': labels,
                    'label_ids': LongTensor(label_ids),
                })
    def __len__(self):
        return len(self.tokens)

    def __getitem__(self, idx):
        embedding = self.embeddings[idx]
        return (embedding['input_ids'],
            embedding['attention_mask'],
            embedding['token_type_ids'],
            embedding['label_ids'])

