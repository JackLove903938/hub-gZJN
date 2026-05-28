import sys
sys.path.insert(0, 'src')

print('step1: tokenizer')
from transformers import BertTokenizer
t = BertTokenizer.from_pretrained('bert-base-chinese')
print('tokenizer ok')

print('step2: load json')
import json
with open('data/train.json', encoding='utf-8') as f:
    data = json.load(f)
print('json loaded:', len(data))

print('step3: create dataset')
from data_utils import BERTDataset
ds = BERTDataset(data[:10], t, 64)
print('dataset created, len:', len(ds))

print('step4: getitem')
item = ds[0]
print('getitem ok')
print('keys:', list(item.keys()))
for k, v in item.items():
    print(f'  {k}: {type(v)}', end='')
    if hasattr(v, 'shape'):
        print(f' shape={v.shape}')
    else:
        print()

print('ALL OK')
