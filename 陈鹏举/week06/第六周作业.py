import os
# 必须在导入 datasets 之前设置镜像
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'

import numpy as np
import pandas as pd
from tqdm import tqdm
import time
import warnings
warnings.filterwarnings('ignore')

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

from transformers import BertTokenizer, BertForSequenceClassification, AdamW
from transformers import get_linear_schedule_with_warmup

from datasets import load_dataset  # 新增导入

SAMPLE_PER_CLASS = 500
RANDOM_SEED = 42
EPOCHS = 5
BATCH_SIZE = 16
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def load_and_prepare_data(sample_per_class=500):
    dataset = load_dataset("ag_news")
    train_data = dataset["train"]
    test_data = dataset["test"]

    train_df = pd.DataFrame({'label': train_data['label'], 'text': train_data['text']})
    test_df = pd.DataFrame({'label': test_data['label'], 'text': test_data['text']})

    # 无需 label-1，因为 datasets 中的 ag_news 标签已经是 0~3

    sampled_train_dfs = []
    num_classes = train_df['label'].nunique()
    for c in range(num_classes):
        c_df = train_df[train_df['label'] == c]
        sample_n = min(sample_per_class, len(c_df))
        sampled_train_dfs.append(c_df.sample(sample_n, random_state=RANDOM_SEED))
    train_df = pd.concat(sampled_train_dfs, ignore_index=True)

    print(f"训练集样本数: {len(train_df)} (每类约{sample_per_class}条)")
    print(f"测试集样本数: {len(test_df)}")
    return train_df, test_df

# ================= 2. 方法一: TF-IDF + Logistic Regression (基线) =================
def train_tfidf_lr(train_df, test_df):
    print("\n" + "="*50)
    print("开始训练 TF-IDF + Logistic Regression")
    start_time = time.time()

    vectorizer = TfidfVectorizer(
        max_features=50000,
        ngram_range=(1, 3),
        sublinear_tf=True
    )
    X_train = vectorizer.fit_transform(train_df['text'].values)
    y_train = train_df['label'].values
    X_test = vectorizer.transform(test_df['text'].values)
    y_test = test_df['label'].values

    clf = LogisticRegression(
        C=1.0,
        max_iter=1000,
        multi_class='ovr',
        random_state=RANDOM_SEED,
        n_jobs=-1
    )
    clf.fit(X_train, y_train)

    y_pred = clf.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    train_time = time.time() - start_time

    print(f"准确率: {acc:.4f}")
    print(f"训练耗时: {train_time:.2f} 秒")
    return {"accuracy": acc, "time": train_time, "name": "TF-IDF+LR"}

# ================= 3. 方法二: TextCNN =================
class TextCNN(nn.Module):
    def __init__(self, vocab_size, embed_dim, num_classes, kernel_sizes=[2,3,4], num_filters=128, dropout=0.5):
        super(TextCNN, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.convs = nn.ModuleList([
            nn.Conv2d(1, num_filters, (k, embed_dim)) for k in kernel_sizes
        ])
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(len(kernel_sizes) * num_filters, num_classes)

    def forward(self, x):
        x = self.embedding(x)
        x = x.unsqueeze(1)
        conv_outs = []
        for conv in self.convs:
            conv_out = torch.relu(conv(x)).squeeze(3)
            pool_out = torch.max_pool1d(conv_out, conv_out.size(2)).squeeze(2)
            conv_outs.append(pool_out)
        x = torch.cat(conv_outs, dim=1)
        x = self.dropout(x)
        return self.fc(x)

# ================= 4. 方法三: BiLSTM =================
class BiLSTM(nn.Module):
    def __init__(self, vocab_size, embed_dim, hidden_dim, num_layers, num_classes, dropout=0.5, bidirectional=True):
        super(BiLSTM, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.lstm = nn.LSTM(
            input_size=embed_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0,
            bidirectional=bidirectional
        )
        self.dropout = nn.Dropout(dropout)
        # 双向LSTM输出维度为 hidden_dim * 2
        self.fc = nn.Linear(hidden_dim * (2 if bidirectional else 1), num_classes)

    def forward(self, x):
        # x: (batch, seq_len)
        embedded = self.embedding(x)          # (batch, seq_len, embed_dim)
        lstm_out, (hidden, cell) = self.lstm(embedded)   # hidden: (num_layers * 2, batch, hidden_dim)
        # 取最后一层的双向隐藏状态拼接
        if self.lstm.bidirectional:
            hidden_last = torch.cat((hidden[-2], hidden[-1]), dim=1)  # (batch, hidden_dim*2)
        else:
            hidden_last = hidden[-1]            # (batch, hidden_dim)
        out = self.dropout(hidden_last)
        return self.fc(out)

# ================= 公用词汇表和数据处理 (供TextCNN和BiLSTM使用) =================
def build_vocab(train_df, max_vocab=50000):
    from collections import Counter
    tokenizer = lambda x: x.split()
    counter = Counter()
    for text in train_df['text']:
        counter.update(tokenizer(text))
    vocab = {word: idx+2 for idx, (word, _) in enumerate(counter.most_common(max_vocab))}
    vocab['<PAD>'] = 0
    vocab['<UNK>'] = 1
    return vocab

def text_to_ids(text, vocab, max_len=256):
    tokens = text.split()
    ids = [vocab.get(token, vocab['<UNK>']) for token in tokens[:max_len]]
    return ids

class TextDataset(Dataset):
    def __init__(self, texts, labels, vocab, max_len=256):
        self.texts = texts
        self.labels = labels
        self.vocab = vocab
        self.max_len = max_len
    def __len__(self):
        return len(self.texts)
    def __getitem__(self, idx):
        ids = text_to_ids(self.texts[idx], self.vocab, self.max_len)
        input_ids = torch.tensor(ids, dtype=torch.long)
        label = torch.tensor(self.labels[idx], dtype=torch.long)
        return input_ids, label

def collate_fn(batch):
    inputs, labels = zip(*batch)
    inputs_padded = pad_sequence(inputs, batch_first=True, padding_value=0)
    labels = torch.stack(labels)
    return inputs_padded, labels

def train_deep_model(model, train_loader, test_loader, epochs, lr, device, model_name):
    """通用深度学习模型训练函数 (TextCNN, BiLSTM)"""
    model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()

    for epoch in range(epochs):
        model.train()
        total_loss = 0
        for batch in tqdm(train_loader, desc=f"{model_name} Epoch {epoch+1}/{epochs}", leave=False):
            inputs, labels = batch
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        print(f"{model_name} Epoch {epoch+1} - Loss: {total_loss/len(train_loader):.4f}")

    # 评估
    model.eval()
    all_preds, all_labels = [], []
    with torch.no_grad():
        for batch in test_loader:
            inputs, labels = batch
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            preds = torch.argmax(outputs, dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    return accuracy_score(all_labels, all_preds)

def train_textcnn(train_df, test_df):
    print("\n" + "="*50)
    print("开始训练 TextCNN")
    start_time = time.time()

    vocab = build_vocab(train_df)
    vocab_size = len(vocab)

    train_texts = train_df['text'].tolist()
    train_labels = train_df['label'].tolist()
    test_texts = test_df['text'].tolist()
    test_labels = test_df['label'].tolist()

    train_dataset = TextDataset(train_texts, train_labels, vocab)
    test_dataset = TextDataset(test_texts, test_labels, vocab)

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate_fn)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, collate_fn=collate_fn)

    model = TextCNN(vocab_size, embed_dim=100, num_classes=4, kernel_sizes=[2,3,4], num_filters=128)
    acc = train_deep_model(model, train_loader, test_loader, EPOCHS, lr=1e-3, device=DEVICE, model_name="TextCNN")
    train_time = time.time() - start_time

    print(f"准确率: {acc:.4f}")
    print(f"训练耗时: {train_time:.2f} 秒")
    return {"accuracy": acc, "time": train_time, "name": "TextCNN"}

def train_bilstm(train_df, test_df):
    print("\n" + "="*50)
    print("开始训练 BiLSTM")
    start_time = time.time()

    vocab = build_vocab(train_df)
    vocab_size = len(vocab)

    train_texts = train_df['text'].tolist()
    train_labels = train_df['label'].tolist()
    test_texts = test_df['text'].tolist()
    test_labels = test_df['label'].tolist()

    train_dataset = TextDataset(train_texts, train_labels, vocab)
    test_dataset = TextDataset(test_texts, test_labels, vocab)

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate_fn)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, collate_fn=collate_fn)

    model = BiLSTM(vocab_size, embed_dim=100, hidden_dim=128, num_layers=2, num_classes=4, dropout=0.5, bidirectional=True)
    acc = train_deep_model(model, train_loader, test_loader, EPOCHS, lr=1e-3, device=DEVICE, model_name="BiLSTM")
    train_time = time.time() - start_time

    print(f"准确率: {acc:.4f}")
    print(f"训练耗时: {train_time:.2f} 秒")
    return {"accuracy": acc, "time": train_time, "name": "BiLSTM"}

# ================= 5. 方法四: BERT =================
def train_bert(train_df, test_df):
    print("\n" + "="*50)
    print("开始训练 BERT")
    start_time = time.time()

    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    model = BertForSequenceClassification.from_pretrained(
        'bert-base-uncased',
        num_labels=4,
        output_attentions=False,
        output_hidden_states=False
    ).to(DEVICE)

    def encode_texts(texts, max_len=128):
        return tokenizer(
            texts,
            truncation=True,
            padding='max_length',
            max_length=max_len,
            return_tensors='pt'
        )

    train_encodings = encode_texts(train_df['text'].tolist())
    test_encodings = encode_texts(test_df['text'].tolist())
    train_labels = torch.tensor(train_df['label'].tolist())
    test_labels = torch.tensor(test_df['label'].tolist())

    train_dataset = torch.utils.data.TensorDataset(
        train_encodings['input_ids'],
        train_encodings['attention_mask'],
        train_labels
    )
    test_dataset = torch.utils.data.TensorDataset(
        test_encodings['input_ids'],
        test_encodings['attention_mask'],
        test_labels
    )

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE)

    optimizer = AdamW(model.parameters(), lr=2e-5, correct_bias=False)
    total_steps = len(train_loader) * EPOCHS
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=int(0.1 * total_steps),
        num_training_steps=total_steps
    )

    for epoch in range(EPOCHS):
        model.train()
        total_loss = 0
        for batch in tqdm(train_loader, desc=f"BERT Epoch {epoch+1}/{EPOCHS}", leave=False):
            input_ids, attention_mask, labels = [b.to(DEVICE) for b in batch]
            optimizer.zero_grad()
            outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
            loss = outputs.loss
            loss.backward()
            optimizer.step()
            scheduler.step()
            total_loss += loss.item()
        print(f"BERT Epoch {epoch+1} - Loss: {total_loss/len(train_loader):.4f}")

    model.eval()
    all_preds, all_labels = [], []
    with torch.no_grad():
        for batch in test_loader:
            input_ids, attention_mask, labels = [b.to(DEVICE) for b in batch]
            outputs = model(input_ids, attention_mask=attention_mask)
            preds = torch.argmax(outputs.logits, dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    acc = accuracy_score(all_labels, all_preds)
    train_time = time.time() - start_time
    print(f"准确率: {acc:.4f}")
    print(f"训练耗时: {train_time:.2f} 秒")
    return {"accuracy": acc, "time": train_time, "name": "BERT"}

# ================= 6. 主实验 =================
def run_experiment(sample_per_class, data_desc):
    print(f"\n{'#'*60}")
    print(f"## 数据规模: {data_desc} (每类 {sample_per_class} 条训练样本)")
    print(f"{'#'*60}")

    train_df, test_df = load_and_prepare_data(sample_per_class)

    results = []
    results.append(train_tfidf_lr(train_df, test_df))  # 基线
    results.append(train_textcnn(train_df, test_df))
    results.append(train_bilstm(train_df, test_df))
    results.append(train_bert(train_df, test_df))

    print("\n" + "="*50)
    print(f"【{data_desc}数据集实验结果】")
    print("="*50)
    for res in results:
        print(f"{res['name']:20} | 准确率: {res['accuracy']:.4f} | 耗时: {res['time']:.2f}秒")
    return results

if __name__ == "__main__":
    run_experiment(SAMPLE_PER_CLASS, f"Medium (每类{SAMPLE_PER_CLASS}条)")
