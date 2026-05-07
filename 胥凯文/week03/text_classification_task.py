"""
对一个任意包含“你”字的五个字的文本，“你”在第几位，就属于第几类。
"""

import random
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt

EMBED_DIM   = 64
HIDDEN_DIM  = 64
g_model_type = 'rnn'

CHARS = '的一是了我不在人有他这大来上国个到说们为子和你地出道理也时年得就那要下以生会自着去之过家学对可她里后小么心多天而能好都然没于来起还发成事只作当看文样点相真其经把等最但分明公力外现已关机主进使部头本反'
vocab ={"<PAD>": 0}
for i, char in enumerate(CHARS):
    vocab[char] = i + 1
vocab["<UNK>"] = len(vocab)

def encode(text):
    ids = [vocab.get(ch, vocab["<UNK>"]) for ch in text]
    return ids

def generate_5word_sentences():
    pos = random.randint(0, 4)
    char_list = []
    for i in range(5):
        if i == pos:
            char_list.append("你")
        else:
            char_list.append(random.choice(CHARS))
    return char_list, pos

def build_dataset(total_sample_num):
    X = []
    Y = []
    for i in range(total_sample_num):
        x, y = generate_5word_sentences()
        X.append(encode(x))
        Y.append(y)
    return torch.LongTensor(X), torch.LongTensor(Y) 

class text_classification_task(nn.Module):
    def __init__(self, vocab_size, embed_dim=64, hidden_dim=64, dropout=0.3):
        global g_model_type
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        if g_model_type == 'rnn':
            self.rnn = nn.RNN(embed_dim, hidden_dim, batch_first=True)
        else:
            self.lstm = nn.LSTM(embed_dim, hidden_dim, batch_first=True)
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_dim, 5)
        self.loss = nn.functional.cross_entropy

    def forward(self, x, y=None):
        if g_model_type == 'rnn':
            e, _ = self.rnn(self.embedding(x))
        else:
            e, _ = self.lstm(self.embedding(x))
        last_hidden = e[:, -1, :]
        pooled = self.dropout(last_hidden)
        y_pred = self.fc(pooled)
        if y is not None:
            return self.loss(y_pred, y)
        return y_pred

def evaluate(model): 
    model.eval()
    test_sample_num = 100 
    x, y = build_dataset(test_sample_num) 
    class_counts = torch.bincount(y, minlength=5)
    print("本次预测集中各类别样本数：", class_counts.tolist())
    correct, wrong = 0, 0 
    with torch.no_grad(): 
        y_pred = model(x)
        for y_p, y_t in zip(y_pred, y):
                    if torch.argmax(y_p) == int(y_t):
                        correct += 1
                    else:
                        wrong += 1

    print("正确预测个数：%d, 正确率：%f" % (correct, correct / (correct + wrong))) 
    return correct / (correct + wrong)

def train():
    epoch_num = 30
    batch_size = 20
    train_sample = 1000
    embed_dim = EMBED_DIM
    hidden_dim = HIDDEN_DIM
    learning_rate = 0.001
    log=[]
    train_x, train_y = build_dataset(train_sample) 
    global g_model_type
    g_model_type = 'lstm'
    model     = text_classification_task(len(vocab), embed_dim, hidden_dim)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    for epoch in range(epoch_num):
        model.train()
        watch_loss = []
        for batch_index in range(train_sample // batch_size):
            x = train_x[batch_index * batch_size : (batch_index + 1) * batch_size]
            y = train_y[batch_index * batch_size : (batch_index + 1) * batch_size]
            loss = model(x, y)  # 计算loss
            loss.backward()  # 计算梯度
            optimizer.step()  # 更新权重
            optimizer.zero_grad()  # 梯度归零
            watch_loss.append(loss.item())
        print("=========\n第%d轮平均loss:%f" % (epoch + 1, np.mean(watch_loss)))
        acc = evaluate(model)
        log.append([acc, float(np.mean(watch_loss))])

    torch.save(model.state_dict(), "model.bin")
    print(log)
    plt.plot(range(len(log)), [l[0] for l in log], label="acc")
    plt.plot(range(len(log)), [l[1] for l in log], label="loss")
    plt.legend()
    plt.savefig("training_result.png")
    return

def predict(model_path, input_vec):
    model = text_classification_task(len(vocab), EMBED_DIM, HIDDEN_DIM)
    model.load_state_dict(torch.load(model_path))
    x_predict = []
    for sentence in input_vec:
        x_predict.append(encode(sentence))

    with torch.no_grad():
        result = model(torch.LongTensor(x_predict))
        print("-" * 60)
        for vec, res in zip(input_vec, result):
            pred_class = torch.argmax(res).item()
            print(f"输入：{vec}, 预测结果: {pred_class}")
        print("-" * 60)

if __name__ == '__main__':
    train()
    test_vec = ['你真的很棒',
                '你吃饭了吗',
                '给你一朵花',
                '请你帮帮我',
                '拥抱你一下',
                '拉着你手走',
                '我想见你了',
                '他喜欢你去',
                '我见到了你',
                '明天来找你']
    predict("model.bin", test_vec)
