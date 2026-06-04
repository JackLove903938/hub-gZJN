import torch
from torch.optim import Adam
from torch.utils.data import DataLoader
from transformers import BertTokenizer
from seqeval.metrics import f1_score, precision_score, recall_score

from dataset_div import PeoplesDailyDataset
from model_div import ModelDiv

LABEL_SIZE = 7
LR         = 1e-3
epochs     = 100

if __name__ == '__main__':
    tokenizer = BertTokenizer.from_pretrained(r'/Users/cxl/hub-gZJN/陈小林/work06/bert-base-chinese')
    dataset_train = PeoplesDailyDataset('train.json', 128, tokenizer=tokenizer)
    dataset_valid = PeoplesDailyDataset('validation.json', 128, tokenizer=tokenizer)
    model = ModelDiv(label_size=LABEL_SIZE)
    optimizer = Adam(model.parameters(), lr=LR)

    train_loader = DataLoader(dataset_train, shuffle=True, batch_size=32)
    valid_loader = DataLoader(dataset_valid, shuffle=False, batch_size=32)
    id2label = dataset_train.id2label

    for epoch in range(epochs):
        model.train()
        losses = []
        for batch in train_loader:
            optimizer.zero_grad()
            input_ids, attention_mask, token_type_ids, labels = batch
            outputs, loss = model(input_ids, attention_mask, token_type_ids, labels)
            loss.backward()
            optimizer.step()
            losses.append(loss.item())
        print(f'Epoch {epoch}: Loss {torch.mean(torch.tensor(losses)):.4f}')

        if (epoch + 1) % 5 == 0:
            model.eval()
            all_preds, all_golds = [], []
            val_losses = []
            with torch.no_grad():
                for batch in valid_loader:
                    input_ids, attention_mask, token_type_ids, labels = batch
                    outputs, loss = model(input_ids, attention_mask, token_type_ids, labels)
                    val_losses.append(loss.item())

                    pred_ids = model.crf.decode(outputs, attention_mask.bool())

                    for i in range(len(input_ids)):
                        gold_seq, pred_seq = [], []
                        for j, gold_id in enumerate(labels[i].tolist()):
                            if gold_id == -100:
                                continue
                            gold_seq.append(id2label[gold_id])
                            pred_seq.append(id2label.get(pred_ids[i][j], "O") if j < len(pred_ids[i]) else "O")
                        all_golds.append(gold_seq)
                        all_preds.append(pred_seq)

            avg_val_loss = torch.mean(torch.tensor(val_losses))
            f1 = f1_score(all_golds, all_preds)
            p = precision_score(all_golds, all_preds)
            r = recall_score(all_golds, all_preds)
            print(f'Validation — Loss: {avg_val_loss:.4f}, P: {p:.4f}, R: {r:.4f}, F1: {f1:.4f}')

    # 测试集评估
    dataset_test = PeoplesDailyDataset('test.json', 128, tokenizer=tokenizer)
    test_loader = DataLoader(dataset_test, shuffle=False, batch_size=32)
    model.eval()
    all_preds, all_golds = [], []
    with torch.no_grad():
        for batch in test_loader:
            input_ids, attention_mask, token_type_ids, labels = batch
            outputs, _ = model(input_ids, attention_mask, token_type_ids, labels)
            pred_ids = model.crf.decode(outputs, attention_mask.bool())
            for i in range(len(input_ids)):
                gold_seq, pred_seq = [], []
                for j, gold_id in enumerate(labels[i].tolist()):
                    if gold_id == -100:
                        continue
                    gold_seq.append(id2label[gold_id])
                    pred_seq.append(id2label.get(pred_ids[i][j], "O") if j < len(pred_ids[i]) else "O")
                all_golds.append(gold_seq)
                all_preds.append(pred_seq)
    f1 = f1_score(all_golds, all_preds)
    p = precision_score(all_golds, all_preds)
    r = recall_score(all_golds, all_preds)
    print(f'\nTest — P: {p:.4f}, R: {r:.4f}, F1: {f1:.4f}')
