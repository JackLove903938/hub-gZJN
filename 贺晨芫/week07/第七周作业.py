import json
import os
import random
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizer, BertForTokenClassification, get_linear_schedule_with_warmup
from sklearn.metrics import classification_report
import numpy as np
from tqdm import tqdm

# ===================== 1. 配置参数 =====================
class Config:
    def __init__(self):
        # 分别指定训练集、验证集、测试集路径
        self.train_path = r"D:\百度网盘\学习资料\week7序列标注问题\week7 序列标注问题\序列标注项目\data\peoples_daily\train.json"
        self.val_path = r"D:\百度网盘\学习资料\week7序列标注问题\week7 序列标注问题\序列标注项目\data\peoples_daily\validation.json"
        self.test_path = r"D:\百度网盘\学习资料\week7序列标注问题\week7 序列标注问题\序列标注项目\data\peoples_daily\test.json"
        self.model_name = r"E:\VScode项目\八斗-AI算法\专业代码\课程代码\week6\text_classification项目\pretrain_models\bert-base-chinese"
        self.batch_size = 8
        self.epochs = 10
        self.lr = 2e-5
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.max_seq_len = 128
        self.seed = 42
        self.gradient_clip = 1.0
        self.warmup_ratio = 0.1
        self.save_dir = "./checkpoints"
        # NER标签映射（BIO格式）
        self.label2id = {
            "O": 0,
            "B-LOC": 1, "I-LOC": 2,
            "B-PER": 3, "I-PER": 4,
            "B-ORG": 5, "I-ORG": 6
        }
        self.id2label = {v: k for k, v in self.label2id.items()}
        self.num_labels = len(self.label2id)

config = Config()


def set_seed(seed):
    """设置随机种子，保证可复现"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True


# ===================== 2. 数据加载与预处理 =====================
class NERDataset(Dataset):
    def __init__(self, data_path, tokenizer, config):
        self.tokenizer = tokenizer
        self.config = config
        self.data = self.load_data(data_path)

    def load_data(self, path):
        """加载JSON格式的NER数据"""
        with open(path, "r", encoding="utf-8") as f:
            raw_data = json.load(f)

        processed_data = []
        for item in raw_data:
            tokens = item["tokens"]
            ner_tags = item["ner_tags"]

            input_ids = []
            attention_mask = []
            label_ids = []

            # [CLS]
            input_ids.append(self.tokenizer.cls_token_id)
            attention_mask.append(1)
            label_ids.append(-100)  # CLS位置标签设为-100，不计入loss

            # 正文token
            for token, tag in zip(tokens, ner_tags):
                # 使用tokenize处理单个token，更健壮地处理sub-token拆分
                sub_tokens = self.tokenizer.tokenize(token)
                if len(sub_tokens) == 0:
                    continue  # 跳过无法分词的token

                sub_token_ids = self.tokenizer.convert_tokens_to_ids(sub_tokens)

                # 第一个sub-token使用真实标签，后续sub-token设为-100（不计入loss）
                input_ids.extend(sub_token_ids)
                attention_mask.extend([1] * len(sub_token_ids))
                label_ids.append(self.config.label2id.get(tag, 0))
                for _ in range(len(sub_token_ids) - 1):
                    label_ids.append(-100)

                # 截断：预留[SEP]的位置
                if len(input_ids) >= self.config.max_seq_len - 1:
                    input_ids = input_ids[:self.config.max_seq_len - 1]
                    attention_mask = attention_mask[:self.config.max_seq_len - 1]
                    label_ids = label_ids[:self.config.max_seq_len - 1]
                    break

            # [SEP]
            input_ids.append(self.tokenizer.sep_token_id)
            attention_mask.append(1)
            label_ids.append(-100)  # SEP位置标签设为-100，不计入loss

            # 填充到最大长度
            padding_len = self.config.max_seq_len - len(input_ids)
            if padding_len > 0:
                input_ids += [self.tokenizer.pad_token_id] * padding_len
                attention_mask += [0] * padding_len
                label_ids += [-100] * padding_len

            # 安全检查：确保长度一致
            assert len(input_ids) == self.config.max_seq_len
            assert len(attention_mask) == self.config.max_seq_len
            assert len(label_ids) == self.config.max_seq_len

            input_ids = torch.tensor(input_ids, dtype=torch.long)
            attention_mask = torch.tensor(attention_mask, dtype=torch.long)
            label_ids = torch.tensor(label_ids, dtype=torch.long)

            processed_data.append({
                "input_ids": input_ids,
                "attention_mask": attention_mask,
                "labels": label_ids
            })
        return processed_data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]


# ===================== 3. 模型定义 =====================
class NERModel(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.bert = BertForTokenClassification.from_pretrained(
            config.model_name,
            num_labels=config.num_labels
        )

    def forward(self, input_ids, attention_mask, labels=None):
        outputs = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels
        )
        return outputs


# ===================== 4. 训练与验证函数 =====================
def train_epoch(model, train_loader, optimizer, scheduler, config):
    model.train()
    total_loss = 0.0
    total_steps = 0

    for batch in tqdm(train_loader, desc="Training"):
        input_ids = batch["input_ids"].to(config.device)
        attention_mask = batch["attention_mask"].to(config.device)
        labels = batch["labels"].to(config.device)

        optimizer.zero_grad()
        outputs = model(input_ids, attention_mask, labels)
        loss = outputs.loss
        loss.backward()

        # 梯度裁剪，防止梯度爆炸
        torch.nn.utils.clip_grad_norm_(model.parameters(), config.gradient_clip)

        optimizer.step()
        scheduler.step()

        total_loss += loss.item()
        total_steps += 1

    avg_loss = total_loss / total_steps
    return avg_loss


@torch.no_grad()
def evaluate(model, eval_loader, config, desc="Evaluating"):
    """评估函数：同时返回评估loss、token-level报告和entity-level报告"""
    model.eval()
    total_loss = 0.0
    total_steps = 0
    all_preds = []
    all_labels = []

    for batch in tqdm(eval_loader, desc=desc):
        input_ids = batch["input_ids"].to(config.device)
        attention_mask = batch["attention_mask"].to(config.device)
        labels = batch["labels"].to(config.device)

        # 计算loss
        outputs = model(input_ids, attention_mask, labels)
        total_loss += outputs.loss.item()
        total_steps += 1

        # 预测
        logits = outputs.logits
        preds = torch.argmax(logits, dim=-1)

        # 只保留有效token（label != -100）的预测和标签
        for i in range(len(preds)):
            label_cpu = labels[i].cpu().numpy()
            pred_cpu = preds[i].cpu().numpy()
            mask = label_cpu != -100
            all_preds.extend(pred_cpu[mask].tolist())
            all_labels.extend(label_cpu[mask].tolist())

    avg_loss = total_loss / total_steps

    # Token-level分类报告
    report = classification_report(
        all_labels,
        all_preds,
        target_names=[config.id2label[i] for i in range(config.num_labels)],
        zero_division=0
    )

    # Entity-level评估
    entity_report = entity_level_evaluate(all_preds, all_labels, config)

    return avg_loss, report, entity_report


def entity_level_evaluate(preds, labels, config):
    """
    Entity-level评估：从BIO标签序列中提取实体，计算实体级别的P/R/F1
    """
    id2label = config.id2label
    pred_tags = [id2label[p] for p in preds]
    gold_tags = [id2label[l] for l in labels]

    def extract_entities(tags):
        """从BIO标签序列中提取实体"""
        entities = []
        i = 0
        while i < len(tags):
            if tags[i].startswith("B-"):
                entity_type = tags[i][2:]
                start = i
                i += 1
                while i < len(tags) and tags[i] == f"I-{entity_type}":
                    i += 1
                entities.append((entity_type, start, i))
            else:
                i += 1
        return set(entities)

    pred_entities = extract_entities(pred_tags)
    gold_entities = extract_entities(gold_tags)

    correct = len(pred_entities & gold_entities)
    precision = correct / len(pred_entities) if len(pred_entities) > 0 else 0.0
    recall = correct / len(gold_entities) if len(gold_entities) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

    report = (
        f"Entity-Level Evaluation:\n"
        f"  Gold entities: {len(gold_entities)}, Pred entities: {len(pred_entities)}, Correct: {correct}\n"
        f"  Precision: {precision:.4f}, Recall: {recall:.4f}, F1: {f1:.4f}"
    )
    return report


# ===================== 5. 推理函数 =====================
def predict(model, tokenizer, text, config):
    """对单条文本进行NER预测，返回实体列表"""
    model.eval()

    tokens = list(text)
    encoding = tokenizer(
        tokens,
        is_split_into_words=True,
        max_length=config.max_seq_len,
        padding="max_length",
        truncation=True,
        return_tensors="pt"
    )

    input_ids = encoding["input_ids"].to(config.device)
    attention_mask = encoding["attention_mask"].to(config.device)

    with torch.no_grad():
        outputs = model(input_ids, attention_mask)
        logits = outputs.logits
        preds = torch.argmax(logits, dim=-1)[0].cpu().numpy()

    # 获取word_ids映射，只取每个词的第一个sub-token
    word_ids = encoding.word_ids(0)
    results = []
    previous_word_id = None
    for idx, word_id in enumerate(word_ids):
        if word_id is None or word_id == previous_word_id:
            continue
        previous_word_id = word_id
        pred_label = config.id2label[preds[idx]]
        if pred_label != "O":
            results.append({
                "token": tokens[word_id],
                "position": word_id,
                "label": pred_label
            })

    # 合并连续的B/I标签为完整实体
    entities = []
    current_entity = None
    for r in results:
        if r["label"].startswith("B-"):
            if current_entity:
                entities.append(current_entity)
            current_entity = {
                "type": r["label"][2:],
                "start": r["position"],
                "end": r["position"],
                "text": r["token"]
            }
        elif r["label"].startswith("I-") and current_entity and r["label"][2:] == current_entity["type"]:
            current_entity["end"] = r["position"]
            current_entity["text"] += r["token"]
        else:
            if current_entity:
                entities.append(current_entity)
            current_entity = None
    if current_entity:
        entities.append(current_entity)

    return entities


# ===================== 6. 主函数 =====================
def main():
    # 0. 设置随机种子
    set_seed(config.seed)

    # 创建保存目录
    os.makedirs(config.save_dir, exist_ok=True)

    # 1. 加载tokenizer
    tokenizer = BertTokenizer.from_pretrained(config.model_name)

    # 2. 分别加载训练集、验证集、测试集
    train_dataset = NERDataset(config.train_path, tokenizer, config)
    val_dataset = NERDataset(config.val_path, tokenizer, config)
    test_dataset = NERDataset(config.test_path, tokenizer, config)

    print(f"训练集大小: {len(train_dataset)}")
    print(f"验证集大小: {len(val_dataset)}")
    print(f"测试集大小: {len(test_dataset)}")

    train_loader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=config.batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=config.batch_size, shuffle=False)

    # 3. 初始化模型和优化器
    model = NERModel(config)
    model.to(config.device)

    optimizer = optim.AdamW(model.parameters(), lr=config.lr, weight_decay=0.01)

    # 学习率调度器：线性warmup + 线性衰减
    total_steps = len(train_loader) * config.epochs
    warmup_steps = int(total_steps * config.warmup_ratio)
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=total_steps
    )

    # 4. 训练循环
    best_val_f1 = 0.0

    for epoch in range(config.epochs):
        print(f"\n{'='*60}")
        print(f"Epoch {epoch+1}/{config.epochs}")
        print(f"{'='*60}")

        # 训练（使用训练集）
        train_loss = train_epoch(model, train_loader, optimizer, scheduler, config)
        print(f"Train Loss: {train_loss:.4f}")

        # 验证（使用验证集）
        val_loss, token_report, entity_report = evaluate(model, val_loader, config, desc="Validating")
        print(f"Val Loss:   {val_loss:.4f}")
        print(f"\n{token_report}")
        print(entity_report)

        # 从entity_report中提取F1
        val_f1 = 0.0
        for line in entity_report.split("\n"):
            if "F1:" in line:
                val_f1 = float(line.split("F1:")[1].strip())
                break

        # 基于验证集entity F1保存最优模型
        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            save_path = os.path.join(config.save_dir, "best_ner_model.pt")
            torch.save(model.state_dict(), save_path)
            print(f"✅ Saved best model! (Val Loss: {val_loss:.4f}, Entity F1: {val_f1:.4f})")

    # 5. 加载最优模型，在测试集上进行最终评估
    print(f"\n{'='*60}")
    print("Loading best model for final evaluation on TEST set...")
    print(f"{'='*60}")
    save_path = os.path.join(config.save_dir, "best_ner_model.pt")
    model.load_state_dict(torch.load(save_path, weights_only=True))
    test_loss, test_token_report, test_entity_report = evaluate(model, test_loader, config, desc="Testing")
    print(f"Test Loss: {test_loss:.4f}")
    print(f"\n{test_token_report}")
    print(test_entity_report)

    # 6. 推理演示
    print(f"\n{'='*60}")
    print("Inference Demo:")
    print(f"{'='*60}")
    test_texts = [
        "北京市长王安顺出席会议",
        "中国银行在上海开设了新分行",
        "李克强总理访问北京大学"
    ]
    for text in test_texts:
        entities = predict(model, tokenizer, text, config)
        print(f"\n文本: {text}")
        if entities:
            for ent in entities:
                print(f"  实体: {ent['text']} | 类型: {ent['type']} | 位置: [{ent['start']}, {ent['end']}]")
        else:
            print("  未识别到实体")


if __name__ == "__main__":
    main()
