import json
from pathlib import Path
import argparse
import random
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
from transformers import AutoTokenizer, AutoModelForCausalLM

import time
from tqdm import tqdm

ROOT = Path(__file__).parent.parent
DATA_DIR = ROOT / 'data' / "peoples_daily"
MODEL_PATH = Path(r"D:\AAA八斗\2.202604期直播\week6文本分类问题\week6 文本分类问题\week6 文本分类问题\pretrain_models\Qwen2.5-0.5B-Instruct")
OUTPUT_DIR = ROOT / "output"

try:
    from peft import get_peft_model, LoraConfig, TaskType
    PEFT_AVAILABLE = True
except ImportError:
    PEFT_AVAILABLE = False

ENTITY_TYPES = [
    "PER", "ORG", "LOC",
]

SYSTEM_PROMPT = (
    "你是一个命名实体识别助手。从文本中识别命名实体，以 JSON 格式输出。\n"
    "实体类型（英文标识）：LOC（地点）、PER（人名）、ORG（公司）\n"
    '输出格式（严格遵守，不输出其他内容）：{"entities": [{"text": "实体文本", "type": "实体类型"}]}\n'
    '无实体时输出：{"entities": []}'
)


def record_to_target(record: dict) -> str:
    """把 peoples_daily BIO 格式转为 SFT 目标 JSON 字符串。
        输入：{"tokens": ["海", "钓", ...], "ner_tags": ["O", "O", "B-LOC", "I-LOC", ...]}
        输出：'{"entities": [{"text": "厦门", "type": "LOC"}, ...]}'
    """
    entities = []
    tokens = record.get("tokens", [])
    ner_tags = record.get("ner_tags", [])
    current_entity = []
    current_type = None
    for token, tag in zip(tokens, ner_tags):
        if tag.startswith("B-"):
            # 如果当前正在构建实体，先将已构建的实体存入列表
            if current_entity:
                # 将当前实体组合成完整字符串并添加到实体列表
                # 数据样式：entities.append({"text": "厦门", "type": "LOC"})
                entities.append({"text": "".join(current_entity), "type": current_type})
                # 重置当前实体字符列表
                current_entity = []
            
            # 提取实体类型（去掉 "B-" 前缀）
            # 数据样式：tag="B-LOC" → current_type="LOC"
            current_type = tag[2:]
            
            # 将当前 token 添加到正在构建的实体中
            # 数据样式：current_entity = ["厦"]
            current_entity.append(token)
        
        elif tag.startswith("I-"):
            # 只有当前正在构建实体且类型匹配时，才继续添加字符
            # 数据样式：current_type="LOC", tag[2:]="LOC" → 匹配成功
            if current_entity and current_type == tag[2:]:
                # 继续追加字符到当前实体
                # 数据样式：current_entity = ["厦", "门"]
                current_entity.append(token)
        
        # 当前标签为非实体标签（O）或其他情况
        else:
            # 如果当前正在构建实体，先将其存入列表
            if current_entity:
                # 数据样式：entities.append({"text": "厦门", "type": "LOC"})
                entities.append({"text": "".join(current_entity), "type": current_type})
                # 重置当前实体字符列表
                current_entity = []
                # 重置当前实体类型
                current_type = None
    
    # 循环结束后检查是否还有未完成的实体，
    # 通常出现在最后结尾处。后面无法跟O，也就是没有走到上面else里面的逻辑，所以这里判断一下再添加到entities
    if current_entity:
        # 将剩余的实体添加到列表
        # 数据样式：entities.append({"text": "金门", "type": "LOC"})
        entities.append({"text": "".join(current_entity), "type": current_type})
    
    # 将实体列表转为 JSON 字符串输出，ensure_ascii=False 保留中文
    # 返回数据样式：'{"entities": [{"text": "厦门", "type": "LOC"}, {"text": "金门", "type": "LOC"}]}'
    return json.dumps({"entities": entities}, ensure_ascii=False)


# ══════════════════════════════════════════════════════════════════════════════
# Dataset
# ══════════════════════════════════════════════════════════════════════════════

class SFTDataset(Dataset):
    """
        把 cluener NER 数据转换为 chat-format SFT 训练样本。

        与分类任务的关键区别：
          - 分类：TARGET = "科技"（1~2 个 token，极短）
          - NER：TARGET = '{"entities": [...]}' （20~150 个 token，结构化 JSON）

        Loss mask 结构：
          ┌──────────────────────────────────────────────────────────────┐
          │ <system>...<user>{text}<assistant>\n                         │  → -100
          │ {"entities": [{"text": "浙商银行", "type": "ORG"}]} <EOS>│  → 真实 id
          └──────────────────────────────────────────────────────────────┘
        """
    def __init__(self, data, tokenizer, max_length=256):
        self.data = data
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        target = record_to_target(item)

        # ── Step 1：构建 prompt 文本（tokenize=False 兼容 transformers 5.x）──
        prompt_text = self.tokenizer.apply_chat_template(
            [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": "".join(item["tokens"])},
            ],
            tokenize=False,
            add_generation_prompt=True,
        )
        prompt_ids = self.tokenizer.encode(prompt_text, add_special_tokens=False)

        # ── Step 2：response = JSON 字符串 + EOS ──────────────────────────────
        response_ids = (
                self.tokenizer.encode(target, add_special_tokens=False)
                + [self.tokenizer.eos_token_id]
        )

        # ── Step 3：拼接 + 截断 ───────────────────────────────────────────────
        input_ids = (prompt_ids + response_ids)[: self.max_length]

        # ── Step 4：loss mask：prompt 全 -100，只在 JSON 部分计算 loss ──────
        labels = ([-100] * len(prompt_ids) + response_ids)[: self.max_length]

        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "labels": torch.tensor(labels, dtype=torch.long),
        }


def collate_fn(batch, pad_id):
    """
    自定义批处理函数：将一批样本补齐到相同长度并堆叠成 tensor。

    Args:
        batch: list of dict，每个 dict 包含单样本的 input_ids、labels
        pad_id: padding token 的 id，用于补齐短序列

    Returns:
        dict: 包含 input_ids、labels、attention_mask 的 batch tensor
    """
    # 1. 找到当前 batch 中最长序列的长度
    max_len = max(item["input_ids"].size(0) for item in batch)

    # 2. 初始化三个列表，分别存储补齐后的 input_ids、labels、attention_mask
    input_ids_list, labels_list, mask_list = [], [], []

    # 3. 遍历每个样本，进行补齐处理
    for item in batch:
        n = item["input_ids"].size(0)  # 当前样本的长度
        pad = max_len - n  # 需要补齐的长度

        # 3.1 input_ids：用 pad_id 补齐到 max_len
        input_ids_list.append(torch.cat([item["input_ids"],
                                         torch.full((pad,), pad_id, dtype=torch.long)]))

        # 3.2 labels：补齐部分设为 -100（loss 计算时会被忽略）
        labels_list.append(torch.cat([item["labels"],
                                      torch.full((pad,), -100, dtype=torch.long)]))

        # 3.3 attention_mask：有效部分为 1，padding 部分为 0
        mask_list.append(torch.cat([torch.ones(n, dtype=torch.long),
                                    torch.zeros(pad, dtype=torch.long)]))

    # 4. 将列表堆叠成 batch tensor（形状: [batch_size, max_len]）
    return {
        "input_ids": torch.stack(input_ids_list),
        "labels": torch.stack(labels_list),
        "attention_mask": torch.stack(mask_list),
    }



# ══════════════════════════════════════════════════════════════════════════════
# Training
# ══════════════════════════════════════════════════════════════════════════════

def parse_args():
    parser = argparse.ArgumentParser(description="LLM SFT NER 训练（LoRA / 全量微调）")
    parser.add_argument("--model_path",  default=str(MODEL_PATH))
    parser.add_argument("--data_dir",    default=str(DATA_DIR))
    parser.add_argument("--output_dir",  default=str(OUTPUT_DIR))
    parser.add_argument("--num_train",   default=-1,   type=int,
                        help="训练样本数，-1 使用全部 10748 条（默认）")
    parser.add_argument("--epochs",      default=3,    type=int)
    parser.add_argument("--batch_size",  default=2,    type=int)
    parser.add_argument("--grad_accum",  default=4,    type=int)
    parser.add_argument("--lr",          default=None, type=float,
                        help="学习率；默认 LoRA=2e-4，全量=2e-5（自动判断）")
    parser.add_argument("--max_length",  default=256,  type=int,
                        help="序列最大长度；NER 的 JSON 输出比分类长，建议 256")
    # 全量微调开关
    parser.add_argument("--full_ft",     action="store_true",
                        help="全量微调：跳过 LoRA，更新所有 495M 参数（需显存 ≥ 16GB）")
    # LoRA 超参（full_ft 时忽略）
    parser.add_argument("--lora_r",      default=8,    type=int)
    parser.add_argument("--lora_alpha",  default=16,   type=int)
    parser.add_argument("--seed",        default=42,   type=int)
    return parser.parse_args()

def main():
    args = parse_args()
    random.seed(args.seed)
    torch.manual_seed(args.seed)

    if args.lr is None:
        args.lr = 2e-5 if args.full_ft else 2e-4

    data_dir = Path(args.data_dir)
    output_dir = Path(args.output_dir)
    ckpt_dir = output_dir / ("sft_full_ckpt_homework" if args.full_ft else "sft_adapter_homework")
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    mode_str = "全量微调" if args.full_ft else "LoRA 微调"
    print(f"使用设备: {device}  |  微调模式: {mode_str}")

    # ── 加载数据 ──────────────────────────────────────────────────────────────
    with open(data_dir /"train.json", encoding="utf-8") as f:
        train_raw = json.load(f)

    with open(data_dir /"validation.json", encoding="utf-8") as f:
        val_raw = json.load(f)

    if args.num_train > 0:
        train_raw = random.sample(train_raw, min(args.num_train, len(train_raw)))
    print(f"训练集: {len(train_raw)} 条 | 验证集（前300条）: 300 条")

    # ── 加载 Tokenizer ─────────────────────────────────────────────────────────
    print(f"\n加载 tokenizer: {args.model_path}")
    tokenizer = AutoTokenizer.from_pretrained(
        str(Path(args.model_path).resolve()), trust_remote_code=True
    )
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    # ── 构建数据集 ─────────────────────────────────────────────────────────────
    train_dataset = SFTDataset(train_raw, tokenizer, args.max_length)
    val_dataset   = SFTDataset(val_raw[:300], tokenizer, args.max_length)

    # 创建 collate_fn 的偏函数：固定 pad_token_id 参数
    # _collate(batch) 等价于 collate_fn(batch, tokenizer.pad_token_id)
    _collate = lambda b: collate_fn(b, tokenizer.pad_token_id)

    # 创建训练数据加载器
    # batch_size: 每批样本数, shuffle: 训练时打乱顺序, collate_fn: 自定义批处理
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size,
                              shuffle=True, collate_fn=_collate)
    # 创建验证数据加载器（batch_size 翻倍，因为验证不需要反向传播，显存占用小）
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size * 2,
                            shuffle=False, collate_fn=_collate)

    # ── 加载模型 ───────────────────────────────────────────────────────────────
    print(f"加载 base model: {args.model_path}")
    model = AutoModelForCausalLM.from_pretrained(
        str(Path(args.model_path).resolve()),
        torch_dtype=torch.float32,  # transformers 5.x 用 dtype= 不用 torch_dtype=
        # dtype=torch.float32,  # transformers 5.x 用 dtype= 不用 torch_dtype=
        trust_remote_code=True,
    )

    # ── LoRA 或全量微调 ────────────────────────────────────────────────────────
    if args.full_ft:
        total = sum(p.numel() for p in model.parameters())
        print(f"trainable params: {total:,} || all params: {total:,} || trainable%: 100.0000")
    else:
        if not PEFT_AVAILABLE:
            raise ImportError("LoRA 模式需要 peft 库：pip install peft>=0.14.0")
        lora_config = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            r=args.lora_r,
            lora_alpha=args.lora_alpha,
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
            lora_dropout=0.05,
            bias="none",
        )
        # 加载换成lora的模型
        model = get_peft_model(model, lora_config)
        model.print_trainable_parameters()

    model = model.to(device)

    # ── 优化器 ────────────────────────────────────────────────────────────────
    optimizer = AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    total_steps = len(train_loader) * args.epochs // args.grad_accum
    print(f"总训练步数: {total_steps}（batch={args.batch_size}, "
          f"grad_accum={args.grad_accum}, epochs={args.epochs}, lr={args.lr}）\n")

    # ── 训练循环 ──────────────────────────────────────────────────────────────
    best_val_loss = float("inf")
    log_records = []

    for epoch in range(1, args.epochs + 1):
        model.train()
        total_loss, total_tokens = 0.0, 0
        optimizer.zero_grad()
        t0 = time.time()

        pbar = tqdm(train_loader, desc=f"Epoch {epoch}/{args.epochs} [Train]", leave=False)
        for step, batch in enumerate(pbar):
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)

            outputs = model(input_ids=input_ids,
                            attention_mask=attention_mask,
                            labels=labels)
            loss = outputs.loss

            (loss / args.grad_accum).backward()
            if (step + 1) % args.grad_accum == 0:
                nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
                optimizer.zero_grad()

            n_tokens = (labels != -100).sum().item()
            total_loss += loss.item() * n_tokens
            total_tokens += n_tokens
            pbar.set_postfix(loss=f"{loss.item():.4f}")

        avg_train_loss = total_loss / max(total_tokens, 1)

        # ── 验证 loss ─────────────────────────────────────────────────────────
        model.eval()
        val_loss, val_tokens = 0.0, 0
        with torch.no_grad():
            for batch in tqdm(val_loader, desc="Val", leave=False):
                input_ids = batch["input_ids"].to(device)
                attention_mask = batch["attention_mask"].to(device)
                labels = batch["labels"].to(device)
                outputs = model(input_ids=input_ids,
                                attention_mask=attention_mask,
                                labels=labels)
                n_tokens = (labels != -100).sum().item()
                val_loss += outputs.loss.item() * n_tokens
                val_tokens += n_tokens
        avg_val_loss = val_loss / max(val_tokens, 1)

        elapsed = time.time() - t0
        print(f"Epoch {epoch}/{args.epochs} | "
              f"train_loss={avg_train_loss:.4f}  val_loss={avg_val_loss:.4f} | {elapsed:.0f}s")

        log_records.append({
            "epoch": epoch, "train_loss": avg_train_loss,
            "val_loss": avg_val_loss, "elapsed_s": elapsed,
        })

        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            model.save_pretrained(ckpt_dir)
            tokenizer.save_pretrained(ckpt_dir)
            ckpt_label = "完整模型" if args.full_ft else "LoRA adapter"
            print(f"  ✓ 最优{ckpt_label}已保存 → {ckpt_dir}  (val_loss={avg_val_loss:.4f})")

    # ── 保存训练日志 ──────────────────────────────────────────────────────────
    log_tag = "full_ft" if args.full_ft else "sft"
    log_path = output_dir / "logs" / f"homework_train_{log_tag}.json"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with open(log_path, "w", encoding="utf-8") as f:
        json.dump(log_records, f, ensure_ascii=False, indent=2)

    ckpt_label = "完整模型" if args.full_ft else "LoRA adapter"
    print(f"\n训练完成。最优 val_loss={best_val_loss:.4f}")
    print(f"训练日志 → {log_path}")
    print(f"{ckpt_label} → {ckpt_dir}")

if __name__ == "__main__":
        main()




