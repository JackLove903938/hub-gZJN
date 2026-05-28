"""
llm_sft_pipeline.py —— LLM SFT + LoRA 高效微调

教学重点：
  1. 指令微调格式：system/user/assistant chat 格式
  2. Loss Masking：prompt 部分 label=-100，只在 assistant 回复上算 loss
  3. LoRA 原理：冻结原权重 W，只训低秩旁路 ΔW = B·A
  4. 与全量微调的对比：0.22% 参数、16GB 显存、数小时训练

使用方式：
  from llm_sft_pipeline import train_sft, evaluate_sft
  model, history = train_sft()
  metrics = evaluate_sft()
"""

import json
import os
import random
import time

import torch
import torch.nn as nn
from torch.optim import AdamW
from transformers import AutoModelForCausalLM
from tqdm import tqdm

from config import cfg
from data_utils import build_sft_loaders, SYSTEM_PROMPT

# Windows 多进程 OpenMP 冲突规避
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

try:
    from peft import get_peft_model, LoraConfig, TaskType, PeftModel
    PEFT_AVAILABLE = True
except ImportError:
    PEFT_AVAILABLE = False
    print("[警告] peft 库未安装，LoRA 模式不可用。安装: pip install peft>=0.14.0")


def train_sft(device="cpu"):
    """
    LoRA SFT 训练主函数。

    返回:
      model: 训练好的模型（已合并 LoRA 权重）
      history: list, 每 epoch 的 train_loss, val_loss
    """
    if not PEFT_AVAILABLE:
        raise ImportError("LoRA SFT 需要 peft 库: pip install peft>=0.14.0")

    print("\n" + "=" * 50)
    print("【LLM SFT (LoRA)】开始训练")
    print("=" * 50)

    # 固定随机种子，保证可复现
    random.seed(cfg.sft_seed if hasattr(cfg, "sft_seed") else 42)
    torch.manual_seed(cfg.sft_seed if hasattr(cfg, "sft_seed") else 42)

    # 构建 DataLoader
    train_loader, val_loader, tokenizer = build_sft_loaders(cfg.qwen_model_path)

    # 加载 base model
    print(f"[SFT] 加载 base model: {cfg.qwen_model_path}")
    model = AutoModelForCausalLM.from_pretrained(
        cfg.qwen_model_path,
        dtype=torch.float32,
        trust_remote_code=True,
    )

    # ── LoRA 配置 ──
    lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=cfg.sft_lora_r,
        lora_alpha=cfg.sft_lora_alpha,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
        lora_dropout=0.05,
        bias="none",
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()  # 打印可训练参数占比
    model = model.to(device)

    # ── 优化器 ──
    optimizer = AdamW(model.parameters(), lr=cfg.sft_lr, weight_decay=0.01)
    total_steps = len(train_loader) * cfg.sft_epochs // cfg.sft_grad_accum
    print(f"[SFT] epochs={cfg.sft_epochs}, total_steps={total_steps}, lr={cfg.sft_lr}")

    # ── 训练循环 ──
    best_val_loss = float("inf")
    history = []
    adapter_dir = cfg.checkpoint_dir / "sft_adapter"

    for epoch in range(1, cfg.sft_epochs + 1):
        model.train()
        total_loss, total_tokens = 0.0, 0
        optimizer.zero_grad()
        t0 = time.time()

        pbar = tqdm(train_loader, desc=f"Epoch {epoch}/{cfg.sft_epochs}", leave=False)
        for step, batch in enumerate(pbar):
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)

            # HuggingFace CausalLM 内部自动处理 -100 的 loss masking
            outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
            loss = outputs.loss

            (loss / cfg.sft_grad_accum).backward()
            if (step + 1) % cfg.sft_grad_accum == 0:
                nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
                optimizer.zero_grad()

            # 只统计非 -100 的 token 数
            n_tokens = (labels != -100).sum().item()
            total_loss += loss.item() * n_tokens
            total_tokens += n_tokens
            pbar.set_postfix(loss=f"{loss.item():.4f}")

        avg_train_loss = total_loss / max(total_tokens, 1)

        # ── 验证 ──
        model.eval()
        val_loss, val_tokens = 0.0, 0
        with torch.no_grad():
            for batch in tqdm(val_loader, desc="Val", leave=False):
                input_ids = batch["input_ids"].to(device)
                attention_mask = batch["attention_mask"].to(device)
                labels = batch["labels"].to(device)
                outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
                n_tokens = (labels != -100).sum().item()
                val_loss += outputs.loss.item() * n_tokens
                val_tokens += n_tokens
        avg_val_loss = val_loss / max(val_tokens, 1)

        elapsed = time.time() - t0
        print(f"[Epoch {epoch}] train_loss={avg_train_loss:.4f} val_loss={avg_val_loss:.4f} ({elapsed:.0f}s)")

        history.append({"epoch": epoch, "train_loss": avg_train_loss, "val_loss": avg_val_loss})

        # 保存最优 adapter
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            model.save_pretrained(adapter_dir)
            tokenizer.save_pretrained(adapter_dir)
            print(f"  ✓ 最优 adapter 已保存 → {adapter_dir}")

    print(f"[SFT] 训练完成。最优 val_loss={best_val_loss:.4f}")

    # 合并 LoRA 权重，加速推理
    print("[SFT] 合并 LoRA 权重到 base model ...")
    model = model.merge_and_unload()
    return model, history


def evaluate_sft(model=None, tokenizer=None, num_samples=200, device="cpu", seed=42):
    """
    SFT 模型评估主函数。
    若未传入 model/tokenizer，则自动加载 adapter 并合并。

    返回:
      metrics: dict, 包含 accuracy, macro_f1, per_sample_time 等
    """
    from transformers import AutoTokenizer, AutoModelForCausalLM
    from data_utils import load_split, load_label_map

    print("\n" + "=" * 50)
    print("【LLM SFT 评估】开始")
    print("=" * 50)

    # ── 加载模型（若未传入）──
    if model is None:
        adapter_dir = cfg.checkpoint_dir / "sft_adapter"
        if not adapter_dir.exists():
            print(f"[错误] adapter 不存在: {adapter_dir}")
            print("请先运行 train_sft()")
            return None

        print(f"[SFT] 加载 base model + adapter ...")
        tokenizer = AutoTokenizer.from_pretrained(cfg.qwen_model_path, trust_remote_code=True)
        base_model = AutoModelForCausalLM.from_pretrained(
            cfg.qwen_model_path,
            dtype=torch.float16 if device == "cuda" else torch.float32,
            device_map="auto" if device == "cuda" else None,
            trust_remote_code=True,
        )
        model = PeftModel.from_pretrained(base_model, str(adapter_dir))
        model = model.merge_and_unload()
        if device != "cuda":
            model = model.to(device)
        model.eval()
        print("[SFT] 模型加载完成（LoRA 已合并）")

    # ── 加载数据 ──
    val_data = load_split("val")
    label_map = load_label_map()
    id2name = {int(k): v for k, v in label_map["id2name"].items()}

    random.seed(seed)
    samples = random.sample(val_data, min(num_samples, len(val_data)))

    # ── 推理 ──
    correct, total, unparseable = 0, 0, 0
    t0 = time.time()

    for i, item in enumerate(samples):
        text = item["sentence"]
        true_name = id2name[item["label"]]

        # 构造 prompt 并生成
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": f"新闻标题：{text}\n类别："},
        ]
        encoding = tokenizer.apply_chat_template(
            messages, tokenize=True, add_generation_prompt=True,
            return_tensors="pt", return_dict=True,
        )
        input_ids = encoding["input_ids"].to(device)
        attention_mask = encoding["attention_mask"].to(device)
        prompt_len = input_ids.shape[-1]

        with torch.no_grad():
            output_ids = model.generate(
                input_ids, attention_mask=attention_mask,
                max_new_tokens=8, do_sample=False,
                pad_token_id=tokenizer.eos_token_id,
            )
        new_tokens = output_ids[0][prompt_len:]
        raw_output = tokenizer.decode(new_tokens, skip_special_tokens=True).strip()

        # 解析
        pred_name = None
        for name in cfg.label_names:
            if name in raw_output:
                pred_name = name
                break

        is_correct = (pred_name == true_name)
        if pred_name is None:
            unparseable += 1
        if is_correct:
            correct += 1
        total += 1

        if i < 5:
            status = "✓" if is_correct else ("?" if pred_name is None else "✗")
            print(f"  [{i+1}] {status} 真实:{true_name:4s} 预测:{str(pred_name):4s} | {text[:30]}...")

    elapsed = time.time() - t0
    acc = correct / total if total > 0 else 0

    print(f"\n[SFT 结果] accuracy={acc:.4f} ({correct}/{total})")
    print(f"           无法解析: {unparseable} 条 ({unparseable/total*100:.1f}%)")
    print(f"           均速 {elapsed/total:.2f}s/条")

    # 保存结果
    result = {
        "method": "LLM SFT (LoRA)",
        "accuracy": acc,
        "macro_f1": None,
        "total": total,
        "correct": correct,
        "unparseable": unparseable,
        "elapsed": elapsed,
        "per_sample_time": elapsed / total,
    }
    out_path = cfg.output_dir / "sft_results.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)
    print(f"[输出] 结果已保存 → {out_path}")

    return result
