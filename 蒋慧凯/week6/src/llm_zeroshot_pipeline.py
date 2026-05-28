"""
llm_zeroshot_pipeline.py —— LLM Zero-Shot 文本分类

教学重点：
  1. Prompt 工程：system + user 模板设计
  2. 生成式分类：model.generate() → decode → 解析类别名
  3. 输出兜底：模糊匹配处理模型格式不稳定
  4. 与 BERT 判别式的本质差异（无训练、高延迟、格式需解析）

使用方式：
  from llm_zeroshot_pipeline import run_zeroshot
  metrics = run_zeroshot()
"""

import json
import random
import time

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

from config import cfg
from data_utils import load_split, load_label_map, SYSTEM_PROMPT


def load_qwen_model(device):
    """加载 Qwen2-0.5B-Instruct 模型和 tokenizer。"""
    print(f"[LLM] 加载模型: {cfg.qwen_model_path}")
    tokenizer = AutoTokenizer.from_pretrained(cfg.qwen_model_path, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        cfg.qwen_model_path,
        dtype=torch.float16 if device == "cuda" else torch.float32,
        device_map="auto" if device == "cuda" else None,
        trust_remote_code=True,
    )
    if device != "cuda":
        model = model.to(device)
    model.eval()
    print("[LLM] 模型加载完成")
    return model, tokenizer


def classify_one(text, model, tokenizer, device):
    """
    单条 zero-shot 分类。
    返回模型生成的原始文本（已做 strip）。
    """
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": f"新闻标题：{text}\n类别："},
    ]
    # transformers 5.x 兼容：apply_chat_template 返回 dict，取 input_ids
    encoding = tokenizer.apply_chat_template(
        messages, tokenize=True, add_generation_prompt=True,
        return_tensors="pt", return_dict=True,
    )
    input_ids = encoding["input_ids"].to(device)
    attention_mask = encoding["attention_mask"].to(device)
    prompt_len = input_ids.shape[-1]

    with torch.no_grad():
        output_ids = model.generate(
            input_ids,
            attention_mask=attention_mask,
            max_new_tokens=8,       # 类别名通常只有 1~2 个 token
            do_sample=False,        # greedy decoding，分类任务要确定性
            pad_token_id=tokenizer.eos_token_id,
        )

    # 只取新生成的部分
    new_tokens = output_ids[0][prompt_len:]
    return tokenizer.decode(new_tokens, skip_special_tokens=True).strip()


def parse_prediction(raw_output):
    """
    从模型输出中提取类别名。
    模型可能输出"科技"、"科技新闻"、"科技类"等，做模糊匹配。
    若无法解析，返回 None。
    """
    for name in cfg.label_names:
        if name in raw_output:
            return name
    return None


def run_zeroshot(num_samples=None, device="cpu", seed=42):
    """
    Zero-shot 分类主函数。
    从验证集随机采样 num_samples 条进行评估。

    返回:
      metrics: dict, 包含 accuracy, total, correct, unparseable, elapsed, per_sample_time
    """
    num_samples = num_samples or cfg.zeroshot_num_samples
    print("\n" + "=" * 50)
    print("【LLM Zero-Shot】开始评估")
    print("=" * 50)
    print(f"[配置] 采样数: {num_samples}, 设备: {device}")

    # 加载数据和模型
    val_data = load_split("val")
    label_map = load_label_map()
    id2name = {int(k): v for k, v in label_map["id2name"].items()}

    random.seed(seed)
    samples = random.sample(val_data, min(num_samples, len(val_data)))

    model, tokenizer = load_qwen_model(device)

    # 推理
    correct, total, unparseable = 0, 0, 0
    t0 = time.time()

    for i, item in enumerate(samples):
        text = item["sentence"]
        true_name = id2name[item["label"]]

        raw_output = classify_one(text, model, tokenizer, device)
        pred_name = parse_prediction(raw_output)

        is_correct = (pred_name == true_name)
        if pred_name is None:
            unparseable += 1
        if is_correct:
            correct += 1
        total += 1

        status = "✓" if is_correct else ("?" if pred_name is None else "✗")
        if i < 5:  # 只打印前 5 条详细结果
            print(f"  [{i+1}] {status} 真实:{true_name:4s} 预测:{str(pred_name):4s} | {text[:30]}...")

    elapsed = time.time() - t0
    acc = correct / total if total > 0 else 0

    print(f"\n[Zero-Shot 结果] accuracy={acc:.4f} ({correct}/{total})")
    print(f"                 无法解析: {unparseable} 条 ({unparseable/total*100:.1f}%)")
    print(f"                 总耗时: {elapsed:.1f}s, 均速 {elapsed/total:.2f}s/条")

    # 保存结果
    result = {
        "method": "LLM Zero-Shot",
        "accuracy": acc,
        "macro_f1": None,  # zero-shot 不计算 macro_f1（没有每类统计）
        "total": total,
        "correct": correct,
        "unparseable": unparseable,
        "elapsed": elapsed,
        "per_sample_time": elapsed / total,
    }
    out_path = cfg.output_dir / "zeroshot_results.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)
    print(f"[输出] 结果已保存 → {out_path}")

    return result
