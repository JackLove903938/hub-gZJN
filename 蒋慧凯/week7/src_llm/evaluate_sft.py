"""
加载 SFT checkpoint（LoRA / 全量微调），在验证集上评估 NER entity-level F1
适配 peoples_daily 数据集

使用方式：
  python evaluate_sft.py
  python evaluate_sft.py --n_samples 50 --demo

依赖：
  pip install torch transformers peft
"""

import os
import argparse
import json
import random
import re
import time
from pathlib import Path

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

try:
    from peft import PeftModel
    PEFT_AVAILABLE = True
except ImportError:
    PEFT_AVAILABLE = False

ROOT        = Path(__file__).parent.parent
DATA_DIR    = ROOT / "data" / "peoples_daily"
ADAPTER_DIR = ROOT / "outputs" / "sft_adapter"
LOG_DIR     = ROOT / "outputs" / "logs"

_local_qwen = ROOT.parent.parent / "pretrain_models" / "Qwen2-0.5B-Instruct"
MODEL_PATH = str(_local_qwen) if _local_qwen.exists() else "Qwen/Qwen2-0.5B-Instruct"

ENTITY_TYPES = ["PER", "ORG", "LOC"]

SYSTEM_PROMPT = (
    "你是一个命名实体识别助手。从文本中识别命名实体，以 JSON 格式输出。\n"
    "实体类型（英文标识）：PER（人名）、ORG（组织机构）、LOC（地点）\n"
    '输出格式（严格遵守，不输出其他内容）：{"entities": [{"text": "实体文本", "type": "实体类型"}]}\n'
    '无实体时输出：{"entities": []}'
)


def load_model(model_path: str, ckpt_dir: str, device: torch.device):
    ckpt_path = Path(ckpt_dir)
    is_lora   = (ckpt_path / "adapter_config.json").exists()

    if is_lora:
        if not PEFT_AVAILABLE:
            raise ImportError("加载 LoRA adapter 需要 peft 库")
        print(f"检测到 LoRA adapter，加载 base model: {model_path}")
        tokenizer  = AutoTokenizer.from_pretrained(
            model_path, trust_remote_code=True
        )
        base_model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=torch.float32,
            device_map="auto" if device.type == "cuda" else None,
            trust_remote_code=True,
        )
        print(f"加载 LoRA adapter: {ckpt_dir}")
        model = PeftModel.from_pretrained(base_model, str(ckpt_path))
        model = model.merge_and_unload()
    else:
        print(f"检测到全量微调 checkpoint，直接加载: {ckpt_dir}")
        tokenizer = AutoTokenizer.from_pretrained(
            str(ckpt_path), trust_remote_code=True
        )
        model = AutoModelForCausalLM.from_pretrained(
            str(ckpt_path),
            torch_dtype=torch.float16 if device.type == "cuda" else torch.float32,
            device_map="auto" if device.type == "cuda" else None,
            trust_remote_code=True,
        )

    if device.type != "cuda":
        model = model.to(device)
    model.eval()
    ckpt_type = "LoRA adapter 已合并" if is_lora else "全量微调模型"
    print(f"模型加载完成（{ckpt_type}）\n")
    return model, tokenizer


def generate_ner(text: str, model, tokenizer, device: torch.device,
                 max_new_tokens: int = 256) -> str:
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user",   "content": text},
    ]
    encoding = tokenizer.apply_chat_template(
        messages, tokenize=True, add_generation_prompt=True,
        return_tensors="pt", return_dict=True,
    )
    input_ids      = encoding["input_ids"].to(device)
    attention_mask = encoding["attention_mask"].to(device)
    prompt_len     = input_ids.shape[-1]

    with torch.no_grad():
        output_ids = model.generate(
            input_ids, attention_mask=attention_mask,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
        )
    new_tokens = output_ids[0][prompt_len:]
    return tokenizer.decode(new_tokens, skip_special_tokens=True).strip()


def bio_to_spans(tokens: list[str], ner_tags: list[str]) -> set:
    spans = set()
    i = 0
    n = len(tokens)
    char_pos = [0]
    for tok in tokens:
        char_pos.append(char_pos[-1] + len(tok))
    while i < n:
        tag = ner_tags[i]
        if tag.startswith("B-"):
            etype = tag[2:]
            j = i + 1
            while j < n and ner_tags[j] == f"I-{etype}":
                j += 1
            surface = "".join(tokens[i:j])
            spans.add((surface, etype, char_pos[i], char_pos[j] - 1))
            i = j
        else:
            i += 1
    return spans


def pred_spans_from_output(text: str, raw_output: str) -> set:
    json_match = re.search(r"\{.*\}", raw_output, re.DOTALL)
    if not json_match:
        return set()
    try:
        obj = json.loads(json_match.group())
    except json.JSONDecodeError:
        return set()
    entities = obj.get("entities", [])
    if not isinstance(entities, list):
        return set()
    spans = set()
    for ent in entities:
        if not isinstance(ent, dict):
            continue
        surface = str(ent.get("text", "")).strip()
        etype   = str(ent.get("type", "")).strip()
        if not surface or etype not in ENTITY_TYPES:
            continue
        idx = text.find(surface)
        if idx == -1:
            continue
        spans.add((surface, etype, idx, idx + len(surface) - 1))
    return spans


def compute_span_f1(all_golds: list, all_preds: list) -> dict:
    tp         = sum(len(g & p) for g, p in zip(all_golds, all_preds))
    pred_total = sum(len(p) for p in all_preds)
    gold_total = sum(len(g) for g in all_golds)
    p  = tp / pred_total if pred_total else 0.0
    r  = tp / gold_total if gold_total else 0.0
    f1 = 2 * p * r / (p + r) if (p + r) else 0.0
    return {"precision": p, "recall": r, "f1": f1,
            "tp": tp, "pred_total": pred_total, "gold_total": gold_total}


def parse_args():
    parser = argparse.ArgumentParser(description="LLM SFT NER 评估")
    parser.add_argument("--model_path",  default=MODEL_PATH)
    parser.add_argument("--ckpt_dir",    default=str(ADAPTER_DIR))
    parser.add_argument("--data_dir",    default=str(DATA_DIR))
    parser.add_argument("--n_samples",   default=100, type=int)
    parser.add_argument("--seed",        default=42,  type=int)
    parser.add_argument("--demo",        action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"使用设备: {device}")

    ckpt_dir = Path(args.ckpt_dir)
    if not ckpt_dir.exists():
        print(f"[错误] checkpoint 目录不存在：{ckpt_dir}")
        print("请先运行 train_sft.py 完成训练。")
        return

    with open(Path(args.data_dir) / "validation.json", encoding="utf-8") as f:
        val_data = json.load(f)

    random.seed(args.seed)
    n = 5 if args.demo else args.n_samples
    samples = random.sample(val_data, min(n, len(val_data)))
    print(f"评估样本数: {len(samples)}\n")

    model, tokenizer = load_model(args.model_path, str(ckpt_dir), device)

    all_golds, all_preds = [], []
    detail_records = []
    parse_fail = 0
    t0 = time.time()

    for i, record in enumerate(samples, 1):
        text  = "".join(record["tokens"])
        g_set = bio_to_spans(record["tokens"], record["ner_tags"])
        raw   = generate_ner(text, model, tokenizer, device)
        p_set = pred_spans_from_output(text, raw)

        if not re.search(r"\{.*entities.*\}", raw, re.DOTALL):
            parse_fail += 1

        all_golds.append(g_set)
        all_preds.append(p_set)
        detail_records.append({
            "text": text,
            "gold":  [{"text": s, "type": t} for s, t, *_ in g_set],
            "pred":  [{"text": s, "type": t} for s, t, *_ in p_set],
            "raw_output": raw,
        })

        tp_here  = len(g_set & p_set)
        status   = "OK" if g_set == p_set else ("~" if tp_here > 0 else "X")
        gold_str = ",".join(f"{s}({t})" for s, t, *_ in list(g_set)[:3])
        print(f"[{i:3d}/{len(samples)}] {status}  gold:{gold_str or '无'}  |  {text[:30]}")

    elapsed = time.time() - t0
    metrics = compute_span_f1(all_golds, all_preds)

    print(f"\n{'='*65}")
    print(f"LLM SFT NER 评估结果")
    print(f"{'='*65}")
    print(f"  样本数      : {len(samples)}")
    print(f"  Precision   : {metrics['precision']:.4f}")
    print(f"  Recall      : {metrics['recall']:.4f}")
    print(f"  F1          : {metrics['f1']:.4f}")
    print(f"  JSON 解析失败: {parse_fail} 条 ({parse_fail/len(samples)*100:.1f}%)")
    print(f"  总耗时      : {elapsed:.1f}s，均值 {elapsed/len(samples):.2f}s/条")

    out_path = LOG_DIR / "eval_sft.json"
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump({
            "metrics": {k: (v if isinstance(v, (int, float)) else v)
                        for k, v in metrics.items()},
            "n_samples": len(samples), "parse_fail": parse_fail,
            "detail": detail_records,
        }, f, ensure_ascii=False, indent=2)
    print(f"结果已保存 → {out_path}")


if __name__ == "__main__":
    main()
