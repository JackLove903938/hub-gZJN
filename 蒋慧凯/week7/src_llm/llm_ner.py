"""
使用 DeepSeek API 做 NER：zero-shot vs few-shot 对比（peoples_daily 数据集）

教学重点：
  1. LLM 做 NER 的 prompt 设计
  2. 结构化输出解析（JSON提取 + 容错处理）
  3. span 级别 F1 计算

使用方式：
  # 方式1：环境变量（推荐，安全）
  export DEEPSEEK_API_KEY="sk-xxx"
  python llm_ner.py

  # 方式2：命令行传入（仅本地测试）
  python llm_ner.py --api_key sk-xxx --n_samples 50

依赖：
  pip install openai
"""

import os
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import json
import time
import random
import argparse
import re
from pathlib import Path
from collections import defaultdict

from openai import OpenAI

ROOT = Path(__file__).parent.parent
DATA_DIR = ROOT / "data" / "peoples_daily"
LOG_DIR = ROOT / "outputs" / "logs"

ENTITY_TYPE_ZH = {
    "PER": "人名",
    "ORG": "组织机构",
    "LOC": "地点",
}

ENTITY_TYPES_EN = list(ENTITY_TYPE_ZH.keys())


def build_client(api_key: str = None) -> OpenAI:
    """构建 DeepSeek 客户端。优先使用传入的 key，其次环境变量。"""
    key = api_key or os.getenv("DEEPSEEK_API_KEY")
    if not key:
        raise EnvironmentError(
            "请设置 DeepSeek API Key：\n"
            "  方式1（推荐）: export DEEPSEEK_API_KEY='sk-xxx'\n"
            "  方式2（测试）: python llm_ner.py --api_key sk-xxx"
        )
    return OpenAI(api_key=key, base_url="https://api.deepseek.com/v1")


def bio_to_spans(tokens: list[str], ner_tags: list[str]) -> set[tuple[str, str, int, int]]:
    """将 BIO 标签序列转为 span 集合 {(text, type, start, end)}。"""
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
            start = char_pos[i]
            j = i + 1
            while j < n and ner_tags[j] == f"I-{etype}":
                j += 1
            end = char_pos[j] - 1
            surface = "".join(tokens[i:j])
            spans.add((surface, etype, start, end))
            i = j
        else:
            i += 1
    return spans


def pred_spans_from_response(text: str, response_text: str) -> set[tuple[str, str, int, int]]:
    """从 LLM 输出中解析 span，格式：{(surface, type, start, end)}。"""
    json_match = re.search(r"\{.*\}", response_text, re.DOTALL)
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
        etype = str(ent.get("type", "")).strip()
        if not surface or etype not in ENTITY_TYPES_EN:
            continue
        idx = text.find(surface)
        if idx == -1:
            continue
        spans.add((surface, etype, idx, idx + len(surface) - 1))

    return spans


def compute_span_f1(all_golds: list, all_preds: list) -> dict:
    """计算 span-level 精确率、召回率、F1。"""
    tp = sum(len(g & p) for g, p in zip(all_golds, all_preds))
    pred_total = sum(len(p) for p in all_preds)
    gold_total = sum(len(g) for g in all_golds)
    p = tp / pred_total if pred_total else 0.0
    r = tp / gold_total if gold_total else 0.0
    f1 = 2 * p * r / (p + r) if (p + r) else 0.0
    return {"precision": p, "recall": r, "f1": f1, "tp": tp, "pred_total": pred_total, "gold_total": gold_total}


SYSTEM_PROMPT = """你是一个命名实体识别（NER）专家，专门处理中文文本。
请从用户输入的文本中识别以下3类实体，并以 JSON 格式输出结果：
- PER：人名
- ORG：组织机构名称
- LOC：地点名称

输出格式（严格遵守，不要包含其他文字）：
{"entities": [{"text": "实体文本", "type": "实体类型英文名"}, ...]}

如果没有实体，输出：{"entities": []}"""

FEW_SHOT_EXAMPLES = [
    {
        "text": "海钓比赛地点在厦门与金门之间的海域。",
        "output": '{"entities": [{"text": "厦门", "type": "LOC"}, {"text": "金门", "type": "LOC"}]}'
    },
    {
        "text": "这座依山傍水的博物馆由国内一流的设计师主持设计，整个建筑群精美而恢宏。",
        "output": '{"entities": []}'
    },
    {
        "text": "正在这时，马奈的黑色天鹅绒大衣出现在餐厅门口。",
        "output": '{"entities": [{"text": "马奈", "type": "PER"}]}'
    },
]


def zero_shot_prompt(text: str) -> list[dict]:
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": text},
    ]


def few_shot_prompt(text: str) -> list[dict]:
    messages = [{"role": "system", "content": SYSTEM_PROMPT}]
    for ex in FEW_SHOT_EXAMPLES:
        messages.append({"role": "user", "content": ex["text"]})
        messages.append({"role": "assistant", "content": ex["output"]})
    messages.append({"role": "user", "content": text})
    return messages


def call_api(client: OpenAI, messages: list[dict], model: str) -> str:
    """调用 DeepSeek API，返回文本输出，带简单重试。"""
    for attempt in range(3):
        try:
            resp = client.chat.completions.create(
                model=model,
                messages=messages,
                temperature=0.0,
                max_tokens=512,
            )
            return resp.choices[0].message.content or ""
        except Exception as e:
            if attempt < 2:
                time.sleep(2 ** attempt)
            else:
                print(f"  API 调用失败：{e}")
                return ""
    return ""


def sample_records(n: int, seed: int = 42) -> list[dict]:
    """从验证集中采样 n 条，尽量覆盖所有实体类型。"""
    with open(DATA_DIR / "validation.json", "r", encoding="utf-8") as f:
        records = json.load(f)

    random.seed(seed)
    by_type = defaultdict(list)
    for r in records:
        spans = bio_to_spans(r["tokens"], r["ner_tags"])
        for _, etype, _, _ in spans:
            by_type[etype].append(r)

    selected = set()
    selected_list = []

    per_type = max(1, n // len(ENTITY_TYPES_EN))
    for etype in ENTITY_TYPES_EN:
        candidates = [r for r in by_type[etype] if id(r) not in selected]
        if candidates:
            chosen = random.sample(candidates, min(per_type, len(candidates)))
            for r in chosen:
                if len(selected_list) < n and id(r) not in selected:
                    selected.add(id(r))
                    selected_list.append(r)

    remaining = [r for r in records if id(r) not in selected]
    random.shuffle(remaining)
    for r in remaining:
        if len(selected_list) >= n:
            break
        selected_list.append(r)

    return selected_list[:n]


def main():
    args = parse_args()

    try:
        client = build_client(args.api_key)
    except EnvironmentError as e:
        print(f"[跳过] {e}")
        return

    records = sample_records(args.n_samples)
    print(f"采样 {len(records)} 条验证集样本")

    zero_shot_golds = []
    zero_shot_preds = []
    few_shot_golds = []
    few_shot_preds = []

    detail_records = []

    for i, record in enumerate(records, 1):
        text = "".join(record["tokens"])
        gold = bio_to_spans(record["tokens"], record["ner_tags"])

        # Zero-shot
        zs_resp = call_api(client, zero_shot_prompt(text), args.model)
        zs_pred = pred_spans_from_response(text, zs_resp)

        # Few-shot
        fs_resp = call_api(client, few_shot_prompt(text), args.model)
        fs_pred = pred_spans_from_response(text, fs_resp)

        zero_shot_golds.append(gold)
        zero_shot_preds.append(zs_pred)
        few_shot_golds.append(gold)
        few_shot_preds.append(fs_pred)

        detail_records.append({
            "text": text,
            "gold": [{"text": s, "type": t} for s, t, _, _ in gold],
            "zero_shot": [{"text": s, "type": t} for s, t, _, _ in zs_pred],
            "few_shot": [{"text": s, "type": t} for s, t, _, _ in fs_pred],
        })

        if i % 10 == 0 or i == len(records):
            print(f"  已处理 {i}/{len(records)} 条")

    zs_metrics = compute_span_f1(zero_shot_golds, zero_shot_preds)
    fs_metrics = compute_span_f1(few_shot_golds, few_shot_preds)

    print("\n" + "=" * 60)
    print(f"DeepSeek NER 对比结果（模型：{args.model}，样本：{len(records)} 条）")
    print("=" * 60)
    print(f"{'方案':<20} {'Precision':>10} {'Recall':>10} {'F1':>10}")
    print("-" * 52)
    print(f"{'Zero-shot':<20} {zs_metrics['precision']:>10.4f} {zs_metrics['recall']:>10.4f} {zs_metrics['f1']:>10.4f}")
    print(f"{'Few-shot (3例)':<20} {fs_metrics['precision']:>10.4f} {fs_metrics['recall']:>10.4f} {fs_metrics['f1']:>10.4f}")

    LOG_DIR.mkdir(parents=True, exist_ok=True)
    result = {
        "model": args.model,
        "n_samples": len(records),
        "zero_shot": zs_metrics,
        "few_shot": fs_metrics,
        "detail": detail_records,
    }

    def _to_python(v):
        return v.item() if hasattr(v, "item") else v

    result["zero_shot"] = {k: _to_python(v) for k, v in result["zero_shot"].items()}
    result["few_shot"] = {k: _to_python(v) for k, v in result["few_shot"].items()}

    out_path = LOG_DIR / "eval_llm.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)
    print(f"\nLLM 评估结果已保存 → {out_path}")


def parse_args():
    parser = argparse.ArgumentParser(description="DeepSeek zero-shot/few-shot NER 对比")
    parser.add_argument("--n_samples", type=int, default=100)
    parser.add_argument("--model", type=str, default="deepseek-chat",
                        help="模型名称：deepseek-chat (默认，V3) 或 deepseek-reasoner (R1)。"
                             "注意：deepseek-v4-flash / deepseek-v4-pro 将于 2026/07/24 启用，"
                             "当前 deepseek-chat 对应 v4-flash 非思考模式")
    parser.add_argument("--api_key", type=str, default=None,
                        help="DeepSeek API Key（仅本地测试使用，生产环境请用环境变量）")
    return parser.parse_args()


if __name__ == "__main__":
    main()
