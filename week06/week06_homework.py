import sys
import io
import json
from pathlib import Path
from collections import Counter

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")

# ══════════════════════════════════════════════════════════════════════════════
# 一、加载各方法的实验结果
# ══════════════════════════════════════════════════════════════════════════════

ROOT = Path(__file__).parent / "text_classification项目"
OUTPUT_DIR = ROOT / "outputs"
DATA_DIR = ROOT / "data"


def load_results():
    """加载所有方法的训练日志和评估结果。"""
    results = {}

    # 1. BERT fine-tune (CLS池化，普通Loss)
    log_cls = OUTPUT_DIR / "train_log_cls.json"
    if log_cls.exists():
        with open(log_cls, encoding="utf-8") as f:
            results["bert_cls"] = json.load(f)

    # 2. BERT fine-tune (CLS池化，加权Loss)
    log_cls_w = OUTPUT_DIR / "train_log_cls_weighted.json"
    if log_cls_w.exists():
        with open(log_cls_w, encoding="utf-8") as f:
            results["bert_cls_weighted"] = json.load(f)

    # 3. LLM Zero-shot
    zs_path = OUTPUT_DIR / "llm_zero_shot_results.json"
    if zs_path.exists():
        with open(zs_path, encoding="utf-8") as f:
            results["llm_zero_shot"] = json.load(f)

    # 4. LLM SFT (LoRA)
    sft_log = OUTPUT_DIR / "train_log_sft.json"
    if sft_log.exists():
        with open(sft_log, encoding="utf-8") as f:
            results["llm_sft_log"] = json.load(f)

    sft_eval = OUTPUT_DIR / "llm_sft_results.json"
    if sft_eval.exists():
        with open(sft_eval, encoding="utf-8") as f:
            results["llm_sft_eval"] = json.load(f)

    return results


# ══════════════════════════════════════════════════════════════════════════════
# 二、分析与对比
# ══════════════════════════════════════════════════════════════════════════════

def print_header(title):
    print(f"\n{'═' * 70}")
    print(f"  {title}")
    print(f"{'═' * 70}")


def analyze_bert_training(results):
    """分析 BERT Fine-tuning 训练过程与效果。"""
    print_header("方法一：BERT Fine-tuning（判别式分类）")

    print("""
【方法原理】
  - 模型：bert-base-chinese (110M 参数)
  - 结构：BertModel → 池化层(CLS/Mean/Max) → Dropout → Linear → 15类logits
  - 训练策略：
    · 分层学习率：BERT层 lr=2e-5，分类头 lr=1e-4
    · 优化器：AdamW (weight_decay=0.01)
    · 调度器：Linear Warmup (10%) + Decay
    · 梯度裁剪：max_norm=1.0
  - 数据：全量 53,360 条训练数据
""")

    if "bert_cls" in results:
        print("  【CLS 池化 - 普通 Loss 训练曲线】")
        print(f"  {'Epoch':<8} {'Train Loss':<12} {'Train Acc':<12} {'Val Acc':<10} {'Val F1':<10} {'Time':<8}")
        print(f"  {'-' * 58}")
        for rec in results["bert_cls"]:
            print(f"  {rec['epoch']:<8} {rec['train_loss']:<12.4f} "
                  f"{rec['train_acc']:<12.4f} {rec['val_acc']:<10.4f} "
                  f"{rec['val_macro_f1']:<10.4f} {rec['elapsed_s']:<8.0f}s")

        best = max(results["bert_cls"], key=lambda x: x["val_acc"])
        print(f"\n  ★ 最优结果（Epoch {best['epoch']}）：")
        print(f"    Val Accuracy = {best['val_acc']:.4f}")
        print(f"    Val Macro F1 = {best['val_macro_f1']:.4f}")

    if "bert_cls_weighted" in results:
        print("\n  【CLS 池化 - 加权 Loss 训练曲线】")
        print(f"  {'Epoch':<8} {'Train Loss':<12} {'Train Acc':<12} {'Val Acc':<10} {'Val F1':<10} {'Time':<8}")
        print(f"  {'-' * 58}")
        for rec in results["bert_cls_weighted"]:
            print(f"  {rec['epoch']:<8} {rec['train_loss']:<12.4f} "
                  f"{rec['train_acc']:<12.4f} {rec['val_acc']:<10.4f} "
                  f"{rec['val_macro_f1']:<10.4f} {rec['elapsed_s']:<8.0f}s")

        best_w = max(results["bert_cls_weighted"], key=lambda x: x["val_acc"])
        print(f"\n  ★ 最优结果（Epoch {best_w['epoch']}）：")
        print(f"    Val Accuracy = {best_w['val_acc']:.4f}")
        print(f"    Val Macro F1 = {best_w['val_macro_f1']:.4f}")

    # 对比加权 vs 非加权
    if "bert_cls" in results and "bert_cls_weighted" in results:
        best_plain = max(results["bert_cls"], key=lambda x: x["val_acc"])
        best_weighted = max(results["bert_cls_weighted"], key=lambda x: x["val_acc"])
        print(f"""
  【加权 Loss 效果分析】
    普通 Loss:  Accuracy={best_plain['val_acc']:.4f}, Macro F1={best_plain['val_macro_f1']:.4f}
    加权 Loss:  Accuracy={best_weighted['val_acc']:.4f}, Macro F1={best_weighted['val_macro_f1']:.4f}

    Accuracy 变化: {best_weighted['val_acc'] - best_plain['val_acc']:+.4f}
    Macro F1 变化: {best_weighted['val_macro_f1'] - best_plain['val_macro_f1']:+.4f}

  ⇒ 分析：加权 Loss 虽然整体 Accuracy 可能略降（因为牺牲了部分多数类准确率），
    但 Macro F1 提升说明少数类（如证券类，仅 257 条训练样本）的 Recall 得到改善。
    这体现了 Accuracy 在类别不均衡时的"欺骗性"——它被多数类主导。
""")


def analyze_llm_zero_shot(results):
    """分析 LLM Zero-shot 效果。"""
    print_header("方法二：LLM Zero-shot（无需训练）")

    print("""
【方法原理】
  - 模型：Qwen2-0.5B-Instruct (500M 参数)
  - 方法：设计 prompt，让 LLM 直接输出类别名称
  - System Prompt：指定角色 + 列出 15 个可选类别
  - User Prompt：  "新闻标题：{text}\\n类别："
  - 解码策略：Greedy decoding（do_sample=False）
  - 后处理：模糊匹配提取类别名
  - 训练数据需求：0 条（零样本）
""")

    if "llm_zero_shot" in results:
        zs = results["llm_zero_shot"]
        print(f"  【评估结果（val 集随机采样 {zs['total']} 条）】")
        print(f"    准确率    : {zs['correct']}/{zs['total']} = {zs['accuracy']:.4f}")
        print(f"    无法解析  : {zs['unparseable']} 条 ({zs['unparseable']/zs['total']*100:.1f}%)")

        # 统计各类别表现
        if "results" in zs:
            true_counts = Counter()
            correct_counts = Counter()
            for r in zs["results"]:
                true_counts[r["true_label"]] += 1
                if r["correct"]:
                    correct_counts[r["true_label"]] += 1

            print(f"\n  【各类别准确率（样本数 ≥ 5 的类别）】")
            print(f"    {'类别':<6} {'正确/总数':<12} {'准确率':<8}")
            print(f"    {'-' * 30}")
            for label in sorted(true_counts.keys(),
                                key=lambda x: correct_counts[x] / max(true_counts[x], 1),
                                reverse=True):
                if true_counts[label] >= 5:
                    acc = correct_counts[label] / true_counts[label]
                    print(f"    {label:<6} {correct_counts[label]:>3}/{true_counts[label]:<8} {acc:.3f}")

        print("""
  ⇒ 分析：
    1. Zero-shot 准确率仅 36%，远低于 BERT fine-tune (56%)
    2. 无法解析率高达 29%——模型输出了训练数据中不存在的类别名（如"房地产"而非"房产"）
    3. Qwen2-0.5B 模型参数量较小(500M)，知识储备有限
    4. 优势：无需任何训练数据和训练时间，适合快速验证和冷启动场景
""")


def analyze_llm_sft(results):
    """分析 LLM SFT (LoRA) 效果。"""
    print_header("方法三：LLM SFT — LoRA 指令微调（生成式分类）")

    print("""
【方法原理】
  - 基座模型：Qwen2-0.5B-Instruct (495M 总参数)
  - 微调方法：LoRA (Low-Rank Adaptation)
    · 冻结原始权重 W，旁路添加 ΔW = B·A（r=8）
    · 目标模块：q_proj, k_proj, v_proj, o_proj
    · 可训练参数：1,081,344 (0.22%)
  - 数据格式：Chat format (system/user/assistant)
  - Loss 策略：仅对 assistant 回复部分计算 loss（prompt 部分 label=-100）
  - 训练数据：5,000 条（仅占 BERT 训练数据的 9.4%）
  - 优化器：AdamW, lr=2e-4, grad_accum=4（等效 batch=16）
""")

    if "llm_sft_log" in results:
        print("  【训练过程】")
        print(f"  {'Epoch':<8} {'Train Loss':<14} {'Val Loss':<12} {'Time':<8}")
        print(f"  {'-' * 42}")
        for rec in results["llm_sft_log"]:
            print(f"  {rec['epoch']:<8} {rec['train_loss']:<14.4f} "
                  f"{rec['val_loss']:<12.4f} {rec['elapsed_s']:<8.0f}s")

        best_epoch = min(results["llm_sft_log"], key=lambda x: x["val_loss"])
        print(f"\n  ★ 最优 Epoch: {best_epoch['epoch']}（val_loss={best_epoch['val_loss']:.4f}）")
        print(f"    注意：Epoch 3 val_loss 回升，出现轻微过拟合")

    if "llm_sft_eval" in results:
        sft = results["llm_sft_eval"]
        print(f"\n  【评估结果（val 集采样 {sft['total']} 条）】")
        print(f"    准确率    : {sft['correct']}/{sft['total']} = {sft['accuracy']:.4f}")
        print(f"    无法解析  : {sft['unparseable']} 条 ({sft['unparseable']/sft['total']*100:.1f}%)")

    print("""
  ⇒ 分析：
    1. LoRA 仅用 5K 数据 + 0.22% 参数更新，就达到 58% 准确率
    2. 对比 Zero-shot (36%)，SFT 提升了 22 个百分点——微调的价值巨大
    3. 无法解析率从 29% 降至 1%——模型学会了严格按格式输出
    4. 训练效率极高：18 分钟 (GPU)，显存仅需 ~3GB
""")


def comprehensive_comparison(results):
    """三种方法综合对比。"""
    print_header("综合对比分析")

    print("""
┌─────────────────┬───────────────────┬───────────────────┬───────────────────┐
│     对比维度     │  BERT Fine-tune   │  LLM Zero-shot    │  LLM SFT (LoRA)   │
├─────────────────┼───────────────────┼───────────────────┼───────────────────┤
│ 分类范式         │ 判别式（logits）   │ 生成式（文本）     │ 生成式（文本）     │
│ 基座模型         │ bert-base (110M)  │ Qwen2-0.5B (500M) │ Qwen2-0.5B (500M) │
│ 可训练参数       │ 110M (100%)       │ 0                 │ 1.08M (0.22%)     │
│ 训练数据量       │ 53,360 条         │ 0 条              │ 5,000 条          │
│ 训练时间         │ ~33 min (GPU×3ep) │ 0                 │ ~18 min (GPU×3ep) │
│ Val Accuracy     │ 56.18%            │ 36.00%            │ 58.00%            │
│ Val Macro F1     │ 54.99%            │ -                 │ -                 │
│ 无法解析率       │ 0%                │ 29.0%             │ 1.0%              │
│ 推理速度         │ ~5ms/条 (GPU)     │ ~2s/条 (CPU)      │ ~60ms/条 (GPU)    │
│ 显存需求(训练)   │ ~4GB              │ 0                 │ ~3GB              │
└─────────────────┴───────────────────┴───────────────────┴───────────────────┘
""")

    # 获取实际数据
    bert_acc = None
    if "bert_cls" in results:
        bert_acc = max(results["bert_cls"], key=lambda x: x["val_acc"])["val_acc"]

    zs_acc = results.get("llm_zero_shot", {}).get("accuracy")
    sft_acc = results.get("llm_sft_eval", {}).get("accuracy")

    if bert_acc and zs_acc and sft_acc:
        print(f"  实测准确率汇总：")
        print(f"    BERT Fine-tune (CLS, 53K数据, 3ep):  {bert_acc:.2%}")
        print(f"    LLM Zero-shot (无训练):              {zs_acc:.2%}")
        print(f"    LLM SFT (LoRA, 5K数据, 3ep):        {sft_acc:.2%}")


def key_insights():
    """核心洞察与结论。"""
    print_header("核心洞察与结论")

    print("""
【洞察一：数据效率】
  LLM SFT 用 5K 条数据（BERT 的 9.4%）就超越了 BERT 全量训练的效果。
  这体现了大模型预训练知识的迁移效率优势——预训练阶段已学到丰富的语言理解能力，
  SFT 只需少量数据"激活"任务相关能力即可。

【洞察二：训练 vs 不训练的鸿沟】
  Zero-shot (36%) vs SFT (58%)：差距 22 个百分点。
  即使是极少量的标注数据+微调，效果也远超 prompt engineering。
  这说明对于需要精确类别体系的任务，纯 prompt 方法有明显天花板。

【洞察三：判别式 vs 生成式】
  - BERT 优势：推理极快(5ms)、无解析问题、确定性输出
  - LLM SFT 优势：数据效率高、可复用基座、可扩展多任务
  - 工程选择取决于：数据量、延迟要求、是否需要灵活扩展

【洞察四：类别不均衡处理】
  加权 Loss 的效果：Accuracy 可能微降，但 Macro F1 提升。
  本质是在"整体正确率"和"对所有类公平"之间做 trade-off。
  证券类（仅 257 条训练样本 vs 科技类 5955 条，比例 1:23）受益最大。

【洞察五：无法解析问题】
  生成式方法的固有缺陷——模型可能输出非预设类别名。
  Zero-shot 无法解析率 29%（输出"房地产"而非"房产"）。
  SFT 后降至 1%——微调有效约束了输出格式。
  判别式 BERT 则完全不存在此问题（直接输出 logits）。

【洞察六：过拟合迹象】
  SFT Epoch 2 的 val_loss 最低(0.6386)，Epoch 3 回升(0.6523)。
  BERT 3 epoch 内 val_acc 持续上升但增速放缓。
  说明小数据集 + 大模型容易过拟合，early stopping 策略很重要。
""")


def practical_recommendations():
    """实践建议。"""
    print_header("实践建议：如何选择训练方法")

    print("""
┌─────────────────────────────────────────────────────────────────────────┐
│                        方法选择决策树                                     │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                         │
│  Q1: 是否有标注数据？                                                    │
│   ├─ 否 → LLM Zero-shot / Few-shot                                     │
│   └─ 是 → Q2                                                           │
│                                                                         │
│  Q2: 标注数据量多大？                                                    │
│   ├─ < 1K 条    → LLM SFT (LoRA)，大模型迁移能力弥补数据不足             │
│   ├─ 1K~10K 条  → LLM SFT (LoRA) 优先，BERT fine-tune 也可             │
│   └─ > 10K 条   → BERT fine-tune（充分数据下判别式模型上限更高）          │
│                                                                         │
│  Q3: 推理延迟要求？                                                      │
│   ├─ < 10ms     → BERT（判别式，一次前向传播）                           │
│   └─ > 100ms OK → LLM（生成式，需 auto-regressive decoding）            │
│                                                                         │
│  Q4: 是否需要多任务扩展？                                                │
│   ├─ 是 → LLM SFT（一个模型 + 不同 prompt = 多任务）                    │
│   └─ 否 → 按 Q2/Q3 选择                                                │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘

  场景示例：
  1. 线上商品分类（万级标注，低延迟）→ BERT fine-tune
  2. 客服工单分类（千级标注，可容忍延迟）→ LLM SFT
  3. 新品类快速验证（无标注）→ LLM Zero-shot + 人工校验
  4. 学术论文分类（少量标注 + 多标签）→ LLM SFT with LoRA
""")


def training_details_comparison():
    """训练细节对比。"""
    print_header("训练细节深度对比")

    print("""
┌──────────────────────┬─────────────────────────┬─────────────────────────┐
│       技术细节        │      BERT Fine-tune      │      LLM SFT (LoRA)     │
├──────────────────────┼─────────────────────────┼─────────────────────────┤
│ 输入格式             │ [CLS] text [SEP] + pad  │ Chat template           │
│                      │ → input_ids [B, 64]     │ (system/user/assistant) │
│                      │                         │ → input_ids [B, 128]    │
├──────────────────────┼─────────────────────────┼─────────────────────────┤
│ 输出形式             │ logits [B, 15]          │ 生成 token sequence     │
│                      │ → argmax 取预测类别     │ → 解码文本 → 匹配类别  │
├──────────────────────┼─────────────────────────┼─────────────────────────┤
│ Loss 计算            │ CrossEntropy(logits, y)  │ CrossEntropy(仅 asst 部 │
│                      │ 所有样本等权            │ 分)，prompt 部分 =-100  │
├──────────────────────┼─────────────────────────┼─────────────────────────┤
│ 参数更新策略         │ 全参数更新              │ 仅更新 LoRA 旁路矩阵   │
│                      │ 分层 lr (BERT < Head)   │ B·A, r=8, target=QKV+O │
├──────────────────────┼─────────────────────────┼─────────────────────────┤
│ 正则化               │ Dropout(0.1) + L2       │ LoRA dropout(0.05) + L2 │
│                      │ + Warmup + Grad Clip    │ + Grad Clip             │
├──────────────────────┼─────────────────────────┼─────────────────────────┤
│ Checkpoint           │ best_{pool}.pt          │ adapter_model.safetens  │
│                      │ (~440MB 全量)           │ (~4MB adapter only)     │
├──────────────────────┼─────────────────────────┼─────────────────────────┤
│ 类别不均衡处理       │ class_weight (sklearn)  │ 无专门处理              │
│                      │ → 加权 CrossEntropy     │ (预训练知识部分缓解)    │
└──────────────────────┴─────────────────────────┴─────────────────────────┘
""")


# ══════════════════════════════════════════════════════════════════════════════
# 三、主程序
# ══════════════════════════════════════════════════════════════════════════════

def main():
    print("=" * 70)
    print("   作业：对比文本分类不同训练方法效果")
    print("   数据集：TNEWS (CLUE)，15类中文新闻标题分类")
    print("=" * 70)

    results = load_results()

    if not results:
        print("\n[警告] 未找到实验结果文件，请先运行训练脚本。")
        print("  cd text_classification项目/src")
        print("  python train.py --pool cls --epochs 3")
        print("  python train.py --pool cls --epochs 3 --use_class_weight")
        print("  cd ../src_llm")
        print("  python classify_llm.py --num_samples 200")
        print("  python train_sft.py")
        print("  python evaluate_sft.py --num_samples 200")
        return

    # 各方法分析
    analyze_bert_training(results)
    analyze_llm_zero_shot(results)
    analyze_llm_sft(results)

    # 综合对比
    comprehensive_comparison(results)
    training_details_comparison()

    # 洞察与建议
    key_insights()
    practical_recommendations()

    print_header("总结")
    print("""
  本次作业通过在同一数据集(TNEWS)上对比三种文本分类方法，得出以下核心结论：

  1. 准确率排序：LLM SFT (58%) > BERT Fine-tune (56%) > LLM Zero-shot (36%)

  2. 数据效率：LLM SFT 以 9.4% 的数据量超越 BERT 全量训练，
     体现了大模型预训练知识的强大迁移能力

  3. 训练成本：三种方法各有适用场景——
     · 零成本快速验证 → Zero-shot
     · 低成本高效微调 → LoRA SFT
     · 高吞吐生产部署 → BERT Fine-tune

  4. 工程 trade-off：没有"最好"的方法，只有"最适合"的方法。
     选择取决于数据量、延迟要求、可扩展性、部署环境等约束条件。
""")


if __name__ == "__main__":
    main()
