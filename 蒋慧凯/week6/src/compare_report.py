"""
compare_report.py —— 三种方法统一对比与可视化

功能：
  1. 读取三种方法的评估结果（JSON）
  2. 生成对比表格（终端输出 + Markdown 文件）
  3. 绘制柱状图对比（准确率、推理速度、训练成本）

输出文件：
  outputs/comparison_table.md
  outputs/comparison_chart.png
"""

import json
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import numpy as np

from config import cfg

matplotlib.rcParams["axes.unicode_minus"] = False

# ── 中文字体自动检测 ──
def _find_chinese_font():
    candidates = ["SimHei", "Microsoft YaHei", "PingFang SC", "Noto Sans CJK SC", "WenQuanYi Micro Hei"]
    available = {f.name for f in fm.fontManager.ttflist}
    for name in candidates:
        if name in available:
            return name
    return None

_CN_FONT = _find_chinese_font()
if _CN_FONT:
    plt.rcParams["font.family"] = _CN_FONT


def load_result(filename):
    """加载单个方法的评估结果 JSON。"""
    path = cfg.output_dir / filename
    if not path.exists():
        return None
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def build_comparison_table(results):
    """
    构建对比表格（Markdown 格式）。
    
    results: dict, key 为方法名，value 为评估结果 dict
    """
    lines = []
    lines.append("# 文本分类方法对比报告\n")
    lines.append("## 1. 核心指标对比\n")
    lines.append("| 方法 | 准确率 | 训练成本 | 推理速度 | 数据需求 | 输出可控性 |")
    lines.append("|------|--------|----------|----------|----------|------------|")

    # BERT
    r = results.get("bert")
    if r:
        lines.append(f"| BERT Fine-tune | {r.get('accuracy', 0):.4f} | 中等（CPU/GPU 均可） | ~5ms/条 | 需标注（53K） | 高（固定类别） |")

    # Zero-Shot
    r = results.get("zeroshot")
    if r:
        lines.append(f"| LLM Zero-Shot | {r.get('accuracy', 0):.4f} | 无（零训练） | ~{r.get('per_sample_time', 0):.2f}s/条 | 零标注 | 低（需解析兜底） |")

    # SFT
    r = results.get("sft")
    if r:
        lines.append(f"| LLM SFT (LoRA) | {r.get('accuracy', 0):.4f} | 低（0.22% 参数） | ~{r.get('per_sample_time', 0):.2f}s/条 | 需标注（5K 即可） | 中（需解析） |")

    lines.append("\n## 2. 方法选型建议\n")
    lines.append("- **延迟敏感 + 有标注** → BERT Fine-tune（<20ms，100% 可控）")
    lines.append("- **零标注 + 快速验证** → LLM Zero-Shot（分钟级上线）")
    lines.append("- **大模型能力 + 少量标注** → LLM SFT + LoRA（单卡可训）")
    lines.append("- **类别频繁变动** → LLM Zero-Shot / Few-Shot（改 prompt 即可）")
    lines.append("\n## 3. 关键洞察\n")
    lines.append("1. BERT 用 53K 数据达到 ~57-62% 准确率，是工程落地的首选（延迟低、成本低、输出稳）。")
    lines.append("2. LLM Zero-Shot 无需训练即可达到可用水平，适合快速原型和零标注启动。")
    lines.append("3. LLM SFT (LoRA) 仅用 5K 数据（BERT 的 9.4%）即可逼近 BERT 效果，体现大模型预训练知识的迁移效率。")
    lines.append("4. 生成式分类（LLM）存在'无法解析'风险，生产系统需加兜底逻辑；判别式分类（BERT）无此问题。")

    return "\n".join(lines)


def plot_comparison_chart(results, save_path):
    """
    绘制三种方法的对比柱状图。
    左图：准确率对比
    右图：单条推理时间对比（对数坐标）
    """
    methods = []
    accs = []
    times = []
    colors = []

    color_map = {"bert": "#4C72B0", "zeroshot": "#DD8452", "sft": "#55A868"}

    for key, label in [("bert", "BERT Fine-tune"), ("zeroshot", "LLM Zero-Shot"), ("sft", "LLM SFT (LoRA)")]:
        r = results.get(key)
        if r:
            methods.append(label)
            accs.append(r.get("accuracy", 0) * 100)  # 转为百分比
            times.append(r.get("per_sample_time", 0))
            colors.append(color_map[key])

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # 左图：准确率
    bars1 = axes[0].bar(methods, accs, color=colors, edgecolor="white", linewidth=0.5)
    axes[0].set_ylim(0, 100)
    axes[0].set_ylabel("准确率 (%)")
    axes[0].set_title("三种方法准确率对比")
    for bar, acc in zip(bars1, accs):
        axes[0].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
                     f"{acc:.1f}%", ha="center", va="bottom", fontsize=11)

    # 右图：推理速度（对数坐标，BERT ms 级 vs LLM s 级差距太大）
    bars2 = axes[1].bar(methods, times, color=colors, edgecolor="white", linewidth=0.5)
    axes[1].set_yscale("log")
    axes[1].set_ylabel("单条推理时间 (秒, 对数坐标)")
    axes[1].set_title("三种方法推理速度对比")
    for bar, t in zip(bars2, times):
        unit = "ms" if t < 0.1 else "s"
        val = t * 1000 if t < 0.1 else t
        axes[1].text(bar.get_x() + bar.get_width()/2, bar.get_height() * 1.2,
                     f"{val:.1f}{unit}", ha="center", va="bottom", fontsize=11)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"[图表] 对比图已保存 → {save_path}")


def generate_report():
    """
    主入口：加载三种方法结果，生成对比表格和图表。
    """
    print("\n" + "=" * 50)
    print("【对比报告生成】")
    print("=" * 50)

    results = {
        "bert": load_result("bert_results.json"),
        "zeroshot": load_result("zeroshot_results.json"),
        "sft": load_result("sft_results.json"),
    }

    # 检查哪些方法已完成
    completed = [k for k, v in results.items() if v is not None]
    print(f"[报告] 已完成的方法: {completed}")

    if not completed:
        print("[警告] 没有找到任何评估结果，请先运行各方法！")
        return

    # 生成 Markdown 报告
    md_content = build_comparison_table(results)
    md_path = cfg.output_dir / "comparison_table.md"
    with open(md_path, "w", encoding="utf-8") as f:
        f.write(md_content)
    print(f"[报告] Markdown 报告已保存 → {md_path}")

    # 生成可视化图表
    chart_path = cfg.figure_dir / "comparison_chart.png"
    plot_comparison_chart(results, chart_path)

    # 终端打印摘要
    print("\n" + "=" * 50)
    print("【对比摘要】")
    print("=" * 50)
    for key, label in [("bert", "BERT"), ("zeroshot", "Zero-Shot"), ("sft", "SFT(LoRA)")]:
        r = results.get(key)
        if r:
            acc = r.get("accuracy", 0)
            t = r.get("per_sample_time", 0)
            print(f"  {label:12s}: accuracy={acc:.4f}, 均速={t:.3f}s/条")
    print("=" * 50)
