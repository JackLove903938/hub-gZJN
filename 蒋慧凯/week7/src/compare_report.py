"""
对比报告生成：BERT+Linear vs BERT+CRF vs LLM API vs LLM SFT

生成内容：
  1. Markdown 表格（控制台打印 + 文件保存）
  2. 可视化柱状图（F1 / Precision / Recall 对比）

使用方式：
  from compare_report import generate_report
  generate_report()
"""

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

# 设置中文字体（优先使用系统常见字体）
plt.rcParams["font.sans-serif"] = ["SimHei", "Microsoft YaHei", "Arial Unicode MS", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False


def generate_report(output_dir: Path = None, figure_dir: Path = None):
    """读取各模型的评估结果，生成 Markdown 报告和可视化图表。"""
    if output_dir is None:
        output_dir = Path(__file__).parent.parent / "outputs"
    if figure_dir is None:
        figure_dir = output_dir / "figures"

    log_dir = output_dir / "logs"
    figure_dir.mkdir(parents=True, exist_ok=True)

    results = {}

    # BERT+Linear
    path = log_dir / "eval_linear_validation.json"
    if path.exists():
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        results["BERT+Linear"] = {
            "precision": data.get("precision", 0),
            "recall": data.get("recall", 0),
            "f1": data.get("f1", 0),
            "illegal": data.get("illegal_stats", {}).get("total_illegal", "N/A"),
        }

    # BERT+CRF
    path = log_dir / "eval_crf_validation.json"
    if path.exists():
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        results["BERT+CRF"] = {
            "precision": data.get("precision", 0),
            "recall": data.get("recall", 0),
            "f1": data.get("f1", 0),
            "illegal": data.get("illegal_stats", {}).get("total_illegal", "N/A"),
        }

    # LLM API (few-shot)
    path = log_dir / "eval_llm.json"
    if path.exists():
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        fs = data.get("few_shot", {})
        results["LLM API few-shot"] = {
            "precision": fs.get("precision", 0),
            "recall": fs.get("recall", 0),
            "f1": fs.get("f1", 0),
            "illegal": "N/A",
        }

    # LLM SFT
    path = log_dir / "eval_sft.json"
    if path.exists():
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        m = data.get("metrics", {})
        results["LLM SFT (LoRA)"] = {
            "precision": m.get("precision", 0),
            "recall": m.get("recall", 0),
            "f1": m.get("f1", 0),
            "illegal": "N/A",
        }

    if not results:
        print("[错误] 没有任何评估结果，无法生成报告。请先运行训练和评估。")
        return

    # ── Markdown 报告 ──
    lines = []
    lines.append("# Week7 作业：序列标注模型对比报告（peoples_daily）\n")
    lines.append("## 评估结果对比\n")
    lines.append("| 模型 | Precision | Recall | F1 | 非法序列数 |")
    lines.append("|------|-----------|--------|----|-----------|")

    for name, data in results.items():
        illegal = data.get("illegal", "N/A")
        lines.append(
            f"| {name} | {data['precision']:.4f} | {data['recall']:.4f} | "
            f"{data['f1']:.4f} | {illegal} |"
        )

    lines.append("")
    lines.append("## 核心结论\n")

    if "BERT+Linear" in results and "BERT+CRF" in results:
        linear_f1 = results["BERT+Linear"]["f1"]
        crf_f1 = results["BERT+CRF"]["f1"]
        diff = crf_f1 - linear_f1
        lines.append(f"1. **F1 对比**：BERT+CRF（{crf_f1:.4f}）vs BERT+Linear（{linear_f1:.4f}），差异 {diff:+.4f}")
        linear_illegal = results["BERT+Linear"].get("illegal", 0)
        if isinstance(linear_illegal, int):
            lines.append(f"2. **非法序列**：BERT+Linear 产生 {linear_illegal} 条非法序列，BERT+CRF 保证 0 条。")
        lines.append("3. **CRF 的价值**：通过转移矩阵和 Viterbi 解码，CRF 在数学上保证输出合法 BIO 序列。")

    if "LLM API few-shot" in results:
        lines.append(f"4. **LLM API**：few-shot F1 = {results['LLM API few-shot']['f1']:.4f}，"
                     "无需训练即可快速验证，但依赖 API 成本和延迟。")

    if "LLM SFT (LoRA)" in results:
        lines.append(f"5. **LLM SFT**：LoRA 微调后 F1 = {results['LLM SFT (LoRA)']['f1']:.4f}，"
                     "可本地部署，但训练时间和显存要求高于 BERT。")

    report_path = output_dir / "comparison_table.md"
    with open(report_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    print(f"[报告] Markdown 报告已保存 → {report_path}")

    # ── 可视化图表 ──
    if len(results) >= 2:
        names = list(results.keys())
        metrics = ["precision", "recall", "f1"]
        values = {m: [results[n][m] for n in names] for m in metrics}

        x = np.arange(len(metrics))
        width = 0.8 / len(names)

        fig, ax = plt.subplots(figsize=(10, 6))
        for i, name in enumerate(names):
            offset = width * (i - (len(names) - 1) / 2)
            bars = ax.bar(x + offset, [values[m][i] for m in metrics], width, label=name)
            for bar in bars:
                height = bar.get_height()
                ax.annotate(f"{height:.3f}",
                            xy=(bar.get_x() + bar.get_width() / 2, height),
                            xytext=(0, 3), textcoords="offset points",
                            ha="center", va="bottom", fontsize=8)

        ax.set_ylabel("Score")
        ax.set_title("序列标注模型对比（peoples_daily NER）")
        ax.set_xticks(x)
        ax.set_xticklabels([m.capitalize() for m in metrics])
        ax.set_ylim(0, 1.1)
        ax.legend()
        ax.grid(axis="y", linestyle="--", alpha=0.5)

        fig.tight_layout()
        chart_path = figure_dir / "comparison_chart.png"
        fig.savefig(chart_path, dpi=150)
        plt.close(fig)
        print(f"[图表] 对比图已保存 → {chart_path}")


if __name__ == "__main__":
    generate_report()
