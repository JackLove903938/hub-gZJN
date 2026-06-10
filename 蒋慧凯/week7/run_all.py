"""
run_all.py —— 一键运行 BERT+Linear、BERT+CRF、LLM API、LLM SFT 对比

使用方式：
  # 运行全部（BERT + LLM API + SFT + 报告）
  python run_all.py --all

  # 只运行 BERT+Linear
  python run_all.py --linear

  # 只运行 BERT+CRF
  python run_all.py --crf

  # 只运行 LLM API（需 DASHSCOPE_API_KEY）
  python run_all.py --llm

  # 只运行 LLM SFT（LoRA）
  python run_all.py --sft

  # 只生成对比报告
  python run_all.py --report

依赖：
  pip install -r requirements.txt
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent / "src"))
sys.path.insert(0, str(Path(__file__).parent / "src_llm"))

import torch
from transformers import BertTokenizerFast

from config import cfg
from dataset import build_label_schema, build_dataloaders
from model import build_model
from train_eval import train_model, evaluate_model
from compare_report import generate_report


def parse_args():
    parser = argparse.ArgumentParser(description="peoples_daily NER 序列标注作业")
    parser.add_argument("--all", action="store_true", help="运行全部流程")
    parser.add_argument("--linear", action="store_true", help="只运行 BERT+Linear")
    parser.add_argument("--crf", action="store_true", help="只运行 BERT+CRF")
    parser.add_argument("--llm", action="store_true", help="只运行 LLM API")
    parser.add_argument("--sft", action="store_true", help="只运行 LLM SFT (LoRA)")
    parser.add_argument("--report", action="store_true", help="只生成对比报告")
    parser.add_argument("--epochs", type=int, default=None, help="覆盖默认 epochs")
    parser.add_argument("--batch_size", type=int, default=None, help="覆盖默认 batch_size")
    parser.add_argument("--sft_epochs", type=int, default=1, help="SFT 训练 epoch 数")
    parser.add_argument("--sft_num_train", type=int, default=-1, help="SFT 训练样本数，-1=全部")
    parser.add_argument("--api_key", type=str, default=None, help="DeepSeek API Key（仅 LLM 评估用）")
    parser.add_argument("--llm_model", type=str, default="deepseek-chat", help="LLM 模型：deepseek-chat / deepseek-reasoner")
    return parser.parse_args()


def main():
    args = parse_args()

    # 若用户未指定任何参数，默认运行 BERT 两种模型
    bert_only = not any([args.all, args.linear, args.crf, args.llm, args.sft, args.report])
    if bert_only:
        args.linear = True
        args.crf = True
        args.report = True

    epochs = args.epochs if args.epochs is not None else cfg.epochs
    batch_size = args.batch_size if args.batch_size is not None else cfg.batch_size

    device = torch.device(cfg.device)
    print(f"[系统] 使用设备: {device}")
    cfg.print_config()

    # ── 准备标签体系和数据 ──
    labels, label2id, id2label = build_label_schema(cfg.entity_types)
    num_labels = len(labels)
    print(f"\nBIO 标签数：{num_labels}（O + {len(labels) - 1} 个实体标签）")

    tokenizer = BertTokenizerFast.from_pretrained(cfg.bert_path)
    train_loader, val_loader, test_loader = build_dataloaders(
        tokenizer=tokenizer,
        label2id=label2id,
        batch_size=batch_size,
        max_length=cfg.max_length,
        data_dir=cfg.data_dir,
    )

    # ── 运行 BERT+Linear ──
    if args.all or args.linear:
        try:
            print("\n" + "=" * 60)
            print("【BERT + Linear】开始训练")
            model_linear = build_model(
                use_crf=False,
                bert_path=cfg.bert_path,
                num_labels=num_labels,
                dropout=cfg.dropout,
            ).to(device)

            model_linear, info_linear = train_model(
                model=model_linear,
                train_loader=train_loader,
                val_loader=val_loader,
                device=device,
                id2label=id2label,
                use_crf=False,
                epochs=epochs,
                lr=cfg.lr,
                head_lr_mult=cfg.head_lr_mult,
                warmup_ratio=cfg.warmup_ratio,
                grad_accum=cfg.grad_accum,
                ckpt_dir=cfg.ckpt_dir,
                log_dir=cfg.log_dir,
            )

            print("\n【BERT + Linear】开始评估")
            evaluate_model(
                model=model_linear,
                loader=val_loader,
                id2label=id2label,
                device=device,
                use_crf=False,
                split_name="validation",
                log_dir=cfg.log_dir,
            )
        except Exception as e:
            print(f"[错误] BERT+Linear 运行失败: {e}")
            import traceback
            traceback.print_exc()

    # ── 运行 BERT+CRF ──
    if args.all or args.crf:
        try:
            print("\n" + "=" * 60)
            print("【BERT + CRF】开始训练")
            model_crf = build_model(
                use_crf=True,
                bert_path=cfg.bert_path,
                num_labels=num_labels,
                dropout=cfg.dropout,
            ).to(device)

            model_crf, info_crf = train_model(
                model=model_crf,
                train_loader=train_loader,
                val_loader=val_loader,
                device=device,
                id2label=id2label,
                use_crf=True,
                epochs=epochs,
                lr=cfg.lr,
                head_lr_mult=cfg.head_lr_mult,
                warmup_ratio=cfg.warmup_ratio,
                grad_accum=cfg.grad_accum,
                ckpt_dir=cfg.ckpt_dir,
                log_dir=cfg.log_dir,
            )

            print("\n【BERT + CRF】开始评估")
            evaluate_model(
                model=model_crf,
                loader=val_loader,
                id2label=id2label,
                device=device,
                use_crf=True,
                split_name="validation",
                log_dir=cfg.log_dir,
            )
        except Exception as e:
            print(f"[错误] BERT+CRF 运行失败: {e}")
            import traceback
            traceback.print_exc()

    # ── 运行 LLM API ──
    if args.all or args.llm:
        try:
            print("\n" + "=" * 60)
            print("【LLM API (DeepSeek)】开始评估")
            from llm_ner import main as llm_main
            import llm_ner
            old_argv = sys.argv
            sys_argv = ["llm_ner.py", "--n_samples", "100", "--model", args.llm_model]
            if args.api_key:
                sys_argv += ["--api_key", args.api_key]
            sys.argv = sys_argv
            llm_main()
            sys.argv = old_argv
        except EnvironmentError as e:
            print(f"[跳过] LLM API: {e}")
        except Exception as e:
            print(f"[错误] LLM API 运行失败: {e}")
            import traceback
            traceback.print_exc()

    # ── 运行 LLM SFT (LoRA) ──
    if args.all or args.sft:
        try:
            print("\n" + "=" * 60)
            print("【LLM SFT (LoRA)】开始训练")
            from train_sft import main as sft_main
            import train_sft
            old_argv = sys.argv
            sys.argv = ["train_sft.py", "--epochs", str(args.sft_epochs)]
            if args.sft_num_train > 0:
                sys.argv += ["--num_train", str(args.sft_num_train)]
            sft_main()
            sys.argv = old_argv

            print("\n【LLM SFT】开始评估")
            from evaluate_sft import main as eval_sft_main
            import evaluate_sft
            old_argv = sys.argv
            sys.argv = ["evaluate_sft.py", "--n_samples", "100"]
            eval_sft_main()
            sys.argv = old_argv
        except Exception as e:
            print(f"[错误] LLM SFT 运行失败: {e}")
            import traceback
            traceback.print_exc()

    # ── 生成对比报告 ──
    if args.all or args.report or args.linear or args.crf or args.llm or args.sft:
        try:
            print("\n" + "=" * 60)
            print("【对比报告生成】")
            generate_report(output_dir=cfg.output_dir, figure_dir=cfg.figure_dir)
        except Exception as e:
            print(f"[错误] 报告生成失败: {e}")

    print("\n[系统] 全部完成！")
    print(f"  结果目录: {cfg.output_dir}")
    print(f"  报告文件: {cfg.output_dir / 'comparison_table.md'}")
    print(f"  图表文件: {cfg.figure_dir / 'comparison_chart.png'}")


if __name__ == "__main__":
    main()
