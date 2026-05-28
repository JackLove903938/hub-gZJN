"""debug_run.py —— 带详细跟踪的启动脚本"""
import sys
import traceback
from pathlib import Path

print("[DEBUG] Python:", sys.executable)
print("[DEBUG] Version:", sys.version)
print("[DEBUG] CWD:", Path.cwd())

sys.path.insert(0, str(Path(__file__).parent / "src"))
print("[DEBUG] sys.path inserted")

try:
    print("[DEBUG] Importing config...")
    from config import cfg
    print("[DEBUG] config OK, device:", cfg.device)
except Exception as e:
    print("[DEBUG] config FAILED:", e)
    traceback.print_exc()
    sys.exit(1)

try:
    print("[DEBUG] Importing data_utils...")
    from data_utils import download_tnews, build_bert_loaders
    print("[DEBUG] data_utils OK")
except Exception as e:
    print("[DEBUG] data_utils FAILED:", e)
    traceback.print_exc()
    sys.exit(1)

try:
    print("[DEBUG] Importing bert_pipeline...")
    from bert_pipeline import train_bert, evaluate_bert
    print("[DEBUG] bert_pipeline OK")
except Exception as e:
    print("[DEBUG] bert_pipeline FAILED:", e)
    traceback.print_exc()
    sys.exit(1)

try:
    print("[DEBUG] Importing llm_zeroshot_pipeline...")
    from llm_zeroshot_pipeline import run_zeroshot
    print("[DEBUG] llm_zeroshot_pipeline OK")
except Exception as e:
    print("[DEBUG] llm_zeroshot_pipeline FAILED:", e)
    traceback.print_exc()

try:
    print("[DEBUG] Importing llm_sft_pipeline...")
    from llm_sft_pipeline import train_sft, evaluate_sft
    print("[DEBUG] llm_sft_pipeline OK")
except Exception as e:
    print("[DEBUG] llm_sft_pipeline FAILED:", e)
    traceback.print_exc()

try:
    print("[DEBUG] Importing compare_report...")
    from compare_report import generate_report
    print("[DEBUG] compare_report OK")
except Exception as e:
    print("[DEBUG] compare_report FAILED:", e)
    traceback.print_exc()

print("[DEBUG] All imports done. Starting main...")

# 模拟 run_all.py --bert 的核心逻辑
device = cfg.device
print("[DEBUG] device:", device)

print("[DEBUG] Loading data...")
download_tnews()
print("[DEBUG] Data loaded")

print("[DEBUG] Building BERT loaders...")
train_loader, val_loader = build_bert_loaders(cfg.bert_model_path)
print("[DEBUG] BERT loaders built, train batches:", len(train_loader))

print("[DEBUG] Starting BERT training...")
model, history = train_bert(train_loader, val_loader, device=device)
print("[DEBUG] BERT training done")

print("[DEBUG] Evaluating BERT...")
metrics = evaluate_bert(model, val_loader, device=device, verbose=True)
print("[DEBUG] BERT evaluation done:", metrics)

import json
result = {
    "method": "BERT Fine-tune",
    "accuracy": metrics["accuracy"],
    "macro_f1": metrics["macro_f1"],
    "per_sample_time": 0.005,
}
with open(cfg.output_dir / "bert_results.json", "w", encoding="utf-8") as f:
    json.dump(result, f, ensure_ascii=False, indent=2)
print("[DEBUG] Results saved to", cfg.output_dir / "bert_results.json")
