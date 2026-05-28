"""
check_env.py —— 环境诊断脚本
快速检查：路径、模型、依赖是否正常
"""

import sys
from pathlib import Path

print("=" * 50)
print("【环境诊断】")
print("=" * 50)

# 1. 检查当前工作目录
print(f"\n[1] 当前工作目录: {Path.cwd()}")
print(f"     期望路径应包含 'homework'")

# 2. 检查项目路径解析
sys.path.insert(0, str(Path(__file__).parent / "src"))
try:
    from config import cfg
    print(f"\n[2] 项目根目录: {cfg.project_root}")
    print(f"     数据目录: {cfg.data_dir}")
    print(f"     输出目录: {cfg.output_dir}")
    print(f"     BERT模型: {cfg.bert_model_path}")
    print(f"     Qwen模型: {cfg.qwen_model_path}")
except Exception as e:
    print(f"\n[2] 配置加载失败: {e}")

# 3. 检查依赖
print("\n[3] 依赖检查:")
deps = ["torch", "transformers", "datasets", "sklearn", "peft"]
for dep in deps:
    try:
        if dep == "sklearn":
            __import__("sklearn")
        else:
            __import__(dep)
        print(f"     ✓ {dep}")
    except ImportError:
        print(f"     ✗ {dep} (未安装)")

# 4. 检查数据是否存在
print("\n[4] 数据检查:")
data_files = ["train.json", "val.json", "label_map.json"]
for f in data_files:
    p = Path("data") / f
    if p.exists():
        print(f"     ✓ {f}")
    else:
        print(f"     ✗ {f} (缺失)")

# 5. 检查模型路径是否存在（本地）
print("\n[5] 模型路径检查:")
for name, path in [("BERT", cfg.bert_model_path), ("Qwen", cfg.qwen_model_path)]:
    if Path(path).exists():
        print(f"     ✓ {name}: 本地路径存在")
    else:
        print(f"     → {name}: 将自动从 HuggingFace 下载 '{path}'")

# 6. 设备检查
print(f"\n[6] 设备: {cfg.device}")

try:
    import torch
    print(f"     PyTorch版本: {torch.__version__}")
    print(f"     CUDA可用: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"     CUDA设备: {torch.cuda.get_device_name(0)}")
except:
    pass

print("\n" + "=" * 50)
