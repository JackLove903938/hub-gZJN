import json
from pathlib import Path
import statistics

ROOT = Path(__file__).parent.parent
DATA_DIR = ROOT / 'data' / "peoples_daily"


def compute_stats(lengths):
    lengths_sorted = sorted(lengths)
    n = len(lengths_sorted)
    # 中位数
    median = statistics.median(lengths_sorted)

    # 累计分布比例（<=阈值）
    def proportion(threshold):
        count = sum(1 for l in lengths_sorted if l <= threshold)
        return count / n * 100

    # 分位数（给定百分比，返回对应的长度值）
    def percentile(p):
        # p 是 0-100 之间的数
        idx = int(p / 100 * (n - 1))
        return lengths_sorted[idx]

    print(f"样本总数: {n}")
    print(f"最大长度: {lengths_sorted[-1]}")
    print(f"中位数 (50%分位数): {median}")
    print("\n--- 累积分布 (长度 <= 阈值) ---")
    for thresh in [90, 95, 98]:
        print(f"长度 ≤ {thresh}: {proportion(thresh):.2f}%")
    print("\n--- 分位数 (百分比对应的长度值) ---")
    for p in [90, 95, 98]:
        print(f"{p}% 分位数: {percentile(p)}")


if __name__ == "__main__":
    length_data = []
    for split in ["train.json", "test.json", "validation.json"]:
        with open(DATA_DIR / split) as f:
            for data in json.load(f):
                length_data.append(len(data["tokens"]))

    compute_stats(length_data)
