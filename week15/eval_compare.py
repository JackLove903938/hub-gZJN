"""
Parallel vs Serial 量化对比（凸显 subagent 并行优势）

教学重点：
  同一组调研问题，主 agent 派发的 subagent 分别用「并行(ThreadPool)」和
  「串行(for 循环)」两种方式执行，对比 wall-clock，量化并行加速。

  并行的意义不是少做事，而是把 N 个独立子任务的墙钟时间从 sum 压到 max。
  本项目的 dispatch_subagents 用 ThreadPoolExecutor 实现并行，
  serial=True 时退化为串行（eval 基线）。

使用方式：
  python eval_compare.py            # 默认 4 题，parallel vs serial
  python eval_compare.py --limit 2  # 快速版
"""
import os, sys, time, json, logging, argparse
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
logging.basicConfig(level=logging.WARNING)
logger = logging.getLogger(__name__)

EVAL_QUESTIONS = [
    "2024年中国新能源汽车市场调研：销量规模、主要厂商竞争格局、政策趋势",
    "中国咖啡市场调研：市场规模、主要品牌、消费趋势",
    "中国扫地机器人市场调研：市场规模、主要品牌、技术趋势",
    "中国宠物经济调研：市场规模、主要品类、消费趋势",
]

# —— 房地产楼盘三维分析测试题 ——
ESTATE_PROJECT_QUESTIONS = [
    "分析深圳市南山区华润城润府这个楼盘",
    "分析上海市浦东新区汤臣一品这个楼盘",
    "分析广州市天河区珠江新城保利天悦",
    "分析杭州市西湖区融创河滨之城",
]


def run_one(question, serial, system_prompt=None, subagent_roles=None):
    """跑一次调研，返回 (wall_clock, n_subagents, dispatched)。
    serial=True/False 控制子 agent 执行方式。"""
    import agents
    t0 = time.time()
    r = agents.run_research(question, serial=serial,
                            system_prompt=system_prompt,
                            subagent_roles=subagent_roles)
    wall = time.time() - t0
    ps = r["parallel_stats"][-1] if r["parallel_stats"] else None
    return {
        "wall": round(wall, 2),
        "n_subagents": ps["n_subagents"] if ps else 0,
        "dispatch_wall": ps["wall_clock"] if ps else 0,
        "serial_sum": ps["serial_sum"] if ps else 0,
        "speedup": ps["speedup"] if ps else 0,
        "dispatched": len(r["dispatches"]) > 0,
    }


def main():
    parser = argparse.ArgumentParser(description="parallel vs serial 对比")
    parser.add_argument("--limit", type=int, default=0, help="题目数量（0=全部）")
    parser.add_argument("--domain", type=str, default="market",
                        choices=["market", "estate_project"],
                        help="领域：market=市场调研（默认）, estate_project=楼盘三维分析")
    args = parser.parse_args()

    import agents
    if args.domain == "estate_project":
        qs = ESTATE_PROJECT_QUESTIONS[:args.limit] if args.limit else ESTATE_PROJECT_QUESTIONS
        sys_prompt = agents.REAL_ESTATE_MAIN_SYSTEM
        subagent_roles = agents.ESTATE_SUBAGENT_ROLES
        domain_label = "楼盘三维分析"
    else:
        qs = EVAL_QUESTIONS[:args.limit] if args.limit else EVAL_QUESTIONS
        sys_prompt = None
        subagent_roles = None
        domain_label = "市场调研"

    print(f"\n{'='*60}\n  领域: {domain_label} | 题目数: {len(qs)}\n{'='*60}")

    results = []
    for i, q in enumerate(qs):
        logger.warning(f"[{i+1}/{len(qs)}] {q}")
        p = run_one(q, serial=False, system_prompt=sys_prompt, subagent_roles=subagent_roles)
        s = run_one(q, serial=True, system_prompt=sys_prompt, subagent_roles=subagent_roles)
        results.append({"question": q, "parallel": p, "serial": s})
        print(f"  {q[:32]:<34} 并行 {p['wall']}s vs 串行 {s['wall']}s "
              f"(subagent {p['n_subagents']}, 加速 {p['speedup']}×)")

    avg_p = sum(r["parallel"]["wall"] for r in results) / len(results)
    avg_s = sum(r["serial"]["wall"] for r in results) / len(results)
    avg_spd = sum(r["parallel"]["speedup"] for r in results) / len(results)

    print(f"\n{'='*60}\nParallel vs Serial 对比（{len(results)} 题）\n{'='*60}")
    print(f"{'指标':<16} {'并行(ThreadPool)':<18} {'串行(for循环)':<18}")
    print(f"{'平均墙钟(s)':<16} {avg_p:<18.2f} {avg_s:<18.2f}")
    print(f"{'平均加速':<16} {avg_spd:<18.2f}× {'—':<18}")
    print(f"\n结论：subagent 并行把 N 个独立子任务的墙钟从 sum 压到 ≈max，"
          f"平均加速 {avg_spd:.2f}×")

    out = {"summary": {"avg_parallel_s": round(avg_p, 2),
                        "avg_serial_s": round(avg_s, 2),
                        "avg_speedup": round(avg_spd, 2)},
           "details": results}
    (Path(__file__).parent.parent / "outputs" / "eval_compare.json").write_text(
        json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
