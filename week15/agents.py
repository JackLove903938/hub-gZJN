"""
主 Agent + 并行 Subagent 编排

教学重点：
  1. 主 agent 自己是 ReAct 循环，有 2 个工具：
     - web_search：单次联网搜索（简单问题直接用）
     - dispatch_subagents：派发多个 subagent 并行调研（多侧面研究问题用）
     主 agent 根据 query 自行决定用哪个——不是固定拓扑，是 LLM 自主路由
  2. 并行优势凸显：dispatch_subagents 一次派发 N 个 subagent，
     ThreadPoolExecutor 并行跑，wall-clock ≈ max(单agent时长)，
     而非 sum——这就是 subagent 并行的核心价值
  3. 每个 subagent 也是 ReAct 循环（只 web_search 工具），
     trace 全程捕获存入 shared_state，供可视化「点节点看 ReAct 过程」

架构对应 PPT Part 6.3 的 Orchestrator-Workers 拓扑（动态：主 agent 决定派几个）。
"""

import os, time, json, logging, uuid
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Callable

from react_loop import ReActLoop
from tavily_search import tavily_search, format_search_result

logger = logging.getLogger(__name__)

MAIN_SYSTEM = """你是市场调研主分析师。你有 2 个工具：
- web_search：联网搜索一次（参数=查询词）。仅用于单一事实可一次答出的问题
- dispatch_subagents：派发多个子调研员并行调研（参数=用 | 分隔的多个子课题）

【关键决策原则】
- 只要问题涉及 2 个及以上侧面（如「市场调研」「竞品分析」「行业分析」「XX 概况/现状/趋势」等），
  必须用 dispatch_subagents 把各侧面拆给子调研员并行处理，不要自己串行 web_search 多次。
  示例："新能源汽车市场调研：销量、竞争、政策" → Action: dispatch_subagents
        Action Input: 2024年中国新能源汽车销量规模 | 主要厂商竞争格局 | 政策与补贴趋势
- 只有单一事实问题（如"2024年比亚迪销量"）才直接 web_search
- 拿到子调研结果后，综合成结构化报告

报告要求：分维度组织，每个要点带来源，末尾给结论与不确定性说明。

【示例】
Question: 2023中国咖啡市场调研：市场规模、主要品牌、消费趋势
Thought: 这是多维度市场调研（3个侧面），必须派发子调研员并行收集，不能自己串行搜索
Action: dispatch_subagents
Action Input: 2023年中国咖啡市场规模与增长 | 中国咖啡主要品牌竞争格局 | 中国咖啡消费趋势与人群
Observation: 并行调研完成：3 个子调研员...（各子课题结果）
Thought: 已收齐三个维度的并行调研结果，综合成报告
Final Answer: （分维度报告）"""


def _dispatch_subagents(action_input: str, shared_state: dict = None,
                        on_subagent_step: Callable = None,
                        on_subagent_done: Callable = None,
                        on_dispatch: Callable = None,
                        serial: bool = False,
                        subagent_roles: list = None) -> str:
    """dispatch_subagents 工具实现。
    action_input: "子课题1 | 子课题2 | ..."（管道分隔）
    派发 N 个 subagent 并行（ThreadPoolExecutor），收齐返回汇总文本。
    serial=True 时改成串行执行（eval A/B 对比用，凸显并行加速）。
    并行优势量化：wall_clock vs sum_durations。
    ⚠️ 用真实 subagent id 发 dispatch 事件（与 subagent_step 事件的 id 一致），
       否则前端拓扑节点和步骤对不上。"""
    subtopics = [s.strip() for s in action_input.split("|") if s.strip()][:6]
    if not subtopics:
        return "未解析出子课题"
    shared_state = shared_state if shared_state is not None else {}
    shared_state.setdefault("subagents", {})

    # ── 角色匹配：根据 topic 关键词匹配角色 → 返回 (角色名, 专属系统提示) ──
    def _match_role(topic: str) -> tuple:
        if subagent_roles:
            for keywords, role_name, sys_prompt in subagent_roles:
                if any(kw in topic for kw in keywords):
                    return role_name, sys_prompt
        return "子调研员", None   # 默认角色

    # 构造 (sid, subagent, subtopic) 三元组
    defs = []
    for topic in subtopics:
        sid = f"sub_{uuid.uuid4().hex[:6]}"
        role_name, role_sys = _match_role(topic)
        # agent_label = 角色名 + 短 ID，前端拓扑图显示更直观
        agent_label = f"{role_name}_{sid[-4:]}"
        sub = ReActLoop(
            agent_name=agent_label,
            tools={"web_search": (lambda q, **_: format_search_result(tavily_search(q)),
                                  "联网搜索，参数是查询词")},
            max_steps=4, model_tag="deepseek-chat(子)",
            system_prompt=role_sys)   # ← 每个 subagent 注入专属系统提示
        defs.append((sid, sub, topic))

    # 记录派发（拓扑可视化用：主→N 个子节点）—— 用真实 subagent id
    dispatch_info = {"subtopics": subtopics,
                     "subagent_ids": [sid for sid, _, _ in defs]}
    shared_state.setdefault("dispatches", []).append(dispatch_info)
    if on_dispatch:
        on_dispatch(dispatch_info)   # 真实 id，前端加的节点和后续 subagent_step 对得上

    t0 = time.time()
    results = {}
    # ── 执行：serial=False 并行(ThreadPool) / serial=True 串行(for 循环) ──
    def _run_one(sid=sid, sub=sub, topic=topic):
        return sid, sub.run(topic, on_step=(
            lambda step, sid=sid: on_subagent_step(sid, step) if on_subagent_step else None))

    if serial:
        # 串行：一个接一个，凸显并行的意义（eval A/B 对比基线）
        for sid, sub, topic in defs:
            sid, res = _run_one(sid, sub, topic)
            topic = next(t for s, _, t in defs if s == sid)
            results[sid] = (topic, res)
            shared_state["subagents"][sid] = {
                "subtopic": topic, "trace": res["trace"],
                "duration": res["duration"], "final_answer": res["final_answer"]}
            if on_subagent_done:
                on_subagent_done(sid, res["duration"], topic)
    else:
        # 并行（凸显 subagent 并行优势的核心）
        with ThreadPoolExecutor(max_workers=len(defs)) as pool:
            futs = {pool.submit(_run_one, sid, sub, topic): sid for sid, sub, topic in defs}
            for fut in as_completed(futs):
                sid, res = fut.result()
                topic = next(t for s, _, t in defs if s == sid)
                results[sid] = (topic, res)
                shared_state["subagents"][sid] = {
                    "subtopic": topic, "trace": res["trace"],
                    "duration": res["duration"], "final_answer": res["final_answer"]}
                if on_subagent_done:
                    on_subagent_done(sid, res["duration"], topic)

    wall = round(time.time() - t0, 2)
    serial_sum = round(sum(r["duration"] for _, r in results.values()), 2)
    shared_state.setdefault("parallel_stats", []).append({
        "n_subagents": len(defs), "wall_clock": wall, "serial_sum": serial_sum,
        "speedup": round(serial_sum / wall, 2) if wall else 0})

    # 汇总文本（喂回主 agent 当 Observation，每个子结果截短避免主 agent context 过长）
    parts = [f"【子课题: {topic}】(用时{r['duration']}s)\n{r['final_answer'][:500]}"
             for sid, (topic, r) in results.items()]
    stats = shared_state["parallel_stats"][-1]
    return (f"并行调研完成：{len(defs)} 个子调研员，wall-clock {wall}s "
            f"(串行需 {serial_sum}s，加速 {stats['speedup']}×)\n\n" + "\n\n".join(parts))


def run_research(question: str, on_main_step: Callable = None,
                 on_subagent_step: Callable = None,
                 on_subagent_done: Callable = None,
                 on_dispatch: Callable = None,
                 serial: bool = False,
                 system_prompt: str = None,
                 subagent_roles: list = None) -> dict:
    """执行一次调研。返回 {final_answer, main_trace, subagents, parallel_stats}。
    system_prompt: 主 Agent 自定义系统提示（不传用默认 MAIN_SYSTEM）。
    subagent_roles: 子代理角色映射 [(关键词列表, 角色名, 系统提示), ...]。"""
    shared_state = {"subagents": {}, "dispatches": [], "parallel_stats": []}

    def dispatch_tool(action_input, shared_state=None):
        info = shared_state or {}
        return _dispatch_subagents(action_input, shared_state=info,
                                   on_subagent_step=on_subagent_step,
                                   on_subagent_done=on_subagent_done,
                                   on_dispatch=on_dispatch,
                                   serial=serial,
                                   subagent_roles=subagent_roles)

    main = ReActLoop(
        agent_name="main",
        tools={
            "web_search": (lambda q, **_: format_search_result(tavily_search(q)),
                           "联网搜索一次，参数=查询词"),
            "dispatch_subagents": (dispatch_tool,
                                   "派发多个子调研员并行调研，参数=用 | 分隔的多个子课题"),
        },
        max_steps=8,
        model_tag="deepseek-chat(主)",
        system_prompt=system_prompt or MAIN_SYSTEM,
    )
    # 把 shared_state 注入主 agent run
    result = main.run(question, on_step=on_main_step, shared_state=shared_state)
    return {
        "final_answer": result["final_answer"],
        "main_trace": result["trace"],
        "subagents": shared_state["subagents"],
        "parallel_stats": shared_state["parallel_stats"],
        "dispatches": shared_state["dispatches"],
    }


# ═══════════════════════════════════════════════════════════
# 房地产楼盘三维分析 Agent
# 3 个 SubAgent 并行：楼盘本体 / 配套规划 / 市场政策
# ═══════════════════════════════════════════════════════════

# ── 主 Agent 系统提示（引导 LLM 按 3 个维度拆任务） ──
REAL_ESTATE_MAIN_SYSTEM = """你是房地产楼盘分析师。你有 2 个工具：
- web_search：联网搜索一次（参数=查询词）。仅用于单一事实
- dispatch_subagents：派发 3 个子代理并行调研（参数=用 | 分隔的 3 个调研维度）

【关键决策原则】
- 只要用户询问具体楼盘/项目，必须用 dispatch_subagents 派发 3 个子代理
- 固定派发以下 3 个维度（缺一不可）：
  1. 楼盘本体分析 | 2. 配套与规划分析 | 3. 市场与政策分析
- 每个子代理的调研任务必须包含楼盘名称/地段，确保搜到精准信息
- 示例：
  Question: 分析深圳华润城润府这个楼盘
  Action Input: 华润城润府楼盘本体参数户型价格 | 华润城润府周边配套交通学校规划 | 深圳南山区房价走势房贷政策
- 拿到 3 个子报告后，综合成结构化楼盘分析报告

报告要求：
- 分三个维度（楼盘本体 / 配套规划 / 市场政策）
- 每个维度列关键数据、带来源
- 末尾给【综合评估】和【购房建议】"""

# ── SubAgent-A：楼盘本体子代理 ──
PROPERTY_AGENT_SYSTEM = """你是楼盘本体调研员。专注分析指定楼盘的基本面。

重点关注：
- 开发商背景、项目规模（占地/建面/栋数/户数）
- 户型产品（面积段、户型配比、得房率、层高）
- 售价与价格走势（开盘价、均价、近期成交均价）
- 建筑参数（容积率、绿化率、车位配比）
- 交付时间、物业费、物业公司

搜索策略：
- 先用楼盘全名搜基本信息（楼盘百科/新房网）
- 再搜户型图/实测/业主评价
- 价格数据参考链家/贝壳/安居客等成交平台

输出要求：分点列出数据要点，标注信息来源。如某项数据搜不到，明确说明。"""

# ── SubAgent-B：配套与规划子代理 ──
AMENITIES_AGENT_SYSTEM = """你是配套与规划调研员。专注分析指定楼盘的周边配套和未来规划。

重点关注：
- 交通：最近地铁线路/站点距离、公交枢纽、高速出入口、通勤时间
- 教育：对口公立学校、学区等级、私立名校距离
- 商业：周边商圈、购物中心/超市距离与档次
- 医疗：附近三甲医院距离
- 环境：公园绿地、噪音源（高架/高铁/机场）、不利因素
- 未来规划：附近地铁延伸线、拆迁/旧改、新建学校/医院规划

搜索策略：
- 地图类搜 "楼盘名 + 地铁/学校/医院"
- 规划类搜 "楼盘所在区域 + 规划/拆迁/地铁"
- 对口学校查当地教育局公告

输出要求：分类列出配套，标注距离/等级/来源。特别标注不利因素。"""

# ── SubAgent-C：市场与政策子代理 ──
MARKET_AGENT_SYSTEM = """你是市场与政策调研员。专注分析指定楼盘所在区域的市场趋势和政策环境。

重点关注：
- 区域房价：所在行政区近 1 年房价走势、同片区楼盘对比
- 成交数据：片区月度成交量、库存去化周期（月）
- 房贷政策：当前房贷利率（首套/二套）、首付比例、LPR 变化
- 限购政策：所在城市限购限贷、落户政策、人才购房补贴
- 土地市场：近期同区域土拍成交价/楼面价，判断未来供应
- 趋势预判：该区域短期（半年）/中期（2 年）走势判断

搜索策略：
- "城市名 + 区域 + 房价走势 2024"
- "城市名 + 房贷利率 限购 2024"
- "城市名 + 土拍 成交 楼面价"

输出要求：分点列数据，带来源。趋势判断需说明依据。"""

# ── 角色映射：topic 关键词 → 角色名 + 专属系统提示 ──
ESTATE_SUBAGENT_ROLES = [
    # SubAgent-A：匹配楼盘本体相关 topic
    (["楼盘", "本体", "参数", "户型", "项目", "规模", "开发商"],
     "楼盘本体子代理", PROPERTY_AGENT_SYSTEM),
    # SubAgent-B：匹配配套规划相关 topic
    (["配套", "规划", "周边", "交通", "学校", "环境", "商业", "医疗"],
     "配套规划子代理", AMENITIES_AGENT_SYSTEM),
    # SubAgent-C：匹配市场政策相关 topic
    (["市场", "政策", "价格", "走势", "成交", "房贷", "均价", "限购", "房价"],
     "市场政策子代理", MARKET_AGENT_SYSTEM),
]


def run_real_estate_project(question: str,
                            on_main_step: Callable = None,
                            on_subagent_step: Callable = None,
                            on_subagent_done: Callable = None,
                            on_dispatch: Callable = None,
                            serial: bool = False) -> dict:
    """房地产楼盘三维分析 Agent。流程：
    1. 主 agent 发一个 '派发' 步骤（可视化）
    2. 直接强制派发 3 个 subagent 并行调研（跳过 LLM 决策）
    3. 主 agent 发 '收齐结果' 步骤（可视化）
    4. 用 LLM 综合子报告生成最终结构化报告
    5. 主 agent 发 'Final Answer' 步骤（可视化）
    """
    from llm_client import llm_chat

    shared_state = {"subagents": {}, "dispatches": [], "parallel_stats": []}

    # ── 1. 主 agent 发决策步骤（可视化） ──
    subtopics = [
        f"{question} 楼盘本体参数户型价格",
        f"{question} 周边配套交通学校规划",
        f"{question} 所在区域房价走势房贷政策",
    ]
    action_input = " | ".join(subtopics)

    if on_main_step:
        on_main_step({"idx": 0, "agent": "main",
                      "thought": f"收到楼盘分析请求，涉及楼盘本体、配套规划、市场政策 3 个维度，派发 3 个子代理并行调研。",
                      "action": "dispatch_subagents",
                      "action_input": action_input,
                      "observation": None, "final": False})

    # ── 2. 强制派发 3 个 subagent ──
    dispatch_obs = _dispatch_subagents(
        action_input,
        shared_state=shared_state,
        on_subagent_step=on_subagent_step,
        on_subagent_done=on_subagent_done,
        on_dispatch=on_dispatch,
        serial=serial,
        subagent_roles=ESTATE_SUBAGENT_ROLES,
    )

    # ── 3. 主 agent 展示派发结果（可视化） ──
    if on_main_step:
        on_main_step({"idx": 0, "agent": "main",
                      "thought": f"3 个子代理已完成并行调研，现在综合结果生成报告。",
                      "action": "dispatch_subagents",
                      "action_input": action_input,
                      "observation": dispatch_obs[:400],
                      "final": False})

    # ── 4. LLM 综合子报告生成最终答案 ──
    try:
        sub_reports_text = "\n\n".join(
            f"=== {info['subtopic']} ===\n{info['final_answer']}"
            for sid, info in shared_state["subagents"].items()
        )

        summarize_prompt = f"""你是楼盘分析师。请根据以下 3 个子代理的并行调研结果，
综合成一份简洁的结构化楼盘分析报告。

报告格式（必须按此格式输出）：
## 一、楼盘本体分析
（户型、价格、参数等关键数据，3-5 条）

## 二、配套与规划分析
（交通、教育、商业、环境等，3-5 条）

## 三、市场与政策分析
（房价走势、房贷政策、趋势判断等，3-5 条）

## 四、综合评估与购房建议
（总结优势劣势，给出明确购房建议）

---
用户问题：{question}

子代理调研结果：

{sub_reports_text}"""

        llm_out = llm_chat(
            "你是专业的房地产分析师，擅长综合多源信息撰写简洁的结构化报告。",
            summarize_prompt,
            temperature=0.0, max_tokens=4096,
        )

        final_answer = llm_out.strip()
    except Exception as e:
        # 综合失败时，把 3 个子报告直接拼接返回，保证有结果
        logger.error(f"综合报告生成失败: {e}")
        sub_reports_text = "\n\n".join(
            f"## {info['subtopic']}\n\n{info['final_answer']}"
            for sid, info in shared_state["subagents"].items()
        )
        final_answer = f"（综合报告生成失败，以下为各子代理的原始调研结果）\n\n{sub_reports_text}"

    # ── 5. 主 agent 发最终报告步骤（可视化） ──
    if on_main_step:
        on_main_step({"idx": 1, "agent": "main",
                      "thought": "3 个子代理结果已收齐，综合成结构化楼盘分析报告。",
                      "action": "Final Answer",
                      "action_input": final_answer[:300],
                      "observation": None, "final": True})

    return {
        "final_answer": final_answer,
        "main_trace": [
            {"idx": 0, "agent": "main",
             "thought": "派发 3 个子代理并行调研。",
             "action": "dispatch_subagents",
             "action_input": action_input,
             "observation": dispatch_obs[:400],
             "final": False},
            {"idx": 1, "agent": "main",
             "thought": "综合子报告生成最终答案。",
             "action": "Final Answer",
             "action_input": final_answer[:300],
             "observation": None, "final": True},
        ],
        "subagents": shared_state["subagents"],
        "parallel_stats": shared_state["parallel_stats"],
        "dispatches": shared_state["dispatches"],
    }


if __name__ == "__main__":
    import logging as _l
    _l.basicConfig(level=_l.WARNING)

    # ── 演示：房地产楼盘三维并行分析 ──
    print("=" * 60)
    print("【房地产楼盘三维分析 Agent】")
    q = "分析深圳市南山区华润城润府这个楼盘"
    r = run_real_estate_project(q)
    print(f"主 agent 动作: {[s['action'] for s in r['main_trace']]}")
    print(f"派发次数: {len(r['dispatches'])} | subagent 数: {len(r['subagents'])}")
    for sid, info in r["subagents"].items():
        agent_label = info['trace'][0]['agent'] if info['trace'] else '?'
        print(f"  {sid} [{agent_label}]: "
              f"{info['subtopic'][:35]}... → {info['final_answer'][:80]}")
    print(f"\n并行统计: {r['parallel_stats']}")
    print(f"\n{'='*60}")
    print(f"综合报告:\n{r['final_answer'][:800]}")
