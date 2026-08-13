---
name: Inquiry-dtc
description: >-
  车载故障码（DTC）诊断查询：输入 DTC 故障码，自动生成暗色科技风 HTML 诊断报告（含故障解释、可能诱因、排查步骤、维修建议、相关故障码）。
  Use when the user asks to diagnose / look up a vehicle DTC code, e.g. "P0301 是什么故障"、"帮我查下 P0420 的原因"、"查故障码 P0171 怎么修"。
---

# Inquiry-dtc 车载故障码诊断查询

输入 DTC 码 → 查知识库 → 生成暗色科技风 HTML 诊断报告并打开浏览器。**不连接实车硬件**。

## 用法
1. 提取故障码（`p0301`→`P0301`，支持多码），在**当前工作目录**执行：
   `python skills/Inquiry-dtc/scripts/make_dtc_report.py <CODE1> [CODE2 ...]`
   自动完成：查库 → 未命中族匹配提示 → 生成 `dtc_<CODE>.html` → 打开浏览器；可加 `-o` 指定输出路径、`--no-open` 不自动打开。
2. 未命中知识库时，回复说明该码不在库中并建议核对。
3. 追问"可能是什么坏了"时，基于 `causes`/`steps` 归纳故障部件排序，提示 `related` 相关码交叉诊断。

## 知识库
`skills/Inquiry-dtc/data/dtc_knowledge_base.json`（`dtcs` 数组：code/name_zh/description/causes/steps/repairs/related/source）。

## 注意
- 报告输出到**当前工作目录**，暗色科技风，含故障解释/可能诱因（按概率排序）/排查步骤/维修建议/相关码（可点击跳转）/来源。
- 纯推理查询类 Skill：不连接 OBD、不读取实车数据、不执行维修动作；输出仅供参考，实际维修以原厂维修手册与专业设备为准。
