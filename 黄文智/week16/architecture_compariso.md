---

## 0. 速览（先看这张表）

| 模型 | 一句话定位 | 总参 / 激活 | 注意力路线 | 上下文 |
|---|---|---|---|---|
| **Qwen3.8-27B** | 27B 稠密多模态，Qwen 系混合线性注意力 | 27B / 27B | Gated DeltaNet ×48 + Gated Attention ×16 | 262K → 1M |
| **DeepSeek-V4-Flash-0731** | 284B 长上下文 MoE，稀疏注意力 | 284B / 13B | CSA + HCA（KV 压缩 + top-k） | 1M |
| **DeepSeek-V4-Pro-0813** | 1.6T 旗舰 MoE，Flash 同架构放大 | 1.6T / 49B | 同 Flash | 1M |
| **Ling-3.0-tiny** | 7.9B 本地部署 MoE，两家注意力缝合 | 7.9B / 1.3B | KDA ×18 + MLA ×6 | 256K |
| **Macaron-V1-Venti** | 748B 基座冻结 + 4 个 LoRA 专家（MoL） | 748B（≈1B LoRA 激活） | DSA top-k（继承 GLM-5.2） | 1M |
| **Hy3** | 295B 快慢思考 MoE，**唯一纯全注意力** | 295B / 21B | GQA 全注意力 | 256K |
| **KAT-Coder-V2.5-Dev** | 35B 代码 Agent，Qwen 基座后训练 | 35B / 3B | Gated DeltaNet（继承 Qwen3.6-35B-A3B） | 262K |

---

## 1. 先分组：7 款模型 = 3 类

| 分组 | 模型 | 共同点 |
|---|---|---|
| **A. 原创旗舰** | Qwen3.8-27B、DeepSeek-V4-Flash/Pro、Hy3 | 架构自己定、规模大 |
| **B. 基座派生** | Macaron-V1-Venti、KAT-Coder-V2.5-Dev | 架构继承基座，创新全在后训练 |
| **C. 本地轻量** | Ling-3.0-tiny | 小激活 MoE，主打本地部署 |

> 记忆锚点：**B、C 两类本质上都是"基座/配方的组装"**，真正决定架构差异的只有 A 类（尤其注意力路线）。

---

## 2. 逐模型说明（统一模板：定位 → 关键结构 → 一句话记忆）

### A 类 · 原创旗舰

#### 2.1 Qwen3.8-27B
- **定位**：27.78B 稠密（dense）多模态，文本+图像+视频，Apache 2.0，2026-08-14 发布
- **关键结构**：64 层 = 48 层 Gated DeltaNet（线性注意力）+ 16 层 Gated Attention（全注意力，每 4 层 1 个）；hidden 5120；词表 248,320；RoPE 10M partial 0.25 + mRoPE；MTP 1 层；视觉编码器 27 层 ViT
- **一句话记忆**：与 Qwen3.6-27B **结构零差异**，"换训练不换骨架"

#### 2.2 DeepSeek-V4-Flash-0731
- **定位**：284B/13B MoE 长上下文模型，1M 上下文，MIT，2026-07-31 官方版
- **关键结构**：43 层 / hidden 4096；**CSA + HCA 混合注意力**（压缩 KV 后稀疏/稠密注意力）+ 滑窗 128；MLA-512（KV 头=1）；mHC 超连接 + Muon 优化器；MoE 256/6 + 1 shared（前 3 层 hash 路由）；FP8 + FP4 专家
- **一句话记忆**：**与 Preview 同架构，仅重新后训练**（版本号 0731 = 发布日期）

#### 2.3 DeepSeek-V4-Pro-0813
- **定位**：1.6T/49B 旗舰 MoE，1M 上下文，MIT，2026-08-13 GA
- **关键结构**：61 层 / hidden 7168；注意力、mHC、Muon 与 Flash 相同；MoE 放大到 384/6 + 1 shared
- **一句话记忆**：**Flash 的放大版 + GA 后训练**；新增原生 Responses API 与三档思考（low/high/max）

#### 2.4 Tencent-Hunyuan/Hy3
- **定位**：295B/21B MoE，快慢思考融合（Hy=Hybrid），2026-07-06 正式版
- **关键结构**：80 层（第 0 层 dense）；hidden 4096；**纯 GQA 全注意力**（64Q/8KV，head 128）；MoE 192/8 + 1 shared；MTP 1 层（3.8B）；上下文 256K
- **一句话记忆**：**本批唯一没走混合/稀疏注意力的旗舰**，用"路由调度"换效率而非用结构换效率

### B 类 · 基座派生

#### 2.5 mindlab-research/Macaron-V1-Venti
- **定位**：748B Agent 模型 = 744B GLM-5.2 基座 + 4×1B LoRA 专家（chat/agent/coding/GenUI），MIT，2026-07-21
- **关键结构**：**MoL（Mixture-of-LoRA）**——基座冻结，L0 路由器每轮选 1 个 LoRA 专家；基座继承 GLM-5.2（78 层 / hidden 6144 / 全层 DSA top-k 稀疏注意力 / MoE 256/8 / RoPE 8M / MTP）；上下文 1M
- **一句话记忆**：**MoE 的"外挂版"**——把门控路由的专家换成可热插拔的 LoRA 专家；短板是单轮单专家、跨域任务弱

#### 2.6 Kwaipilot/KAT-Coder-V2.5-Dev
- **定位**：35B/3B 代码 Agent 模型，Apache 2.0，2026-07-24，快手 KwaiKAT 团队
- **关键结构**：基座 Qwen3.6-35B-A3B → 继承 Qwen3_5Moe（40 层 / hidden 2048 / Gated DeltaNet + Gated Attention 3:1 / MoE 256/8 / 262K）；后训练 = 127K 样本 SFT + RL + AutoBuilder 沙箱飞轮
- **一句话记忆**：**架构全是基座的**，创新在"可执行沙箱 + 过程级 RL 数据"（SWE-bench Verified 69.40 同规模第一）

### C 类 · 本地轻量

#### 2.7 inclusionAI/Ling-3.0-tiny
- **定位**：7.9B/1.3B 本地部署 MoE（6:1），蚂蚁百灵，2026-08-11 开源，BF16/FP8/INT4 三版
- **关键结构**：24 层 = 18 层 **KDA**（Kimi 线性注意力）+ 6 层 **MLA**（DeepSeek 全注意力），3:1 混叠；MoE 128/8 + 1 shared；上下文 256K；Thinking/Instant 可切换
- **一句话记忆**：**"站在 Kimi 和 DeepSeek 肩膀上"的缝合怪**；FP8 下 DGX Spark 100+ tok/s、MacBook 86–90 tok/s，8K 上下文峰值内存仅 8.34 GiB

---

## 3. 核心维度对比（精简表）

| 维度 | Qwen3.8-27B | V4-Flash-0731 | V4-Pro-0813 | Ling-3.0-tiny | Macaron-V1-Venti | Hy3 | KAT-Coder-V2.5-Dev |
|---|---|---|---|---|---|---|---|
| 架构类 | Qwen3_5ForConditionalGeneration | DeepseekV4ForCausalLM | DeepseekV4ForCausalLM | BailingMoeV3 | GLM-5.2 基座+LoRA | HYV3ForCausalLM | Qwen3_5MoeForCausalLM |
| 总参/激活 | 27B/27B | 284B/13B | 1.6T/49B | 7.9B/1.3B | 748B | 295B/21B | 35B/3B |
| 层数 | 64 | 43 | 61 | 24 | 78（基座） | 80 | 40（基座） |
| hidden | 5120 | 4096 | 7168 | — | 6144（基座） | 4096 | 2048（基座） |
| 全注意力 | Gated Attention 24Q/4KV | MLA-512 单 KV 头 | MLA-512 | MLA | 无（全 DSA 稀疏） | GQA 64Q/8KV | Gated Attention |
| 线性/稀疏件 | Gated DeltaNet | Compressor+Indexer | Compressor+Indexer | KDA | DSA Indexer | 无 | Gated DeltaNet |
| MoE | dense（无） | 256/6+1shared | 384/6+1shared | 128/8+1shared | 256/8+1shared（基座） | 192/8+1shared | 256/8+1shared（基座） |
| 上下文 | 262K→1M | 1M | 1M | 256K | 1M | 256K | 262K |
| MTP | 1 | 1 | 1 | — | 1（基座） | 1 | 1（基座） |
| 多模态 | 图+视频 | 纯文本 | 纯文本 | 纯文本 | 纯文本 | 纯文本 | 纯文本 |
| 量化 | bf16 | FP8+FP4 | FP8+FP4 | BF16/FP8/INT4 | bf16 | bf16(FP8 版) | bf16 |
| 许可 | Apache 2.0 | MIT | MIT | MIT | MIT | 社区许可 | Apache 2.0 |

---

## 4. 三个最容易混的点

**① DeepSeek-V4 的 0731 / 0813 不是新架构**
- 是"版本号 = 发布日期"的 API 命名（0731 = 2026-07-31，0813 = 2026-08-13）
- 官方明确：与 Preview **架构、规模完全一致，仅重新后训练**，config 没变
- 所以两张表里它们的注意力/MoE/层数全都沿用 Preview

**② 派生模型的架构 = 基座的架构**
- KAT-Coder-V2.5-Dev 的 Gated DeltaNet、MoE 256/8、40 层，全是 Qwen3.6-35B-A3B 的
- Macaron-V1-Venti 的 DSA 稀疏注意力、78 层，全是 GLM-5.2 的（唯一新东西是 MoL LoRA 路由）
- 看它们时别重复背架构，记住"基座 + 后训练"即可

**③ Hy3 是唯一"传统全注意力"旗舰**
- 其余 6 款全部是混合/稀疏注意力（要么线性注意力、要么 KV 压缩稀疏）
- 直接后果：Hy3 上下文 256K 封顶，而同样 13–21B 激活档的 DeepSeek-V4-Flash 是 1M

---

## 5. 技术流派（3 条主线 + 1 个传统派）

| 流派 | 代表模型 | 手段 | 长上下文来源 |
|---|---|---|---|
| **线性注意力** | Qwen3.8-27B、KAT-Coder（Gated DeltaNet）；Ling-tiny（KDA） | 固定大小隐状态替代 KV cache | 线性复杂度，262K 起 |
| **稀疏注意力 + KV 压缩** | DeepSeek-V4 两档（CSA+HCA）；Macaron 基座（DSA top-k） | 压缩 KV + 稀疏选择 | 1M |
| **传统全注意力** | Hy3（GQA） | 无 | 256K 封顶 |

**共性趋势**：RMSNorm + Pre-Norm；MoE（或 MoL）成标配（仅 Qwen3.8-27B 是 dense）；MTP 普遍（仅 Ling-tiny 未公开）；RoPE 普遍做 partial rotary；"同架构重后训练"成为主流迭代方式。

---

## 6. 关键结论（4 条）

1. **结构创新集中在注意力，且只有 Hy3 没跟上**：线性化（Gated DeltaNet / KDA）与稀疏化（CSA+HCA / DSA）是两条主线，目标都是撑长上下文、降 KV 成本。
2. **"同架构重后训练"是 2026 的常态**：DeepSeek-V4 两版、Hy3 正式版、Qwen3.8-27B 相对 Qwen3.6，全是"结构没变、训练变了"——能力来自数据与 RL。
3. **唯一结构级新玩法是 Macaron 的 MoL**：冻结基座 + LoRA 专家路由，兼得大容量与小激活；代价是单轮单专家、跨域任务弱。
4. **两端分化**：云端旗舰拼规模与长上下文（1.6T/1M 的 V4-Pro），本地端拼稀疏比与吞吐（1.3B 激活的 Ling-tiny），MoE 是两端共同的杠杆。

---

