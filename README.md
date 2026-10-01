# W26 — Agent 规划 & 反思

Week 26 of the Month 7 (Agent 工程化) track. 目标：让 Agent **会自我批判、会任务分解、有双记忆**。

承接 Month 6 Self-RAG 的 critique token —— 从「检索侧评判检索内容」延伸到「批判自己的执行过程」。

## 本周验收

| 判据 | 结果 |
|---|---|
| 多步任务完成率 ≥ 70% | ⚠️ 未达成 —— 任务对当前模型已过期（见下） |
| Agent 自主调用 ≥3 工具（含 RAG）且调用链连贯 | ✅ **2/2 通过** |

## 产出清单

| Day | 主题 | 产出 | 验收 |
|:---:|---|---|:---:|
| D1 | 自我反思（Reflexion） | `reflection.py` / `reflection_quiz.py` / `reflection_quiz2.py` | ⚠️ 见下 |
| D1·重做 | 真实触发 + 严格归因 | `reflection_v2.py` | ⚠️ 任务过期 |
| D2 | 规划器 Planner | `planner.py` | ✅ |
| D2 | 任务型 Reflexion（环境注入） | `reflection_env.py` | ✅ |
| D3 | 单 vs 多 Agent 对比 | `multi_agent.py` + 报告 | ✅ |
| D4 | Agent 双记忆 | `memory.py` | ✅ recall 4/4 |
| D5 | 记忆增强决策 | `agent_v2.py` | ⚠️ 增益 +0 |
| D5·补 | 行为模式触发器 | `behavior_triggers.py` | ⚠️ 机制成立 |
| D6 | Reflexion 论文卡片 | `cards/W26-Day6-*.md` | ✅ |

## 核心结论

### ✅ 已证明的

**1. 环境注入能让失败真实发生**（`reflection_env.py`）
Reflexion 要在**任务型场景**验收：多步工具 + 中途环境变化/局部失败。

**2. 经验可以编译成「控制流」**（`behavior_triggers.py`）
不是把教训当提示塞 prompt（被动、靠概率），而是编译成 `event → condition → action`
触发器，命中信号即**强制执行**（主动、确定性）。

**3. Agent 自主编排 RAG 工具链**（`verify_rag_chain.py`）
```
rag_search → search_papers → read_paper → save_note
```
4 工具、含 RAG、原样连贯、无未解决错误。

**4. 记忆要按类型分流**（`memory.py`）
- 事实型（"库里有几篇"）→ 结论快照，**会过期**，强制带版本戳
- 方法型（"key 要用连字符"）→ 操作处方，**不过期**，跨版本召回

### ⚠️ 未达成的（同一个根因）

**「完成率提升」在 Day1 / Day5 / D5·补 三处全部为 +0。**

根因不是机制无效，而是**评测任务对当前模型已全部过期**：
```
第0步 裸跑探测：三个任务裸跑就全过（0/3 有区分度）
```
模型碰壁后能**自愈**（503 → 自己重试、参数错 → 自己改），
Reflexion / 触发器根本没有用武之地。

> **这是本周最有价值的产出：第一次把「任务过期」量化出来了。**

## 三个真 bug（可复用教训）

**1. 失败判据把「成功自愈」判成失败**
`has_error()`（轨迹里出现过任何 error 即失败）把模型「碰壁后自己重试成功」判成失败，
导致 A/B/C 三组全部误判为 0。
→ 修正：`has_unresolved_error()` 只看该 error 之后**同类调用是否被解决**。
→ **教训：失败判据必须看最终状态，不能看过程噪声。**

**2. 判据里的「标准答案」自己没验证过**
P1 判据要求点名 RAPTOR，但那篇摘要里根本没有 "RAG" 字样 —— 把正确答案判成了失败。
→ **教训：判据里的标准答案本身也必须先验证。**

**3. 参数接入只改了主路径**
`scope` 接入时只改了 `_run_with_memory`，漏了 `_seed_memory` / `_probe_difficulty`
→ 沉淀日志显示 `kind=? scope=—`，机制**静默失效**。
→ **教训：一门参数接入后，必须搜出所有调用点逐个检查。**

## 目录结构

```
.
├── reflection.py            # D1 Reflexion 骨架
├── reflection_quiz.py       # D1 开放任务自纠错
├── reflection_quiz2.py      # D1b 维度对模型不可见
├── reflection_v2.py         # D1 重做：A/B/C 三组对照 + 环境注入
├── reflection_env.py        # D2 任务型 Reflexion（4 场景）
├── planner.py               # D2 规划器 + rag_search 工具
├── multi_agent.py           # D3 单 vs 多 Agent
├── memory.py                # D4 双记忆（短期滑窗 + 长期 BGE/FAISS）
├── behavior_triggers.py     # D5·补 行为模式触发器
├── behavior_patterns.py     # 内置触发器集
├── agent_v2.py              # D5 五步闭环 + 跨会话 A/B
├── verify_rag_chain.py      # 月度指标验证（≥3 工具含 RAG）
├── DAY3-单vs多Agent对比报告.md
└── cards/                   # 每日验收复盘（9 张）
```

## 环境依赖

```bash
export LLM_API_KEY=...      # DeepSeek API
export LLM_BASE_URL=https://api.deepseek.com
export LLM_MODEL=deepseek-v4-flash
```

本地模型（记忆 embedding）：`BAAI/bge-small-zh-v1.5`
RAG 检索复用 Month 6 的 `rag-kb`（`HybridRetriever`：Flat 叶子 + RAPTOR 摘要 + RRF 融合）。

## 运行

```bash
python3 reflection_v2.py              # D1 重做：A/B/C 对照
python3 reflection_env.py             # D2 环境注入 Reflexion
python3 memory.py                     # D4 双记忆自测
python3 agent_v2.py                   # D5 闭环 A/B
DAY5_ROUNDS=3 python3 agent_v2.py     # 稳定性验证
python3 verify_rag_chain.py           # 月度指标：≥3 工具含 RAG
```

## ⚠️ API 坑（勿再犯）

`deepseek-v4-flash` 是 reasoning 模型：
- 请求体必须带 `{"thinking": {"type": "disabled"}}`，否则思维链吃光 `max_tokens` → `content` 为空
- 生成/判据预算一律 ≥1500
- 任何「截断后判断」都会产生**假失败**

## 下一步（W27）

真正卡住的不是「记忆能不能用」，而是**「找不到模型现在还会犯的错」**。
W27 评测流程必须先解决**任务难度滚动更新**：每轮验收前先测裸跑完成率，
>80% 的任务直接淘汰换坑。

---

论文对齐：Reflexion (Shinn et al., 2023) —— 语言强化 + trial loop
