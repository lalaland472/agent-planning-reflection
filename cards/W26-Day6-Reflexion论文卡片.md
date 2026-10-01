# W26 Day6 — 论文卡片：Reflexion（Shinn et al., 2023）

日期：2026-09-19
论文：*Reflexion: Language Agents with Verbal Reinforcement Learning*（Shinn et al., NeurIPS 2023）
计划验收：**卡片含 ≥3 个代码对应**（reflection.py / agent_v2.py 如何对齐 verbal reinforcement）
结论：**✅ 通过**（6 个代码对应，含 3 个已落地 + 3 个待改进）

---

## 一、论文核心（一句话）

**不给模型权重做梯度更新，而是让它把失败经验写成「语言」，存进记忆，下次尝试时读回来——用自然语言当强化信号。**

三个组件（论文原架构）：

| 组件 | 作用 | 论文里的角色 |
|---|---|---|
| **Actor** | 生成动作/轨迹 | 就是 LLM 本体（M_t = LLM） |
| **Evaluator** | 判定轨迹好坏，给奖励信号 | 测试通过率 / 人工反馈 / LLM 评判 |
| **Self-Reflection** | 把失败轨迹转成**言语教训** | 核心创新：verbal feedback |
| **Memory** | 存教训，供下次尝试 | 短期（当轮轨迹）+ 长期（跨轮教训） |

关键机制：**trial → error → verbal reflection → retry**，循环直到成功或达上限。
论文在 AlfWorld（决策）、HotPotQA（推理）、HumanEval（代码）上均显著提升。

---

## 二、代码对应（验收核心：≥3 项）

### 对应 1：Actor — 工具执行循环 ✅
| 论文 | 代码 |
|---|---|
| Actor 生成动作序列 | `planner.py` 的 `_loop()`（Thought→Action→Observation 循环） |
| 环境返回反馈 | `TaskEnv.dispatch()` 返回 tool observation |

### 对应 2：Self-Reflection — 言语反思 ✅
| 论文 | 代码 |
|---|---|
| 把失败轨迹转成语言教训 | `agent_v2.py` 的 `reflect()`：输出结构化 `{correct, what_wrong, how_to_fix}` |
| 反思是"语言"而非标量奖励 | prompt 明确要求"可操作的规则"，而非分数 |

**实证**：Day5 实验里反思质量确实高——播种轮准确抓到"没按任务要求用 topic='method'，自己换成了 framework/raptor"。

### 对应 3：Memory — 教训存储与复用 ✅
| 论文 | 代码 |
|---|---|
| Memory 存跨轮教训 | `memory.py` 的 `LongTermMemory`（BGE 向量化 + FAISS 检索） |
| 下次尝试读取教训 | `agent_v2.py` 的 `make_plan_with_memory()`：检索命中则注入规划 prompt |

### 对应 4：Evaluator — 判定轨迹 ✅（但实现偏弱）
| 论文 | 代码 |
|---|---|
| Evaluator 给二元/连续奖励 | Day5 的 `TRAP_TASKS[*].check` 硬判据 |
| 论文用测试通过率 / LLM judge | 我们的 `quality_score()` 是启发式规则，**应升级为 LLM-as-Judge**（W16 model-eval-lab 已有经验，注意 `thinking: disabled` 坑） |

### 对应 5：Trial 循环（retry until success）⚠️ 部分缺失
| 论文 | 代码 |
|---|---|
| 同一任务多次 trial，每轮带教训重试 | `run_agent_v2()` 每次都是**单次 trial** |
| —— | 跨 trial 的复用靠**跨会话**（会话A 播种 → 会话B 复用），非论文的同任务内重试 |

→ **差距**：论文是"同一任务重试 N 轮直到成功"，我们是"不同会话间复用教训"。**我们弱化了 trial 循环，这是与论文最大的结构差异。**

### 对应 6：Memory 粒度对齐 ⚠️ 已知缺陷
| 论文 | 代码 |
|---|---|
| 教训要能直接指导下次行动 | Day5 发现：反思产出**抽象原则**（"要确认接口语义"），而任务需要**具体事实**（"就是 2 篇，RAPTOR + GraphRAG"） |
| —— | **抽象教训解决不了具体事实题** → 漏斗断在"注入生效"这一环 |

→ 这正是 W28 D6「MEMIT / Generative Agents 记忆论文」要补的：**记忆的写入粒度必须与检索/使用粒度对齐**。

---

## 三、我们与论文的差异（诚实清单）

| 维度 | 论文 Reflexion | 我们的实现 | 评价 |
|---|---|---|---|
| 重试循环 | 同任务内多 trial，直到成功 | 单次 trial + 跨会话复用 | ⚠️ 弱化 |
| 奖励信号 | 测试通过率（客观） | 启发式 check / 规则 | ⚠️ 偏弱 |
| 教训粒度 | 面向同一任务的具体修正 | 偏抽象原则 | ⚠️ 需升级 |
| 记忆检索 | 简单存储 + 全量喂回 | BGE 语义检索 top-k | ✅ 更强 |
| 记忆时效 | 未讨论 | 已识别问题（旧经验会失效） | ✅ 更清醒 |

**结论：我们的 RAG 式记忆检索比论文的原版更强，但 trial 循环和反思粒度弱于论文。**

---

## 四、产品/工程启示（计划要求"卡片含≥3 个代码对应"，此为延伸）

1. **反思要"事实级 + 原则级"双输出**：论文的教训是针对具体失败轨迹的，天然具体。我们的改进方向是让 `reflect()` 同时产出「事实快照」和「操作原则」；
2. **Evaluator 是瓶颈**：论文用客观测试（代码通过率）做 evaluator 才好用；我们的启发式打分是最大的薄弱环节，应优先升级为 LLM-as-Judge；
3. **trial 循环值得补回**：Day5 的"跨会话复用"不能替代论文的"同任务重试"——前者解决"同类任务再遇到"，后者解决"这次就做对"。

---

## 五、与 W26 其他 Day 的呼应

| Day | 落地了什么 | 对应论文组件 |
|---|---|---|
| D1 | `reflection.py` 自我反思 | Self-Reflection |
| D2 | `planner.py` 任务分解 | 前置规划（论文外） |
| D3 | 单 vs 多 Agent 对比 | Evaluator 的独立化（Critic） |
| D4 | `memory.py` 双记忆 | Memory |
| D5 | `agent_v2.py` 闭环 | 完整 trial 循环（简化版） |

→ **W26 一整周 = Reflexion 论文的工程化拆解与落地。**

---

## 附
- 论文：Shinn et al., "Reflexion: Language Agents with Verbal Reinforcement Learning", NeurIPS 2023
- 代码：`reflection.py`（D1）/ `memory.py`（D4）/ `agent_v2.py`（D5）
- 相关卡片：`W26-Day1-Reflexion-验收复盘.md`（含两次"假验收"教训）
