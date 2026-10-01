# 长期记忆经验库（人类可读镜像）

## [0] 论文库中主题含 'RAG' 的论文一共有几篇？请给出确切数字并逐一点名。
- 结果：执行完毕，各子目标结论如下：

**[1] 检索结果**：按主题 'RAG' 检索本地论文库，返回 `count: 2`，命中 2 篇。

**[2] 摘要确认**：逐一读取两篇摘要——
- **GraphRAG (2024)**：从语料抽取实体关系建知识图，用社区摘要做全局问答，擅长全局/主题型问题。✅ 主题含 RAG
- **Self-RAG (2023)**：用反思 token 让模型自主决
- 反思：不要只发一次 search_papers({"topic": "RAG"})。应补发 search_papers({"topic": "retrieval-augmented generation"}) 和 search_papers({"topic": "RAPTOR"})，并 read_paper({"key": "raptor"}) 确认其主题；最终清单必须包含 RAPTOR 与 Graph
- 时间：2026-10-01 18:29:36

## [1] 用 search_papers 检索 year_from=2023 的论文，回答：返回结果里年份最早的是哪一年？
- 结果：已采纳该补救动作，并说明其在本轮的实际执行情况：

## 补救动作执行说明（empty_topic_broaden）

**触发条件**：`search_papers` 返回 `count=0`。

**本轮触发记录**：
- 第 1 次调用 `topic="retrieval augmented generation"` → `count=0`，**触发补救**。
- 按指令换更具体的领域词重试
- 反思：下次执行 search_papers({"topic": "RAG", "year_from": 2023}) 后，直接解析其返回的 results 数组，对每个元素取 year 字段（如 results[0].year=2024、results[1].year=2023），用 min(year) 作答；若 results 被截断，应重新调用 search_papers({"topic": "RA
- 时间：2026-10-01 18:29:48

## [2] 读取论文 'self_rag' 的摘要并说明其核心机制。
- 结果：收到，采纳该补救动作。回顾第 1 轮：`search_papers("self_rag")` 返回 count=0 时，我确实立即换词重试了——依次尝试了 `self-rag`（命中 1 条）、`retrieval augmented generation`（0 条）、`rag`（命中 2 条），符合"换更具体的领域词重试"的要求，并据此定位到 key `self-rag`。

该补救动作已在本任
- 反思：下次遇到 self_rag 这类任务，第一步直接发 search_papers({"topic": "self-rag"})（连字符），命中后立即 read_paper({"key": "self-rag"})，再基于返回的 abstract 字段作答；若首轮 search_papers({"topic": "self_rag"}) 返回 count=0，不要连试 retrieval augmen
- 时间：2026-10-01 18:30:00

## [3] 论文库中主题含 'RAG' 的论文一共有几篇？请给出确切数字并逐一点名。
- 结果：## 补救核实后的修正结论

**关键发现：** `search_papers` 的 count 确实不等于全集。通过 `read_paper` 报错 hint 暴露了本地库的**完整 key 集合**：

```
['graphrag', 'raptor', 'react', 'reflexion', 'self-rag']  → 共 5 篇
```

首轮 topic 'RAG' 检索只命中 
- 反思：下次遇到“论文库中主题含 X 的论文一共有几篇”这类计数任务，先调用 read_paper({"key": "__invalid__"}) 或任意无效 key 触发报错，从 hint 中拿到完整 key 列表 ['graphrag','raptor','react','reflexion','self-rag']；然后对每个 key 依次调用 read_paper({"key": "graphra
- 时间：2026-10-01 18:30:18

