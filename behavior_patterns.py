"""
W26 Day5·补 — 内置行为模式触发器集（behavior_triggers.py 的兜底/样板）
====================================================================

来源：Day5 三轮实验 + W26 D1-D4 实测中**真实踩到**的坑（每条都标注出处）。
作用：
  ① 当 LLM 编译器（compile_trigger）没编译出触发器时兜底；
  ② 作为「触发器长什么样」的人工样板，方便对照 LLM 的编译质量。

字段含义（见 behavior_triggers.Trigger）：
  when       —— 检测器名，必须存在于 DETECTORS
  action     —— inject / force_tool / block
  payload    —— force_tool 时的工具名与参数模板
  instruction—— inject / block 时的具体指令（必须含工具名或参数值）
  source_task—— 该触发器来自哪次失败（可追溯，便于人审）
"""

BUILTIN_PATTERNS = [
    # ── 坑 1：key 字面量（Day5 P3 实测：self_rag 全 0，self-rag 才命中）──
    {
        "name": "key_hyphen_retry",
        "when": "wrong_key",
        "action": "force_tool",
        "payload": {"tool": "read_paper", "args": {"key": "$bad"}},
        "instruction": "read_paper 的 key 必须小写连字符（self-rag / graphrag / raptor），"
                       "下划线或大写一律报 unknown paper，立即用连字符变体重试。",
        "source_task": "W26 D5 P3：任务给错 key self_rag → 0 条",
    },
    # ── 坑 2：单关键词漏检（Day5 P1：topic='RAG' 只命中 2 篇，模型虚报 3 篇）──
    {
        "name": "count_crosscheck",
        "when": "underclaim_count",
        "action": "inject",
        "payload": {},
        "instruction": "禁止把篇数说成大于 search_papers 返回的 count；"
                       "若 count<3 要如实说明「库内仅 N 篇」，不得凑数。",
        "source_task": "W26 D5 P1：模型报 3 篇，实际 count=2",
    },
    # ── 坑 3：检索空结果不换词（D5 经验：'method' 泛词 count=0 后未换词）──
    {
        "name": "empty_topic_broaden",
        "when": "empty_result",
        "action": "inject",
        "payload": {},
        "instruction": "search_papers 返回 count=0 时，立即换更具体的领域词重试"
                       "（如 method → RAG / agent / retrieval），最多重试 3 次。",
        "source_task": "W26 D4 自测：topic='method' 返回 0",
    },
    # ── 坑 4：不读论文凭记忆答（W25 D1 教训：模型能心算就不调工具）──
    {
        "name": "must_read_before_answer",
        "when": "final_missing_tool",
        "action": "inject",
        "payload": {},
        "instruction": "最终答案必须基于 read_paper 读到的摘要原文，"
                       "答案里要出现论文标题；不得凭训练记忆作答。",
        "source_task": "W25 D1：模型对可回忆内容不触发工具",
    },
    # ── 坑 5：一次失败就放弃（Reflexion 论文核心：带教训重试）──
    {
        "name": "retry_once_with_lesson",
        "when": "needs_retry",
        "action": "inject",
        "payload": {},
        "instruction": "任务尚未产生任何工具结果，请先调用 search_papers 或 read_paper "
                       "获取真实数据，再作答。",
        "source_task": "Reflexion (2023) trial loop 语义",
    },
]
