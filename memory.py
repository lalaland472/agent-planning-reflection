"""
W26 Day4 — Agent 双记忆（短期 + 长期）
========================================

计划要求（Month 7 W26 D4）：
    短期（窗口 + 摘要压缩）+ 长期（复用 FAISS/BGE 存经验）
    复用：agent-planner 记忆 + rag-kb 向量检索
    产出：memory.py
    验收：长期记忆检索 recall 命中相关旧经验

设计：
    ShortTermMemory —— 滑动窗口 + 超限摘要压缩（LLM 把旧消息压成一条摘要）
    LongTermMemory  —— BGE 向量化经验 → FAISS 索引 → 语义检索旧经验
                       经验 = 一次任务执行的「任务 + 结果 + 反思」三元组

存储布局（与 rag-kb 的 FAISS 用法对齐，但独立目录避免污染主索引）：
    w26_plan_reflect/memory_store/
        experience.faiss     — 经验向量索引（IndexFlatIP，经验量少不需 HNSW）
        experience.json      — 经验元数据（与 faiss 行号一一对应）
        experience_notes.md  — 人类可读镜像（便于人工核查）
"""

import os
import re
import sys
import json
import time
import hashlib

import numpy as np

BASE = os.path.dirname(os.path.abspath(__file__))
STORE = os.path.join(BASE, "memory_store")
os.makedirs(STORE, exist_ok=True)

EMB_MODEL_NAME = "BAAI/bge-small-zh-v1.5"
EMB_DIM = 512

# ⚠️ 延迟加载：BGE 加载约 2-3 秒，不在 import 时做
_EMB = None


def _emb():
    global _EMB
    if _EMB is None:
        from sentence_transformers import SentenceTransformer
        _EMB = SentenceTransformer(EMB_MODEL_NAME)
    return _EMB


def embed(texts, is_query=False):
    """BGE 编码。bge-*-zh 检索时 query 需加指令前缀（官方建议）。"""
    model = _emb()
    if isinstance(texts, str):
        texts = [texts]
    if is_query:
        texts = [f"为这个句子生成表示以用于检索相关文章：{t}" for t in texts]
    v = model.encode(texts, normalize_embeddings=True, show_progress_bar=False)
    return np.asarray(v, dtype="float32")


# ══════════════════════════════════════════════════
# 一、短期记忆：滑动窗口 + 摘要压缩
# ══════════════════════════════════════════════════
FACT_SCOPE_PLACEHOLDER = "fact@unversioned"   # 事实型经验的默认版本戳（未显式指定时）

_FACT_PAT = re.compile(r"[0-9]+\s*(篇|个|条|项|次|篇论文)|几篇|多少篇|count|总数|共有")


def infer_kind(task, outcome=""):
    """推断经验类型（Day5 遗留③）。

    fact   : 任务在问「有多少 / 几篇 / 总数」这类**会随环境变化的事实**
    method : 其余（怎么做、参数怎么写、流程怎么走）—— 操作方法不过期
    """
    t = str(task) + " " + str(outcome)[:120]
    return "fact" if _FACT_PAT.search(t) else "method"


def _bad_reflection(reflection):
    """经验质量门禁（Day4 遗留②）：返回拒收原因，可写时返回空字符串。"""
    r = (reflection or "").strip()
    if not r:
        return "empty_reflection"
    if any(p in r for p in ("(解析失败)", "解析失败", "(解析失败", "None")):
        return "parse_failed_placeholder"
    if len(r) < 8:
        return "reflection_too_short"
    return ""


# ⚠️ D4 计划要求「复用 rag-kb 向量检索」——
#    原先 memory.py 自建了 FAISS+BGE，从未真正调用 rag-kb 的检索能力（重造而非复用）。
#    此处接上 rag-kb 的 HybridRetriever（Flat 叶子 + RAPTOR 摘要 + RRF 融合）。
RAGKB_SCRIPTS = os.path.join(BASE, "..", "..", "rag-kb", "scripts")
if RAGKB_SCRIPTS not in sys.path:
    sys.path.insert(0, RAGKB_SCRIPTS)

_RAG_RETRIEVER = None


def get_rag_retriever():
    """懒加载 rag-kb 的 HybridRetriever（共享模型与索引，避免重复加载）。"""
    global _RAG_RETRIEVER
    if _RAG_RETRIEVER is None:
        try:
            from retriever import HybridRetriever
            _RAG_RETRIEVER = HybridRetriever()
        except Exception as e:
            print(f"  ⚠️ rag-kb 检索器加载失败（降级为本地经验库）: {e}")
            _RAG_RETRIEVER = False
    return _RAG_RETRIEVER or None


class RagKBRetriever:
    """D4「复用 rag-kb 向量检索」的适配层。

    与 LongTermMemory 的区别（两者互补，不是重复）：
      LongTermMemory —— 存 **Agent 自己的经验**（任务/结果/反思），供跨会话复用；
      RagKBRetriever —— 复用 **rag-kb 的论文知识库**（4521 chunks + 91 RAPTOR 摘要），
                        用于「这题以前读过哪篇论文可以参照」。
    计划 D4 的「复用」指的就是后者：不必自建，直接调用 rag-kb 已验证的检索管道。
    """

    def __init__(self):
        self.retriever = get_rag_retriever()

    @property
    def available(self):
        return self.retriever is not None

    def search(self, query, k=3, mode="flat", rerank=False):
        """检索论文库，返回 [{source, doc_id, chunk_id, score, mode}]。

        mode='flat'   叶子精确召回（高精度）
        mode='hybrid' Flat + RAPTOR 摘要融合（强主题检索）
        """
        if not self.retriever:
            return []
        out = self.retriever.retrieve(query, k=k, mode=mode, rerank=rerank)
        hits = out.get("flat_top") or []
        return [{"source": h.get("source", h.get("doc_id")),
                 "doc_id": h.get("doc_id"), "chunk_id": h.get("chunk_id"),
                 "score": round(float(h.get("score", 0)), 3),
                 "mode": out.get("mode", mode)} for h in hits]

    def as_hint(self, query, k=3, mode="flat"):
        """渲染成给 Agent 的「可参照论文」提示（供规划/作答前注入）。"""
        hits = self.search(query, k=k, mode=mode)
        if not hits:
            return ""
        lines = ["【rag-kb 论文库·相关参考（来自向量检索）】"]
        for h in hits:
            lines.append(f"- {h['source']} (chunk {h['chunk_id']}, 距离{h['score']})")
        return "\n".join(lines)


class ShortTermMemory:
    """滑窗 + 摘要压缩的短期记忆。

    超出 `max_turns` 的旧消息不再直接进 prompt，而是被 LLM 压缩成一条
    『历史摘要』保留关键信息（任务意图、已做动作、关键结论），避免上下文线性膨胀。
    """

    def __init__(self, max_turns=4, keep_recent=3, min_compress_chars=500):
        self.max_turns = max_turns      # 触发压缩的阈值（消息条数，不含 system）
        self.keep_recent = keep_recent  # 压缩后保留的最近消息条数
        self.min_compress_chars = min_compress_chars  # 最小压缩量阈值（Day4 遗留①）
        self.buffer = []                # 原始消息 [{role, content}]
        self.summary = ""               # 压缩后的历史摘要
        self.compress_count = 0         # 压缩次数（验收可观测）
        self.skipped_compress = 0       # 因过短被跳过的次数（可观测）

    def add(self, role, content):
        self.buffer.append({"role": role, "content": content})

    def needs_compress(self):
        """是否需要压缩。

        ⚠️ Day4 遗留①：短历史下摘要压缩是**负优化**（实测 201 字 → 221 字）。
           LLM 生成的摘要本身有开销，只有被压的旧内容足够长时才划算。
           因此除了消息条数，还须满足「待压内容 ≥ min_compress_chars」。
        """
        if len(self.buffer) <= self.max_turns:
            return False
        old = self.buffer[:-self.keep_recent]
        old_chars = sum(len(m["content"]) for m in old)
        if old_chars < self.min_compress_chars:
            self.skipped_compress += 1
            return False
        return True

    def compress(self, env, usage=None):
        """把 buffer 里较旧的消息压成摘要，只保留最近 keep_recent 条。"""
        from planner import plain_llm

        old = self.buffer[:-self.keep_recent]
        recent = self.buffer[-self.keep_recent:]
        old_text = "\n".join(f"{m['role']}: {m['content'][:300]}" for m in old)

        prompt = (
            "把下面这段 Agent 执行历史压缩成一条简洁摘要，保留：①任务意图 ②已完成的动作与结论 "
            "③失败/未完成的部分 ④关键数据。只输出摘要正文，不要前缀。\n\n" + old_text
        )
        new_sum = plain_llm(env, [{"role": "user", "content": prompt}],
                            usage=usage, max_tokens=300, temperature=0.2)
        # 摘要叠加：新旧摘要拼接（旧摘要也需保留，否则长任务会丢早期信息）
        self.summary = (self.summary + "\n" + new_sum).strip() if self.summary else new_sum
        self.buffer = recent
        self.compress_count += 1
        return self.summary

    def to_context(self):
        """渲染成可塞进 prompt 的文本。"""
        parts = []
        if self.summary:
            parts.append(f"【历史摘要（已压缩）】\n{self.summary}")
        if self.buffer:
            parts.append("【最近对话】\n" + "\n".join(
                f"{m['role']}: {m['content'][:300]}" for m in self.buffer))
        return "\n\n".join(parts) or "(空)"

    def total_tokens_estimate(self):
        """粗略 token 估算（中文约 1 字 1 token）。"""
        return len(self.summary) + sum(len(m["content"]) for m in self.buffer)


# ══════════════════════════════════════════════════
# 二、长期记忆：BGE + FAISS 存经验
# ══════════════════════════════════════════════════
class LongTermMemory:
    """向量化经验库：一次任务执行的 (任务, 结果, 反思) 存成一条经验。

    检索时用语义相似度找「过去做过类似的事」，把教训喂给当前执行。
    """

    INDEX_PATH = os.path.join(STORE, "experience.faiss")
    META_PATH = os.path.join(STORE, "experience.json")
    NOTES_PATH = os.path.join(STORE, "experience_notes.md")

    def __init__(self):
        self.meta = []          # [{id, task, outcome, reflection, ts, hash}]
        self.index = None
        self._load()

    # ── 持久化 ────────────────────────────────────
    def _load(self):
        import faiss
        if os.path.exists(self.META_PATH):
            with open(self.META_PATH, encoding="utf-8") as f:
                self.meta = json.load(f)
        if os.path.exists(self.INDEX_PATH) and self.meta:
            self.index = faiss.read_index(self.INDEX_PATH)
        else:
            self.index = faiss.IndexFlatIP(EMB_DIM)   # 内积 = 余弦（向量已归一化）

    def _save(self):
        import faiss
        faiss.write_index(self.index, self.INDEX_PATH)
        with open(self.META_PATH, "w", encoding="utf-8") as f:
            json.dump(self.meta, f, ensure_ascii=False, indent=2)
        with open(self.NOTES_PATH, "w", encoding="utf-8") as f:
            f.write("# 长期记忆经验库（人类可读镜像）\n\n")
            for m in self.meta:
                f.write(f"## [{m['id']}] {m['task'][:60]}\n")
                f.write(f"- 结果：{m['outcome'][:200]}\n")
                f.write(f"- 反思：{m['reflection'][:200]}\n")
                f.write(f"- 时间：{m['ts']}\n\n")

    # ── 写入 ──────────────────────────────────────
    @staticmethod
    def _hash(task, outcome):
        return hashlib.md5((task + "|" + outcome).encode()).hexdigest()[:12]

    def remember(self, task, outcome, reflection="", dedup=True, quality_gate=True,
                 scope="", kind=None):
        """存一条经验。dedup=True 时按 (task, outcome) 哈希去重。

        ⚠️ Day4 遗留②：经验质量门禁。
           实测出现过 `(解析失败)` 这类空教训被写入 —— 垃圾进垃圾出。
           quality_gate=True 时，拒收：反思为空 / 反思是解析失败占位 / 反思过短（<8 字）。

        ⚠️ Day5 遗留③：按任务类型分流存储（kind）。
           - kind="fact"  事实型（如「库里有几篇 RAG 论文」）→ outcome 就是**结论快照**。
             这类经验**会过期**（库内容一变就废），必须带 scope，检索时严格看版本。
           - kind="method" 方法型（如「read_paper 的 key 要用连字符」）→ reflection 是**操作处方**。
             这类经验**不过期**，跨 scope 也适用。
           kind=None 时自动推断（含数字/计数词的判定为 fact）。
           动机：Day1 重做实测发现「错误教训比没教训更糟」——
                 很大一部分就是把会过期的事实当成不过期的方法来复用了。
        """
        if quality_gate:
            bad = _bad_reflection(reflection)
            if bad:
                return {"stored": False, "reason": bad, "id": None}
        kind = kind or infer_kind(task, outcome)
        # 事实型经验强制要求 scope（否则过期后无法识别，会污染检索）
        if kind == "fact" and not scope:
            scope = FACT_SCOPE_PLACEHOLDER
        h = self._hash(task, outcome)
        if dedup and any(m["hash"] == h for m in self.meta):
            return {"stored": False, "reason": "duplicate", "id": h}

        text = f"任务：{task}\n结果：{outcome}\n反思：{reflection}"
        vec = embed(text)
        self.index.add(vec)
        rec = {"id": len(self.meta), "task": task, "outcome": outcome,
               "reflection": reflection, "ts": time.strftime("%Y-%m-%d %H:%M:%S"),
               "hash": h, "scope": scope, "kind": kind}
        self.meta.append(rec)
        self._save()
        return {"stored": True, "id": rec["id"], "total": len(self.meta)}

    # ── 检索 ──────────────────────────────────────
    def recall(self, query, top_k=3, threshold=0.0, scope="", kind=None):
        """语义检索旧经验，返回 [{score, task, outcome, reflection}]（按分数降序）。

        ⚠️ Day4 遗留③：经验过期/失效机制。
           库内容变化后旧经验可能失效（如论文库新增论文，「只有 2 篇」就过期了）。
           scope 是经验所属的**环境版本戳**；只召回 scope 匹配的经验。
        ⚠️ Day5 遗留③：方法型经验不过期（跨 scope 召回），事实型经验严格看版本。
        """
        if self.index is None or self.index.ntotal == 0:
            return []
        q = embed(query, is_query=True)
        k = min(top_k * 3, self.index.ntotal)   # 多取候选，过滤过期后再截断
        scores, idxs = self.index.search(q, k)
        out = []
        for sc, i in zip(scores[0], idxs[0]):
            if i < 0 or sc < threshold:
                continue
            m = self.meta[i]
            if kind and m.get("kind") != kind:
                continue
            # 过早过滤：只有**事实型**经验会因为环境变化而失效；
            # 方法型经验（操作处方）跨版本依然有效。
            if scope and m.get("kind") == "fact" \
                    and m.get("scope") and m["scope"] != scope:
                continue
            out.append({"score": round(float(sc), 4), "id": m["id"],
                        "task": m["task"], "outcome": m["outcome"],
                        "reflection": m["reflection"], "scope": m.get("scope", ""),
                        "kind": m.get("kind", "")})
            if len(out) >= top_k:
                break
        return out

    def as_hint(self, query, top_k=2, threshold=0.5, scope=""):
        """把检索结果渲染成给执行 Agent 的经验提示文本。低相似度不注入。"""
        hits = self.recall(query, top_k=top_k, threshold=threshold, scope=scope)
        if not hits:
            return ""
        lines = ["【长期记忆·过往类似任务的教训】"]
        for h in hits:
            lines.append(f"- (相似度{h['score']:.2f}) 上次任务：{h['task'][:60]}")
            if h["outcome"]:
                lines.append(f"  结果：{h['outcome'][:120]}")
            if h["reflection"]:
                lines.append(f"  反思：{h['reflection'][:120]}")
        return "\n".join(lines)

    def stats(self):
        return {"experiences": len(self.meta), "index_ntotal":
                (self.index.ntotal if self.index is not None else 0)}

    def expire(self, scope):
        """Day4 遗留③：把某个 scope 的旧经验全部作废（环境版本变更时调用）。

        返回被作废的条数。物理删除需重建 FAISS 索引（Flat 不支持删单条），
        成本高；此处采用「移出 meta + 重建索引」的简单做法。
        """
        keep = [m for m in self.meta if m.get("scope") != scope]
        dropped = len(self.meta) - len(keep)
        if dropped == 0:
            return 0
        self.meta = keep
        self._rebuild()
        self._save()
        return dropped

    def _rebuild(self):
        """按当前 meta 重建向量索引（删除/过期后调用）。"""
        import faiss
        self.index = faiss.IndexFlatIP(EMB_DIM)
        if self.meta:
            vecs = embed([f"任务：{m['task']}\n结果：{m['outcome']}\n反思：{m['reflection']}"
                          for m in self.meta])
            self.index.add(vecs)


# ══════════════════════════════════════════════════
# 三、验收自测
# ══════════════════════════════════════════════════
def _selftest():
    from planner import load_env, Usage

    print("=" * 84)
    print("W26 Day4 — 双记忆自测（短期滑窗摘要 + 长期向量经验）")
    print("=" * 84)
    env = load_env()

    # ── 1. 短期记忆压缩 ───────────────────────────
    print("\n【1】短期记忆：滑窗 + 摘要压缩")
    stm = ShortTermMemory(max_turns=4, keep_recent=3)
    u = Usage()
    script = [
        ("user", "帮我查 RAG 领域 3 篇论文并总结"),
        ("assistant", "我用 search_papers 检索 topic=RAG，返回 count=2"),
        ("tool", "read_paper('raptor') → RAPTOR (2024) 递归聚类层次树"),
        ("tool", "read_paper('graphrag') → GraphRAG (2024) 实体关系图+社区摘要"),
        ("assistant", "发现库内 RAG 论文只有 2 篇，凑不够 3 篇"),
        ("tool", "save_note('已写入 2 条笔记')"),
    ]
    for r, c in script:
        stm.add(r, c)
        if stm.needs_compress():
            print(f"  buffer={len(stm.buffer)} 超阈值 → 触发压缩")
            stm.compress(env, usage=u)
            print(f"  压缩后 buffer={len(stm.buffer)}，摘要长度={len(stm.summary)}")
    print(f"  压缩次数={stm.compress_count}（可观测指标）")
    tok = stm.total_tokens_estimate()
    raw = sum(len(c) for _, c in script)
    print(f"  上下文规模：原始 {raw} 字 → 压缩后 {tok} 字（{tok / raw:.0%}）")
    print(f"  摘要预览：{stm.summary[:150].replace(chr(10), ' ')}")

    # ── 1b. 遗留① 验证：最小压缩阈值（短会话不压缩）───
    print("\n【1b】遗留①：最小压缩阈值（短历史不压缩，避免负优化）")
    stm_short = ShortTermMemory(max_turns=4, keep_recent=3, min_compress_chars=500)
    short_script = [("user", "你好"), ("assistant", "你好，需要什么帮助？"),
                    ("user", "帮我看看"), ("assistant", "好的"), ("user", "谢谢")]
    for r, c in short_script:
        stm_short.add(r, c)
    need_short = stm_short.needs_compress()
    short_chars = sum(len(c) for _, c in short_script[:-3])
    print(f"  消息 {len(stm_short.buffer)} 条（>阈值 4），待压 {short_chars} 字 < 500 "
          f"→ needs_compress={need_short}")
    print(f"  {'✅ 正确跳过（避免负优化）' if not need_short else '❌ 不应压缩'}"
          f" | 跳过计数={stm_short.skipped_compress}")

    # ── 1c. 遗留① 验证：内容够长时正常压缩 ────────
    # ⚠️ 注意：needs_compress() 只看**待压部分**（buffer[:-keep_recent]），
    #    所以要用足够长的旧消息才能撑到阈值。
    stm_long = ShortTermMemory(max_turns=4, keep_recent=3, min_compress_chars=200)
    long_script = [("user", "帮我查 RAG 领域 3 篇论文并总结，重点关注检索增强生成的最新进展、"
                            "看它们在长文档场景下的具体做法和实测效果对比"),
                   ("assistant", "我用 search_papers 检索 topic=RAG，返回 count=2；"
                                  "又试了 retrieval 和 long-context 两个关键词，各补充了一批候选，"
                                  "目前共找到 4 篇候选论文"),
                   ("tool", "read_paper('raptor') → RAPTOR (2024) 递归聚类层次树，"
                            "用摘要树解决长文档多跳问答，自顶向下检索克服单层切分的信息丢失问题"),
                   ("tool", "read_paper('graphrag') → GraphRAG (2024) 实体关系图+社区摘要"),
                   ("assistant", "发现库内 RAG 论文只有 2 篇，凑不够 3 篇"),
                   ("tool", "save_note('已写入 2 条笔记')")]
    for r, c in long_script:
        stm_long.add(r, c)
    need_long = stm_long.needs_compress()
    long_chars = sum(len(c) for _, c in long_script[:-3])
    print(f"  长内容（待压 {long_chars} 字 ≥ 200）→ needs_compress={need_long}"
          f" {'✅' if need_long else '❌'}")

    # ── 2. 长期记忆写入 ───────────────────────────
    print("\n【2】长期记忆：写入经验")
    # 用独立测试库，避免污染正式库
    import shutil
    test_store = os.path.join(STORE, "selftest")
    os.makedirs(test_store, exist_ok=True)
    LongTermMemory.INDEX_PATH = os.path.join(test_store, "experience.faiss")
    LongTermMemory.META_PATH = os.path.join(test_store, "experience.json")
    LongTermMemory.NOTES_PATH = os.path.join(test_store, "experience_notes.md")
    for p in (LongTermMemory.INDEX_PATH, LongTermMemory.META_PATH):
        if os.path.exists(p):
            os.remove(p)

    ltm = LongTermMemory()
    exps = [
        ("查询 RAG 领域论文并总结",
         "库内 RAG 主题仅 2 篇，如实说明不足，未凑数",
         "本地论文库 RAG 相关只有 2 篇；检索词换 'retrieval' 可补充；不要为了凑数编造"),
        ("对比 Self-RAG 与 Reflexion 的自我纠错机制",
         "两篇摘要读取成功，机制差异对比完成",
         "read_paper 的 key 必须用连字符小写：self-rag / reflexion"),
        ("计算 2023+2024 并用工具验证",
         "add(2023, 2024)=4047",
         "算术题要用 add 工具，不要让模型心算"),  # 故意写错，测语义检索能否命中

        ("检索 2023 年后方法论文",
         "topic='method' 返回 count=0，改用宽泛主题后才命中",
         "论文库主题词与论文标题强相关，'method' 这类泛词检索为空，应改用具体方向词"),
    ]
    for t, o, r in exps:
        res = ltm.remember(t, o, r)
        print(f"  写入: {t[:30]:32} → {res}")
    print(f"  库统计: {ltm.stats()}")

    # ── 3. 验收核心：recall 是否命中相关旧经验 ───
    print("\n【3】验收：长期记忆 recall 命中相关旧经验")
    probes = [
        ("我要写一个 RAG 论文综述", "查询 RAG 领域论文并总结"),
        ("Self-RAG 和 Reflexion 有什么区别", "对比 Self-RAG 与 Reflexion 的自我纠错机制"),
        ("帮我算一下 2023 加 2024 等于多少", "计算 2023+2024 并用工具验证"),
        ("想找几篇 2023 年之后的论文看看", "检索 2023 年后方法论文"),
    ]
    hit = 0
    for q, expect in probes:
        hits = ltm.recall(q, top_k=3)
        top1 = hits[0] if hits else None
        ok = bool(top1 and expect[:12] in top1["task"])
        hit += 1 if ok else 0
        print(f"  {'✅' if ok else '❌'} 查询: {q}")
        if top1:
            print(f"      top1(score={top1['score']:.3f}): {top1['task']}")
        for h in hits[1:3]:
            print(f"      #{h['id']} (score={h['score']:.3f}): {h['task'][:44]}")
    print(f"\n  验收判据『recall 命中相关旧经验』：{hit}/{len(probes)} "
          f"（top-1 命中率 {hit / len(probes):.0%}）")

    # ── 4. 经验注入效果演示 ───────────────────────
    print("\n【4】经验提示注入演示")
    hint = ltm.as_hint("我想查 RAG 论文写总结", top_k=2, threshold=0.4)
    print("  " + (hint.replace("\n", "\n  ") if hint else "(无相关经验，不注入)"))

    # ══════════════════════════════════════════════
    # Day4 遗留项验收（2026-10-01 补）
    # ══════════════════════════════════════════════
    print("\n" + "=" * 84)
    print("Day4 遗留项验收")
    print("=" * 84)

    # ── 遗留② 经验质量门禁 ───────────────────────
    print("\n【遗留②】经验质量门禁：拒收空/占位教训")
    gate_cases = [
        ("空教训", "", "empty_reflection"),
        ("解析失败占位", "(解析失败)", "parse_failed_placeholder"),
        ("过短教训", "重试", "reflection_too_short"),
        ("正常教训", "read_paper 的 key 必须用连字符小写，如 self-rag", ""),
    ]
    gate_ok = 0
    for name, refl, expect in gate_cases:
        res = ltm.remember(f"门禁测试-{name}", "测试结果", refl)
        handled = (res["stored"] is False and res.get("reason") == expect) if expect \
            else res["stored"] is True
        gate_ok += 1 if handled else 0
        print(f"  {'✅' if handled else '❌'} {name:<12} → 期望 {expect or '写入'} "
              f"| 实际 {res.get('reason') or '已写入'}")
    print(f"  门禁判定：{gate_ok}/{len(gate_cases)}")
    # 清理门禁测试写入的正常经验
    ltm.meta = [m for m in ltm.meta if not m["task"].startswith("门禁测试")]
    ltm._rebuild()

    # ── 遗留③ 经验过期/版本戳 ────────────────────
    print("\n【遗留③】经验过期/版本戳：环境变更后不召回旧经验")
    # ⚠️ 用**独立干净库**验证，避免主测试库里 scope="" 的旧经验干扰
    #    （scope="" 视为「无版本约束」，会匹配任何 scope —— 向下兼容旧数据的设计）
    import shutil
    scope_store = os.path.join(STORE, "selftest_scope")
    shutil.rmtree(scope_store, ignore_errors=True)
    os.makedirs(scope_store, exist_ok=True)
    LongTermMemory.INDEX_PATH = os.path.join(scope_store, "experience.faiss")
    LongTermMemory.META_PATH = os.path.join(scope_store, "experience.json")
    LongTermMemory.NOTES_PATH = os.path.join(scope_store, "experience_notes.md")
    ltm_s = LongTermMemory()
    ltm_s.remember("论文库有多少篇 RAG 论文", "库内仅 2 篇",
                   "单发 topic='RAG' 会漏掉 Self-RAG，需二次检索", scope="paper_db@v1")
    q = "论文库有多少篇 RAG 论文"
    hit_v1 = ltm_s.recall(q, top_k=1, threshold=0.0, scope="paper_db@v1")
    hit_v2 = ltm_s.recall(q, top_k=1, threshold=0.0, scope="paper_db@v2")
    print(f"  同版本 v1    → 命中 {len(hit_v1)} 条 {'✅' if len(hit_v1) == 1 else '❌'}")
    print(f"  新版本 v2    → 命中 {len(hit_v2)} 条 {'✅ 已过期不召回' if len(hit_v2) == 0 else '❌ 不应召回'}")
    dropped = ltm_s.expire("paper_db@v1")
    print(f"  expire('paper_db@v1') → 作废 {dropped} 条，余 {ltm_s.stats()['experiences']} 条")
    # 清理临时库，恢复主测试库路径
    LongTermMemory.INDEX_PATH = os.path.join(test_store, "experience.faiss")
    LongTermMemory.META_PATH = os.path.join(test_store, "experience.json")
    LongTermMemory.NOTES_PATH = os.path.join(test_store, "experience_notes.md")
    shutil.rmtree(scope_store, ignore_errors=True)

    # ── 总结 ──────────────────────────────────────
    print("\n" + "=" * 84)
    print("Day4 遗留项完成情况：")
    print(f"  ① 最小压缩阈值    {'✅ 已加（短会话跳过、长会话正常压缩）' if (not need_short and need_long) else '❌'}")
    print(f"  ② 经验质量门禁    {'✅ 已加' if gate_ok == len(gate_cases) else '❌'}")
    print(f"  ③ 经验过期/版本戳 {'✅ 已加（scope 过滤 + expire）' if len(hit_v2) == 0 and len(hit_v1) == 1 else '❌'}")


if __name__ == "__main__":
    _selftest()
