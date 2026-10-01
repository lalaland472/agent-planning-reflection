"""
W26 Day5 — 记忆增强决策（整合 planner + reflection + memory 闭环）
====================================================================

计划要求（Month 7 W26 D5）：
    长期记忆检索结果喂给规划/反思。整合 planner+reflection+memory。
    产出：agent_v2.py
    验收：跨会话任务能复用上次经验，完成率提升

设计（闭环）：
    ① 回忆   长期记忆检索「过去做过类似任务吗」→ 命中则把教训注入规划 prompt
    ② 规划   planner 参考过往经验拆子目标（避免重蹈覆辙，如「别用 method 这种泛词」）
    ③ 执行   工具循环（短期记忆负责上下文压缩）
    ④ 反思   执行后自我批判：对错？哪错？怎么改？
    ⑤ 沉淀   把 (任务, 结果, 反思) 写回长期记忆 → 下次可复用

⚠️ 2026-10-01 对齐（W26 Day5·补 → agent_v2）：
    Day5 复盘发现结构性断点 —— ①②③ 把经验当「**提示**」塞进 prompt，模型读不读全靠概率。
    behavior_triggers.py 证明经验可以编译成「**控制流**」（信号命中即强制执行）。
    本次把触发器层接进 agent_v2 的实际执行环：
      - 执行阶段由 _loop_triggers 驱动：先发一轮 → 检测信号 → 命中则强制执行补救 → 再发一轮
      - 沉淀阶段除文字经验外，把**反思编译成触发器**一并写回（经验→控制流）
      - 复用阶段：经验库里历史表现好的触发器直接装载
      - 效果记录：本轮与历史命中写回 SQLite（触发器置信度/成本可跨会话累积）
    完整闭环：
      回忆 → 规划 → 执行(触发器兜底) → 反思 → 沉淀(文字经验 + 可执行触发器) → 复用
    开关：use_triggers=False 可退回纯 Day5 行为，便于 A/B 消融。

验收实验（A/B 对照，跨会话）：
    用一个「会踩坑」的任务集，先跑 baseline（无记忆）暴露失败与教训，
    再把教训写入记忆库，然后跑 memory-on（有记忆）看完成率是否提升。
    同时跑 memory-on-without-seed 作为对照，排除「单纯多跑一次」的运气因素。
"""

import os
import re
import sys
import json
import sqlite3
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from planner import (  # noqa: E402
    load_env, Usage, TaskEnv, tool_schemas, _loop, tool_llm, plain_llm,
    make_plan, PlanTracker, TASKS, NOTES,
)
from memory import ShortTermMemory, LongTermMemory, STORE  # noqa: E402
from behavior_triggers import (  # noqa: E402
    Trigger, run_with_triggers, compile_trigger, validate_trigger,
    _round_hint, _last_obs,
)
from behavior_patterns import BUILTIN_PATTERNS  # noqa: E402


# ══════════════════════════════════════════════════
# 反思 Agent（Day1 Reflexion 的复用形态）
# ══════════════════════════════════════════════════
REFLECT_SYS = (
    "你是【反思者】。任务失败后，对这次执行做自我批判。\n"
    "⚠️ 核心要求（对齐 Reflexion 论文的反思形式）：反思必须**对准某个具体动作开处方**，"
    "而不是写抽象原则。参考论文原句：'I searched the wrong title ... I should have searched "
    "the show\'s main character, Gorden Kaye'。\n"
    "写完后自检：如果 next_action 里没有出现**具体的工具名 / 参数值 / 论文 key**，"
    "就说明写得太抽象，必须重写。\n"
    "只输出 JSON：\n"
    '{"correct": true/false, '
    '"wrong_step": "第几步、哪个具体调用错了（带上工具名和参数值）", '
    '"root_cause": "为什么会这么错", '
    '"next_action": "下次遇到同类任务，具体该发哪个调用、传什么参数（可直接照抄执行）"}'
)


def reflect(env, task, final, calls, usage=None, eval_feedback=""):
    """执行后自我批判 → 结构化反思（对齐论文：对准具体动作开处方）。

    ⚠️ 2026-09-19 改进（精读 Reflexion 论文后落地）：
      旧版输出 what_wrong / how_to_fix，实测产出**抽象格言**（"要确认接口语义"），
      而任务需要**具体事实/动作**（"就是 2 篇，RAPTOR + GraphRAG"）。
      论文的反思是"I should have searched Gorden Kaye"级别的**具体动作处方**。
      改进：① 三段式 wrong_step / root_cause / next_action；
            ② 把 Evaluator 的失败原因作为输入喂进来（论文里反思紧跟单测反馈）；
            ③ prompt 强制自检 next_action 必须含具体工具名/参数值。
    """
    trace = "\n".join(
        f"- {c['name']}({json.dumps(c['args'], ensure_ascii=False)[:80]}) → "
        f"{json.dumps(c['obs'], ensure_ascii=False)[:100]}"
        for c in calls[-10:])
    fb = f"\n【Evaluator 反馈（这是判失败的具体原因）】\n{eval_feedback}\n" if eval_feedback else ""
    prompt = (f"任务：{task}\n\n工具调用轨迹：\n{trace}\n{fb}\n"
              f"最终答案：{final[:600]}\n\n请反思。")
    out = plain_llm(env, [{"role": "system", "content": REFLECT_SYS},
                          {"role": "user", "content": prompt}],
                    usage=usage, max_tokens=500, temperature=0.2)
    t = out.strip()
    if "```" in t:
        for s in t.split("```"):
            s = s.replace("json", "", 1).strip()
            if s.startswith("{"):
                t = s
                break
    i, j = t.find("{"), t.rfind("}")
    if i >= 0 and j > i:
        try:
            d = json.loads(t[i:j + 1])
            return {"correct": bool(d.get("correct")),
                    "wrong_step": str(d.get("wrong_step", d.get("what_wrong", "")))[:250],
                    "root_cause": str(d.get("root_cause", ""))[:250],
                    "next_action": str(d.get("next_action", d.get("how_to_fix", "")))[:300]}
        except json.JSONDecodeError:
            pass
    return {"correct": True, "wrong_step": "(解析失败)", "root_cause": "",
            "next_action": ""}


# ══════════════════════════════════════════════════
# 记忆增强的规划（核心：把经验喂给规划器）
# ══════════════════════════════════════════════════
def make_plan_with_memory(env, task, ltm, usage=None, threshold=0.45, scope=""):
    """检索长期记忆 → 命中则把教训注入规划 prompt。

    ⚠️ Day5 遗留④：接入 scope（环境版本戳）。
       事实型经验会随环境过期（如「库里只有 2 篇」），必须按当前版本过滤；
       方法型经验不过期，跨版本依然注入（由 memory.recall 内部区分）。
    """
    hint = (ltm.as_hint(task, top_k=2, threshold=threshold, scope=scope)
            if ltm else "")
    real_tools = [t["function"]["name"] for t in tool_schemas()]
    tool_desc = "\n".join(
        f"  - {t['function']['name']}: {t['function']['description']}" for t in tool_schemas())
    mem_block = (f"\n\n{hint}\n"
                 "⚠️ 以上是过去同类任务的经验，请据此调整你的计划，避免重复踩坑。\n"
                 if hint else "")
    prompt = (
        f"任务：{task}{mem_block}\n"
        f"可用工具（tool 字段只能从以下枚举中选，不得自创）：\n{tool_desc}\n\n"
        "请把这个任务拆成 3-5 个**具体、可执行**的子目标，形成执行顺序。\n"
        "规则：\n"
        "1) 每个子目标如果有对应工具，tool 必须精确等于上面的工具名之一；\n"
        "2) 纯汇总/作答类子目标不需要工具，tool 填 null；\n"
        "3) 至少 2 个子目标要带真实工具。\n"
        "只输出 JSON，无多余文字：\n"
        '{"goal": "对任务的复述", "subgoals": [{"id": 1, "desc": "子目标描述", '
        '"tool": "工具名或null"}]}'
    )
    out = plain_llm(env, [{"role": "user", "content": prompt}], usage=usage,
                    max_tokens=700, temperature=0.2)
    cleaned = out.strip()
    if cleaned.startswith("```"):
        cleaned = cleaned.split("\n", 1)[1].rsplit("```", 1)[0].strip()
    try:
        d = json.loads(cleaned)
        subs = d.get("subgoals", [])
    except Exception:
        return [], task, hint, ""
    for g in subs:
        t = g.get("tool")
        if t in (None, "", "null", "None"):
            g["tool"] = None
        elif t not in real_tools:
            g["tool_illegal"] = t
            g["tool"] = None
    return subs, d.get("goal", task), hint, prompt


# ══════════════════════════════════════════════════
# agent_v2：完整闭环
# ══════════════════════════════════════════════════
def _loop_triggers(env, task, exec_msgs, te, triggers, usage=None, max_rounds=2,
                   max_steps=12):
    """带触发器的执行环：执行 → 检测信号 → 命中强制补救 → 再执行（trial loop）。

    与 planner._loop 的区别：不动工具调用协议（方案 B），只在**轮次之间**做检测
    与强制动作 —— 发一轮，扫真实调用轨迹，命中则 apply() 强制执行补救，把
    「系统已强制执行的动作」以正式消息形式插回对话，再发下一轮让模型采纳。

    过滤策略：只装载 should_fire() 为真的触发器（置信度/成本门槛）。
    返回 (final, all_calls, steps, fired_log)。
    """
    u = usage if usage is not None else Usage()
    active = [tr for tr in triggers if tr.should_fire()]
    skipped = [tr.name for tr in triggers if tr not in active]
    all_calls = []
    fired_log = []
    final = ""
    steps = 0
    msgs = list(exec_msgs)

    for rnd in range(max_rounds):
        final, calls, steps = _loop(env, msgs, te, usage=u, max_steps=max_steps)
        all_calls.extend(calls)
        if rnd == max_rounds - 1:
            break
        hits = [tr for tr in active if tr.match(all_calls, final, rnd)]
        if not hits:
            break
        new_logs = []
        for tr in hits:
            did, note = tr.apply(env, te, all_calls, final, rnd, usage=u)
            if did:
                tr.fire_count += 1
                new_logs.append({"round": rnd + 1, "trigger": tr.name,
                                 "action": tr.action, "note": note})
        if not new_logs:
            break
        # 把补救动作（含强制工具调用的真实结果）以正式消息插回对话，让下一轮看到
        supplement = ("【系统已强制执行的补救动作（必须采纳，不要忽略）】\n"
                      + "\n".join(f"  - 第{x['round']}轮触发 [{x['trigger']}]：{x['note']}"
                                  for x in new_logs))
        fired_log.extend(new_logs)
        msgs.append({"role": "user", "content": supplement})

    return final, all_calls, steps, fired_log


# ══════════════════════════════════════════════════
# 触发器库（SQLite 持久化：让「经验→控制流」可跨会话复用）
# ══════════════════════════════════════════════════
# ⚠️ 2026-10-01：Day5 复盘遗留①「rescue_count 未生效」的落地——
#    触发器不再是一次实验的临时对象，而是有历史命中数/成功率/成本的**资产**。
#    存放于 STORE/day5/triggers.sqlite，与记忆库同目录。

TRIGGER_DB = os.path.join(STORE, "day5", "triggers.sqlite")


class TriggerStore:
    """触发器持久化库：编译产物入库，装载时回填历史统计。"""

    def __init__(self, path=None):
        self.path = path or TRIGGER_DB
        os.makedirs(os.path.dirname(self.path), exist_ok=True)
        self.conn = sqlite3.connect(self.path)
        self.conn.execute("""
            CREATE TABLE IF NOT EXISTS triggers (
                name TEXT PRIMARY KEY, pattern TEXT, source_task TEXT,
                hits INTEGER DEFAULT 0, successes INTEGER DEFAULT 0,
                retry_cost REAL DEFAULT 0, created_at TEXT
            )""")
        self.conn.commit()

    def upsert(self, tr: Trigger, task_id=None):
        """入库（同名则累加统计，不覆盖历史）。"""
        row = self.conn.execute("SELECT hits, successes, retry_cost FROM triggers "
                                "WHERE name=?", (tr.name,)).fetchone()
        if row:
            hits, succ, cost = row
            self.conn.execute(
                "UPDATE triggers SET hits=?, successes=?, retry_cost=?, pattern=? "
                "WHERE name=?", (hits + tr.hits, succ + tr.successes,
                                 tr.retry_cost or cost, json.dumps(tr.to_pattern()),
                                 tr.name))
        else:
            self.conn.execute(
                "INSERT INTO triggers (name, pattern, source_task, hits, "
                "successes, retry_cost, created_at) VALUES (?,?,?,?,?,?,?)",
                (tr.name, json.dumps(tr.to_pattern()),
                 task_id or tr.source_task, tr.hits, tr.successes,
                 tr.retry_cost, time.strftime("%Y-%m-%d %H:%M:%S")))
        self.conn.commit()

    def record_effect(self, name, success, tokens=0.0):
        """回落一次效果记录到持久库（跨会话累积）。"""
        row = self.conn.execute("SELECT hits, successes, retry_cost FROM triggers "
                                "WHERE name=?", (name,)).fetchone()
        if not row:
            return
        hits, succ, cost = row
        hits += 1
        succ += 1 if success else 0
        if tokens > 0:
            a = 0.4 if not cost else 0.3
            cost = (1 - a) * cost + a * tokens
        self.conn.execute("UPDATE triggers SET hits=?, successes=?, retry_cost=? "
                          "WHERE name=?", (hits, succ, cost, name))
        self.conn.commit()

    def load(self):
        """装载全部触发器，回填历史统计（供 should_fire 决策）。"""
        out = []
        for name, pattern, src, hits, succ, cost in self.conn.execute(
                "SELECT name, pattern, source_task, hits, successes, retry_cost "
                "FROM triggers"):
            try:
                d = json.loads(pattern)
            except (TypeError, json.JSONDecodeError):
                continue
            if not validate_trigger(d):
                continue
            out.append(Trigger(name=name, when=d.get("when"),
                               action=d.get("action"),
                               payload=d.get("payload") or {},
                               instruction=str(d.get("instruction", "")),
                               source_task=src or "", hits=hits,
                               successes=succ, retry_cost=cost))
        return out

    def stats(self):
        r = self.conn.execute("SELECT COUNT(*), COALESCE(SUM(hits),0), "
                              "COALESCE(SUM(successes),0) FROM triggers").fetchone()
        return {"triggers": r[0], "hits": r[1], "successes": r[2]}

    def close(self):
        self.conn.close()


def load_triggers(store=None):
    """装载触发器集：库里优先，空库回退内置触发器集（同构产品能力）。"""
    if store is not None:
        trs = store.load()
        if trs:
            return trs
    return [Trigger(**p) for p in BUILTIN_PATTERNS]


def save_trigger(store, tr, task_id=None):
    """编译产物入库（先 schema 校验，挡幻觉参数）。"""
    if not validate_trigger(tr.to_pattern()):
        return False
    store.upsert(tr, task_id=task_id)
    return True


def record_trigger_effect(store, tr, success, tokens=0.0, task_id=None):
    """记录命中效果（本轮 + 历史），让 rescue 归因可跨会话累积。"""
    store.record_effect(tr.name, success, tokens=tokens)


# ══════════════════════════════════════════════════
# agent_v2：完整闭环
# ══════════════════════════════════════════════════
def run_agent_v2(env, task, ltm=None, usage=None, use_memory=True, use_stm=False,
                 reflect_and_store=True, threshold=0.45, verbose=False,
                 eval_feedback="", use_triggers=True, triggers=None,
                 trigger_store=None, task_id=None, max_rounds=2, judge=None,
                 scope=""):
    """记忆增强 Agent：回忆 → 规划 → 执行(触发器兜底) → 反思 → 沉淀。

    use_triggers=True 时，③ 执行阶段由 _loop_triggers 驱动：
    先跑一轮，检测信号，命中则强制执行补救并再跑一轮（Reflexion 的 trial loop）。
    triggers 为 None 时，自动从触发器库（trigger_store）或内置集装载。
    """
    te = TaskEnv()
    u = usage if usage is not None else Usage()
    info = {"memory_hit": "", "plan": "", "reflection": None, "stored": None,
            "compress": 0, "fired": [], "triggers_used": [],
            "compiled": 0, "tokens": 0}

    # ── ① 回忆 + ② 规划 ───────────────────────────
    if use_memory and ltm:
        subs, goal, hint, prompt = make_plan_with_memory(env, task, ltm, usage=u,
                                                         threshold=threshold,
                                                         scope=scope)
        info["memory_hit"] = hint
    else:
        subs, goal = make_plan(env, task, usage=u)
        hint = ""
    tracker = PlanTracker(subs)
    info["plan"] = "\n".join(f"  {i+1}. {it['desc']} [tool={it['tool']}]"
                             for i, it in enumerate(subs))

    # 装载触发器（经验 → 控制流）
    tr_list = []
    if use_triggers:
        tr_list = triggers if triggers is not None else load_triggers(trigger_store)

    # ── ③ 执行 ───────────────────────────────────
    sys_msg = ("你是会用工具的助手，正在**按既定计划**执行任务。"
               "按顺序推进子目标，完成后说明每个子目标的结论。"
               "用工具获取真实数据，禁止编造；库内信息不足要如实说明。")
    ctx = tracker.to_context()
    stm = ShortTermMemory(max_turns=6, keep_recent=3) if use_stm else None
    exec_msgs = [{"role": "system", "content": sys_msg},
                 {"role": "user", "content":
                  f"任务：{task}\n\n{ctx}\n\n请开始执行，按子目标顺序推进。"}]
    if stm:
        stm.add("user", f"任务：{task}")
    if tr_list:
        tok0 = u.total_tokens
        final, call_list, steps, fired = _loop_triggers(
            env, task, exec_msgs, te, tr_list, usage=u, max_rounds=max_rounds)
        info["fired"] = fired
        retry_tok = u.total_tokens - tok0
    else:
        final, call_list, steps = _loop(env, exec_msgs, te, usage=u)
        retry_tok = 0
    info["calls"] = len(call_list)
    info["triggers_used"] = [tr.name for tr in tr_list]
    info["tokens"] = u.total_tokens

    # ── ④ 反思 ───────────────────────────────────
    refl = None
    if reflect_and_store:
        refl = reflect(env, task, final, call_list, usage=u, eval_feedback=eval_feedback)
        info["reflection"] = refl

        # ── ⑤ 沉淀到长期记忆（文字经验 + 可执行触发器）────
        if ltm is not None:
            outcome = (final or "")[:200]
            # ⚠️ 2026-09-19：存「动作处方」（next_action）而非抽象原则——
            #    检索命中后要能直接照抄执行，不能只给一句格言。
            lesson = (refl.get("next_action") or refl.get("root_cause")
                      or refl.get("wrong_step") or "")
            # ⚠️ Day5 遗留③④：事实型/方法型分流 + 带环境版本戳
            #   memory.remember 会自动推断 kind：
            #     fact（会过期）→ 强制带 scope，检索时严格过滤
            #     method（不过期）→ 跨版本仍可召回
            res = ltm.remember(task, outcome, lesson, scope=scope)
            info["stored"] = res
            if verbose:
                print(f"      ↳ 沉淀经验: {res} (kind={res.get('kind','?')}, scope={scope or '—'})")

        # 经验 → 触发器（控制流）：把这次反思编译成可执行触发器入库
        if use_triggers and trigger_store is not None and refl is not None \
                and not refl.get("correct"):
            trs = compile_trigger(env, task, call_list, refl, usage=u)
            for tr in trs:
                if save_trigger(trigger_store, tr, task_id=task_id):
                    info["compiled"] += 1
                    if verbose:
                        print(f"      ⚙️ 编译并入库触发器 [{tr.name}] "
                              f"when={tr.when} action={tr.action}")

    # ── 组装结果 dict（判据函数需要 final/call_list，先建好再记效果）────
    result = {"final": final, "call_list": call_list, "calls": len(call_list),
              "steps": steps, "usage": u, **info}

    # ── 效果记录（跨会话累积：触发器置信度/成本）─────────────
    if use_triggers and trigger_store is not None and info["fired"]:
        # ⚠️ 判据函数（task["check"]）接受的是**完整结果 dict**（含 final/call_list），
        #    不能只传 info（info 里没有 final）。
        outcome_ok = judge(result) if judge else None
        if outcome_ok is None:
            outcome_ok = bool(refl and refl.get("correct"))
        # 成本只算触发器带来的额外轮次（trial loop 的固有开销）
        per = retry_tok / max(len(info["fired"]), 1)
        for tr in tr_list:
            if tr.fire_count > 0:
                record_trigger_effect(trigger_store, tr, success=bool(outcome_ok),
                                      tokens=per, task_id=task_id)

    return result


# ══════════════════════════════════════════════════
# 验收实验：跨会话复用经验，完成率是否提升
# ══════════════════════════════════════════════════
# 「深坑」任务集（v2 收紧版，2026-09-19）
# ─────────────────────────────────────────────────
# 上一版（P1-P3）判据太宽松，无记忆对照也全过（3/3），测不出增益。
# 本版改为「反直觉约束」：这些坑只有真的踩过（或从经验里知道）才能避开，
# 靠常识绕不过去。每条的 check 都是**硬判据**，无关记忆必失败。
#
# 关键设计：坑本身在工具环境里是**真实存在**的（非编造），
# 所以无记忆时模型只能靠试错，而有记忆时可直接照抄结论。
TRAP_TASKS = [
    {
        "id": "P1",
        "desc": "精确总数（坑：`RAG` 关键词只命中 2 篇，正确答案是 2 且必须点名两篇）",
        "task": "论文库中主题含 'RAG' 的论文一共有几篇？请给出确切数字并逐一点名。",
        # 硬判据：必须同时给出 2、两个名字（RAPTOR/GraphRAG 或对应中文），且不含 3/4/5 篇的说法
        "check": lambda r: _p1_strict(r["final"]),
        "check_desc": "总数=2 且点名 RAPTOR+GraphRAG，不虚报篇数",
        # ⚠️ 事实型任务：答案（「2 篇」）随论文库内容变化 → 必须带版本戳
        "kind": "fact",
    },
    {
        "id": "P2",
        "desc": "反直觉 year_from（坑：year_from 是**下界闭区间**，year_from=2023 应含 2023 本身）",
        "task": "用 search_papers 检索 year_from=2023 的论文，回答：返回结果里年份最早的是哪一年？",
        # 硬判据：答案必须是 2022（因为 reflexion/react 是 2022/2023，而 year_from=2023 会过滤掉 react(2022),
        # 真实最早是 2023 —— 但模型常误以为 year_from 是「大于」导致答 2024，或误答 2022）
        "check": lambda r: _p2_strict(r),
        "check_desc": "必须真的调用过 search_papers(year_from=2023) 并据实答年份",
        # 方法型：year_from 的语义不随环境变化，是工具契约
        "kind": "method",
    },
    {
        "id": "P3",
        "desc": "key 字面量（坑：`self-rag` 是连字符小写，`self_rag`/`SelfRAG` 全部报错）",
        "task": "读取论文 'self_rag' 的摘要并说明其核心机制。",
        # 硬判据：任务给的是错误 key（下划线），模型必须自己试出正确写法 self-rag 才可能成功
        "check": lambda r: _p3_strict(r),
        "check_desc": "必须用 self-rag（连字符）成功读到，且最后一篇答案基于真实摘要",
        # 方法型：key 命名规范不随环境变化
        "kind": "method",
    },
]


def _p1_strict(text):
    """P1：总数必须=2，且点名两篇，且不含把篇数说成 3/4/5 的表述。"""
    t = str(text)
    has_two = ("2 篇" in t or "两篇" in t or "2篇" in t or "共 2" in t
               or "只有 2" in t or "仅 2" in t)
    named = (("RAPTOR" in t or "raptor" in t.lower())
             and ("GraphRAG" in t or "graphrag" in t.lower()))
    overclaim = any(k in t for k in ("3 篇", "三篇", "4 篇", "四篇", "5 篇", "五篇"))
    return has_two and named and not overclaim


def _p2_strict(r):
    """P2：必须真的用 year_from=2023 调过 search_papers，且答案给出年份并自洽。"""
    called = any(c["name"] == "search_papers"
                 and int(c["args"].get("year_from") or 0) == 2023
                 for c in r["call_list"])
    if not called:
        return False
    t = str(r["final"])
    return "2023" in t


def _p3_strict(r):
    """P3：必须用 self-rag（连字符）读成功，且答案里出现该论文真实摘要的关键词。"""
    ok_read = False
    for c in r["call_list"]:
        if c["name"] != "read_paper":
            continue
        obs = c.get("obs")
        if isinstance(obs, dict) and obs.get("title") and "error" not in obs:
            if "self" in str(obs.get("title", "")).lower():
                ok_read = True
    if not ok_read:
        return False
    t = str(r["final"])
    # 真实摘要关键词：反思 token / 是否检索 / 相关性 / 忠实性
    return any(k in t for k in ("反思 token", "反思token", "相关性", "忠实性", "是否检索"))


def _count_ok(text):
    """P2：是否如实给出库内论文总数 2（RAG 口径）。"""
    t = str(text)
    return ("2 篇" in t or "两篇" in t or "共2" in t or "2篇" in t
            or "只有 2" in t or "仅 2" in t)


def _read_ok(r):
    """P3：read_paper 是否成功命中 real key（obs 无 error）。"""
    for c in r["call_list"]:
        if c["name"] != "read_paper":
            continue
        obs = c.get("obs")
        if isinstance(obs, dict) and "error" not in obs and obs.get("title"):
            return True
    return False


def _probe_difficulty(env, u):
    """⭐ 任务难度滚动更新（Day5 复盘遗留①）——
    验收前先测「当前模型裸跑完成率」。若某任务已被模型追平（>80%），
    它就不再具备区分度，本轮增益会被它污染成 0。

    → 这是从 Day5「两组都满分→增益不可测」直接推出的流程修正。
    """
    print(f"\n{'━' * 88}")
    print("【第 0 步】任务难度探测：先测裸跑完成率（无记忆/无触发器）")
    print(f"{'━' * 88}")
    alive = []
    for t in TRAP_TASKS:
        r = run_agent_v2(env, t["task"], ltm=None, usage=u, use_memory=False,
                         use_triggers=False, reflect_and_store=False,
                         scope=t.get("scope", ""))
        ok = bool(t["check"](r))
        tag = "✅ 裸跑已过（无区分度）" if ok else "❌ 仍有坑（保留）"
        print(f"  [{t['id']}] {tag} | token {r['usage'].total_tokens} | {t['desc']}")
        if not ok:
            alive.append(t["id"])
    if alive:
        print(f"  → 有区分度的任务：{alive}，本轮增益应以它为准")
    else:
        print("  ⚠️ 全部任务已被模型追平 → 需换一批坑（难度滚动更新）")
    return alive


def _seed_memory(env, ltm, u, tag):
    """第 1 轮：无记忆跑一遍，把踩到的坑沉淀成经验。

    ⚠️ 2026-09-19：把 Evaluator 的**失败原因**作为输入喂给反思器（对齐论文：
       反思紧跟单测/评测反馈），否则反思只能猜“哪里错了”，产出抽象格言。
    """
    print(f"\n{'━' * 88}")
    print(f"【{tag}】第 1 轮：无记忆执行（暴露坑） → 反思 → 沉淀经验")
    print(f"{'━' * 88}")
    for t in TRAP_TASKS:
        r = run_agent_v2(env, t["task"], ltm=ltm, usage=u,
                         use_memory=False, reflect_and_store=True, verbose=True,
                         eval_feedback=f"判据：{t['check_desc']}",
                         scope=t.get("scope", ""))
        ok = bool(t["check"](r))
        print(f"  [{t['id']}] {'✅' if ok else '❌'} {t['desc']}")
        refl = r.get("reflection") or {}
        print(f"       反思: 对={refl.get('correct')}")
        print(f"       错在哪步: {refl.get('wrong_step','')[:110]}")
        print(f"       根因: {refl.get('root_cause','')[:110]}")
        print(f"       下次动作: {refl.get('next_action','')[:150]}")
        print(f"       沉淀: {r.get('stored')}")


def _run_with_memory(env, ltm, u, label, use_memory, use_triggers=False,
                     trigger_store=None, tasks=None):
    """第 2 轮：带/不带记忆（+触发器）重跑，看完成率。"""
    tasks = tasks or TRAP_TASKS
    hits = 0
    detail = []
    for t in tasks:
        r = run_agent_v2(env, t["task"], ltm=ltm, usage=u,
                         use_memory=use_memory, reflect_and_store=False,
                         use_triggers=use_triggers, trigger_store=trigger_store,
                         task_id=t["id"], judge=t["check"],
                         scope=t.get("scope", ""))
        ok = bool(t["check"](r))
        hit = bool(r.get("memory_hit"))
        hits += 1 if ok else 0
        detail.append((t["id"], ok, hit, r["calls"], r["usage"].total_tokens,
                       len(r.get("fired", []))))
        print(f"  [{t['id']}] {'✅' if ok else '❌'} 记忆注入={'有' if hit else '无'}"
              f" | 触发 {len(r.get('fired', []))} | 工具 {r['calls']} "
              f"| token {r['usage'].total_tokens} | {t['desc']}")
    print(f"  → {label} 完成率: {hits}/{len(tasks)}")
    return hits, detail


def _reset_day5_store():
    """清空 day5 实验库（稳定性验证前后用）。"""
    for p in (LongTermMemory.INDEX_PATH, LongTermMemory.META_PATH):
        if os.path.exists(p):
            os.remove(p)


def _stability_check(env, u, tasks, rounds):
    """Day5 遗留⑤：单次样本会漂 —— 同任务重复 N 次，看结论是否稳定。

    对每个任务、每组（有记忆 / 无记忆）分别跑 rounds 次，统计：
      - 通过率（均值）
      - 方差/极差（稳定性）
      结论才可归因：若两组区间重叠，说明单次差异是噪声。
    """
    import statistics
    for t in tasks:
        row = {"mem": [], "ctrl": []}
        for _ in range(rounds):
            # 有记忆（独立库，每次重建避免累积干扰）
            _reset_day5_store()
            ltm_m = LongTermMemory()
            ltm_m.remember(t["task"], "(种子)", "去试正确做法，不要卡在报错上")
            rm = run_agent_v2(env, t["task"], ltm=ltm_m, usage=u, use_memory=True,
                              reflect_and_store=False, use_triggers=False,
                              scope=t.get("scope", ""))
            row["mem"].append(1 if t["check"](rm) else 0)
            # 无记忆
            _reset_day5_store()
            ltm_c = LongTermMemory()
            rc = run_agent_v2(env, t["task"], ltm=ltm_c, usage=u, use_memory=False,
                              reflect_and_store=False, use_triggers=False,
                              scope=t.get("scope", ""))
            row["ctrl"].append(1 if t["check"](rc) else 0)
        m, c = row["mem"], row["ctrl"]
        print(f"  [{t['id']}] 有记忆 {sum(m)}/{rounds} "
              f"(均值{statistics.mean(m):.2f} 极差{max(m)-min(m)}) | "
              f"无记忆 {sum(c)}/{rounds} (均值{statistics.mean(c):.2f}) | {t['desc'][:30]}")
    print("  → 若两组极差/均值接近，说明单次差异属噪声（验譁单次结论不可靠）")


def main():
    env = load_env()
    if not env["api_key"]:
        print("❌ 未找到 LLM_API_KEY")
        return

    # 独立实验库
    exp_store = os.path.join(STORE, "day5")
    os.makedirs(exp_store, exist_ok=True)
    LongTermMemory.INDEX_PATH = os.path.join(exp_store, "experience.faiss")
    LongTermMemory.META_PATH = os.path.join(exp_store, "experience.json")
    LongTermMemory.NOTES_PATH = os.path.join(exp_store, "experience_notes.md")
    for p in (LongTermMemory.INDEX_PATH, LongTermMemory.META_PATH):
        if os.path.exists(p):
            os.remove(p)

    print("=" * 88)
    print("W26 Day5 — 记忆增强决策：跨会话经验复用 A/B 实验")
    print("=" * 88)

    u = Usage()
    ltm = LongTermMemory()
    tstore = TriggerStore()   # 触发器库（持久化，跨会话复用）

    # ── 第 0 步：任务难度探测（难度滚动更新）──────────
    alive = _probe_difficulty(env, u)
    tasks = [t for t in TRAP_TASKS if t["id"] in alive] or TRAP_TASKS

    # ⚠️ Day5 遗留⑤：单次样本会漂 —— 每个任务重复 R 次取均值/一致性
    ROUNDS = int(os.environ.get("DAY5_ROUNDS", "1"))
    if ROUNDS > 1:
        print(f"\n{'=' * 88}\n📊 稳定性验证：每任务重复 {ROUNDS} 次（遗留⑤）\n{'=' * 88}")
        _stability_check(env, u, tasks, ROUNDS)
        # 稳定性验证用独立库，避免污染；完成后重建主库
        _reset_day5_store()
        ltm = LongTermMemory()

    # ── 会话 A：无记忆（baseline，同时播种经验 + 编译触发器）──
    _seed_memory(env, ltm, u, "会话 A")
    print(f"\n  经验库: {ltm.stats()}")

    # 从经验库编译触发器（对齐 Day5 补：经验 → 控制流）
    print(f"\n{'━' * 88}")
    print("【编译】从失败经验编译行为模式触发器")
    print(f"{'━' * 88}")
    compiled = 0
    for t in tasks:
        r = run_agent_v2(env, t["task"], ltm=ltm, usage=u, use_memory=True,
                         reflect_and_store=True, use_triggers=True,
                         trigger_store=tstore, task_id=t["id"],
                         eval_feedback=f"判据：{t['check_desc']}",
                         scope=t.get("scope", ""))
        compiled += r.get("compiled", 0)
        print(f"  [{t['id']}] {'✅' if t['check'](r) else '❌'} "
              f"编译入库 {r.get('compiled', 0)} 条触发器")
    print(f"  → 触发器库: {tstore.stats()}")

    # ── 会话 B：有记忆 + 有触发器（复用会话 A 沉淀的资产）──
    print(f"\n{'━' * 88}")
    print("【会话 B】第 2 轮：有记忆 + 触发器（复用上次经验→控制流）")
    print(f"{'━' * 88}")
    hits_mem, det_mem = _run_with_memory(env, ltm, u, "有记忆+触发器",
                                         use_memory=True, use_triggers=True,
                                         trigger_store=tstore, tasks=tasks)

    # ── 对照：清库后无记忆无触发器重跑（排除运气因素）──
    print(f"\n{'━' * 88}")
    print("【对照】清库后无记忆重跑（排除运气因素）")
    print(f"{'━' * 88}")
    for p in (LongTermMemory.INDEX_PATH, LongTermMemory.META_PATH):
        if os.path.exists(p):
            os.remove(p)
    ltm2 = LongTermMemory()
    hits_ctrl, _ = _run_with_memory(env, ltm2, u, "无记忆(对照)", use_memory=False,
                                    use_triggers=False, tasks=tasks)
    print(f"\n  触发器库最终状态: {tstore.stats()}")

    # ── 汇总 ─────────────────────────────────────
    n = len(tasks)
    print(f"\n{'=' * 88}\n📊 验收结果：跨会话经验复用（经验→提示 + 经验→控制流）\n")
    print(f"{'组别':<24}{'完成率':<10}{'说明':<40}")
    print(f"{'第0步 裸跑探测':<24}{f'{n-len(alive)}/{len(TRAP_TASKS)}':<10}"
          f"{'任务难度滚动更新（有区分度的留存）':<40}")
    print(f"{'会话A 无记忆(播种)':<24}{'—':<10}{'暴露坑 + 沉淀经验 + 编译触发器':<40}")
    print(f"{'会话B 有记忆+触发器':<24}{f'{hits_mem}/{n}':<10}{'复用经验与触发器':<40}")
    print(f"{'对照 无记忆(清库)':<24}{f'{hits_ctrl}/{n}':<10}{'排除运气因素':<40}")

    delta = hits_mem - hits_ctrl
    print(f"\n  记忆增益 = 有记忆({hits_mem}) − 无记忆对照({hits_ctrl}) = "
          f"{'+' if delta >= 0 else ''}{delta}")
    if delta > 0:
        print("  ✅ 验收通过：跨会话经验复用带来完成率提升")
    elif hits_mem == n:
        print("  ⚠️ 有记忆已满分，但对照也达标 → 无法区分是记忆作用还是任务太简单")
    else:
        print("  ❌ 未观察到提升")

    print(f"\n  明细（会话B）：")
    print(f"  {'任务':<6}{'通过':<6}{'记忆注入':<10}{'触发':<6}{'工具':<6}{'token':<8}")
    for tid, ok, hit, calls, tok, fired in det_mem:
        print(f"  {tid:<6}{'✅' if ok else '❌':<6}{'有' if hit else '无':<10}"
              f"{fired:<6}{calls:<6}{tok:<8}")
    print(f"\n  触发器台账（跨会话累积，供下次 should_fire 决策）：")
    for tr in tstore.load():
        print(f"    - {tr.name:<22} when={tr.when:<16} hits={tr.hits:<3} "
              f"救回={tr.successes:<3} 置信度={tr.confidence():.2f} "
              f"成本≈{tr.retry_cost:.0f}tok 触发={tr.should_fire()}")
    tstore.close()


if __name__ == "__main__":
    main()
