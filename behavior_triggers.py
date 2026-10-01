"""
W26 Day5 · 补 — 行为模式触发器（Behavior Pattern Triggers）
================================================================

问题（Day5 复盘揭示的结构性断点）：
    Day5 四段链路 ①反思质量 ✅ ②记忆检索 ✅ ③注入生效 ✅ ④**模型照做** ❓
    断点定位在 ④：记忆命中了、教训也精确到参数值了，但**只把经验当"提示"塞进 prompt**，
    模型读不读、照不照做，全靠概率。

    论文 Reflexion 为何能生效？因为它把教训**变成控制流**——
    "上次这样错了" → "这次必须那样做"，然后**强制重试**。
    而不是把教训打印在 prompt 里让模型"参考"。

本模块的答案：把「经验」编译成「**可执行触发器**」（event → condition → action）。

设计（三段式，对齐 Reflexion 的 trial loop 语义）：
    ① 触发条件（when）   在什么信号出现时触发 —— 从经验/反思里抽取的可检测条件
    ② 触发动作（then）   触发后**强制执行**什么 —— 注入指令 / 改参数重试 / 拦截返回
    ③ 效果记录（record） 触发了几次、救回了几个 —— 让"记忆到底起没起作用"可量化

与 Day5 的本质区别：
    Day5：经验 → 一段文字 → 塞进 prompt → 模型自行决定（被动，靠概率）
    本模块：经验 → 一个触发器 → 命中信号就**自动执行**（主动，确定性）

验收标准（对齐计划 D5 原判据的精神）：
    同一批"深坑"任务，带触发器组 vs 无触发器组：
    ① 完成率提升（≥1 个坑被触发器救回），或
    ② 完成率持平但触发器的**确定性**可证（命中即执行，不依赖模型自觉）

⚠️ API 坑：deepseek-v4-flash 是 reasoning 模型，请求体必须
   {"thinking": {"type": "disabled"}}，否则 content 为空（见 planner.plain_llm）。

运行：
    python3 behavior_triggers.py            # 单测 + 实跑对比
    python3 behavior_triggers.py --selftest # 只跑离线自测（不花 token）
"""

import os
import sys
import json
import re
import time
import copy

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from planner import (  # noqa: E402
    load_env, Usage, TaskEnv, tool_schemas, _loop, plain_llm, PAPER_DB,
)
from behavior_patterns import BUILTIN_PATTERNS  # noqa: E402


# ══════════════════════════════════════════════════
# 〇、工具白名单（供编译期 schema 校验用）
# ══════════════════════════════════════════════════
# ⚠️ 2026-10-01 对齐 agent_v2：编译出的触发器参数必须先过工具 schema 再入库。
#    上轮实测 LLM 编译出幻觉参数 match="contains"（见 Day5 复盘遗留②）。
VALID_TOOL_NAMES = tuple(t["function"]["name"] for t in tool_schemas())
_PLACEHOLDER_RE = re.compile(r"^\$(bad|last\.[A-Za-z_][A-Za-z0-9_]*)$")
WARMUP_HITS = 2   # 前 N 次命中无条件触发（先拿到成本/成功率样本）


# ══════════════════════════════════════════════════
# 一、信号检测器（把「经验里说的坑」变成「可观测的信号」）
# ══════════════════════════════════════════════════
# 关键设计：触发器不能依赖模型「自报」，必须能对**真实调用轨迹**做检测。
# 每个 detector 接收 (call_list, final, round_no)，返回 True/False。

def det_empty_result(calls, final, rnd):
    """信号：某次工具调用返回 count=0 / error / 空 results。"""
    for c in calls:
        obs = c.get("obs")
        if isinstance(obs, dict):
            if obs.get("count") == 0 or "error" in obs:
                return True
        elif obs in (None, "", [], {}):
            return True
    return False


def det_underclaim_count(calls, final, rnd):
    """信号：模型给出的篇数与 search_papers 真实返回的 count 不一致（漏检/虚报）。

    ⚠️ 2026-09-29：原实现只看「模型说 3/4/5 篇」—— 这是**过拟合旧案例**。
    真正的信号应该是「模型的篇数与库内 count 不符」，不限定在 3/4/5。
    """
    counts = [c["obs"].get("count") for c in calls
              if c["name"] == "search_papers" and isinstance(c.get("obs"), dict)
              and c["obs"].get("count") is not None]
    if not counts:
        return False
    real = max(counts)
    t = str(final)
    # 模型报的篇数（抓 "N 篇 / N篇 / 共 N 篇"）
    nums = {int(m) for m in re.findall(r"(\d+)\s*篇", t)}
    if not nums:
        return False
    return any(n != real for n in nums)


def det_wrong_key(calls, final, rnd):
    """信号：read_paper 用了不存在的 key（unknown paper）→ 说明 key 格式写错。"""
    for c in calls:
        if c["name"] != "read_paper":
            continue
        obs = c.get("obs")
        if isinstance(obs, dict) and "error" in obs:
            return True
    return False


def det_needs_retry(calls, final, rnd):
    """信号：任务未完成（无实质工具结果）且还没到重试上限。"""
    return rnd == 0 and not calls


def det_final_missing_tool(calls, final, rnd):
    """信号：最终回答里没有出现任何真实读过的论文标题（答案不基于库内事实）。"""
    read_titles = []
    for c in calls:
        if c["name"] == "read_paper":
            obs = c.get("obs")
            if isinstance(obs, dict) and obs.get("title"):
                read_titles.append(str(obs["title"]).split()[0].lower())
    if not read_titles:
        return True
    t = str(final).lower()
    return not any(tt in t for tt in read_titles)


DETECTORS = {
    "empty_result": det_empty_result,
    "underclaim_count": det_underclaim_count,
    "wrong_key": det_wrong_key,
    "needs_retry": det_needs_retry,
    "final_missing_tool": det_final_missing_tool,
}


# ══════════════════════════════════════════════════
# 二、触发器：event → condition → action
# ══════════════════════════════════════════════════
class Trigger:
    """一个行为模式触发器。

    when   : 检测器名（在 DETECTORS 里查）
    action : 命中后的动作类型
             - "inject"    把 instruction 追加进下一轮 prompt（软，但**强制可见**）
             - "force_tool"强制先发一次指定工具调用（硬，绕过模型自觉）
             - "block"     拦截该次结果，要求重做
    payload: 动作参数（如强制调用的工具名+参数模板）
    """

    def __init__(self, name, when, action, payload=None, instruction="",
                 source_task="", fire_count=0, rescue_count=0,
                 retry_cost=0.0, hits=0, successes=0):
        self.name = name
        self.when = when
        self.action = action
        self.payload = payload or {}
        self.instruction = instruction
        self.source_task = source_task     # 来源经验（可追溯）
        self.fire_count = fire_count       # 本轮实验触发次数
        self.rescue_count = rescue_count   # 本轮实验救回次数
        # ── 跨会话累积统计（Day5 复盘遗留①：低成本利用历史）──────
        self.retry_cost = retry_cost       # 每次命中额外花的 token（指数移动均值）
        self.hits = hits                   # 历史命中次数
        self.successes = successes         # 历史「命中后任务转✅」次数

    # ── 置信度 / 成本门槛（Day5 复盘遗留③）──────────────
    def confidence(self):
        """命中后带来成功的经验概率（Beta 先验 1/1，避免早期过自信）。"""
        return (self.successes + 1.0) / (self.hits + 2.0)

    def should_fire(self, min_conf=0.40, max_cost=4000):
        """是否值得触发：warm-up 期必触发；有历史后按置信度/成本决策。

        动机：trial loop 命中即多跑一轮 = 纯成本（见 Day5 复盘）。
        历史告诉我们「这条触发器命中后基本没救回任务」时，就不该再花这一轮。
        """
        if self.hits < WARMUP_HITS:
            return True
        if self.retry_cost > max_cost:
            return False
        return self.confidence() >= min_conf

    def record_outcome(self, success, tokens_spent=0):
        """记录一次命中结果 → 更新累积统计（跨会话可复用）。"""
        self.hits += 1
        self.successes += 1 if success else 0
        self.fire_count += 1
        self.rescue_count += 1 if success else 0
        if tokens_spent > 0:
            a = 0.4 if self.retry_cost == 0 else 0.3
            self.retry_cost = (1 - a) * self.retry_cost + a * tokens_spent

    def match(self, calls, final, rnd):
        det = DETECTORS.get(self.when)
        return bool(det and det(calls, final, rnd))

    def apply(self, env, te, calls, final, rnd, usage=None):
        """执行动作，返回 (did_something, note)。"""
        if self.action == "inject":
            return True, f"注入指令: {self.instruction[:80]}"
        if self.action == "force_tool":
            name = self.payload.get("tool")
            args = copy.deepcopy(self.payload.get("args", {}))
            # ── 占位符解析（让触发器能引用「上次调用的实际参数」）──────
            #   $last.<field>  取上一次同名工具调用的参数值
            #   $bad           取上一次**失败**调用的同名参数（自动重映射）
            #
            # ⚠️ 2026-09-29 修复：原版没有解析 $bad，导致 payload 里写 {"key": "$bad"}
            #    时直接把字符串 '$bad' 当 key 发出去（实测报 unknown paper '$bad'）——
            #    触发器形状对但参数没落地，是典型的「假执行」。
            for k, v in list(args.items()):
                if not isinstance(v, str) or not v.startswith("$"):
                    continue
                if v == "$bad":
                    failed = _last_failed_arg(calls, name, k)
                    if failed is not None:
                        args[k] = failed
                elif v.startswith("$last."):
                    last = _last_obs(calls, name)
                    args[k] = (last or {}).get(v[6:], "")
            # 修正 key 类参数：把非法 key 按 variant map 重写
            if "key" in args and args["key"] not in PAPER_DB:
                fixed = _fix_key(args["key"])
                if fixed:
                    args["key"] = fixed
                else:
                    return False, f"无法重映射 key={args['key']}，跳过"
            obs = te.dispatch(name, args)
            return True, f"强制执行 {name}({json.dumps(args, ensure_ascii=False)}) → {str(obs)[:70]}"
        if self.action == "block":
            return True, f"拦截返回，要求重做: {self.instruction[:60]}"
        return False, "(未知动作)"

    def to_dict(self):
        return {"name": self.name, "when": self.when, "action": self.action,
                "payload": self.payload, "instruction": self.instruction,
                "source_task": self.source_task, "fire_count": self.fire_count,
                "rescue_count": self.rescue_count, "hits": self.hits,
                "successes": self.successes, "retry_cost": round(self.retry_cost, 1)}

    def to_pattern(self):
        """导出为 behavior_patterns.BUILTIN_PATTERNS 同构的 dict（可持久化）。"""
        return {"name": self.name, "when": self.when, "action": self.action,
                "payload": self.payload, "instruction": self.instruction,
                "source_task": self.source_task, "hits": self.hits,
                "successes": self.successes, "retry_cost": self.retry_cost}


def _last_obs(calls, tool_name=None):
    for c in reversed(calls):
        if tool_name is None or c["name"] == tool_name:
            if isinstance(c.get("obs"), dict):
                return c["obs"]
    return None


def _last_failed_arg(calls, tool_name, arg_key):
    """取上一次**失败**调用（obs 含 error）的同名参数值。

    用途：key 类坑的处方是「用连字符变体重试」——触发器必须能知道
    「上次传的是哪个错 key」，才能把它重映射后再发一次。
    """
    for c in reversed(calls):
        if c["name"] != tool_name:
            continue
        obs = c.get("obs")
        if isinstance(obs, dict) and "error" in obs:
            v = c.get("args", {}).get(arg_key)
            if v is not None:
                return v
    return None


def _fix_key(bad_key):
    """把常见错误 key 写法修正到库内真实 key（key 格式坑的通用处方）。"""
    norm = re.sub(r"[^a-z]", "", str(bad_key).lower())
    for real in PAPER_DB:
        if re.sub(r"[^a-z]", "", real.lower()) == norm:
            return real
    return {"self_rag": "self-rag", "selfrag": "self-rag",
            "graph_rag": "graphrag", "snap": "raptor"}.get(bad_key)


# ══════════════════════════════════════════════════
# 三、经验 → 触发器（编译）
# ══════════════════════════════════════════════════
# 这是本模块最"产品化"的一步：不是手写触发器，而是**从反思/经验自动编译**。
# 用 LLM 读「任务 + 反思处方」，产出 0-2 个结构化触发器。
COMPILE_SYS = (
    "你是【经验编译器】。下面给你一次 Agent 失败任务的执行痕迹与反思处方。\n"
    "请把它编译成 **行为模式触发器** —— 下次遇到同类信号时能自动执行的动作，而不是一句提醒。\n"
    "可用检测器（when，只能从中选）：\n"
    "  - empty_result       某次工具调用返回 count=0 或 error\n"
    "  - underclaim_count   模型报的篇数大于 search_papers 真实返回的 count\n"
    "  - wrong_key          read_paper 用了不存在的 key\n"
    "  - final_missing_tool 最终答案里没有出现的真实读过的论文\n"
    "可用动作（action）：\n"
    "  - force_tool  强制先发一次工具调用（payload 含 tool 与 args 参数模板）\n"
    "  - inject      把 instruction 强制追加到下一轮（instruction ≤60 字，必须含具体工具名/参数值）\n"
    f"可用的工具名（payload.tool 只能从中选）：{', '.join(VALID_TOOL_NAMES)}\n"
    "⚠️ payload.args 的键必须是该工具的**真实参数名**，不得自创参数（如 match/format 均不存在）；\n"
    "⚠️ 参数值可写占位符：$bad（上次失败调用的同名参数值）、$last.<字段>（上次调用返回字段）。\n"
    "只输出 JSON 数组（可为空数组），无多余文字：\n"
    '[{"name":"简短英文名","when":"检测器名","action":"force_tool或inject",'
    '"payload":{"tool":"工具名","args":{...}},"instruction":"具体到可照抄的指令"}]'
)


def compile_trigger(env, task, calls, reflection, usage=None):
    """把一次「失败 + 反思」编译成触发器列表。"""
    trace = "\n".join(f"- {c['name']}({json.dumps(c['args'], ensure_ascii=False)}) → "
                      f"{json.dumps(c['obs'], ensure_ascii=False)[:100]}"
                      for c in calls[-8:])
    prompt = (f"任务：{task}\n\n执行轨迹：\n{trace}\n\n"
              f"反思处方：{json.dumps(reflection, ensure_ascii=False)}\n\n请编译触发器。")
    out = plain_llm(env, [{"role": "system", "content": COMPILE_SYS},
                          {"role": "user", "content": prompt}],
                    usage=usage, max_tokens=600, temperature=0.1)
    t = out.strip()
    i, j = t.find("["), t.rfind("]")
    if i < 0 or j <= i:
        return []
    try:
        arr = json.loads(t[i:j + 1])
    except json.JSONDecodeError:
        return []
    out_list = []
    for d in arr:
        if not isinstance(d, dict):
            continue
        when = d.get("when")
        action = d.get("action")
        if when not in DETECTORS or action not in ("inject", "force_tool", "block"):
            continue
        if not validate_trigger(d, verbose=bool(os.environ.get("TRIGGER_DEBUG"))):
            continue
        out_list.append(Trigger(name=d.get("name", f"trig_{when}"), when=when,
                                action=action, payload=d.get("payload") or {},
                                instruction=str(d.get("instruction", ""))[:120],
                                source_task=task[:80]))
    return out_list


def validate_trigger(d, verbose=False):
    """编译期 schema 校验（Day5 复盘遗留②）：把幻觉参数挡在入库之前。

    返回 True/False。校验项：
      ① force_tool 的 tool 必须在真实工具白名单里；
      ② args 的**键**必须是该工具 schema 的真实参数名（挡 match/format 这类幻觉参数）；
      ③ 值要么是字面量，要么是合法占位符（$bad / $last.<field>）；
      ④ inject/block 必须有非空 instruction。
    """
    action = d.get("action")
    payload = d.get("payload") or {}
    instruction = str(d.get("instruction", "")).strip()

    if action in ("inject", "block") and not instruction:
        if verbose:
            print(f"       ⛔ schema 校验：{action} 缺 instruction，丢弃")
        return False
    if action == "force_tool":
        tool = payload.get("tool")
        if tool not in VALID_TOOL_NAMES:
            if verbose:
                print(f"       ⛔ schema 校验：工具 '{tool}' 不在白名单，丢弃")
            return False
        real_args = _tool_arg_names(tool)
        for k, v in (payload.get("args") or {}).items():
            if k not in real_args:
                if verbose:
                    print(f"       ⛔ schema 校验：{tool} 无参数 '{k}'（幻觉参数），丢弃")
                return False
            if isinstance(v, str) and v.startswith("$") and not _PLACEHOLDER_RE.match(v):
                if verbose:
                    print(f"       ⛔ schema 校验：非法占位符 '{v}'，丢弃")
                return False
    return True


def _tool_arg_names(tool_name):
    """取某工具 schema 里的真实参数名集合。"""
    for t in tool_schemas():
        fn = t["function"]
        if fn["name"] == tool_name:
            return set((fn.get("parameters") or {}).get("properties", {}).keys())
    return set()


# ══════════════════════════════════════════════════
# 四、带触发器的执行循环（trial loop：命中即动作，最多 N 轮）
# ══════════════════════════════════════════════════
TRIGGER_SYS = (
    "你是会用工具的助手。用工具获取真实数据，禁止编造；"
    "完成后给出中文最终答案。"
    "\n⚠️ 系统已装载【行为模式触发器】：当检测到特定失败信号时，"
    "系统会强制执行补救动作并把结果插回对话。请务必采纳这些结果，不要忽略。"
)


def run_with_triggers(env, task, triggers, usage=None, max_rounds=2,
                      max_steps=12, verbose=False):
    """执行 → 命中触发器 → 强制补救 → 再执行 …（最多 max_rounds 轮）。

    返回 {final, calls, rounds, fired, log}
    """
    te = TaskEnv()
    u = usage if usage is not None else Usage()
    all_calls = []
    fired_log = []
    final = ""
    rounds = 0

    for rnd in range(max_rounds):
        rounds = rnd + 1
        msgs = [{"role": "system", "content": TRIGGER_SYS},
                {"role": "user", "content":
                 f"任务：{task}" + _round_hint(fired_log)}]
        final, calls, steps = _loop(env, msgs, te, usage=u, max_steps=max_steps)
        all_calls.extend(calls)

        # ── 信号检测 → 触发器命中 → 强制执行 ──────────
        hits = [tr for tr in triggers if tr.match(all_calls, final, rnd)]
        if not hits or rnd == max_rounds - 1:
            break
        for tr in hits:
            tr.fire_count += 1
            did, note = tr.apply(env, te, all_calls, final, rnd, usage=u)
            if did:
                fired_log.append({"round": rnd + 1, "trigger": tr.name,
                                  "action": tr.action, "note": note})
                if verbose:
                    print(f"      ⚡ 触发器命中 [{tr.name}] ({tr.action}) → {note}")
        # 把本轮新增的真实调用也纳入下一轮的检测视野
        all_calls = all_calls  # noqa: PLW0127

    return {"final": final, "calls": all_calls, "rounds": rounds,
            "fired": fired_log, "usage": u}


def _round_hint(fired_log):
    if not fired_log:
        return ""
    lines = ["\n\n【系统已强制执行的补救动作（必须采纳，不要忽略）】"]
    for f in fired_log:
        lines.append(f"  - 第{f['round']}轮触发 [{f['trigger']}]：{f['note']}")
    return "\n".join(lines)


# ══════════════════════════════════════════════════
# 五、验收实验：同一批深坑任务，无触发器 vs 有触发器
# ══════════════════════════════════════════════════
TASKS = [
    {
        "id": "P1",
        # ⚠️ 2026-09-29 修正：原任务「主题含 'RAG' 的论文有几篇」已被证明**无真坑**——
        #    Day5 时代模型会点名 RAPTOR（那是幻觉，库里 RAPTOR 摘要不含 "RAG" 字样）；
        #    现在模型按 count 如实答「2 篇：GraphRAG + Self-RAG」，判据里的「必须点名 RAPTOR」
        #    反而把**正确答案判成失败**（假失败）。教训：判据里的「正确答案」本身要先验证。
        #    改为真正有区分度的坑：**自指检索循环**（Self-RAG 用自己检索自己）。
        "desc": "精确总数（坑：'RAG' 检索命中 RAPTOR 但**不含 Self-RAG**，需二次检索才全）",
        "task": "论文库中有哪些论文是用 RAG 机制解决长文本问题的？请逐一点名。",
        "check": lambda r: _p1_ok(r["final"]),
    },
    {
        "id": "P2",
        "desc": "key 字面量（坑：任务故意给错 key self_rag）",
        "task": "读取论文 'self_rag' 的摘要并说明其核心机制。",
        "check": lambda r: _p2_ok(r),
    },
    {
        "id": "P3",
        "desc": "答案须基于库内事实（坑：不读论文直接凭记忆答）",
        "task": "Self-RAG 和 RAPTOR 各自用什么机制解决长文本问题？请分别说明。",
        "check": lambda r: _p3_ok(r),
    },
]


def _p1_ok(text):
    """P1 判据（2026-09-29 修正）：
    任务问「哪些论文用 RAG 机制解决长文本问题」——库内真正相关的是 RAPTOR +
    GraphRAG + Self-RAG 三篇。坑在于：单发 topic='RAG' 只命中 RAPTOR/GraphRAG
    （Self-RAG 的摘要里没有 "RAG" 字样），必须换词（self-rag / 反思 token）才能补全。
    故判据要求：**三篇都点到名**，只点两篇 = 漏检 = 失败。
    """
    t = str(text).lower()
    hit = {"raptor": "raptor" in t,
           "graphrag": "graphrag" in t or "graph rag" in t,
           "self-rag": "self-rag" in t or "selfrag" in t or "self rag" in t}
    return all(hit.values())


def _p2_ok(r):
    for c in r["calls"]:
        if c["name"] == "read_paper":
            obs = c.get("obs")
            if isinstance(obs, dict) and obs.get("title") and "error" not in obs:
                if "self" in str(obs.get("title", "")).lower():
                    return True
    return False


def _p3_ok(r):
    titles = []
    for c in r["calls"]:
        if c["name"] == "read_paper":
            obs = c.get("obs")
            if isinstance(obs, dict) and obs.get("title"):
                titles.append(str(obs["title"]).split()[0].lower())
    if len(set(titles)) < 2:
        return False
    t = str(r["final"]).lower()
    return all(tt in t for tt in set(titles))


def _run_group(env, label, triggers, u, verbose=False):
    hits = 0
    detail = []
    for t in TASKS:
        r = run_with_triggers(env, t["task"], triggers, usage=u, verbose=verbose)
        ok = bool(t["check"](r))
        hits += 1 if ok else 0
        detail.append({"id": t["id"], "ok": ok, "rounds": r["rounds"],
                       "fired": len(r["fired"]), "calls": len(r["calls"]),
                       "tokens": r["usage"].total_tokens})
        print(f"  [{t['id']}] {'✅' if ok else '❌'} 轮次 {r['rounds']} "
              f"| 触发 {len(r['fired'])} | 工具 {len(r['calls'])} "
              f"| token {r['usage'].total_tokens} | {t['desc']}")
    print(f"  → {label} 完成率: {hits}/{len(TASKS)}")
    return hits, detail


def main():
    env = load_env()
    if not env["api_key"]:
        print("❌ 未找到 LLM_API_KEY")
        return
    selftest_only = "--selftest" in sys.argv

    print("=" * 88)
    print("W26 Day5·补 — 行为模式触发器：把经验从『提示』变成『控制流』")
    print("=" * 88)

    u = Usage()

    # ── 离线自测：检测器 + 触发器（不花 token）──────────
    print("\n【0】离线自测：检测器与触发器")
    _selftest()
    if selftest_only:
        return

    # ── 第 1 步：无触发器跑一遍，收集失败与信号 ─────────
    print(f"\n{'━' * 88}")
    print("【第 1 步】无触发器执行（暴露坑）→ 从反思自动编译触发器")
    print(f"{'━' * 88}")
    compiled = []
    for t in TASKS:
        r = run_with_triggers(env, t["task"], [], usage=u)
        ok = bool(t["check"](r))
        print(f"  [{t['id']}] {'✅' if ok else '❌'} 无触发器 | {t['desc']}")
        if not ok:
            from agent_v2 import reflect
            refl = reflect(env, t["task"], r["final"], r["calls"], usage=u,
                           eval_feedback=f"判据未通过：{t['desc']}")
            trs = compile_trigger(env, t["task"], r["calls"], refl, usage=u)
            print(f"       反思处方: {refl.get('next_action','')[:90]}")
            for tr in trs:
                print(f"       ⚙️ 编译出触发器 [{tr.name}] when={tr.when} action={tr.action}")
            compiled.extend(trs)

    # 兜底：如果 LLM 编译不出（或全过），用内置触发器（体现「同构产品能力」）
    if not compiled:
        print("  （LLM 未编译出触发器 → 装载内置触发器集）")
        compiled = [Trigger(**p) for p in BUILTIN_PATTERNS]

    print(f"\n  触发器集共 {len(compiled)} 条：")
    for tr in compiled:
        print(f"    - {tr.name:<22} when={tr.when:<18} action={tr.action:<12} "
              f"← {tr.source_task[:36]}")

    # ── 第 2 步：有触发器重跑 ───────────────────────
    print(f"\n{'━' * 88}")
    print("【第 2 步】装载触发器执行（命中即强制补救）")
    print(f"{'━' * 88}")
    hits_trig, det_trig = _run_group(env, "有触发器", compiled, u, verbose=True)

    # ── 第 3 步：对照（内置触发器 vs 无）──────────────
    print(f"\n{'━' * 88}")
    print("【第 3 步】对照：无触发器重跑（同批次，排除运气）")
    print(f"{'━' * 88}")
    hits_none, _ = _run_group(env, "无触发器", [], u)

    # ── 汇总 ────────────────────────────────────
    n = len(TASKS)
    print(f"\n{'=' * 88}\n📊 验收结果：行为模式触发器的增益\n")
    print(f"{'组别':<20}{'完成率':<10}{'说明':<40}")
    print(f"{'无触发器':<20}{f'{hits_none}/{n}':<10}{'被动：经验只是提示':<40}")
    print(f"{'有触发器':<20}{f'{hits_trig}/{n}':<10}{'主动：命中信号即强制补救':<40}")
    delta = hits_trig - hits_none
    print(f"\n  触发器增益 = {hits_trig} − {hits_none} = {'+' if delta >= 0 else ''}{delta}")

    total_fired = sum(tr.fire_count for tr in compiled)
    print(f"  触发器触发次数：{total_fired}（确定性证据：命中即执行，不看模型脸色）")
    if delta > 0:
        print("  ✅ 验收通过：触发器把经验变成控制流，完成率提升")
    elif total_fired > 0:
        print("  ⚠️ 完成率持平，但触发器确实命中并强制执行 → "
              "确定性成立；提升被任务难度/模型能力上限吃掉")
    else:
        print("  ❌ 触发器一次未命中 → 检测器与该任务失败模式不匹配")

    print(f"\n  明细：")
    print(f"  {'任务':<6}{'通过':<6}{'轮次':<6}{'触发':<6}{'工具':<6}{'token':<8}")
    for d in det_trig:
        print(f"  {d['id']:<6}{'✅' if d['ok'] else '❌':<6}{d['rounds']:<6}"
              f"{d['fired']:<6}{d['calls']:<6}{d['tokens']:<8}")
    print(f"\n  语言触发器集（供人审阅）：")
    for tr in compiled:
        print(f"    {tr.to_dict()}")


# ══════════════════════════════════════════════════
# 六、离线自测
# ══════════════════════════════════════════════════
def _selftest():
    ok_all = True

    # 1. 空结果检测器
    calls = [{"name": "search_papers", "args": {"topic": "self_rag"},
              "obs": {"topic": "self_rag", "count": 0, "results": []}}]
    r1 = det_empty_result(calls, "", 0)
    print(f"  {'✅' if r1 else '❌'} empty_result 检出 count=0：{r1}")
    ok_all &= r1

    # 2. key 错误检测器
    calls2 = [{"name": "read_paper", "args": {"key": "self_rag"},
               "obs": {"error": "unknown paper 'self_rag'"}}]
    r2 = det_wrong_key(calls2, "", 0)
    print(f"  {'✅' if r2 else '❌'} wrong_key 检出未知 key：{r2}")
    ok_all &= r2

    # 3. 虚报篇数检测器（真实 count=2，模型却说 3 篇）
    calls3 = [{"name": "search_papers", "args": {"topic": "RAG"},
               "obs": {"count": 2, "results": [1, 2]}}]
    r3 = det_underclaim_count(calls3, "库里有 3 篇相关论文", 0)
    print(f"  {'✅' if r3 else '❌'} underclaim_count 检出虚报：{r3}")
    ok_all &= r3
    r3b = det_underclaim_count(calls3, "库里有 2 篇相关论文", 0)
    print(f"  {'✅' if not r3b else '❌'} underclaim_count 不误报（2 篇正确）：{r3b}")
    ok_all &= (not r3b)
    # 3b. 泛化：真实 count=3，模型说 2 篇也要能检出（不能只盯 3/4/5）
    calls3c = [{"name": "search_papers", "args": {"topic": "RAG"},
                "obs": {"count": 3, "results": [1, 2, 3]}}]
    r3c = det_underclaim_count(calls3c, "库里有 2 篇相关论文", 0)
    print(f"  {'✅' if r3c else '❌'} underclaim_count 泛化（count=3 说 2 篇）：{r3c}")
    ok_all &= r3c

    # 4. key 修正映射
    fixed = _fix_key("self_rag")
    r4 = fixed == "self-rag"
    print(f"  {'✅' if r4 else '❌'} _fix_key('self_rag') → {fixed}")
    ok_all &= r4

    # 5. 触发器对象命中 + 执行（用真实 TaskEnv，不发 LLM）
    tr = Trigger(name="key_retry", when="wrong_key", action="force_tool",
                 payload={"tool": "read_paper", "args": {"key": "self_rag"}})
    te = TaskEnv()
    did, note = tr.apply(None, te, calls2, "", 0)
    got = [c for c in te.calls if c["name"] == "read_paper"
           and isinstance(c.get("obs"), dict) and obs_ok(c)]
    r5 = did and len(got) == 1
    print(f"  {'✅' if r5 else '❌'} 触发器强制执行修正后 key 并读成功：{note[:70]}")
    ok_all &= r5

    # 6. 内置触发器集可加载
    r6 = len(BUILTIN_PATTERNS) >= 3
    print(f"  {'✅' if r6 else '❌'} 内置触发器集 {len(BUILTIN_PATTERNS)} 条")
    ok_all &= r6

    print(f"\n  离线自测：{'✅ 全部通过' if ok_all else '❌ 有失败项'}")
    return ok_all


def obs_ok(c):
    o = c.get("obs")
    return isinstance(o, dict) and "error" not in o and o.get("title")


if __name__ == "__main__":
    main()
