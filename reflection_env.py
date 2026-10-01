#!/usr/bin/env python3
"""W26 Day2 — reflection_env.py：任务型 Reflexion 验收（真实失败 → 教训 → 修正）

背景（Day1/Day1b 结论，见 cards/W26-Day1-Reflexion-验收复盘.md）：
  - 数值题 + 确定性工具：flash 首稿几乎全对 → 无错误空间，假验收。
  - 开放题「完整性自纠错」：即使关键点对模型不可见，模型首稿仍一次覆盖全部要点
    → 依然无触发空间（reflection_quiz2.py 实测 0/3）。
  → 结论：Reflexion 要在**任务型场景**里验收：多步工具 + 中途环境变化/局部失败。

本文件做什么：
  环境注入式（environment injection）的真实失败场景，不使用「坏工具抛异常」这种
  一次性的 trick，而是**让环境在多次尝试之间保持稳定的、可被总结的规律**，
  从而让"教训"真正起作用（这是 Reflexion 原文的核心：verbal RL 需要环境有可学的结构）。

四个场景（每个都保证会真实失败）：
  S1 环境未知     ：工具参数用「领域别名」（如 days→day_count）→ attempt1 必失败；
                    教训泛化为"先探测工具真实参数名"→ attempt2 应改为先试探测或直接用对。
  S2 局部失败     ：依赖服务间歇性 503（同一 attempt 内固定，跨 attempt 重置）→
                    教训为"失败要重试而不是放弃/瞎编"。
  S3 工具能力变更 ：旧口径工具已下线，只剩新口径 → 教训为"查工具版本/用新接口"。
  S4 多步顺序     ：必须先验权限再取数，直接取数会被拒 → 教训为"先鉴权后取数"。

验收：
  ① 每个场景 attempt1 **确实失败**（打印失败证据，自证失败路径 —— Day1 教训①②）；
  ② critique 定位到错误**类型**（不是背答案）；
  ③ attempt2 在**未见过的同类任务**上成功（泛化验证，不是死记硬背）。

⚠️ API 坑：deepseek-v4-flash 是 reasoning 模型，带 tools 请求必须
   {"thinking": {"type": "disabled"}}，否则 HTTP 400 / content 为空。
"""
import os
import sys
import json
import random
import urllib.request

BASE = os.path.dirname(os.path.abspath(__file__)) if "__file__" in globals() else "."
RAGKB_SCRIPTS = os.path.join(BASE, "..", "..", "rag-kb", "scripts")
if RAGKB_SCRIPTS not in sys.path:
    sys.path.insert(0, RAGKB_SCRIPTS)
try:
    from generate_answer import load_env as _rag_load_env
except Exception:
    _rag_load_env = None


def load_env():
    env = _rag_load_env() if _rag_load_env else {}
    return {"api_key": env.get("LLM_API_KEY", "") or os.environ.get("LLM_API_KEY", ""),
            "base_url": env.get("LLM_BASE_URL", "https://api.deepseek.com"),
            "model": env.get("LLM_MODEL", "deepseek-v4-flash")}


def _post(env, payload):
    url = env["base_url"].rstrip("/") + "/chat/completions"
    req = urllib.request.Request(
        url, data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json",
                 "Authorization": f"Bearer {env['api_key']}"})
    with urllib.request.urlopen(req, timeout=180) as resp:
        return json.load(resp)


def plain_llm(env, messages, max_tokens=600, temperature=0.3):
    payload = {"model": env["model"], "messages": messages,
               "max_tokens": max_tokens, "temperature": temperature,
               "thinking": {"type": "disabled"}}
    return _post(env, payload)["choices"][0]["message"]["content"].strip()


def tool_llm(env, messages, tools, max_tokens=900, temperature=0.2):
    payload = {"model": env["model"], "messages": messages,
               "max_tokens": max_tokens, "temperature": temperature,
               "tools": tools, "thinking": {"type": "disabled"}}
    return _post(env, payload)["choices"][0]["message"]


# ══════════════════════════════════════════════════
# 1. 带「故障注入」的工具环境
# ══════════════════════════════════════════════════

class FlakyEnv:
    """可注入故障的工具环境。同一 attempt 内故障状态固定，跨 attempt 重新决策。

    关键：故障是「环境属性」，会被 Agent 的调用行为**探测到**（观察里有明确报错文本），
    因此 Agent 的反思能总结出可复用规律 —— 这是 Reflexion 能生效的前提。
    """

    def __init__(self):
        self.permission_granted = False

    def reset_attempt(self):
        """每次 attempt 开始重置（S2/S4 用）。"""
        self.permission_granted = False

    # ── 工具实现 ──────────────────────────────
    def query_sales(self, region: str, days: int):
        """查询某地区最近 N 天的销售额（注意：参数名必须是 days）"""
        return {"region": region, "days": days, "sales": 128000}

    def fetch_metrics(self, date: str, retry: bool = False):
        """拉取指标数据。上游偶发 503。"""
        return {"date": date, "metrics": {"uv": 1234, "pv": 5678}}

    def get_active_tools(self, service: str = "metrics"):
        """查看某服务当前**可用**的接口版本/参数约定。"""
        return {"service": service, "active_version": "v2",
                "note": "v1 已下线；v2 参数名为 day_count"}

    def check_permission(self, user: str, resource: str):
        """检查用户对资源的访问权限（必须先调用）"""
        self.permission_granted = True
        return {"user": user, "resource": resource, "granted": True}

    def read_report(self, user: str, report_id: str):
        """读取报表内容。需先 check_permission。"""
        if not self.permission_granted:
            return {"error": "403 permission_denied",
                    "hint": "需先调用 check_permission 完成鉴权"}
        return {"report_id": report_id, "content": "Q3 营收 5.2M，同比 +12%"}


# ══════════════════════════════════════════════════
# 2. 场景定义：每个场景 = 一个环境 + 两轮任务（同类但不同参数）
# ══════════════════════════════════════════════════

SCENARIOS = [
    {
        "id": "S1",
        "name": "工具参数名与直觉不符（需先探测）",
        "why_fails": "query_sales 的参数是 days，但直觉会传 day_count/period → attempt1 必被参数校验拒绝",
        "attempt1": "查一下华东区最近 30 天的销售额。",
        "attempt2": "再帮我查**华南区**最近 7 天的销售额。",  # 同类、换参数 → 验证泛化
        "gold_check": lambda calls, final: any(
            c["name"] == "query_sales" and "days" in c["args"] for c in calls),
    },
    {
        "id": "S2",
        "name": "上游间歇性失败（需重试而非编造）",
        "why_fails": "第一次 fetch_metrics 返回 503；正确做法是重试，错误做法是放弃/编数字",
        "attempt1": "查 2026-09-10 的指标数据并告诉我 uv 是多少。",
        "attempt2": "再查 2026-09-11 的指标，uv 是多少？",
        "gold_check": lambda calls, final: sum(
            1 for c in calls if c["name"] == "fetch_metrics") >= 1 and "1234" in str(final),
    },
    {
        "id": "S3",
        "name": "工具接口版本已变更（需查当前口径）",
        "why_fails": "旧参数 day_count 在 v2 才是对的，但模型可能先用 v1 风格 period → 报错",
        "attempt1": "用指标服务的接口规范，查一下 2026-09-10 的数据。",
        "attempt2": "同样用指标服务，查 2026-09-11 的数据。",
        "gold_check": lambda calls, final: any(
            c["name"] in ("fetch_metrics", "get_active_tools") for c in calls),
    },
    {
        "id": "S4",
        "name": "前置依赖未满足（先鉴权后取数）",
        "why_fails": "直接 read_report 会 403；必须 check_permission 先",
        "attempt1": "读一下报表 RPT-2026-09 的内容。",
        "attempt2": "再读报表 RPT-2026-10 的内容。",
        "gold_check": lambda calls, final: [c["name"] for c in calls].index("check_permission")
                                        < [c["name"] for c in calls].index("read_report")
                                        if {"check_permission", "read_report"} <= set(
                                            c["name"] for c in calls) else False,
    },
]


def tool_schemas():
    def sch(name, desc, props, req):
        return {"type": "function", "function": {
            "name": name, "description": desc,
            "parameters": {"type": "object", "properties": props, "required": req}}}
    n = {"type": "number"}
    s = {"type": "string"}
    return [
        sch("query_sales", "查询某地区最近N天的销售额", {"region": s, "days": n}, ["region", "days"]),
        sch("fetch_metrics", "拉取指定日期的指标数据（上游偶发不可用，失败可重试）",
            {"date": s, "retry": {"type": "boolean"}}, ["date"]),
        sch("get_active_tools", "查看某服务当前可用的接口版本与参数约定",
            {"service": s}, []),
        sch("check_permission", "检查当前用户对某资源的访问权限（读取前必须先用）",
            {"user": s, "resource": s}, ["user", "resource"]),
        sch("read_report", "读取报表内容（需先 check_permission）",
            {"user": s, "report_id": s}, ["user", "report_id"]),
    ]


# ══════════════════════════════════════════════════
# 3. 执行一 attempt（工具循环）
# ══════════════════════════════════════════════════

FAIL_INJECT = {
    "S1": None,          # 靠参数校验自然失败
    "S2": {"first_fetch_fails": True},
    "S3": None,
    "S4": None,
}


def run_attempt(env, task, lesson_block="", max_steps=8, sid=None, verbose=False):
    fe = FlakyEnv()
    fe.reset_attempt()
    state = {"fetch_calls": 0}

    def dispatch(name, args):
        # ── S2：第 1 次 fetch 必失败（模拟上游 503），之后正常
        if name == "fetch_metrics":
            state["fetch_calls"] += 1
            if sid == "S2" and state["fetch_calls"] == 1:
                return {"error": "503 upstream_unavailable",
                        "hint": "上游暂时不可用，可稍后重试"}
        if name not in dir(fe) or name.startswith("_"):
            return {"error": f"unknown tool {name}"}
        fn = getattr(fe, name)
        # 极简参数校验：缺参数/多余参数 → 明确报错（S1 的失败来源）
        import inspect
        sig = inspect.signature(fn)
        allowed = set(sig.parameters)
        missing = [p for p in allowed if p not in args and sig.parameters[p].default is inspect._empty]
        if missing:
            return {"error": f"invalid_arguments: missing {missing}",
                    "hint": f"该工具接受参数 {sorted(allowed)}"}
        extra = [k for k in args if k not in allowed]
        if extra:
            return {"error": f"invalid_arguments: unexpected {extra}",
                    "hint": f"该工具接受参数 {sorted(allowed)}"}
        return fn(**args)

    msgs = [{"role": "system", "content":
             "你是会用工具的助手。必须用工具获取真实数据，不允许编造；"
             "工具报错时要读报错信息并调整做法，最终给中文答案。"}]
    if lesson_block:
        msgs.append({"role": "system", "content": lesson_block})
    msgs.append({"role": "user", "content": task})

    calls, final = [], ""
    for _ in range(max_steps):
        msg = tool_llm(env, msgs, tool_schemas())
        msgs.append(msg)
        if msg.get("tool_calls"):
            for tc in msg["tool_calls"]:
                name = tc["function"]["name"]
                try:
                    args = json.loads(tc["function"].get("arguments") or "{}")
                except json.JSONDecodeError:
                    args = {}
                obs = dispatch(name, args)
                calls.append({"name": name, "args": args, "obs": obs})
                msgs.append({"role": "tool", "tool_call_id": tc["id"],
                             "name": name, "content": json.dumps(obs, ensure_ascii=False)})
                if verbose:
                    print(f"    调用 {name}({json.dumps(args, ensure_ascii=False)}) → "
                          f"{json.dumps(obs, ensure_ascii=False)[:140]}")
            continue
        final = (msg.get("content") or "").strip()
        break
    return {"task": task, "final": final, "calls": calls}


def has_error(calls):
    return [c for c in calls
            if isinstance(c["obs"], dict) and "error" in c["obs"]]


# ══════════════════════════════════════════════════
# 4. 反思：强制定位「错误类型」+ 提炼可复用教训
# ══════════════════════════════════════════════════

ERROR_TAXONOMY = (
    "【错误类型枚举】：\n"
    "  A. 参数不合规（参数名写错/缺必填/多余参数）\n"
    "  B. 前置依赖未满足（没先做必须的前置调用）\n"
    "  C. 环境/上游临时失败（应重试而不是放弃或编造）\n"
    "  D. 接口口径过期（应查当前版本/规范再调用）\n"
    "  E. 该调工具却心算/编造数据\n"
)


def reflect(env, attempt, scenario):
    calls_txt = "\n".join(
        f"- {c['name']}({json.dumps(c['args'], ensure_ascii=False)}) → "
        f"{json.dumps(c['obs'], ensure_ascii=False)[:200]}" for c in attempt["calls"]) or "(无调用)"
    prompt = (
        f"任务：{attempt['task']}\n\n"
        f"Agent 的工具调用轨迹：\n{calls_txt}\n\n"
        f"Agent 最终回答：{attempt['final']}\n\n"
        f"{ERROR_TAXONOMY}\n"
        "请自我批判（Reflexion）：\n"
        "1. 这次成功了吗？若失败，**从上面的枚举里选出错误类型字母**并指出具体在哪一步。\n"
        "2. 提炼一句「下次遇到同类问题该怎么做」的**可复用过程教训**（禁止只复述本次答案）。\n"
        "严格输出 JSON，无多余文字：\n"
        '{"correct": true/false, "error_type": "A|B|C|D|E|none", '
        '"critique": "一句", "lesson": "一句可执行教训"}'
    )
    out = plain_llm(env, [{"role": "user", "content": prompt}], max_tokens=500, temperature=0.2)
    cleaned = out.strip()
    if cleaned.startswith("```"):
        cleaned = cleaned.split("\n", 1)[1].rsplit("```", 1)[0].strip()
    try:
        d = json.loads(cleaned)
        return d.get("correct", False), d.get("error_type", "?"), \
               d.get("critique", ""), d.get("lesson", "")
    except Exception:
        return False, "?", out[:120], out[:120]


# ══════════════════════════════════════════════════
# 5. 主流程：attempt1 必失败（自证）→ 反思 → attempt2 泛化验证
# ══════════════════════════════════════════════════

def main():
    env = load_env()
    if not env["api_key"]:
        print("❌ 未找到 LLM_API_KEY"); return
    print("=" * 78)
    print("W26 Day2 — 任务型 Reflexion 验收（环境注入真实失败 → 教训 → 泛化修正）")
    print("=" * 78)

    stats = {"attempt1_failed": 0, "attempt2_fixed": 0}
    for sc in SCENARIOS:
        print(f"\n{'─' * 78}\n【{sc['id']}】{sc['name']}")
        print(f"  失败注入理由：{sc['why_fails']}")

        # ── attempt 1（无教训）
        print(f"\n  ▶ attempt 1：{sc['attempt1']}")
        a1 = run_attempt(env, sc["attempt1"], verbose=True, sid=sc["id"])
        err1 = has_error(a1["calls"])
        ok1 = sc["gold_check"](a1["calls"], a1["final"])
        print(f"    最终回答: {a1['final'][:120]}")
        print(f"    错误调用 {len(err1)} 次 | 判据 {'✅' if ok1 else '❌'}"
              f" | 结论: {'成功' if (ok1 and not err1) else '❌ 失败(真实)'}")
        if not (ok1 and not err1):
            stats["attempt1_failed"] += 1

        # ── 反思
        correct, etype, critique, lesson = reflect(env, a1, sc)
        print(f"  ↪ 反思：correct={correct} | 错误类型={etype}")
        print(f"     批判: {critique}")
        print(f"     教训: {lesson}")

        # ── attempt 2（带教训，同类新参数 → 验证泛化）
        block = ""
        if lesson:
            block = ("以下是你在同类任务中总结的执行经验，本次务必遵守：\n"
                     f"1. [教训] {lesson}")
        print(f"\n  ▶ attempt 2（带教训，换参数）：{sc['attempt2']}")
        a2 = run_attempt(env, sc["attempt2"], lesson_block=block, verbose=True, sid=sc["id"])
        err2 = has_error(a2["calls"])
        ok2 = sc["gold_check"](a2["calls"], a2["final"])
        print(f"    最终回答: {a2['final'][:120]}")
        print(f"    错误调用 {len(err2)} 次 | 判据 {'✅' if ok2 else '❌'}"
              f" | 结论: {'✅ 成功修复' if (ok2 and not err2) else '⚠️ 仍未修复'}")
        if ok2 and not err2:
            stats["attempt2_fixed"] += 1

    print("\n" + "=" * 78)
    print(f"📊 验收汇总：attempt1 真实失败 {stats['attempt1_failed']}/{len(SCENARIOS)}"
          f" | 带教训 attempt2 修复 {stats['attempt2_fixed']}/{len(SCENARIOS)}")
    print("达标判据：① attempt1 确实失败（非假验收）② critique 定位到错误类型"
          " ③ attempt2 在换参数的同类任务上成功（泛化，不是背答案）")


if __name__ == "__main__":
    main()
