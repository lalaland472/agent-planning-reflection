"""
W26 Day3 — 单 Agent vs 多 Agent 对比实验
=========================================

计划要求（Month 7 W26 D3）：
    同一多步任务，对比 单Agent vs 多Agent 的 质量分 / 成本(调用次数) / 失败率
    验收：报告明确「多 Agent 何时值得 / 何时不值得」

设计：
    A. single   —— 单 Agent：一个主循环既规划又执行（复用 planner.py 的 ReAct 循环）
    B. multi    —— 多 Agent：三个角色 Agent 分工协作
                     Planner（规划者）→ Executor（执行者）→ Critic（审查者）
                   审查者不满意 → 打回执行者重做（最多 1 轮返工，防止无限循环）

    复用：planner.py 的 TaskEnv / tool_schemas / _loop / TASKS / Usage / load_env
          （不做重复建设，符合 Month7 v2「复用 W9-12 资产」原则）

成本控制（计划「成本 & 预算约束」节）：
    - 每任务 LLM 调用上限 MAX_LLM_CALLS，超限标记 over_budget 计入失败率
    - 子任务决策用 flash（默认模型），不额外升级
"""

import os
import sys
import json
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from planner import (  # noqa: E402
    load_env, Usage, TaskEnv, tool_schemas, _loop, tool_llm, plain_llm,
    TASKS, NOTES,
)

MAX_LLM_CALLS = 15          # 单任务 LLM 调用上限（计划约定）
MAX_REWORK = 1              # 审查者打回执行者的最大轮数


# ══════════════════════════════════════════════════
# A. 单 Agent：一个循环既规划又执行
# ══════════════════════════════════════════════════
def run_single(env, task, usage):
    te = TaskEnv()
    msgs = [{"role": "system", "content":
             "你是会用工具的助手。自己规划步骤并逐步执行，用工具获取真实数据，"
             "禁止编造；完成后给出简洁的中文最终答案。"},
            {"role": "user", "content": task}]
    final, calls, steps = _loop(env, msgs, te, usage=usage)
    return {"final": final, "calls": calls, "steps": steps, "rework": 0,
            "roles": ["solo"]}


# ══════════════════════════════════════════════════
# B. 多 Agent：Planner → Executor → Critic
# ══════════════════════════════════════════════════
PLANNER_SYS = (
    "你是【规划者 Agent】。只做一件事：把用户任务拆成 3-5 条有序、可执行的子目标。"
    "每条一行，格式 '1. ...'。不要调用工具，不要执行，不要解释。"
)

EXECUTOR_SYS = (
    "你是【执行者 Agent】。严格按【规划者】给出的子目标清单逐步执行，"
    "用工具获取真实数据，禁止编造。完成后给出简洁的中文答案，"
    "并在末尾用『## 子目标完成情况』逐条说明哪条完成、哪条未完成及原因。"
)

CRITIC_SYS = (
    "你是【审查者 Agent】。只审不答，严禁调用任何工具。"
    "对照【子目标清单】审查【执行者草稿】，检查三件事：\n"
    "1) 是否每条子目标都有明确交代（完成/未完成+原因）；\n"
    "2) 是否有编造（没调用工具却给出具体数字/结论）；\n"
    "3) 是否如实说明了库内信息不足的情况。\n"
    "只输出一行 JSON：{\"verdict\":\"pass\"或\"rework\",\"reason\":\"...\",\"fix\":\"给执行者的具体修改指令\"}"
)


def _parse_verdict(text):
    """从审查者输出里稳健地抠出 JSON。"""
    t = (text or "").strip()
    if "```" in t:
        seg = t.split("```")
        for s in seg:
            s = s.replace("json", "", 1).strip()
            if s.startswith("{"):
                t = s
                break
    i, j = t.find("{"), t.rfind("}")
    if i >= 0 and j > i:
        try:
            d = json.loads(t[i:j + 1])
            v = str(d.get("verdict", "")).lower().strip()
            return ("rework" if v.startswith("re") else "pass",
                    str(d.get("reason", ""))[:160],
                    str(d.get("fix", ""))[:200])
        except json.JSONDecodeError:
            pass
    # 兜底：关键词判定
    low = t.lower()
    if "rework" in low or "不通过" in t or "打回" in t:
        return "rework", "（JSON 解析失败，按关键词判定）", "请补齐未完成的子目标并修正编造内容"
    return "pass", "（JSON 解析失败，默认放行）", ""


def run_multi(env, task, usage):
    te = TaskEnv()
    roles = []

    # ── 角色 1：规划者 ────────────────────────────
    plan_msg = [{"role": "system", "content": PLANNER_SYS},
                {"role": "user", "content": task}]
    plan = plain_llm(env, plan_msg, usage=usage, max_tokens=300, temperature=0.2)
    roles.append("planner")

    # ── 角色 2：执行者 ────────────────────────────
    exec_msgs = [
        {"role": "system", "content": EXECUTOR_SYS},
        {"role": "user", "content": f"任务：{task}\n\n【规划者的子目标清单】\n{plan}\n\n请开始执行。"},
    ]
    final, calls, steps = _loop(env, exec_msgs, te, usage=usage)
    roles.append("executor")

    # ── 角色 3：审查者（最多打回 1 轮）─────────────
    rework = 0
    for _ in range(MAX_REWORK):
        critic_msg = [
            {"role": "system", "content": CRITIC_SYS},
            {"role": "user", "content":
             f"任务：{task}\n\n【子目标清单】\n{plan}\n\n【执行者草稿】\n{final}\n\n请审查并输出 JSON。"},
        ]
        verdict_raw = plain_llm(env, critic_msg, usage=usage, max_tokens=300, temperature=0.0)
        roles.append("critic")
        verdict, reason, fix = _parse_verdict(verdict_raw)
        if verdict == "pass":
            break
        # 打回：执行者按审查意见重做（沿用同一 messages，保持上下文）
        rework += 1
        exec_msgs.append({"role": "assistant", "content": final})
        exec_msgs.append({"role": "user", "content":
                          f"【审查者反馈】{reason}\n【修改指令】{fix}\n请据此修正并重新给出最终答案。"})
        final, new_calls, steps = _loop(env, exec_msgs, te, usage=usage)
        calls = calls + new_calls
        roles.append("executor(rework)")

    return {"final": final, "calls": calls, "steps": steps,
            "rework": rework, "roles": roles, "plan": plan}


# ══════════════════════════════════════════════════
# 质量分（启发式，0-10）：人工规则的粗略代理
# ══════════════════════════════════════════════════
def quality_score(r, task):
    """对最终答案打一个 0-10 的启发式质量分。

    维度（各 2.5 分）：
      1) 结构：是否有分点/标题（说明交代清楚）
      2) 事实锚定：回答里是否出现工具真实返回过的论文标题
      3) 完整性：是否逐条交代子目标（多 Agent 计划 / 单 Agent 自查）
      4) 诚实性：库内不足时是否如实说明（不编造 3 篇）
    """
    ans = str(r.get("final") or "")
    score = 0.0

    if any(k in ans for k in ("1.", "1、", "##", "**")) :
        score += 2.5

    titles = ["RAPTOR", "GraphRAG", "Self-RAG", "Reflexion", "ReAct"]
    if sum(1 for t in titles if t.lower() in ans.lower()) >= 2:
        score += 2.5

    if "子目标完成情况" in ans or "子目标" in ans or "完成情况" in ans:
        score += 2.5

    if ("2" in ans and ("篇" in ans)) or "不足" in ans or "没有" in ans:
        score += 2.5

    return round(min(score, 10.0), 1)


# ══════════════════════════════════════════════════
# 主实验
# ══════════════════════════════════════════════════
def main():
    env = load_env()
    if not env["api_key"]:
        print("❌ 未找到 LLM_API_KEY")
        return

    print("=" * 88)
    print("W26 Day3 — 单 Agent vs 多 Agent 对比实验")
    print(f"        预算：单任务 LLM 调用 ≤ {MAX_LLM_CALLS}｜多 Agent 最多打回 {MAX_REWORK} 轮")
    print("=" * 88)

    MODES = [("single", run_single), ("multi", run_multi)]
    agg = {m: {"ok": 0, "fail": 0, "over": 0, "calls": 0, "steps": 0,
               "tokens": 0, "q": 0.0, "rework": 0} for m, _ in MODES}
    rows = []

    for t in TASKS:
        print(f"\n{'─' * 88}\n【{t['id']}】{t['desc']}\n  {t['task']}")
        for mname, mfn in MODES:
            u = Usage()
            NOTES.clear()
            t0 = time.time()
            over = False
            try:
                r = mfn(env, t["task"], usage=u)
            except Exception as e:
                print(f"  [{mname:7}] ❌ 异常: {type(e).__name__}: {e}")
                agg[mname]["fail"] += 1
                continue

            dt = time.time() - t0
            ok = bool(t["check"](r))
            if u.calls > MAX_LLM_CALLS:
                over = True
                ok = False          # 超预算计为失败（计划约定）

            q = quality_score(r, t["task"])
            a = agg[mname]
            a["ok"] += 1 if ok else 0
            a["fail"] += 0 if ok else 1
            a["over"] += 1 if over else 0
            a["calls"] += len(r["calls"])
            a["steps"] += r.get("steps", 0)
            a["tokens"] += u.total_tokens
            a["q"] += q
            a["rework"] += r.get("rework", 0)

            flag = "✅" if ok else ("⛔超预算" if over else "❌")
            extra = ""
            if mname == "multi":
                extra = (f" | 角色链 {'→'.join(r['roles'])}"
                         f" | 返工 {r['rework']} 次")
            print(f"  [{mname:7}] {flag} 质量 {q:.1f}/10 | 工具调用 {len(r['calls'])} 次"
                  f" | 步数 {r.get('steps', 0)} | LLM 调用 {u.calls} | "
                  f"tokens {u.total_tokens} | {dt:.1f}s{extra}")
            print(f"            回答: {r['final'][:100].replace(chr(10), ' ')}")
            if mname == "multi":
                print(f"            规划者子目标: {(r.get('plan') or '')[:130].replace(chr(10), ' / ')}")
            rows.append({"task": t["id"], "mode": mname, "ok": ok, "over": over,
                         "q": q, "calls": len(r["calls"]), "llm": u.calls,
                         "tokens": u.total_tokens, "rework": r.get("rework", 0),
                         "sec": round(dt, 1)})

    # ── 汇总 ──────────────────────────────────────
    n = len(TASKS)
    print(f"\n{'=' * 88}\n📊 汇总（{n} 任务 × 2 模式）\n")
    print(f"{'模式':<9}{'成功率':<9}{'失败率':<9}{'超预算':<8}{'平均质量':<10}"
          f"{'工具调用':<10}{'LLM调用':<10}{'tokens':<10}{'返工':<6}")
    for mname, _ in MODES:
        a = agg[mname]
        print(f"{mname:<9}{a['ok']}/{n:<7}{a['fail']}/{n:<7}{a['over']:<8}"
              f"{a['q'] / n:<10.1f}{a['calls']:<10}{a['steps']:<10}"
              f"{a['tokens']:<10}{a['rework']:<6}")

    # ── 成本倍率 ──────────────────────────────────
    s, m = agg["single"], agg["multi"]
    qr = (m["q"] / s["q"]) if s["q"] else 0
    cr = (m["calls"] / s["calls"]) if s["calls"] else 0
    tr = (m["tokens"] / s["tokens"]) if s["tokens"] else 0
    print(f"\n多 Agent 相对单 Agent：质量 {qr:.2f}x ｜ 工具调用 {cr:.2f}x ｜ token {tr:.2f}x")

    print("\n结论提示：")
    print("  · 多 Agent 值得 = 质量提升幅度 > 成本增加幅度（质量/成本比 > 1）")
    print("  · 质量提升不明显而成本成倍 → 只在『需要独立审查/易出错』的任务上才用多 Agent")
    print("  · 3 个任务样本量很小，结论只看方向，勿引用具体数字为稳态值")

    # ── 明细 ─────────────────────────────────────
    print(f"\n{'─' * 88}\n明细\n")
    print(f"{'任务':<6}{'模式':<9}{'通过':<6}{'质量':<7}{'工具':<6}{'LLM':<6}{'tokens':<9}{'返工':<6}{'秒':<6}")
    for r in rows:
        print(f"{r['task']:<6}{r['mode']:<9}{'✅' if r['ok'] else '❌':<6}"
              f"{r['q']:<7.1f}{r['calls']:<6}{r['llm']:<6}{r['tokens']:<9}"
              f"{r['rework']:<6}{r['sec']:<6}")


if __name__ == "__main__":
    main()
