#!/usr/bin/env python3
"""W26 Day1 — reflection.py：Reflexion 自我反思 Agent

Day 1 目标（Month 7 W26：Agent 规划 & 反思）：
  让已能调工具的 Agent（复用 W25 tool_registry）学会「执行 → 自我批判 → 把教训写进记忆 → 带教训重试」。
  即 Reflexion (Shinn et al. 2023) 的 verbal reinforcement 最小落地。

承接（计划要求）：
  - Month 6/Self-RAG 的 critique token(s)：从「检索侧评判检索内容」延伸到「批判自己的执行过程」。
  - W25 tool_registry 的三层 schema 校验 + 超时异常捕获（非法参数被拦，Agent 有机会发现问题的 来源）。
  - 双记忆的长期记忆雏形：把反思教训存成结构化文案，供下次 attempt 读取（Day4 会做成向量检索）。

验收（计划）：
  3 步/多步任务中，能从错误自我修正 ≥2 次（第一次错 → 反思 → 第二次带教训做对/少错 → 再反思 → 第三次干净）。

API 注意事项（踩坑记录，勿再犯）：
  - deepseek-v4-flash 是 reasoning 模型：带 tools 请求会进 thinking mode，若不回传 reasoning_content 会 HTTP 400。
    解法：payload 加 {"thinking": {"type": "disabled"}}。
  - 所有 LLM 调用 max_tokens ≥ 200（reasoning 吃预算）。
"""
import os
import sys
import json
import time
import urllib.request

BASE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(BASE)
# 复用 rag-kb 的 load_env（凭证）与 W25 的相对路径约定保持一致
RAGKB_SCRIPTS = os.path.join(ROOT, "..", "rag-kb", "scripts")
if RAGKB_SCRIPTS not in sys.path:
    sys.path.insert(0, RAGKB_SCRIPTS)
try:
    from generate_answer import load_env as _rag_load_env
except Exception:
    _rag_load_env = None

# 复用 W25 的 @tool 注册表 + 执行器（含参数校验/超时）
W25_DIR = os.path.join(BASE, "..", "w25_tool_call")
if W25_DIR not in sys.path:
    sys.path.insert(0, W25_DIR)


# ══════════════════════════════════════════════════
# 1. 环境与通用 LLM 调用（单发 / 带 tools 两种）
# ══════════════════════════════════════════════════

def load_env():
    env = _rag_load_env() if _rag_load_env else {}
    return {
        "api_key": env.get("LLM_API_KEY", "") or os.environ.get("LLM_API_KEY", ""),
        "base_url": env.get("LLM_BASE_URL", "https://api.deepseek.com"),
        "model": env.get("LLM_MODEL", "deepseek-v4-flash"),
    }


def _post(env, payload):
    url = env["base_url"].rstrip("/") + "/chat/completions"
    req = urllib.request.Request(
        url, data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json",
                 "Authorization": f"Bearer {env['api_key']}"})
    with urllib.request.urlopen(req, timeout=120) as resp:
        return json.load(resp)


def plain_llm(env, messages, max_tokens=600, temperature=0.3):
    """纯文本 LLM 调用（无 tools），返回 content 字符串。"""
    payload = {"model": env["model"], "messages": messages,
               "max_tokens": max_tokens, "temperature": temperature}
    r = _post(env, payload)
    return r["choices"][0]["message"]["content"].strip()


def tool_llm(env, messages, tools, max_tokens=800, temperature=0.2):
    """带 tools 的 LLM 调用。⚠️ deepseek reasoning：必须关闭 thinking 避免 400。"""
    payload = {"model": env["model"], "messages": messages,
               "max_tokens": max_tokens, "temperature": temperature,
               "tools": tools, "thinking": {"type": "disabled"}}
    r = _post(env, payload)
    return r["choices"][0]["message"]


# ══════════════════════════════════════════════════
# 2. 反思记忆（结构化教训存储）——长期记忆雏形
# ══════════════════════════════════════════════════

class ReflectionMemory:
    """按 attempt 累积反思教训。每条 = 一句话可执行教训 + 出错上下文。"""

    def __init__(self):
        self.entries = []  # list[dict]
        self.total_attempts = 0

    def add(self, task, critique, lesson, wrong_calls):
        self.entries.append({
            "attempt": self.total_attempts + 1,
            "task": task,
            "critique": critique,      # 批判原文：对错？哪错？
            "lesson": lesson,          # 提炼的可执行教训
            "wrong_calls": wrong_calls,  # 本次出错的具体工具调用（供归因）
        })

    def to_prompt_block(self):
        """把已有教训拼成喂给下一次 attempt 的 system/additional 上下文。"""
        if not self.entries:
            return ""
        lines = ["以下是之前尝试中总结的执行经验，请在本次严格遵守："]
        for i, e in enumerate(self.entries, 1):
            lines.append(f"{i}. [教训] {e['lesson']}")
        return "\n".join(lines)


# ══════════════════════════════════════════════════
# 3. Agent 单次执行（工具循环，复用 W25 注册表风格但内联）
# ══════════════════════════════════════════════════

# 简化工具集：注册到本地执行器（不需要 import W25 的 @tool 副作用）
def tool_add(a: int, b: int) -> int:
    """整数加法"""
    return a + b


def tool_calculate(expression: str):
    """数学表达式求值"""
    allowed = set("0123456789+-*/().% ")
    if not all(c in allowed for c in expression):
        return "错误：表达式含不允许字符"
    try:
        return eval(expression, {"__builtins__": {}}, {})
    except Exception as e:
        return f"错误：{e}"


def tool_multiply(a: float, b: float) -> float:
    """两个数相乘"""
    return a * b


TOOLS = {
    "add": {"fn": tool_add, "schema": {"type": "object",
            "properties": {"a": {"type": "number"}, "b": {"type": "number"}},
            "required": ["a", "b"]}},
    "calculate": {"fn": tool_calculate, "schema": {"type": "object",
                  "properties": {"expression": {"type": "string"}},
                  "required": ["expression"]}},
    "multiply": {"fn": tool_multiply, "schema": {"type": "object",
                "properties": {"a": {"type": "number"}, "b": {"type": "number"}},
                "required": ["a", "b"]}},
}


def tools_schema():
    return [{"type": "function", "function": {
        "name": n, "description": (t["fn"].__doc__ or n).strip().split("\n")[0],
        "parameters": t["schema"]}} for n, t in TOOLS.items()]


def _exec_tool(name, args):
    if name not in TOOLS:
        return f"错误：未知工具 {name}"
    fn = TOOLS[name]["fn"]
    # 极简必填/类型校验（示范；完整见 W25 tool_registry validate_call）
    try:
        import inspect
        sig = inspect.signature(fn)
        for p in sig.parameters:
            if p not in args:
                return f"错误：缺少参数 {p}"
        # 工具执行 + 异常包装（防崩）
        return fn(**args)
    except TypeError as e:
        return f"错误：参数不符 {e}"
    except Exception as e:
        return f"错误：{type(e).__name__}: {e}"


def run_attempt(env, task, lesson_block="", max_steps=6):
    """执行一次工具循环，返回结构化轨迹供反思。

    tool_calls 记录每步调用的工具/参数，供评价正确性（好拿来反思）。"""
    memory_lines = [{"role": "system", "content": (
        "你是会用工具的助手。需要核算/数学时调用工具而非心算；工具结果回填后给最终中文答案。")}]
    if lesson_block:
        memory_lines.append({"role": "system", "content": lesson_block})
    memory_lines.append({"role": "user", "content": task})

    calls = []          # 实际调用记录
    transcript = list(memory_lines)
    final = ""
    for step in range(max_steps):
        msg = tool_llm(env, transcript, tools_schema())
        transcript.append(msg)
        if msg.get("tool_calls"):
            for tc in msg["tool_calls"]:
                fn = tc["function"]
                name = fn["name"]
                try:
                    args = json.loads(fn.get("arguments") or "{}")
                except json.JSONDecodeError:
                    args = {}
                obs = _exec_tool(name, args)
                calls.append({"name": name, "args": args, "obs": obs})
                transcript.append({"role": "tool", "tool_call_id": tc["id"],
                                   "name": name, "content": str(obs)})
            continue
        final = (msg.get("content") or "").strip()
        break
    return {"task": task, "final": final, "calls": calls, "transcript": transcript}


# ══════════════════════════════════════════════════
# 4. 反思（critique + lesson 提炼）
# ══════════════════════════════════════════════════

def reflect(env, attempt, gold_answer):
    """读本次执行轨迹，让 LLM 自我批判并提炼下次能改的教训。

    输入：attempt（轨迹）+ gold_answer（期望答案，供对错判定）。
    返回 (critique, lesson)。
    """
    # 把实际调用序列简化成文本给 LLM
    calls_txt = "\n".join(
        f"- 调用了 {c['name']}({json.dumps(c['args'], ensure_ascii=False)}) → {c['obs']}"
        for c in attempt["calls"]) or "(未调用任何工具)"
    prompt = (
        f"任务：{attempt['task']}\n"
        f"Agent 实际执行的工具调用：\n{calls_txt}\n\n"
        f"Agent 的最终回答：\n{attempt['final']}\n\n"
        f"期望的正确答案：{gold_answer}\n\n"
        "请从执行过程角度批判（Reflexion）：\n"
        "1. 结果对吗？若不对，错在哪一步？是【该调工具却心算/推理】、【参数给错】、"
        "【串了不相关的工具】还是【用了错误的方法/顺序】？\n"
        "2. 提炼一句「下次遇到同类问题要怎么做」的可执行教训（针对过程，不是背答案）。\n"
        "请严格按如下 JSON 输出，不要多余文字：\n"
        '{"critique": "对错+出错点一句", "correct": true或false, "lesson": "下次怎么做一句"}'
    )
    out = plain_llm(env, [{"role": "user", "content": prompt}], max_tokens=400, temperature=0.2)
    try:
        # 去掉可能的 ```json 围栏
        cleaned = out.strip()
        if cleaned.startswith("```"):
            cleaned = cleaned.split("\n", 1)[1].rsplit("```", 1)[0].strip()
        data = json.loads(cleaned)
        return data.get("critique", out), data.get("lesson", ""), bool(data.get("correct"))
    except Exception:
        return out, out, (gold_answer.replace(" ", "") in attempt["final"].replace(" ", ""))


# ══════════════════════════════════════════════════
# 5. Reflexion 主循环（最多 N 次 attempt，带教训重试）
# ══════════════════════════════════════════════════

def reflexion_loop(env, task, gold_answer, max_attempts=3, verbose=True):
    mem = ReflectionMemory()
    results = []
    for attempt_i in range(1, max_attempts + 1):
        mem.total_attempts = attempt_i
        lesson_block = mem.to_prompt_block()
        res = run_attempt(env, task, lesson_block)
        results.append(res)

        if verbose:
            print(f"\n── attempt {attempt_i}──")
            for c in res["calls"]:
                print(f"  调用 {c['name']}({json.dumps(c['args'], ensure_ascii=False)}) → {c['obs']}")
            print(f"  回答: {res['final']}")

        # 判定是否已对（gold 出现在回答 + 无错误调用即认为成功）
        err_call = [c for c in res["calls"]
                    if isinstance(c["obs"], str) and c["obs"].startswith("错误")]
        gold_hit = gold_answer.replace(" ", "") in res["final"].replace(" ", "")
        if gold_hit and not err_call:
            if verbose:
                print(f"  ✅ attempt {attempt_i} 正确无误")
            # 即使做对也轻反思一次，记录为什么这次对（可选）
            return {"success": True, "attempts_used": attempt_i, "results": results,
                    "memory": mem, "final": res["final"]}

        # 做错/有错 → 反思存教训
        critique, lesson, correct_flag = reflect(env, res, gold_answer)
        mem.add(task, critique, lesson, [c for c in res["calls"]])
        if verbose:
            print(f"  ↪ 反思: {critique}")
            print(f"    教训: {lesson}")
        # 若反思判定已 correct，也可提前收（保守起见仍留一次）

    last = results[-1]
    return {"success": False, "attempts_used": max_attempts, "results": results,
            "memory": mem, "final": last["final"]}


# ══════════════════════════════════════════════════
# 6. 验收 Demo：3 个多步任务，验证能从错误自我修正 ≥2
# ══════════════════════════════════════════════════

DEMO_TASKS = [
    {
        # 心算陷阱：希望模型记得用工具，而 calculate 会（若算错）暴露
        "task": "计算 (48 × 7) + (351 ÷ 13) 的值，只能分步调用工具，最后给出单一数值。",
        "gold": "363",  # 48×7=336, 351÷13=27, 336+27=363
    },
    {
        # 串联：两次工具依次进行（add 之后 multiply）。错误点：可能把加法结果当乘法输入搞反
        "task": "先计算 17+29，再把结果乘以 4，给出最终单一数值。请分两步各用一个工具。",
        "gold": "184",
    },
    {
        # 易错结构：减法 mix，可能贪快心算。期望：(120-38)×3 = 246，且要分步用工具
        "task": "计算 (120 − 38) × 3 的值，请先算括号内减法（用工具），再乘 3（用工具），给出最终单一数值。",
        "gold": "246",
    },
]


def main():
    env = load_env()
    if not env["api_key"]:
        print("❌ 未找到 LLM_API_KEY")
        return
    print("=" * 60)
    print("W26 Day1 — Reflexion 自我反思 Demo（从错误中自我修正）")
    print("=" * 60)
    successes = 0
    for t in DEMO_TASKS:
        print(f"\n### 任务: {t['task']}")
        print(f"    期望: {t['gold']}")
        out = reflexion_loop(env, t["task"], t["gold"], max_attempts=3, verbose=True)
        print(f"  => 结果: {'✅ 成功' if out['success'] else '⚠️ 未在3次内做对'} "
              f"(用了 {out['attempts_used']} 次尝试, 收录 {len(out['memory'].entries)} 条教训)")
        if out["success"]:
            successes += 1
    print("\n" + "=" * 60)
    print(f"验收摘要: {successes}/{len(DEMO_TASKS)} 任务最终自我修正正确")
    # 收尾：能自我修正（首次错→后续对；或至少过程中有"引用教训后行为改变"）即视为达标
    print("（达标判据：多步任务从错误自我修正，见每一步 attempt 轨迹）")


if __name__ == "__main__":
    main()
