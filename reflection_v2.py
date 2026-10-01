#!/usr/bin/env python3
"""W26 Day1（重做）— reflection_v2.py：Reflexion 的真实触发 + 严格归因

═══════════════════════════════════════════════════════════════════
为什么重做 Day1（原版三连败）
═══════════════════════════════════════════════════════════════════
Day1 原判据：**「3 步任务能从错误自我修正 ≥2 次」**。三次实验全部落空：

  reflection.py       数值题 + 确定性工具      → 首稿全对，零错误空间
  reflection_quiz.py  开放题 + 给维度清单      → 清单=答题卡，照着答不会漏
  reflection_quiz2.py 开放题 + 维度不可见      → 关键点仍是默认套路，仍 0/3

→ 真实结论：**Day1 从未取得过一次「真实触发 Reflexion → 成功修正」的证据。**
  不是机制不行，是**验收设计从来没能让失败真实发生**。

Day1 复盘自己已经总结出三条教训，但当时只写了、没用在新实验里：
  ① 验收先自证失败路径（没有错误发生的验收是空的）
  ② 判据要看全文（任何"截断后判断"都造假失败）
  ③ 别用确定性任务测可靠性机制

═══════════════════════════════════════════════════════════════════
本文件的三处改进
═══════════════════════════════════════════════════════════════════
改进 1【自证失败路径】环境注入式失败，不再指望模型"自己犯错"
    每个任务在环境层保证 attempt1 必失败；先跑一遍**证明**失败确实发生，
    否则该任务作废（这就是Day1缺少的"难度滚动探测"）。

改进 2【严谨归因 ⭐ 本轮核心】补上 Day1 完全缺失的**对照组**
    原版只有「attempt1 → 带教训 attempt2」，做对了也不知道是谁的功劳。
    本版三组对照：
      A 组 baseline    ：attempt1 无教训 → attempt2 **仍无教训**（纯多跑一次）
      B 组 lesson      ：attempt1 无教训 → attempt2 **带教训**
      C 组 泛化        ：attempt1 无教训 → attempt2 **带教训，换同类新参数**
    → 只有当 B > A（教训有用）且 C 也高（能泛化，不是背答案）才算真触发。

改进 3【判据与生成解耦 + 看全文】
    判据只看「调用轨迹 + 全文回答」，不做任何截断；
    且区分「行为修正」（调对了工具）与「结果修正」（答对了数），分开统计。

═══════════════════════════════════════════════════════════════════
结果预期（写死判据，避免事后找理由）
═══════════════════════════════════════════════════════════════════
  判定"Reflexion 真实生效"需同时满足：
    (a) attempt1 确实失败   （自证）
    (b) B组 修正率 > A组 修正率  （归因：教训而非多跑一次）
    (c) C组 修正率 ≥ B组 - 阈值  （泛化：换参数也修得动）
  三者缺一 → 如实报告"未能归因"，不许含糊通过。

⚠️ API 坑（Day1 踩过两次，勿再犯）：
  deepseek-v4-flash 是 reasoning 模型：
   - 带 tools 的调用必须 {"thinking": {"type": "disabled"}}，否则 400
   - 纯文本调用也要关，否则思维链吃光 max_tokens → content 为空
   - 生成/判据预算一律 ≥1500
"""
import os
import sys
import json
import urllib.request

BASE = os.path.dirname(os.path.abspath(__file__))
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


def plain_llm(env, messages, max_tokens=1500, temperature=0.2, model=None):
    payload = {"model": model or env["model"], "messages": messages,
               "max_tokens": max_tokens, "temperature": temperature,
               "thinking": {"type": "disabled"}}
    return _post(env, payload)["choices"][0]["message"]["content"].strip()


def tool_llm(env, messages, tools, max_tokens=900, temperature=0.2, model=None):
    m = model or env["model"]
    payload = {"model": m, "messages": messages,
               "max_tokens": max_tokens, "temperature": temperature,
               "tools": tools, "thinking": {"type": "disabled"}}
    return _post(env, payload)["choices"][0]["message"]


# ══════════════════════════════════════════════════════════════════
# 1. 工具环境（带「反直觉约定」，保证 attempt1 真实失败）
# ══════════════════════════════════════════════════════════════════
class ToolEnv:
    """工具环境。所有"坑"都是**稳定、可被总结**的环境属性（Reflexion 生效前提）。

    ⚠️ 2026-10-01 关键修正（v3 → v4）：
       v3 的坑全部**把解药写在报错里**（hint/description 直接告诉怎么做），
       结果两个模型（强+弱）attempt1 全都照提示照做 —— 测的不是「自我修正」，
       是「会不会读提示」。
       本版**撤掉所有 hint 与题面泄漏**：报错只说失败原因，不说怎么解；
       工具描述只说功能，不说前置依赖。模型必须**自己试错**才能过。
       这才是 Reflexion 真正有活干的场景（对齐 Day2 reflection_env 的原始设计）。
    """

    def __init__(self):
        self.permission_granted = False
        self.fetch_calls = 0

    def reset(self):
        self.permission_granted = False
        self.fetch_calls = 0

    # ── 工具 ──────────────────────────────────
    def query_sales(self, region_code: str, days: int):
        """查某地区最近 N 天销售额。region_code 必须用两字母编码：
        华东=EC 华南=SC 华北=NC。传中文地区名一律拒绝。"""
        if str(region_code) not in ("EC", "SC", "NC"):
            # ⚠️ 只说拒绝，不说该传什么（撤掉解药）
            return {"error": "invalid_region_code"}
        return {"region_code": region_code, "days": days, "sales": 128000}

    def fetch_metrics(self, date: str, retry: bool = False):
        """拉取指标数据。上游偶发不可用。"""
        self.fetch_calls += 1
        if self.fetch_calls == 1:
            # ⚠️ v4：只报 503，**不告诉可以重试**（v3 那句 hint 把答案送出去了）
            return {"error": "503 upstream_unavailable"}
        return {"date": date, "uv": 1234, "pv": 5678}

    def get_active_tools(self, service: str = "metrics"):
        """查看某服务当前可用的接口版本。

        ⚠️ v4：只返回版本号，**不再返回参数名**（v3 直接送了 note 说"参数名为 day_count"）。
           模型得自己拿参数去试。
        """
        return {"service": service, "active_version": "v2"}

    def check_permission(self, user: str, resource: str):
        """检查用户对资源的访问权限。"""
        self.permission_granted = True
        return {"user": user, "resource": resource, "granted": True}

    def read_report(self, user: str, report_id: str):
        """读取报表内容。

        ⚠️ v4：描述**不再写**"需先 check_permission"（v3 等于提前告知前置依赖）。
           403 也只报错，不指路。
        """
        if not self.permission_granted:
            return {"error": "403 permission_denied"}
        return {"report_id": report_id, "content": "Q3 营收 5.2M，同比 +12%"}


def tool_schemas():
    def sch(name, desc, props, req):
        return {"type": "function", "function": {
            "name": name, "description": desc,
            "parameters": {"type": "object", "properties": props, "required": req}}}
    s = {"type": "string"}
    n = {"type": "number"}
    b = {"type": "boolean"}
    return [
        sch("query_sales", "查某地区最近N天销售额",
            {"region_code": s, "days": n}, ["region_code", "days"]),
        sch("fetch_metrics", "拉取指定日期的指标数据",
            {"date": s, "retry": b}, ["date"]),
        sch("get_active_tools", "查看某服务当前可用的接口版本",
            {"service": s}, []),
        sch("check_permission", "检查用户对某资源的访问权限",
            {"user": s, "resource": s}, ["user", "resource"]),
        sch("read_report", "读取报表内容",
            {"user": s, "report_id": s}, ["user", "report_id"]),
    ]


def dispatch(te: ToolEnv, name, args):
    """工具分发 + 参数校验（S1 的失败来源：参数名写错会被明确拒绝）。"""
    if name not in ("query_sales", "fetch_metrics", "get_active_tools",
                    "check_permission", "read_report"):
        return {"error": f"unknown tool {name}"}
    import inspect
    fn = getattr(te, name)
    allowed = set(inspect.signature(fn).parameters)
    missing = [p for p in allowed if p not in args
               and inspect.signature(fn).parameters[p].default is inspect._empty]
    if missing:
        # ⚠️ v4：**不再回显**完整参数列表（v3 那句 hint 等于把答案告诉了模型）
        return {"error": f"invalid_arguments: missing {missing}"}
    extra = [k for k in args if k not in allowed]
    if extra:
        return {"error": f"invalid_arguments: unexpected {extra}"}
    return fn(**args)


# ══════════════════════════════════════════════════════════════════
# 2. 场景（每个场景：attempt1 / attempt2 同类换参 / 行为判据）
# ══════════════════════════════════════════════════════════════════
def behavior_ok(sid, calls):
    """行为判据：只看**是否真正拿到了目标数据**，不限定必须调哪几个工具。

    ⚠️ 2026-10-01 修正（v4 → v5）：
       旧版要求 S4「必须同时出现 check_permission 和 read_report 两个调用」。
       但实测：模型卡住时会**直接放弃并反问用户**，一个都不调 → 判据直接 False，
       导致 A/B/C 三组全 0，看起来像「教训无用」，实际是判据把「没试」等同于「试了不行」。
       正确姿势：判「**目标达成**」——即有没有哪个调用真的返回了目标数据（非错误）。
    """
    names = [c["name"] for c in calls]

    def got(tool, field):
        for c in calls:
            if c["name"] != tool:
                continue
            o = c.get("obs")
            if isinstance(o, dict) and "error" not in o and o.get(field) is not None:
                return True
        return False

    if sid == "S1":
        # 目标：query_sales 成功返回销售额（且用的是编码而非中文）
        return got("query_sales", "sales")
    if sid == "S2":
        # 目标：fetch_metrics 成功拿到 uv
        return got("fetch_metrics", "uv")
    if sid == "S3":
        # 目标：fetch_metrics 成功拿到 uv（v2 口径）
        return got("fetch_metrics", "uv")
    if sid == "S4":
        # 目标：read_report 成功拿到报表内容
        return got("read_report", "content")
    return False


def answer_ok(sid, final):
    """结果判据：答案是否给出了正确事实（与行为解耦，分开统计）。"""
    t = str(final)
    if sid == "S2":
        return "1234" in t
    if sid == "S4":
        return "5.2" in t or "营收" in t
    return True  # S1/S3 以行为为准


SCENARIOS = [
    {
        "id": "S1",
        "name": "地区必须用编码而非中文（反直觉参数）",
        "why": "query_sales 的 region_code 只收 EC/SC/NC，传中文地区名一律拒绝；报错不告诉该传什么",
        "attempt1": "查一下华东区最近 30 天的销售额。",
        "attempt2": "再帮我查华南区最近 7 天的销售额。",
    },
    {
        "id": "S2",
        "name": "上游间歇失败（需自己想到重试）",
        "why": "第 1 次 fetch_metrics 恒返 503，报错**不告诉**可以重试；正解是自己重试",
        "attempt1": "查 2026-09-10 的指标数据，告诉我 uv 是多少。",
        "attempt2": "再查 2026-09-11 的指标，uv 是多少？",
    },
    {
        "id": "S3",
        "name": "接口口径已变更（需自己试出参数）",
        "why": "v2 只告诉版本号、**不告诉参数名**；得自己拿参数去试才知道正确写法",
        "attempt1": "用指标服务的接口规范，查一下 2026-09-10 的数据。",
        "attempt2": "同样用指标服务，查 2026-09-11 的数据。",
    },
    {
        "id": "S4",
        "name": "隐藏前置依赖（需自己发现要鉴权）",
        "why": "read_report 描述**不说**前置依赖；403 也不指路，得自己推断要 check_permission",
        "attempt1": "读一下报表 RPT-2026-09 的内容。",
        "attempt2": "再读报表 RPT-2026-10 的内容。",
    },
]

# ══════════════════════════════════════════════════════════════════
# 双模型对照（改进 4）：强模型看「任务是否过期」，弱模型找「真实失败」
# ══════════════════════════════════════════════════════════════════
# Day1 重做的结论：四场景对 v4-flash 全部过期（attempt1 即成功）。
# 同一个坑换个弱模型就未必能一次过 —— 那才是我们要的真实失败。
STRONG_MODEL = "deepseek-v4-flash"
WEAK_MODEL = "deepseek-flash"


# ══════════════════════════════════════════════════════════════════
# 3. 执行一次 attempt
# ══════════════════════════════════════════════════════════════════
def run_attempt(env, task, lesson_block="", max_steps=8, verbose=False, model=None):
    te = ToolEnv()
    te.reset()
    msgs = [{"role": "system", "content":
             "你是会用工具的助手。必须用工具获取真实数据，不允许编造；"
             "工具报错时要读报错信息并调整做法，最终给中文答案。"}]
    if lesson_block:
        msgs.append({"role": "system", "content": lesson_block})
    msgs.append({"role": "user", "content": task})

    calls, final = [], ""
    for _ in range(max_steps):
        msg = tool_llm(env, msgs, tool_schemas(), model=model)
        msgs.append(msg)
        if msg.get("tool_calls"):
            for tc in msg["tool_calls"]:
                name = tc["function"]["name"]
                try:
                    args = json.loads(tc["function"].get("arguments") or "{}")
                except json.JSONDecodeError:
                    args = {}
                obs = dispatch(te, name, args)
                calls.append({"name": name, "args": args, "obs": obs})
                msgs.append({"role": "tool", "tool_call_id": tc["id"],
                             "name": name,
                             "content": json.dumps(obs, ensure_ascii=False)})
                if verbose:
                    print(f"      调用 {name}({json.dumps(args, ensure_ascii=False)}) → "
                          f"{json.dumps(obs, ensure_ascii=False)[:120]}")
            continue
        final = (msg.get("content") or "").strip()
        break
    return {"task": task, "final": final, "calls": calls}


def has_error(calls):
    return [c for c in calls
            if isinstance(c["obs"], dict) and "error" in c["obs"]]


def has_unresolved_error(calls):
    """轨迹中是否存在**未被后续调用解决的** error。

    ⚠️ 2026-10-01 修的判据 bug：
       旧版用 has_error()（轨迹里出现过任何 error）当失败判据 —— 这是错的。
       S2 实测：模型第 1 次拿 503 → **自己带 retry 重试成功**。
       这是「成功自愈」，却被旧判据判成「失败」，导致 A/B/C 三组全部误判为 0。
       等于把「自我修正成功」判成「失败」—— Day1「假失败」病的同一变种。

    修正：只看该 error 之后**同类调用是否被解决**（拿到无 error 的结果）。
        - error 后同类调用成功了 → 不算未解决（模型自愈了）
        - error 一直没人管 / 直接被放弃 → 才算未解决
    """
    for i, c in enumerate(calls):
        if not (isinstance(c["obs"], dict) and "error" in c["obs"]):
            continue
        # 看后续是否有同类调用成功
        resolved = any(
            later["name"] == c["name"]
            and isinstance(later.get("obs"), dict)
            and "error" not in later["obs"]
            for later in calls[i + 1:])
        if not resolved:
            return True
    return False


def attempt_failed(sid, calls, final):
    """统一的失败判据（看**最终状态**，不看过程）。

    一个 attempt 算失败，当且仅当：
      ① 行为判据未过（没用对姿势），或
      ② 结果判据未过（答案不是正确事实），或
      ③ 存在**未解决**的 error（被放弃了 / 没重试）

    只要模型自己碰壁后自愈（error 出现但随后解决）且最终答对 → 算成功。
    """
    if not behavior_ok(sid, calls):
        return True
    if not answer_ok(sid, final):
        return True
    if has_unresolved_error(calls):
        return True
    return False


# ══════════════════════════════════════════════════════════════════
# 4. 反思（critique 定位错误**类型** → lesson 可复用）
# ══════════════════════════════════════════════════════════════════
ERROR_TAXONOMY = (
    "【错误类型枚举】\n"
    "  A. 参数不合规（参数名写错/缺必填/多余参数）\n"
    "  B. 前置依赖未满足（没先做必须的前置调用）\n"
    "  C. 环境/上游临时失败（应重试而不是放弃或编造）\n"
    "  D. 接口口径过期（应查当前版本/规范再调用）\n"
    "  E. 该调工具却心算/编造数据\n"
)


def reflect(env, attempt, max_tokens=1500):
    calls_txt = "\n".join(
        f"- {c['name']}({json.dumps(c['args'], ensure_ascii=False)}) → "
        f"{json.dumps(c['obs'], ensure_ascii=False)[:200]}"
        for c in attempt["calls"]) or "(无调用)"
    prompt = (
        f"任务：{attempt['task']}\n\n"
        f"你的工具调用轨迹：\n{calls_txt}\n\n"
        f"你的最终回答：{attempt['final']}\n\n"
        f"{ERROR_TAXONOMY}\n"
        "请自我批判（Reflexion）：\n"
        "1. 这次成功了吗？若失败，从上面枚举里选出**错误类型字母**，指出具体在哪一步。\n"
        "2. 提炼一句「下次遇到同类问题该怎么做」的**可复用过程教训**"
        "（必须是做法，禁止复述本次具体答案/数字）。\n"
        "严格输出 JSON，无多余文字：\n"
        '{"correct": true/false, "error_type": "A|B|C|D|E|none", '
        '"critique": "一句", "lesson": "一句可执行教训"}'
    )
    out = plain_llm(env, [{"role": "user", "content": prompt}], max_tokens=max_tokens)
    cleaned = out.strip()
    if cleaned.startswith("```"):
        cleaned = cleaned.split("\n", 1)[1].rsplit("```", 1)[0].strip()
    i, j = cleaned.find("{"), cleaned.rfind("}")
    if i >= 0 and j > i:
        try:
            d = json.loads(cleaned[i:j + 1])
            return (bool(d.get("correct")), str(d.get("error_type", "?")),
                    str(d.get("critique", "")), str(d.get("lesson", "")))
        except json.JSONDecodeError:
            pass
    return False, "?", cleaned[:150], cleaned[:150]


# ══════════════════════════════════════════════════════════════════
# 5. ⭐ 三组对照实验（改进 2 的核心）
# ══════════════════════════════════════════════════════════════════
def run_experiment(env, verbose=True, model=None, scenarios=None, tag=""):
    """对每个场景跑三组，产出可归因的统计。

    A baseline : attempt2 无教训（纯"多跑一次"）
    B lesson   : attempt2 带教训（同参数任务）
    C general  : attempt2 带教训（同类新参数，验证泛化）
    """
    scenarios = scenarios or SCENARIOS
    rows = []
    for sc in scenarios:
        sid = sc["id"]
        if verbose:
            print(f"\n{'─' * 74}\n【{sid}】{sc['name']}")
            print(f"  坑：{sc['why']}")

        # ── attempt1：自证失败 ──────────────────
        if verbose:
            print(f"\n  ▶ attempt1（无教训）：{sc['attempt1']}")
        a1 = run_attempt(env, sc["attempt1"], verbose=verbose, model=model)
        ok1_beh = behavior_ok(sid, a1["calls"])
        ok1_ans = answer_ok(sid, a1["final"])
        unresolved1 = has_unresolved_error(a1["calls"])
        success1 = not attempt_failed(sid, a1["calls"], a1["final"])
        if verbose:
            print(f"    回答: {a1['final'][:100]}")
            print(f"    行为{'✅' if ok1_beh else '❌'} 结果{'✅' if ok1_ans else '❌'} "
                  f"未解决错误 {'有' if unresolved1 else '无'} "
                  f"→ {'成功（自愈也算成功）' if success1 else '❌ 失败(真实)'}")

        if success1:
            if verbose:
                print("    ⚠️ attempt1 未失败 → 该场景无区分度，标记作废")
            rows.append({"id": sid, "usable": False})
            continue

        # ── 反思 → 教训 ────────────────────────
        correct, etype, critique, lesson = reflect(env, a1)
        block = ("以下是你在同类任务中总结的执行经验，本次务必遵守：\n"
                 f"1. [教训] {lesson}") if lesson else ""
        if verbose:
            print(f"  ↪ 反思：correct={correct} 错误类型={etype}")
            print(f"     批判: {critique[:100]}")
            print(f"     教训: {lesson[:100]}")

        # ── A 组 baseline：无教训重跑 ────────────
        a_a = run_attempt(env, sc["attempt1"], lesson_block="", model=model)
        ok_a = not attempt_failed(sid, a_a["calls"], a_a["final"])
        # ── B 组 lesson：带教训（同任务）────────
        a_b = run_attempt(env, sc["attempt1"], lesson_block=block, model=model)
        ok_b = not attempt_failed(sid, a_b["calls"], a_b["final"])
        # ── C 组 general：带教训（换参新任务）────
        a_c = run_attempt(env, sc["attempt2"], lesson_block=block, model=model)
        ok_c = not attempt_failed(sid, a_c["calls"], a_c["final"])

        if verbose:
            print(f"\n  ▶ A 组 baseline（无教训重跑）   : "
                  f"{'✅ 通过' if ok_a else '❌ 未过'}")
            print(f"  ▶ B 组 lesson（带教训·同任务）  : "
                  f"{'✅ 通过' if ok_b else '❌ 未过'}")
            print(f"  ▶ C 组 general（带教训·换参数） : "
                  f"{'✅ 通过' if ok_c else '❌ 未过'}")
            print(f"     → B组工具调用 {len(a_b['calls'])} 次 vs A组 {len(a_a['calls'])} 次 "
                  f"（教训是否减少试错）")

        rows.append({"id": sid, "usable": True, "etype": etype, "lesson": lesson,
                     "a": ok_a, "b": ok_b, "c": ok_c})

    return rows


def summarize(rows, title="考核结果：Reflexion 是否真实生效（三组对照）"):
    live = [r for r in rows if r.get("usable")]
    n = len(live)
    if not n:
        print("\n❌ 无任何场景产生真实失败 → 本批任务全部作废（需换更难的坑/换弱模型）")
        return
    a = sum(1 for r in live if r["a"])
    b = sum(1 for r in live if r["b"])
    c = sum(1 for r in live if r["c"])
    print(f"\n{'=' * 74}")
    print(f"📊 {title}\n")
    print(f"  可用场景（attempt1 真实失败）: {n}/{len(rows)}")
    print(f"  {'组别':<32}{'修正率':<12}{'说明'}")
    print(f"  {'A baseline（无教训重跑）':<32}{f'{a}/{n}':<12}{'纯多跑一次的效果'}")
    print(f"  {'B lesson（带教训·同任务）':<32}{f'{b}/{n}':<12}{'教训带来的效果'}")
    print(f"  {'C general（带教训·换参数）':<32}{f'{c}/{n}':<12}{'泛化能力'}")
    d_ab = b - a
    print(f"\n  归因：教训增益 = B − A = {b} − {a} = {'+' if d_ab >= 0 else ''}{d_ab}")
    print("\n  判定（三项须同时成立）：")
    cond_a = n > 0
    cond_b = d_ab > 0
    cond_c = c >= max(b - 1, 0) if n else False
    print(f"    (a) 存在真实失败      : {'✅' if cond_a else '❌'} ({n} 个场景自证 attempt1 失败)")
    print(f"    (b) 教训 > 多跑一次   : {'✅' if cond_b else '❌'} (B={b} vs A={a})")
    print(f"    (c) 能泛化到新参数    : {'✅' if cond_c else '❌'} (C={c} vs B={b})")
    if cond_a and cond_b and cond_c:
        print("\n  ✅ Reflexion 真实生效：可归因、可泛化")
    elif cond_a and cond_b:
        print("\n  ⚠️ 教训有效但泛化不足（B>A 但 C 掉）→ 可能是背答案而非学方法")
    elif cond_a:
        print("\n  ❌ 未观察到教训增益（B≤A）→ 如实报告：本轮未能归因")
    print("\n  明细：")
    print(f"  {'场景':<6}{'错误类型':<10}{'A':<6}{'B':<6}{'C':<6}{'教训'}")
    for r in live:
        print(f"  {r['id']:<6}{r['etype']:<10}{'✅' if r['a'] else '❌':<6}"
              f"{'✅' if r['b'] else '❌':<6}{'✅' if r['c'] else '❌':<6}"
              f"{r['lesson'][:50]}")
    return {"n": n, "a": a, "b": b, "c": c}


def main():
    env = load_env()
    if not env["api_key"]:
        print("❌ 未找到 LLM_API_KEY")
        return
    print("=" * 74)
    print("W26 Day1（重做）— Reflexion 真实触发 + 严格归因（三组对照）")
    print("  改进：① 环境注入自证失败 ② A/B/C 对照组 ③ 判据与生成解耦、看全文")
    print("  ④ 双模型对照：强模型（v4-flash）看任务是否过期 / 弱模型（flash）找真实失败")
    print("=" * 74)

    # ── 第一轮：强模型 ────────────────────────
    print(f"\n{'━' * 74}")
    print("【第一轮】强模型 deepseek-v4-flash（难）+ 原版 deepseek-v4-flash 场景")
    print(f"{'━' * 74}")
    rows_strong = run_experiment(env, verbose=True, model=STRONG_MODEL)
    res_strong = summarize(rows_strong, title="强模型（deepseek-v4-flash）")

    # ── 第二轮：弱模型 ────────────────────────
    print(f"\n\n{'━' * 74}")
    print(f"【第二轮】弱模型 {WEAK_MODEL}（难度降级 → 找真实失败）")
    print(f"{'━' * 74}")
    rows_weak = run_experiment(env, verbose=True, model=WEAK_MODEL)
    res_weak = summarize(rows_weak, title=f"弱模型（{WEAK_MODEL}）")

    # ── 总汇总 ────────────────────────────────
    print(f"\n{'=' * 74}")
    print("📊 双模型总汇总：Day1 判据「从错误自我修正 ≥2 次」是否被满足\n")
    ok = False
    for name, res in (("强模型 v4-flash", res_strong), (WEAK_MODEL, res_weak)):
        if not res:
            print(f"  {name:<22} → 无可用场景（任务全部过期）")
            continue
        verdict = "✅ 满足" if (res["b"] > res["a"] and res["c"] >= max(res["b"] - 1, 0)) \
            else "❌ 未满足"
        print(f"  {name:<22} → A={res['a']}/{res['n']} B={res['b']}/{res['n']} "
              f"C={res['c']}/{res['n']}  {verdict}")
        if verdict == "✅ 满足":
            ok = True
    print()
    if ok:
        print("  ✅ Day1 验收标准被满足：存在真实失败 → 教训可归因（B>A）→ 可泛化（C高）")
    else:
        print("  ❌ 仍未满足：两个难度档都没拿到「真实失败 → 教训增益」的完整链条")
        print("     下一步：继续加难度（5+ 步长链 / 中途状态变更），或换本地小模型")


if __name__ == "__main__":
    main()
