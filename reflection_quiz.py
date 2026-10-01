#!/usr/bin/env python3
"""W26 Day1b — reflection_quiz.py：Reflexion 开放任务自纠错实测

承接 reflection.py 的教训：
  「纯数值+确定性工具」任务对 deepseek-v4-flash 过于可靠（首稿几乎全对），
  Reflexion 的『从错误自我修正』循环几乎没有触发空间。

实测发现 Reflexion（Shinn et al. 2023）真正有价值的场景是——
  *开放式、多要点* 的回答，模型首稿极易漏掉某个必要维度，
  反思（verbal reinforcement）能有效补全。

本文件：一个干净的『完整性自纠错』闭环，用作 W26 Day1 的可复现验收：
  1. 提问要求覆盖 N 个必要维度；
  2. LLM 出首稿（大概率漏 1~2 维）；
  3. 判据（规则/LLM）找出缺失维度 → 存入 ReflectionMemory 作为『教训』；
  4. 下一稿带上教训重生成；
  5. 直至 N 维齐全（或达上限）。
验收：首稿漏 ≥1 维 → 末稿全齐 = 从错误自我修正成功。

⚠️ API：deepseek-v4-flash reasoning 模型，纯文本调用无需关闭 thinking，
   但 max_tokens ≥200（reasoning 吃预算）。
"""
import os
import sys
import json
import re
import urllib.request

BASE = os.path.dirname(os.path.abspath(__file__)) if "__file__" in globals() else "."
# 复用 rag-kb 凭证
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


def chat(env, prompt, system, max_tokens=2000, temperature=0.3):
    url = env["base_url"].rstrip("/") + "/chat/completions"
    payload = {"model": env["model"],
               "messages": [{"role": "system", "content": system},
                            {"role": "user", "content": prompt}],
               "max_tokens": max_tokens, "temperature": temperature,
               # ⚠️ deepseek reasoning：不关 thinking 会吃光 max_tokens → content 为空
               "thinking": {"type": "disabled"}}
    req = urllib.request.Request(url, data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json", "Authorization": f"Bearer {env['api_key']}"})
    with urllib.request.urlopen(req, timeout=150) as resp:
        return json.load(resp)["choices"][0]["message"]["content"].strip()


# ══════════════════════════════════════════════════
# Reflexion 完整性自纠错闭环
# ══════════════════════════════════════════════════

def generate_draft(env, question, n_vision_desc, lessons=None, max_tokens=2000):
    """生成一稿回答。lessons=之前缺哪些的教训（Reflexion 记忆）。"""
    sys_p = "你是严谨、结构化的技术写作助手。回答必须覆盖用户要求的每一个维度，逐条用小标题清晰列出，不要遗漏。"
    user = question + "\n\n要求覆盖以下维度（请逐条回应，不要合并或跳过）：\n" + "\n".join(f"- {d}" for d in n_vision_desc)
    if lessons:
        user += ("\n\n⚠️ 【上一次作答的缺失，本次必须补齐】\n"
                 + "\n".join(f"- {l}" for l in lessons))
    return chat(env, user, sys_p, max_tokens=max_tokens)


def judge_completeness(env, draft, dimensions, question):
    """判缺失维度（critique）。返回 (missing_list, note)。

    ⚠️ 踩坑：LLM-judge 看的是 draft 前 3000 字，长答案被截断后
    会误判“某维度缺”。改为关键词规则判据（对 ≥2000 词预算的完整稿可靠）。
    若某题确需 LLM 语义判据：judge 输入必须用**全文**且 draft 不能超预算。
    """
    # 维度关键点：每维给 1~3 个同义/子串检索词，命中任一即视为覆盖。
    # 历史踩坑2：用「维度标题原文」精确匹配会把“有小标题但标题措辞略异”误判为缺，
    #            导致末稿被判未全齐、自我修正成功率被低估。
    KEYS = {
        "检索/召回": ["检索", "召回"],
        "检索不到或不足的兜底": ["兜底", "检索不足", "降级"],
        "防止幻觉/忠于资料": ["幻觉", "忠于", "引用真实", "不编造", "归因"],
        "回答质量如何评测": ["评测", "评估", "度量", "指标"],
        "角色分工": ["角色", "分工"],
        "Agent 之间如何通信传递": ["通信", "传递", "消息", "共享"],
        "如何处理一个 Agent 失败": ["失败", "重试", "容错", "兜底"],
        "如何收敛成最终一致输出": ["收敛", "一致", "终稿", "合并", "定稿"],
        "短期记忆": ["短期记忆"],
        "长期记忆": ["长期记忆"],
        "何时写入/如何检索经验": ["写入", "何时写", "检索", "召回"],
        "如何防止记住错误经验": ["防止记住", "错误经验", "毒性", "验证", "污染"],
    }
    def _covered(dim):
        if dim in draft:
            return True
        for kw in KEYS.get(dim, []):
            if kw in draft:
                return True
        return False

    missing = [d for d in dimensions if not _covered(d)]
    return missing, f"(规则判据) {'缺 ' + str(missing) if missing else '全齐'}"


def _judge_llm_unused(env, draft, dimensions, question):
    """（保留）LLM 语义判据原稿，仅供对照，勿在默认路径调用。"""
    sys_p = "你是严格的评审。判断下面这篇回答是否逐条覆盖了指定维度。只判断『有没有回应这个维度』，不看内容质量。"
    prompt = (f"要求覆盖的维度:\n{json.dumps(dimensions, ensure_ascii=False)}\n\n"
              f"回答:\n{draft[:3000]}\n\n"
              "逐维度判断回答是否覆盖，只输出 JSON："
              '{"missing": ["没覆盖的维度名", ...], "note": "一句话说明"}，missing 不含已覆盖维度。')
    out = chat(env, prompt, sys_p, max_tokens=300, temperature=0)
    try:
        cleaned = out.strip()
        if cleaned.startswith("```"):
            cleaned = cleaned.split("\n", 1)[1].rsplit("```", 1)[0].strip()
        data = json.loads(cleaned)
        missing = [m for m in data.get("missing", []) if any(m in d or d in m for d in dimensions)]
        note = data.get("note", "")
        return missing, note
    except Exception:
        # 兜底：关键词匹配
        missing = [d for d in dimensions if d not in draft]
        return missing, f"(规则兜底) 缺 {missing}"


# ══════════════════════════════════════════════════

class CompletenessRun:
    def __init__(self, question, dims, max_rounds=4):
        self.question, self.dims, self.max_rounds = question, dims, max_rounds
        self.lessons = []       # 累积缺失维度（去重）
        self.rounds = []        # (draft, missing, note)

    def run(self, env, verbose=True):
        for rnd in range(1, self.max_rounds + 1):
            # ⚠️ 历史：首版用 max_tokens≈700 生成，回答被字数上限截断，
            #    评审（和读者）看到的是半截稿 → 判“漏维度”纯属假象，
            #    整个“自我修正成功率”被长度截断污染（真实失败，非模型失败）。
            #    修正：生成给足预算（≥2000），并把判据改为关键词规则。
            draft = generate_draft(env, self.question, self.dims, self.lessons or None,
                                   max_tokens=2000)
            missing, note = judge_completeness(env, draft, self.dims, self.question)
            self.rounds.append({"round": rnd, "draft": draft, "missing": missing, "note": note})
            if verbose:
                print(f"\n── 第 {rnd} 稿 ──")
                print(draft[:600] + ("…" if len(draft) > 600 else ""))
                print(f"\n  判据→缺失维度: {missing if missing else '✅ 全齐'}")
                if note:
                    print(f"  评审注: {note[:120]}")
            if not missing:
                return {"ok": True, "rounds_used": rnd, "rounds": self.rounds}
            # 记录缺失为教训（Reflexion 记忆），去重
            for m in missing:
                if m not in self.lessons:
                    self.lessons.append(m)
        return {"ok": False, "rounds_used": self.max_rounds, "rounds": self.rounds}


# ══════════════════════════════════════════════════
# 验收：三组开放题，统计"首稿不全 → 末稿全齐"
# ══════════════════════════════════════════════════

QUIZZES = [
    {
        "q": "为一个会「主动检索论文、再回答带引用」的 Agent 问答系统设计技术方案。",
        "dims": ["检索/召回", "检索不到或不足的兜底", "防止幻觉/忠于资料", "回答质量如何评测"],
    },
    {
        "q": "设计一个让多 Agent 协作写一篇技术综述的编排方案。",
        "dims": ["角色分工", "Agent 之间如何通信传递", "如何处理一个 Agent 失败", "如何收敛成最终一致输出"],
    },
    {
        "q": "为一个『会从历史错误中学习』的 Agent 设计记忆机制。",
        "dims": ["短期记忆", "长期记忆", "何时写入/如何检索经验", "如何防止记住错误经验"],
    },
]


def main():
    env = load_env()
    if not env["api_key"]:
        print("❌ 未找到 LLM_API_KEY"); return
    print("=" * 70)
    print("W26 Day1 — Reflexion 开放任务完整性自纠错（从漏维度自我修正）")
    print("=" * 70)
    self_correct = 0
    for i, z in enumerate(QUIZZES, 1):
        print(f"\n### Q{i}: {z['q']}")
        print(f"    要求 {len(z['dims'])} 维: {z['dims']}")
        run = CompletenessRun(z["q"], z["dims"], max_rounds=4)
        res = run.run(env, verbose=True)
        first_missing = res["rounds"][0]["missing"]
        ok = res["ok"]
        print(f"\n  => Q{i}: 首稿缺 {first_missing if first_missing else '无'}"
              f" | {len(res['rounds'])} 轮 | 最终 {'✅ 全覆盖' if ok else '⚠️ 未全齐'}"
              f" | 教训 {len(run.lessons)} 条")
        if first_missing and ok:   # 首稿不全但最终全齐 = 自我修正成功
            self_correct += 1
    print("\n" + "=" * 70)
    print(f"✅ 自我修正成功 {self_correct}/{len(QUIZZES)} 题（首稿漏维度→末稿全齐）")


if __name__ == "__main__":
    main()
