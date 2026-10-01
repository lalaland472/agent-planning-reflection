#!/usr/bin/env python3
"""W26 Day1b（重设计）— reflection_quiz2.py：维度不可见的完整性自纠错

Day1 复盘留下的问题（见 cards/W26-Day1-Reflexion-验收复盘.md）：
  三次调判据都没能触发 Reflexion —— 根因不是判据，而是**任务本身没有错误空间**：
  题目把「要覆盖的维度」直接列给模型，等于给了答题清单，照抄很难漏。

Day2 重设计（本文件）：
  1) 【对模型不可见】只给「目标 + 场景」，**不给维度清单** → 模型必须自己想到关键点，
     首稿漏点才可能真实发生（而不是被截断/判据造出来的假缺失）。
  2) 【判据与生成解耦】关键点清单（KEYPOINTS）只存在于评测脚本里，模型永远看不到；
     匹配用「同义词组」，且**只看全文**（避免截断假失败）。
  3) 【可复现】每轮跑完打印 clean-run 复现命令；首稿漏 → 末稿补齐 = 真实自我修正。

Reflexion 映射：
  执行(首稿) → critique(judge 反查缺失关键点) → 教训写入 ReflectionMemory(缺失点文案)
  → 带教训重生成 → 直到关键点全齐（或达轮次上限）。

⚠️ API 坑（Day1 踩过，勿再犯）：
  - deepseek-v4-flash 是 reasoning 模型：不带 tools 的纯文本调用也要
    {"thinking": {"type": "disabled"}}，否则思维链吃光 max_tokens → content 返回空。
  - 生成与判据都必须给足预算（≥1500），任何"截断后判断"都会造假失败。
"""
import os
import sys
import json
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


def chat(env, prompt, system, max_tokens=2000, temperature=0.3):
    url = env["base_url"].rstrip("/") + "/chat/completions"
    payload = {"model": env["model"],
               "messages": [{"role": "system", "content": system},
                            {"role": "user", "content": prompt}],
               "max_tokens": max_tokens, "temperature": temperature,
               "thinking": {"type": "disabled"}}
    req = urllib.request.Request(url, data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json",
                 "Authorization": f"Bearer {env['api_key']}"})
    with urllib.request.urlopen(req, timeout=180) as resp:
        return json.load(resp)["choices"][0]["message"]["content"].strip()


# ══════════════════════════════════════════════════
# 1. 生成：prompt 里**绝不出现**关键点清单
# ══════════════════════════════════════════════════

def generate_draft(env, question, lessons=None, max_tokens=2000):
    """生成方案稿。lessons = 上一稿被判缺失的关键点（Reflexion 记忆）。

    ⚠️ 首稿（lessons=None）的 prompt 只含「目标 + 场景」，不含任何维度/清单提示，
       这是 Day2 与 Day1 的唯一本质区别 —— 有错误空间，Reflexion 才有事可做。
    """
    sys_p = ("你是资深 AI 系统架构师，为一线的工程团队写落地方案。"
             "要求：结构清晰、给出具体机制名/阈值/流程，不要只说'要保证 XX'这类空话。"
             "直接给方案正文，不要写'以下是方案'之类的开场白。")
    user = question
    if lessons:
        user += ("\n\n⚠️【你的上一版方案被评审指出缺失了以下要点，本版必须补上，"
                 "并保持原有结构完整】：\n" + "\n".join(f"- {x}" for x in lessons))
    return chat(env, user, sys_p, max_tokens=max_tokens)


# ══════════════════════════════════════════════════
# 2. 判据：关键点同义词组（对模型不可见），只看全文，不截断
# ══════════════════════════════════════════════════

def judge_missing(draft, keypoints):
    """返回缺失的关键点 key 列表。

    keypoints: [{"key": 显示名, "syn": [同义词/子串...]}, ...]
    命中任一 syn 即视为覆盖。**不截断 draft**（Day1 教训：截断后判断 = 假失败）。
    """
    missing = []
    for kp in keypoints:
        if any(s and s in draft for s in kp["syn"]):
            continue
        missing.append(kp["key"])
    return missing


# ══════════════════════════════════════════════════
# 3. Reflexion 闭环
# ══════════════════════════════════════════════════

class ReflectionRun:
    def __init__(self, question, keypoints, max_rounds=4):
        self.question, self.keypoints, self.max_rounds = question, keypoints, max_rounds
        self.lessons = []   # 累积缺失关键点（去重）——就是 Reflexion 的 verbal memory
        self.rounds = []

    def run(self, env, verbose=True):
        for rnd in range(1, self.max_rounds + 1):
            draft = generate_draft(env, self.question, self.lessons or None,
                                   max_tokens=2000)
            missing = judge_missing(draft, self.keypoints)
            self.rounds.append({"round": rnd, "draft": draft, "missing": list(missing)})
            if verbose:
                print(f"\n── 第 {rnd} 稿（{len(draft)} 字）──")
                print(draft[:500] + ("…" if len(draft) > 500 else ""))
                print(f"  判据→缺失关键点: {missing if missing else '✅ 全齐'}")
            if not missing:
                return {"ok": True, "rounds_used": rnd, "rounds": self.rounds}
            for m in missing:
                if m not in self.lessons:
                    self.lessons.append(m)
            if verbose and rnd < self.max_rounds:
                print(f"  ↪ 教训入库（第 {len(self.lessons)} 条）→ 带教训重生成")
        return {"ok": False, "rounds_used": self.max_rounds, "rounds": self.rounds}


# ══════════════════════════════════════════════════
# 4. 任务集：只给目标+场景，关键点对模型不可见
# ══════════════════════════════════════════════════

QUIZZES = [
    {
        "q": "我们团队要做一个「面向公司内部文档的问答助手」，用户是运营和客服同学，"
             "文档有 3000 多页历史交接材料、格式很乱。请给出这套系统的落地方案，"
             "说明你会在哪些环节做什么设计、为什么。",
        "keypoints": [
            {"key": "文档解析/清洗（PDF/表格/乱格式怎么处理）",
             "syn": ["解析", "清洗", "PDF", "OCR", "版面", "表格", "乱"]},
            {"key": "分块策略（chunk 大小/重叠/按语义切）",
             "syn": ["分块", "切分", "chunk", "语义切", "chunking"]},
            {"key": "检索方案（向量/关键词/混合）",
             "syn": ["向量", "embedding", "BM25", "关键词", "混合检索", "hybrid"]},
            {"key": "重排（rerank / cross-encoder）",
             "syn": ["重排", "rerank", "reranker", "cross-encoder", "精排"]},
            {"key": "检索不足时的兜底（拒答/转人工/澄清）",
             "syn": ["兜底", "拒答", "转人工", "澄清", "检索不到", "没找到"]},
            {"key": "引用溯源（答案要给出出处）",
             "syn": ["引用", "出处", "来源", "溯源", "标注来源"]},
            {"key": "防幻觉（忠于原文/可核验）",
             "syn": ["幻觉", "忠于", "不编造", "可核验", "归因"]},
            {"key": "效果评测（怎么衡量好坏）",
             "syn": ["评测", "评估", "度量", "指标", "回归测试", "hit rate", "准确率"]},
            {"key": "上线后的迭代机制（用户反馈闭环）",
             "syn": ["反馈", "迭代", "badcase", "bad case", "复盘", "持续优化"]},
        ],
    },
    {
        "q": "我们的内部问答助手已经上线，现在要支持「跨多份文档做对比分析」"
             "（比如对比两家供应商的合同条款差异），请给出实现方案。",
        "keypoints": [
            {"key": "跨文档检索/聚合（同一实体在多文档中找齐）",
             "syn": ["跨文档", "多文档", "聚合", "文档级", "全局"]},
            {"key": "对比维度的抽取（先定对比什么）",
             "syn": ["维度", "对比项", "抽取", "结构化", "schema", "字段"]},
            {"key": "长文档/超上下文处理（分而治之）",
             "syn": ["长文档", "上下文", "分而治之", "map-reduce", "RAPTOR", "摘要树", "分层"]},
            {"key": "输出结构化对比表/对齐",
             "syn": ["表格", "对比表", "对齐", "并排", "矩阵"]},
            {"key": "条款差异的判定标准（怎么算不同）",
             "syn": ["判定", "标准", "差异", "一致", "阈值", "等同"]},
            {"key": "可核验（结论要能指回原文位置）",
             "syn": ["原文", "定位", "页码", "出处", "引用", "回链", "锚点"]},
            {"key": "成本控制（大文档多轮调用很贵）",
             "syn": ["成本", "预算", "token", "缓存", "上限", "花费"]},
            {"key": "人工复核/风险控制（合同场景不能错）",
             "syn": ["人工", "复核", "审核", "风险", "免责", "重要决策"]},
        ],
    },
    {
        "q": "帮我们设计一个「会从历史错误里学乖」的 Agent 记忆机制："
             "同样的错误不要犯第二次，但也不能把偶然的经验当成规律。"
             "请给出机制设计。",
        "keypoints": [
            {"key": "短期记忆（当前任务上下文）",
             "syn": ["短期", "工作记忆", "工作区", "当前任务"]},
            {"key": "长期记忆（跨会话持久化）",
             "syn": ["长期", "持久", "跨会话", "数据库", "向量库"]},
            {"key": "何时写入（什么才算值得记的经验）",
             "syn": ["写入", "何时写", "触发", "值得记", "提炼"]},
            {"key": "如何检索/召回相关经验",
             "syn": ["检索", "召回", "相似", "匹配", "语义搜索"]},
            {"key": "经验的抽象/泛化（不能记具体答案）",
             "syn": ["抽象", "泛化", "规律", "一类", "模式", "原则"]},
            {"key": "防止记住错误经验（验证/置信度）",
             "syn": ["置信", "验证", "复核", "多次", "频次", "确认"]},
            {"key": "遗忘/衰减机制（旧经验要过期）",
             "syn": ["遗忘", "衰减", "过期", "清理", "TTL", "淘汰"]},
            {"key": "记忆冲突处理（新旧经验矛盾）",
             "syn": ["冲突", "矛盾", "覆盖", "优先级", "版本"]},
            {"key": "效果度量（记忆到底有没有帮上忙）",
             "syn": ["度量", "评测", "指标", "提升", "A/B", "对比"]},
        ],
    },
]


def main():
    env = load_env()
    if not env["api_key"]:
        print("❌ 未找到 LLM_API_KEY"); return
    print("=" * 74)
    print("W26 Day1b（重设计）— 维度对模型不可见的完整性自纠错")
    print("  判据：关键点同义词组（模型看不到）｜首稿漏 → 末稿补齐 = 真实自我修正")
    print("=" * 74)
    self_correct = 0
    for i, z in enumerate(QUIZZES, 1):
        print(f"\n### Q{i}: {z['q'][:60]}…")
        print(f"    （模型不可见的关键点 {len(z['keypoints'])} 个）")
        run = ReflectionRun(z["q"], z["keypoints"], max_rounds=4)
        res = run.run(env, verbose=True)
        first = res["rounds"][0]["missing"]
        print(f"\n  => Q{i}: 首稿缺 {len(first)} 点 {first if first else ''}"
              f" | {len(res['rounds'])} 轮 | 最终 {'✅ 全覆盖' if res['ok'] else '⚠️ 未全齐'}"
              f" | 教训 {len(run.lessons)} 条")
        if first and res["ok"]:
            self_correct += 1
    print("\n" + "=" * 74)
    print(f"✅ 真实自我修正成功 {self_correct}/{len(QUIZZES)} 题（首稿漏关键点 → 末稿补齐）")
    if self_correct == 0:
        print("⚠️ 仍无触发：说明题目对 flash 依然太容易（关键点全在模型的默认套路上）。")
        print("   下一步应换成需要模型自己划定范围、或含具体数字约束的任务。")


if __name__ == "__main__":
    main()
