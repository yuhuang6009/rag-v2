# src/generator.py —— 生成：检索到的块 → 拼 prompt → qwen-plus → 答案
# 依赖：data/chunks.json + data/retrieval_results.json + DASHSCOPE_API_KEY
# 功能：
#   build_prompt(q, context)      把【参考资料 + 用户问题】拼成提示词
#   generate_answer(q, context)   调 qwen-plus，返回答案文本
# 命令行用法：
#   python src/generator.py             # 跑 20 题，出答案存 data/answers.json
#   python src/generator.py --top-k 5   # 换喂给大模型的块数
#   python src/generator.py --runs 5    # 同一配置跑 5 轮，存 answers_multirun.json（测噪声带用）
#
# 为什么不用重新算向量：retrieval_results.json 里已经存了 ranked（114 块的完整排序），
# 这步只要「切前 k 名 → 按编号去 chunks.json 取原文」，不碰 embedding（省一半 API 钱）。

# 一句话：噪声 = 结果的随机晃动；噪声带 = 晃动范围；测它 = 先知道「多大的变化才值得当回事」。
#
# 两套产物（别搞混）：
#   answers.json           —— 单次运行。给 faithfulness 用（要 context 快照）。每题一个答案。
#   answers_multirun.json  —— 同一配置跑 N 轮。给噪声带用。每题存 N 个答案的列表。
#   为什么不合一个文件：噪声带要的是「同配置重复多次」，faithfulness 要的是「一份确定的答案」，
#   两个用途的读者不同，混在一起以后会乱。

import json
import os
import sys
from pathlib import Path

import dashscope
from dashscope import Generation

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / ".." / "data"

CHUNKS_JSON = DATA_DIR / "chunks.json"
RETRIEVAL_JSON = DATA_DIR / "retrieval_results.json"
ANSWERS_JSON = DATA_DIR / "answers.json"           # 产物①：单次运行（faithfulness 用）
NOISE_JSON = DATA_DIR / "answers_multirun.json"    # 产物②：多次运行（噪声带用）

# ===== 生成参数区 =====
TOP_K = 3                                     # 喂给大模型几块（第 7 周调参可改这里）
MODEL = "qwen-plus"
NOISE_RUNS = 5                                # --runs 不指定时，测噪声带跑几轮（N≥5）

PROMPT_TEMPLATE = """参考下面资料回答用户问题，尽量使用资料回答，不要编造。
【参考资料】
{context}

【用户问题】
{question}"""


def build_prompt(question, context):
    """把参考资料和问题拼成提示词。context 是已用空行拼好的多段资料。"""
    return PROMPT_TEMPLATE.format(context=context, question=question)


def generate_answer(question, context):
    """调 qwen-plus，返回答案纯文本。"""
    resp = Generation.call(
        model=MODEL, 
        prompt=build_prompt(question, context),
        temperature=0,# 最尖的分布，尽量确定性 
        seed=42,  # 钉死随机序列（关键，光 temperature=0 不够）
        )
    return resp.output.text


def generate_all(top_k, runs=1):
    """跑 20 题 × runs 轮。返回 [{id, question, context, answers: [第1次, 第2次, ...]}]。

    context 每题只存一份 —— 同一题每一轮给大模型的资料【完全相同】（检索是确定性的），
    所以存一份就够，不用存 N 份。变的是答案，不是材料。"""
    chunks = json.load(open(CHUNKS_JSON, encoding="utf-8"))
    retrieval = json.load(open(RETRIEVAL_JSON, encoding="utf-8"))

    # {块编号: 原文}，方便按 ranked 里的编号取文本
    chunk_map = {c["index"]: c["text"] for c in chunks}

    results = []
    for item in retrieval:                                # 每题：id / question / ranked
        qid = item["id"]
        question = item["question"]
        ids = item["ranked"][:top_k]                      # 完整榜单切前 k 名
        context = "\n\n".join(chunk_map[i] for i in ids)  # 取原文，空行隔开

        answers = [generate_answer(question, context) for _ in range(runs)]
        results.append({
            "id": qid,
            "question": question,
            "context": context,                           # 存下给了哪些资料，第 7 周查忠实度要用
            "answers": answers,                           # N 个答案（runs=1 时就是 1 个）
        })
        print(f"#{qid:02d} 已回答 {runs} 次（用 {len(ids)} 块）"
              f"（答案 {len(answers[0])} 字）")
    return results


def _run(top_k):
    """单次运行 → answers.json（每题一个答案，原格式）。"""
    results = generate_all(top_k, runs=1)

    # 把 answers 列表摊平成单个 answer：保持原格式，faithfulness 直接读
    flat = [
        {"id": r["id"], "question": r["question"],
         "context": r["context"], "answer": r["answers"][0]}
        for r in results
    ]
    with open(ANSWERS_JSON, "w", encoding="utf-8") as f:
        json.dump(flat, f, ensure_ascii=False, indent=2)
    print(f"\n20 题答案已存 → {ANSWERS_JSON}")
    return flat


def run_noise(top_k, runs):
    """同一配置跑 runs 轮 → answers_multirun.json（给噪声带用）。

    [!] 要花 runs × 20 次生成调用。runs=5 → 100 次。"""
    results = generate_all(top_k, runs=runs)
    with open(NOISE_JSON, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print(f"\n20 题 × {runs} 轮已存 → {NOISE_JSON}")
    print("接着跑：python src/keyword_recall.py --noise   看噪声带")
    return results


if __name__ == "__main__":
    dashscope.api_key = os.getenv("DASHSCOPE_API_KEY")
    if not dashscope.api_key:
        sys.exit("未设置 DASHSCOPE_API_KEY 环境变量，先 export 再跑")

    top_k = TOP_K
    runs = 1
    for arg in sys.argv:
        if arg.startswith("--top-k"):
            top_k = int(arg.split("=")[1] if "=" in arg else sys.argv[sys.argv.index(arg) + 1])
        if arg.startswith("--runs"):
            runs = int(arg.split("=")[1] if "=" in arg else sys.argv[sys.argv.index(arg) + 1])

    if runs > 1:                       # 显式要测噪声 → 走 multi-run 分支
        run_noise(top_k, runs)
    else:                              # 默认：单次运行，更新 answers.json
        _run(top_k)
