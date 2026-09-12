# src/generator.py —— 生成：检索到的块 → 拼 prompt → qwen-plus → 答案
# 依赖：data/chunks.json + data/retrieval_results.json + DASHSCOPE_API_KEY
# 功能：
#   build_prompt(q, context)      把【参考资料 + 用户问题】拼成提示词
#   generate_answer(q, context)   调 qwen-plus，返回答案文本
# 命令行用法：
#   python src/generator.py             # 跑 20 题，出答案存 data/answers.json
#   python src/generator.py --top-k 5   # 换喂给大模型的块数
#
# 为什么不用重新算向量：retrieval_results.json 里已经存了 ranked（114 块的完整排序），
# 这步只要「切前 k 名 → 按编号去 chunks.json 取原文」，不碰 embedding（省一半 API 钱）。

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
ANSWERS_JSON = DATA_DIR / "answers.json"      # 产物：20 题的答案

# ===== 生成参数区 =====
TOP_K = 3                                     # 喂给大模型几块（第 7 周调参可改这里）
MODEL = "qwen-plus"

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


def _run(top_k):
    """跑 20 题：每题取 ranked 前 k 块 → 拼资料 → 让大模型答 → 收集存盘。"""
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

        answer = generate_answer(question, context)
        results.append({
            "id": qid,
            "question": question,
            "context": context,                           # 存下给了哪些资料，第 7 周查忠实度要用
            "answer": answer,
        })
        print(f"#{qid:02d} 已回答（用了 {len(ids)} 块，答案 {len(answer)} 字）")

    with open(ANSWERS_JSON, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print(f"\n20 题答案已存 → {ANSWERS_JSON}")
    return results


if __name__ == "__main__":
    dashscope.api_key = os.getenv("DASHSCOPE_API_KEY")
    if not dashscope.api_key:
        sys.exit("未设置 DASHSCOPE_API_KEY 环境变量，先 export 再跑")

    top_k = TOP_K
    for arg in sys.argv:  
        if arg.startswith("--top-k"):
            top_k = int(arg.split("=")[1] if "=" in arg else sys.argv[sys.argv.index(arg) + 1])

    _run(top_k)
