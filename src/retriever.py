# src/retriever.py —— 向量化 + top-k 检索（核心手写）
# 依赖 chunks.json（chunker.py 产物）+ DASHSCOPE_API_KEY
# 功能：
#   build_index()       读取 data/chunks.json，逐块 embedding，存 data/vector_store.json
#   retrieve_top_k(q)   把问题转向量，和 114 块算余弦相似度，返回前 k 块的 {index, text, score}
# 命令行用法：
#   python src/retriever.py                      # 首次：建索引 + 跑 20 题检索
#   python src/retriever.py --no-build            # 已有 vector_store.json，跳过建索引直接跑 20 题
#   python src/retriever.py --top-k 5             # 取 top-5 而非默认 top-3
#
# 和第④步的配合：本题返回的 index 就是 chunks.json 里的编号；hit_rate 判断 top-k 里有没有
# gold_chunks.json 里对应的编号。

import json
import os
import sys
from math import sqrt
from pathlib import Path

import dashscope
from dashscope import TextEmbedding

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / ".." / "data"

CHUNKS_JSON = DATA_DIR / "chunks.json"
EVAL_JSON = DATA_DIR / "eval_set.json"
STORE_JSON = DATA_DIR / "vector_store.json"   # 缓存块向量，避免重复调 API

# ===== 检索参数区 =====
TOP_K = 3                    # 给大模型喂几块（第 7 周调参可改这里；evaluator 能自己切 k，不受此限制）
EMBEDDING_MODEL = TextEmbedding.Models.text_embedding_v3


def _embed(text):
    rsp = TextEmbedding.call(model=EMBEDDING_MODEL, input=text)
    return rsp.output["embeddings"][0]["embedding"]


def cosine_similarity(v1, v2):
    up = sum(a * b for a, b in zip(v1, v2))
    v1_norm = sqrt(sum(x * x for x in v1))
    v2_norm = sqrt(sum(x * x for x in v2))
    return up / (v1_norm * v2_norm)


def build_index(force=False):
    if STORE_JSON.exists() and not force:
        store = json.load(open(STORE_JSON, encoding="utf-8"))
        print(f"读缓存索引：{len(store)} 块向量（{STORE_JSON}）")
        print("（若想重新调 API 建索引，跑 python src/retriever.py --rebuild）")
        return store

    chunks = json.load(open(CHUNKS_JSON, encoding="utf-8"))
    store = []
    for i, c in enumerate(chunks):
        vec = _embed(c["text"])
        store.append({"index": c["index"], "text": c["text"], "vector": vec})
        if (i + 1) % 10 == 0 or i == len(chunks) - 1:
            print(f"  embedding {i+1}/{len(chunks)}")
    with open(STORE_JSON, "w", encoding="utf-8") as f:
        json.dump(store, f, ensure_ascii=False)
    print(f"建索引完成：{len(store)} 块 → {STORE_JSON}")
    return store


def retrieve_top_k(store, question, top_k=TOP_K):
    """返回前 top_k 块的 [{"index":.., "text":.., "score":..}]，按相似度降序。
    top_k=None → 返回全部 114 块的完整排序（给 evaluator 算 gold 的精确名次用）。"""
    q_vec = _embed(question)
    scored = []
    for item in store:
        score = cosine_similarity(q_vec, item["vector"])
        scored.append({"index": item["index"], "text": item["text"], "score": score})
    scored.sort(key=lambda x: x["score"], reverse=True)
    return scored if top_k is None else scored[:top_k]


def _run_eval(top_k):
    """跑 20 题，存【完整排序】ranked（114 块的 index，按相似度从高到低）。
    存全排序的好处：evaluator 自己切 top-1/3/5，改 k 不用重跑检索（省 API 调用）。"""
    store = build_index()
    eval_set = json.load(open(EVAL_JSON, encoding="utf-8"))

    results = []
    for i, item in enumerate(eval_set, 1):
        full = retrieve_top_k(store, item["question"], top_k=None)
        ranked = [t["index"] for t in full]
        results.append({"id": i, "question": item["question"], "ranked": ranked})
        print(f"#{i:02d} {item['chapter']} top-{top_k}: {ranked[:top_k]}")

    out = DATA_DIR / "retrieval_results.json"
    with open(out, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print(f"\n20 题检索结果已存 → {out}")
    return results


if __name__ == "__main__":
    dashscope.api_key = os.getenv("DASHSCOPE_API_KEY")
    if not dashscope.api_key:
        sys.exit("未设置 DASHSCOPE_API_KEY 环境变量，先 export 再跑")

    force = "--rebuild" in sys.argv
    no_build = "--no-build" in sys.argv
    top_k = TOP_K
    for arg in sys.argv:
        if arg.startswith("--top-k"):
            top_k = int(arg.split("=")[1] if "=" in arg else sys.argv[sys.argv.index(arg) + 1])

    if no_build:
        # 跳过建索引（直接从缓存读），但仍需 vector_store.json 存在
        store = json.load(open(STORE_JSON, encoding="utf-8"))
        print(f"读缓存索引：{len(store)} 块向量")
    else:
        store = build_index(force=force)

    _run_eval(top_k)
