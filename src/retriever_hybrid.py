# src/retriever_hybrid.py —— 混合检索：BM25 + 向量 → RRF →（可选）重排
#
# 为什么是【新文件】而不是改 retriever.py：
#   retriever.py 产出的 hit@1=0.75 是【对照基准】，它的代码要冻结，不能被碰。
#
# 产出（注意【不同名】，绝不覆盖 dense 那半）：
#   python src/retriever_hybrid.py            → data/retrieval_results_hybrid.json
#   python src/retriever_hybrid.py --rerank   → data/retrieval_results_hybrid_rerank.json
#
# 吃：chunks.json + eval_set.json + retrieval_results.json（= dense 那半：向量【排名】）
# 不吃 API：dense 排名直接读文件；BM25 / 重排都是本地的 → 这一步 0 元
#
# 依赖：pip install rank_bm25 sentence-transformers

import json
import re
import sys
from pathlib import Path

from rank_bm25 import BM25Okapi

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / ".." / "data"

CHUNKS_JSON = DATA_DIR / "chunks.json"
EVAL_JSON   = DATA_DIR / "eval_set.json"
DENSE_JSON  = DATA_DIR / "retrieval_results.json"                # ★ dense 那半（读，不写）
OUT_HYBRID  = DATA_DIR / "retrieval_results_hybrid.json"          # 只 RRF
OUT_RERANK  = DATA_DIR / "retrieval_results_hybrid_rerank.json"   # RRF + 重排

RRF_K = 60                                          # RRF 常数（惯例 60，先不调）
RERANK_TOP_N = 20                                   # 重排只作用于候选集前 N；N 之外保持 RRF 顺序
RERANK_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"


# ══════════════════════════════════════════════════════════════
#  核心函数（★ 三个都是「算法」，要能默写）—— 先看懂，之后遮住重写
# ══════════════════════════════════════════════════════════════

def _tokenize(text):
    """英文分词：转小写 → 只留字母数字（标点当分隔）→ 切词。

    ★ 查询和语料必须走【同一个】_tokenize —— 两边口径不一致，BM25 的分
      就静默偏掉（不报错，最难查）。所以把它抽成一个函数，两处都调它。"""
    return re.findall(r"[a-z0-9]+", text.lower())


def bm25_rank(query, chunks, top_k=None):
    """按 BM25 给全部块打分并排序 → 返回块号列表（高 → 低）。

    · rank_bm25.BM25Okapi：语料 = 每块的 text（各自 _tokenize 后）
    · 查询用同一套 _tokenize
    · 返回【块号 index】，不是分数（下游 RRF 只要排名）
    · top_k=None → 返回全部 302 块号（RRF 要的是完整排名，不是前几个）
    . scores 是一个列表，长度等于 corpus 的长度，每个位置对应一个块的分数。分数越高越相关
    . order 把文档按得分从高到低排好队，然后把它们的编号写在一张列表上
    
    注：302 块很小，每次调用重建索引也无所谓（20 题 ≈ 0.1 秒）。真嫌慢
        再把 BM25Okapi 提到外面建一次。"""
    corpus = [_tokenize(c["text"]) for c in chunks]
    bm25 = BM25Okapi(corpus)
    scores = bm25.get_scores(_tokenize(query))          # scores[j] ↔ chunks[j]
    order = sorted(range(len(chunks)), key=lambda j: scores[j], reverse=True)
    ranked = [chunks[j]["index"] for j in order]
    return ranked if top_k is None else ranked[:top_k]


def rrf_fuse(rankings, k=RRF_K):
    """把多路【排名】融合成一路 → 返回融合后的块号列表（高 → 低）。

    公式：score(块) = Σ 1 / (k + 该块在这一路里的名次)   （名次从 1 开始）
    · 只吃「名次」不吃原始分数 → 两路分数【不用】归一化（这正是 RRF 的好处）
    · 某一路没出现的块，那一路就不给它加分
    · 按 score 降序，返回块号列表

    手算自测：rankings=[[A,B,C],[C,A,B]]，k=60
      A = 1/61 + 1/62 ｜ B = 1/62 + 1/63 ｜ C = 1/63 + 1/61
      → 排序 [A, C, B]（两边都排第 1 的 A 最靠前）"""
    scores = {}
    for ranking in rankings:
        for rank, idx in enumerate(ranking, 1):          # rank 从 1 开始
            scores[idx] = scores.get(idx, 0.0) + 1.0 / (k + rank)
    return sorted(scores, key=lambda idx: scores[idx], reverse=True)


def rerank(query, candidates, chunks, model):
    """用交叉编码器对候选集【精排】→ 返回重排后的候选块号列表。

    · 为每个候选拼 (query, 块文本) 对 → model.predict(pairs) 打分
    · 按分数降序（高分在前）
    · 只重排、不增删 → 返回长度 = 输入候选长度
    model 由管道建好传进来（加载一次 ~90MB，别每题新建）。

    ★ 交叉编码器的意义：它把 query 和「整块文本」一起喂进模型算相关性，
      比向量那种「各自编码再点积」更能抓住细粒度匹配 —— 这正是能把
      「召回了但排低了」的块顶到 rank1 的原因。"""
    text_of = {c["index"]: c["text"] for c in chunks} #建立「块号 → 原文」的字典
    pairs = [(query, text_of[idx]) for idx in candidates] #拼出「问题 + 文本」对
    #这就是交叉编码器的输入格式：它必须同时看到"问题和文档"，才能判断它们有多相关。

    scores = model.predict(pairs) #模型打分
    order = sorted(range(len(candidates)), key=lambda j: scores[j], reverse=True)
    return [candidates[j] for j in order] #返回块号


# ══════════════════════════════════════════════════════════════
#  管道（我写）—— 读文件、循环 20 题、拼流程、写文件
# ══════════════════════════════════════════════════════════════

def _load_json(path):
    return json.load(open(path, encoding="utf-8"))


def _run(use_rerank):
    chunks = _load_json(CHUNKS_JSON)
    eval_set = _load_json(EVAL_JSON)
    dense = _load_json(DENSE_JSON)
    dense_map = {d["id"]: d for d in dense}

    # 重排模型只加载一次（首次会下 ~90MB；国内慢就设 HF_ENDPOINT=https://hf-mirror.com）
    model = None
    if use_rerank:
        from sentence_transformers import CrossEncoder
        print(f"加载重排模型 {RERANK_MODEL} ...")
        model = CrossEncoder(RERANK_MODEL)

    results = []
    for qid, item in enumerate(eval_set, 1):
        q = item["question"]

        # ① dense 那半：直接读 baseline 的向量排名（免费 + 与 baseline 完全一致）
        dense_ranked = dense_map[qid]["ranked"]

        # ② sparse 那半：BM25
        bm25_ranked = bm25_rank(q, chunks)

        # ③ 融合两路排名
        fused = rrf_fuse([dense_ranked, bm25_ranked])

        # ④ 可选：只对候选集前 N 精排，N 之外保持融合顺序（这样 hit@1/3/5 都还能算）
        if use_rerank:
            head = rerank(q, fused[:RERANK_TOP_N], chunks, model)
            final = head + fused[RERANK_TOP_N:]
        else:
            final = fused

        results.append({"id": qid, "question": q, "ranked": final})
        tag = "rerank" if use_rerank else "hybrid"
        print(f"#{qid:02d} [{tag}] top3={final[:3]}")

    out = OUT_RERANK if use_rerank else OUT_HYBRID
    with open(out, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print(f"\n20 题已存 → {out}")
    return results


if __name__ == "__main__":
    _run(use_rerank="--rerank" in sys.argv)
