# src/evaluator.py —— 评测指标：hit_rate（核心手写）
# 用法：python src/evaluator.py  → 逐题诊断 + hit@1/3/5
# 依赖：data/gold_chunks.json + data/retrieval_results.json
# 注：retrieval_results.json 里每题存的是 ranked（114 块的完整排序），
#     所以 hit@k 只是「切前 k 个」，改 k 不用重跑检索。

import json
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / ".." / "data"

GOLD_JSON = DATA_DIR / "gold_chunks.json"
RETRIEVAL_JSON = DATA_DIR / "retrieval_results.json"

TOP_K = 3          # 正式指标的 k（检索 top-3 里含 gold 即命中）


def hit(retrieved, relevant):
    """retrieved（检索到的编号）里有没有 relevant（gold 编号）。"""
    return bool(set(retrieved) & set(relevant))


def gold_rank(ranked, relevant):
    """gold 在完整排序里的最好名次（1-based）；一个都没排上 → None。
    例：ranked=[37,36,15,...]，relevant=[36] → 2（排第 2）。"""
    ranks = [i for i, idx in enumerate(ranked, 1) if idx in relevant]
    return min(ranks) if ranks else None


def compute_hit_rate(gold_data, retrieval_data, top_k=TOP_K):
    """hit@top_k：gold 排进前 top_k 就算这题命中。返回命中率（命中题数 / 总题数）。"""
    # 按 id 对齐：两份数据顺序可能不同，转成 {id: 数据} 才好用 id 配对
    gold_map = {item["id"]: item for item in gold_data}
    retrieval_map = {item["id"]: item for item in retrieval_data}

    hit_count = 0
    for qid in gold_map:
        relevant = gold_map[qid]["gold_chunks"]
        ranked = retrieval_map[qid]["ranked"]
        if hit(ranked[:top_k], relevant):
            hit_count += 1

    return hit_count / len(gold_map)


if __name__ == "__main__":
    gold = json.load(open(GOLD_JSON, encoding="utf-8"))
    retrieval = json.load(open(RETRIEVAL_JSON, encoding="utf-8"))

    gold_map = {item["id"]: item for item in gold}
    retrieval_map = {item["id"]: item for item in retrieval}

    # ① 逐题诊断：rank = gold 在完整排序里排第几（"--" = 排到 114 名开外都没进）
    print("逐题诊断  (rank = gold 在完整排序里的名次)")
    print("-" * 66)
    for qid in gold_map:
        relevant = gold_map[qid]["gold_chunks"]
        ranked = retrieval_map[qid]["ranked"]
        rank = gold_rank(ranked, relevant)
        mark = "HIT " if (rank is not None and rank <= TOP_K) else "MISS"
        rank_s = f"{rank}" if rank else "--"
        print(f"#{qid:02d} [{mark}] gold={relevant} rank={rank_s:>3}  top3={ranked[:TOP_K]}")

    # ② 汇总：同一个检索结果，切不同 k 得到不同口径的分数
    print("-" * 66)
    total = len(gold)
    for k in (1, 3, 5):
        rate = compute_hit_rate(gold, retrieval, top_k=k)
        print(f"hit_rate@{k} = {rate:.2f}  ({round(rate * total)}/{total} 命中)")
