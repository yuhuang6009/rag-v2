# src/evaluator.py —— 评测指标：hit_rate（核心手写）
# 用法：python src/evaluator.py  → 打印 20 题命中率
# 依赖：data/gold_chunks.json + data/retrieval_results.json

import json
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / ".." / "data"

GOLD_JSON = DATA_DIR / "gold_chunks.json"
RETRIEVAL_JSON = DATA_DIR / "retrieval_results.json"


def hit(retrieved, relevant):
    """retrieved（检索到的编号）里有没有 relevant（gold 编号）。"""
    return bool(set(retrieved) & set(relevant))


def compute_hit_rate(gold_data, retrieval_data):
    """逐题判断命中，返回命中率（命中题数 / 总题数）。"""
    # 按 id 对齐：两份数据顺序可能不同，转成 {id: 数据} 才好用 id 配对
    gold_map = {item["id"]: item for item in gold_data}
    retrieval_map = {item["id"]: item for item in retrieval_data}

    hit_count = 0
    for qid in gold_map:
        relevant = gold_map[qid]["gold_chunks"]
        retrieved = retrieval_map[qid]["retrieved"]
        if hit(retrieved, relevant):
            hit_count += 1

    return hit_count / len(gold_map)


if __name__ == "__main__":
    gold = json.load(open(GOLD_JSON, encoding="utf-8"))
    retrieval = json.load(open(RETRIEVAL_JSON, encoding="utf-8"))

    rate = compute_hit_rate(gold, retrieval)
    print(f"hit_rate = {rate:.2f}  ({round(rate * len(gold))}/{len(gold)} 命中)")
