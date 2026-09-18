# src/evaluator.py —— 评测指标：hit_rate（核心手写）
# 用法：python src/evaluator.py            → 逐题诊断 + hit@1/3/5
#       python src/evaluator.py --record   → 额外把这个配置的结果记进 sweep_results.json
# 依赖：data/gold_chunks.json + data/retrieval_results.json
# 注：retrieval_results.json 里每题存的是 ranked（114 块的完整排序），
#     所以 hit@k 只是「切前 k 个」，改 k 不用重跑检索。

import json
import sys
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / ".." / "data"

GOLD_JSON = DATA_DIR / "gold_chunks.json"
RETRIEVAL_JSON = DATA_DIR / "retrieval_results.json"
CONFIG_JSON = DATA_DIR / "chunk_config.json"      # chunker.py 写的「这批块的参数」
SWEEP_JSON = DATA_DIR / "sweep_results.json"      # 扫参总账：每个配置一条记录

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


def _sweep_key(entry):
    """一条记录的身份：换个配置算新记录，同样配置重跑就覆盖。"""
    c = entry["config"]
    return (c.get("chunk_size"), c.get("chunk_overlap"), c.get("top_k"), entry["side"])


def record_sweep(gold, retrieval, top_k=TOP_K):
    """把这次的检索侧结果记进 sweep_results.json。

    存的是什么：
      config       —— 这批块的参数（来自 chunker.py 写的 chunk_config.json）
      metrics      —— hit_rate@1/3/5 三个口径
      per_question —— 每题的 gold_rank。★ 有它就够推出任意 k 的 hit_rate
                      （hit@k == gold_rank <= k），所以「等字符量对比」可以事后算，不用重跑。
    """
    config = json.load(open(CONFIG_JSON, encoding="utf-8")) if CONFIG_JSON.exists() else {}
    config["top_k"] = top_k

    gold_map = {g["id"]: g for g in gold}
    retrieval_map = {r["id"]: r for r in retrieval}

    entry = {
        "config": config,
        "side": "retrieval",
        "metrics": {f"hit_rate@{k}": compute_hit_rate(gold, retrieval, top_k=k)
                    for k in (1, 3, 5)},
        "per_question": [
            {"id": qid,
             "gold_rank": gold_rank(retrieval_map[qid]["ranked"], gold_map[qid]["gold_chunks"])}
            for qid in gold_map
        ],
    }

    sweep = json.load(open(SWEEP_JSON, encoding="utf-8")) if SWEEP_JSON.exists() else []
    key = _sweep_key(entry)
    sweep = [e for e in sweep if _sweep_key(e) != key]     # 同配置的旧记录丢掉，避免重复
    sweep.append(entry)

    with open(SWEEP_JSON, "w", encoding="utf-8") as f:
        json.dump(sweep, f, ensure_ascii=False, indent=2)
    print(f"\n已记入扫参总账（现 {len(sweep)} 条）→ {SWEEP_JSON}")


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

    # ③ 可选：记进扫参总账（扫 chunk_size 时必须开，否则换下个配置就覆盖了）
    if "--record" in sys.argv:
        record_sweep(gold, retrieval)
