# src/make_gold_chunks.py —— 标注每题答案落在哪个 chunk（ground truth）
# 用法：python src/make_gold_chunks.py  → 生成 data/gold_chunks.json
# 依赖：data/chunks.json（chunker.py 的产物）+ data/eval_set.json
# 原理：把每题的 answer 拿去所有 chunk 里找，它完整出现在哪个 chunk，那个 chunk 的 index 就是这题的 gold。

import json
import re
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent      # src/
DATA_DIR = BASE_DIR / ".." / "data"             # data/

CHUNKS_JSON = DATA_DIR / "chunks.json"
EVAL_JSON = DATA_DIR / "eval_set.json"
OUT_JSON = DATA_DIR / "gold_chunks.json"


def norm(s):
    # 空白归一化：把换行、多空格塌成一个空格，再掐头去尾。
    # 因为 PDF 换行会插空格，答案原文可能被拆在相邻两行。
    return re.sub(r"\s+", " ", s).strip()


if __name__ == "__main__":
    chunks = json.load(open(CHUNKS_JSON, encoding="utf-8"))
    eval_set = json.load(open(EVAL_JSON, encoding="utf-8"))

    # 预先把每块归一化，循环里不用重复算
    chunk_norm = [(c["index"], norm(c["text"])) for c in chunks]

    gold = []
    missing = []
    for i, item in enumerate(eval_set, 1):
        ans = norm(item["answer"])
        hits = [idx for idx, cn in chunk_norm if ans in cn]
        gold.append({
            "id": i,
            "chapter": item["chapter"],
            "question": item["question"],
            "gold_chunks": hits,
        })
        if not hits:
            missing.append((i, item["chapter"]))

    with open(OUT_JSON, "w", encoding="utf-8") as f:
        json.dump(gold, f, ensure_ascii=False, indent=2)

    # 汇报
    ok = sum(1 for g in gold if g["gold_chunks"])
    multi = sum(1 for g in gold if len(g["gold_chunks"]) > 1)
    print(f"20 题标注完成：{ok}/20 找到 gold chunk")
    if multi:
        print(f"注意：有 {multi} 题答案落在不止一个 chunk（在重叠区），gold_chunks 里有多个编号——检索到任意一个都算命中")
    for g in gold:
        print(f"  #{g['id']:02d} {g['chapter']} gold_chunks={g['gold_chunks']}")
    if missing:
        print(f"\nMISS: {missing} —— 这几题的答案没完整落在任何 chunk，需要处理")
    print(f"\n已保存 → {OUT_JSON}")
