# src/keyword_recall.py —— 生成侧指标：标准答案的关键词，在生成答案里命中几个
# 依赖：data/eval_set.json（标准答案）+ data/answers.json（生成答案）
# 命令行用法：
#   python src/keyword_recall.py              # 算 20 题的 keyword_recall
#   python src/keyword_recall.py --noise      # 算「噪声带」（要先跑 generator.py --runs N）
#
# 和 evaluator.py 的区别（一份文件管一侧，别混）：
#   evaluator.py      → 检索侧，比的是【文档块】，用 gold_chunks.json
#   keyword_recall.py → 生成侧，比的是【关键词】，用 answers.json
#
# [!]这个指标测的是「字面重叠」，不是「答对了」。已知缺陷（必须知道）：
#   ① 改写/同义词 → 漏判（答案对，但用词不同 → 分低）  ← 最常见的假阴性
#   ② 否定句     → 致命（"...does NOT contribute" 照样满分）
#   ③ 词形变化   → 漏判（mitigation vs mitigating）
#   ④ 它没有语序、逻辑、语义的概念，是个「词袋」
#   所以：它是【警报器】，不是【判决书】。低分要人工看一眼再下结论。

import json
import re
import sys
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / ".." / "data"

EVAL_JSON = DATA_DIR / "eval_set.json"          # 标准答案（20 题）
ANSWERS_JSON = DATA_DIR / "answers.json"        # 生成答案（generator.py 产物，单次）
NOISE_JSON = DATA_DIR / "answers_multirun.json" # 多次运行结果（generator.py --runs N 产物）
CONFIG_JSON = DATA_DIR / "chunk_config.json"    # chunker.py 写的「这批块的参数」
GEN_CONFIG_JSON = DATA_DIR / "gen_config.json"  # generator.py 写的「这批答案的参数」
SWEEP_JSON = DATA_DIR / "sweep_results.json"    # 扫参总账：每个配置一条记录

# ===== 参数区 =====
# 停用词：不算关键词的词。不滤掉的话，任何一段英文都能命中一堆 the/of/and
# → 分数虚高，而且虚得"看起来很正常"（这是最危险的）
STOPWORDS = set("""
a an the and or but if while of to in on at for with by from as
is are was were be been being am
this that these those it its they them their there here
what which who whom whose how why when where
not no nor so than then too very
can could will would shall should may might must
do does did done have has had having
into over under about above below between during through after before
more most some any all each both few other such only own same
i you he she we me him her us my your his our
""".split())


def keywords(text):
    """把一段文本洗成【关键词集合】。
    转小写 → 标点/符号变空格 → 切词 → 去掉停用词和单字符。

    返回 set（不是 list）——因为下面要用集合交集，去重也顺便解决了。"""
    text = text.lower()
    text = re.sub(r"[^a-z0-9\s]", " ", text)   # 标点、引号、破折号 → 空格
    words = text.split()
    return {w for w in words if w not in STOPWORDS and len(w) > 1}


def keyword_recall(gt_answer, gen_answer):
    """单题分数 = 命中的关键词数 ÷ 标准答案的关键词总数。
    命中判定：标准答案的关键词集合 ∩ 生成答案的关键词集合（和 evaluator 里 hit() 一个套路）。
    标准答案没关键词（极端情况）→ None，避免除零。

    ← 单题版，和 evaluator.py 的 hit() 同一个位置：被下面的批量版调用。"""
    gt_kw = keywords(gt_answer)
    if not gt_kw:
        return None
    return len(gt_kw & keywords(gen_answer)) / len(gt_kw)


def compute_keyword_recall(eval_set, answers):
    """20 题平均。返回 (平均分, 逐题明细)。

    eval_set：eval_set.json（list，靠位置对应第 1..20 题）
    answers ：answers.json（list，每项有 id / answer）"""
    ans_map = {a["id"]: a for a in answers}

    rows = []
    for i, item in enumerate(eval_set, 1):
        gt_answer = item["answer"]
        gen_answer = ans_map[i]["answer"]
        # 逐题明细要 hits/total（打印进度条的分母），所以集合在这里也算一次；
        # 算分本身只走 keyword_recall —— 规则只有一份，20 题这点重复可以忽略。
        gt_kw = keywords(gt_answer)
        rows.append({
            "id": i,
            "hits": len(gt_kw & keywords(gen_answer)),
            "total": len(gt_kw),
            "score": keyword_recall(gt_answer, gen_answer),
        })

    valid = [r for r in rows if r["score"] is not None]
    avg = sum(r["score"] for r in valid) / len(valid) if valid else 0.0
    return avg, rows


def _sweep_key(entry):
    """一条记录的身份：换个配置算新记录，同样配置重跑就覆盖。
    必须和 evaluator.py 的同名函数一致 —— 两个 side 的 key 形状不同，
    「同配置」的判断就会跑偏（生成侧漏了 top_k 的话，top_k=3 和 6 会撞车互相覆盖）。"""
    c = entry["config"]
    return (c.get("chunk_size"), c.get("chunk_overlap"), c.get("top_k"), entry["side"])


def record_sweep(avg, rows):
    """把这次的生成侧结果记进 sweep_results.json（和 evaluator 共用一个总账文件）。"""
    # 生成侧同时依赖两套参数：切块参数（哪几块）+ 生成参数（喂几块、哪个模型）
    config = json.load(open(CONFIG_JSON, encoding="utf-8")) if CONFIG_JSON.exists() else {}
    if GEN_CONFIG_JSON.exists():
        config.update(json.load(open(GEN_CONFIG_JSON, encoding="utf-8")))

    entry = {
        "config": config,
        "side": "generation",
        "metrics": {"keyword_recall": avg},
        "per_question": rows,
    }

    sweep = json.load(open(SWEEP_JSON, encoding="utf-8")) if SWEEP_JSON.exists() else []
    key = _sweep_key(entry)
    sweep = [e for e in sweep if _sweep_key(e) != key]
    sweep.append(entry)

    with open(SWEEP_JSON, "w", encoding="utf-8") as f:
        json.dump(sweep, f, ensure_ascii=False, indent=2)
    print(f"已记入扫参总账（现 {len(sweep)} 条）→ {SWEEP_JSON}")


def compute_noise_band(eval_set, multirun):
    """同一配置跑 N 轮的分数波动范围 —— 「噪声带」。
    返回 (每轮分数 list, 最低, 最高)。

    为什么要它：生成不可复现（6 次跑出 5 种答案），
    所以单个数字不可当真。以后改动要【超出这个范围】才算真改善。"""
    n_runs = len(multirun[0]["answers"])
    scores = []
    for r in range(n_runs):
        answers = [{"id": it["id"], "answer": it["answers"][r]} for it in multirun]
        avg, _ = compute_keyword_recall(eval_set, answers)
        scores.append(avg)
    return scores, min(scores), max(scores)


if __name__ == "__main__":
    eval_set = json.load(open(EVAL_JSON, encoding="utf-8"))

    if "--noise" in sys.argv:
        # ===== 模式二：测噪声带 =====
        if not NOISE_JSON.exists():
            sys.exit(f"没有 {NOISE_JSON}\n先跑：python src/generator.py --runs 5")
        multirun = json.load(open(NOISE_JSON, encoding="utf-8"))
        scores, lo, hi = compute_noise_band(eval_set, multirun)

        print(f"同一配置跑了 {len(scores)} 轮，每轮总分：")
        for i, s in enumerate(scores, 1):
            print(f"  第 {i} 轮：{s:.2f}")
        print(f"\n噪声带 = [{lo:.2f}, {hi:.2f}]   （波动 {hi - lo:.2f}）")
        print("→ 以后改动，分数要超出这个范围才算真改善；带内波动是抽签。")
        print("[!]N ≥ 5 才靠谱（n=2 会被运气骗）。")

    else:
        # ===== 模式一：算总分 =====
        if not ANSWERS_JSON.exists():
            sys.exit(f"没有 {ANSWERS_JSON}\n先跑：python src/generator.py")
        answers = json.load(open(ANSWERS_JSON, encoding="utf-8"))
        avg, rows = compute_keyword_recall(eval_set, answers)

        for r in rows:
            bar = "#" * round(r["score"] * 24)
            print(f"#{r['id']:02d}  {r['hits']:>2}/{r['total']:<2} = {r['score']:.2f}  {bar}")

        print(f"\nkeyword_recall = {avg:.2f}   （20 题平均，纯字面匹配）")
        print("[!]单次数字不可当真：生成有噪声。要测噪声带就跑 --noise。")

        if "--record" in sys.argv:
            record_sweep(avg, rows)
