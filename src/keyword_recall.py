# src/keyword_recall.py —— 生成侧指标：标准答案的关键词，在生成答案里命中几个
# 依赖：data/eval_set.json（标准答案）+ data/answers.json（生成答案）
# 命令行用法：
#   python src/keyword_recall.py                        # 算 20 题的 keyword_recall
#   python src/keyword_recall.py --record               # 顺带记进扫参总账（单次）
#   python src/keyword_recall.py --noise                # 算「噪声带」（要先跑 generator.py --runs N）
#   python src/keyword_recall.py --noise --record       # 记噪声带（均值 + 上下界）进总账
#
# ★ 记总账为什么要么带 --noise、要么只是【筛选】：
#   生成有噪声，单次跑只是一个样本，不是「这条配置的成绩」。
#   单次记录只能检出【大于噪声带宽】的差距；差距小的时候它回答不了「谁更好」。
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
    「同配置」的判断就会跑偏（生成侧漏了 top_k 的话，top_k=3 和 6 会撞车互相覆盖）。

    ★ 为什么 runs 也在 key 里：1 轮和 5 轮是【两次不同的测量】，不是同一条配置。
      5 轮贵 5 倍，如果和 1 轮共用一个 key，一次顺手重跑就把它冲掉了。

    ★ 为什么 prompt 也在 key 里：v1 和 short 是【两个不同的实验】。
      不加的话，跑完 v1 再跑 short，key 一模一样 → v1 那条被 short 覆盖
      → 你手里只剩一条记录 → 【实验根本做不出来】。（runs 那个坑的原样重放）

      检索侧没有 runs / prompt → c.get(...) 恒为 None，形状仍然一致。"""
    c = entry["config"]
    return (c.get("chunk_size"), c.get("chunk_overlap"), c.get("top_k"),
            c.get("runs"), c.get("prompt"), entry["side"])


def _data_top_k(records):
    """从产物的 context 反推当时喂了几块。

    context 是 top_k 个块用空行（\\n\\n）拼起来的，而块【内部】没有空行
    （302 块已核实：含空行的 0 个），所以按空行切出来的段数精确等于 top_k。"""
    return len(records[0]["context"].split("\n\n"))


def _check_stamp(records, data_runs):
    """记账前核对：gen_config.json（戳）说的，必须和产物文件里的一致。

    ★ 为什么必须拦：账本的 key 由 config 决定，而分数由产物文件决定。
      两者不是一对的时候，记进去的记录会用【错的 key】覆盖别的配置的真记录，
      而且账本上完全看不出来（数字正常、config 也正常，只是它俩不是一对）。
      9/27 那场覆盖事故就是这么来的。

    对不上就退出，不记 —— 宁可少一条记录，不要一条假记录。
    只在 --record 时调用：不记账的普通查看，戳过期了也允许跑。"""
    if not GEN_CONFIG_JSON.exists():
        sys.exit(f"没有 {GEN_CONFIG_JSON}，无法确认这批数据的身份，拒绝记账")
    stamp = json.load(open(GEN_CONFIG_JSON, encoding="utf-8"))

    problems = []
    if stamp.get("runs") != data_runs:
        problems.append(f"runs ：戳说 {stamp.get('runs')}，产物里是 {data_runs}")
    data_top_k = _data_top_k(records)
    if stamp.get("top_k") != data_top_k:
        problems.append(f"top_k：戳说 {stamp.get('top_k')}，产物的 context 里是 {data_top_k} 块")

    if problems:
        sys.exit("[!] 戳和产物对不上，拒绝记账：\n    " + "\n    ".join(problems)
                 + "\n    → 记下去会用【错的 key】覆盖别的配置的真记录，账本上还看不出来。"
                 "\n    → 先重跑 generator.py 重新生成这批数据，让戳对上，再记账。")


def record_sweep(avg, rows, runs, band=None):
    """把这次的生成侧结果记进 sweep_results.json（和 evaluator 共用一个总账文件）。

    avg  —— 这次的总分（单次跑就是那一次；跑了噪声带就是 N 轮的均值）
    rows —— 逐题明细。单次＝每题一次分数；噪声＝每题 N 轮的均值/上下界
    runs —— 这批数据跑了几轮。★ 故意不给默认值：漏传要【当场报错】。
            若默认成 1，噪声模式忘了传就会被静默记成 1 轮 —— 那正是
            key 撞车、真记录被顺手冲掉的形状（9/27 那场事故）。
    band —— 噪声带 (lo, hi)。单次跑传 None（没测就没有，不编一个出来）

    ★ runs 由【调用者】给，不从 gen_config.json 读：那枚戳是单次/多轮两批
      数据共用的一份，只装得下最后写它的那个 —— 从戳读必错。
    """
    # 生成侧同时依赖两套参数：切块参数（哪几块）+ 生成参数（喂几块、哪个模型、跑几轮）
    config = json.load(open(CONFIG_JSON, encoding="utf-8")) if CONFIG_JSON.exists() else {}
    if GEN_CONFIG_JSON.exists():
        config.update(json.load(open(GEN_CONFIG_JSON, encoding="utf-8")))
    config["runs"] = runs

    metrics = {"keyword_recall": avg}
    if band is not None:
        metrics["noise_band"] = [band[0], band[1]]

    entry = {
        "config": config,
        "side": "generation",
        "metrics": metrics,
        "per_question": rows,
    }

    sweep = json.load(open(SWEEP_JSON, encoding="utf-8")) if SWEEP_JSON.exists() else []
    key = _sweep_key(entry)
    sweep = [e for e in sweep if _sweep_key(e) != key]
    sweep.append(entry)

    with open(SWEEP_JSON, "w", encoding="utf-8") as f:
        json.dump(sweep, f, ensure_ascii=False, indent=2)

    tag = f"{runs} 轮"
    print(f"已记入扫参总账（现 {len(sweep)} 条，本次 runs={tag}）→ {SWEEP_JSON}")


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


def per_question_noise(eval_set, multirun):
    """每题的 N 轮 均值 / 最低 / 最高。

    ★ 为什么总分不够：总分只给一个数，看不出「是哪几题在动」。
      9/15 实测：20 题里只有 #6 #7 #8 #13 #20 会晃，
      #20 一个人在 0.33↔1.00 之间跳，凭它一题就能拉动总分 0.033。
      所以两个配置差 0.03 时，必须翻这张表才知道那是噪声还是真差异。

    total 只有一个值：分母来自【标准答案】，不随生成答案变。"""
    n_runs = len(multirun[0]["answers"])
    per_id = {}                         # id -> {"total": int, "scores": [每轮分数]}
    for r in range(n_runs):
        answers = [{"id": it["id"], "answer": it["answers"][r]} for it in multirun]
        _, rows = compute_keyword_recall(eval_set, answers)
        for row in rows:
            slot = per_id.setdefault(row["id"], {"total": row["total"], "scores": []})
            if row["score"] is not None:
                slot["scores"].append(row["score"])

    out = []
    for i in sorted(per_id):
        sc = per_id[i]["scores"]
        out.append({
            "id": i,
            "total": per_id[i]["total"],
            "score_mean": sum(sc) / len(sc) if sc else None,
            "score_lo": min(sc) if sc else None,
            "score_hi": max(sc) if sc else None,
        })
    return out


if __name__ == "__main__":
    eval_set = json.load(open(EVAL_JSON, encoding="utf-8"))

    if "--noise" in sys.argv:
        # ===== 模式二：测噪声带 =====
        if not NOISE_JSON.exists():
            sys.exit(f"没有 {NOISE_JSON}\n先跑：python src/generator.py --runs 5")
        multirun = json.load(open(NOISE_JSON, encoding="utf-8"))
        scores, lo, hi = compute_noise_band(eval_set, multirun)
        mean = sum(scores) / len(scores)

        print(f"同一配置跑了 {len(scores)} 轮，每轮总分：")
        for i, s in enumerate(scores, 1):
            print(f"  第 {i} 轮：{s:.2f}")
        print(f"\n噪声带 = [{lo:.2f}, {hi:.2f}]   （波动 {hi - lo:.2f}）")
        print(f"均值   = {mean:.2f}   ← 和别的配置比，比的是这个数，不是单轮")
        print("→ 以后改动，分数要超出这个范围才算真改善；带内波动是抽签。")
        print("[!]N >= 5 才靠谱（n=2 会被运气骗）。")

        # 哪几题在动 —— 总分看不出这个，而它才是判「噪声 vs 真差异」的依据
        rows = per_question_noise(eval_set, multirun)
        movers = [r for r in rows if r["score_hi"] - r["score_lo"] > 0.01]
        if movers:
            print(f"\n会动的题（{len(movers)}/{len(rows)} 题，其余 {len(rows)-len(movers)} 题纹丝不动）：")
            for r in movers:
                print(f"  #{r['id']:02d}  {r['score_lo']:.2f} ~ {r['score_hi']:.2f}"
                      f"   (均值 {r['score_mean']:.2f}, 分母 {r['total']} 个关键词)")
            print("→ 分母越小晃得越狠。两个配置一比，先看这几题是不是「同一批人」。")
        else:
            print(f"\n{len(rows)} 题在 {len(scores)} 轮里分数完全一致（这次没抽到噪声）。")

        if "--record" in sys.argv:
            _check_stamp(multirun, data_runs=len(scores))
            record_sweep(mean, rows, runs=len(scores), band=(lo, hi))

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
            _check_stamp(answers, data_runs=1)      # answers.json 是单次产物 → 必为 1 轮
            record_sweep(avg, rows, runs=1)
