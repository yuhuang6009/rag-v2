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
#   ★ 两条路【都】写 gen_config.json，且都带 runs。漏写的话，总账会把
#     「1 轮」和「5 轮」当成同一条配置，贵的那个被顺手重跑冲掉。
#
# 抗断（长任务必备）：
#   ① 重试    —— 单次请求遇到网络抖动，自己等 1/2/4 秒重试，最多 4 次
#   ② 断点续跑 —— 每答完一题存一次 _progress.json；中途挂了重跑同一条命令，
#                从断点接着跑，已花掉的 API 调用不重花
#   ★ 为什么断点不直接写进正式产物：正式文件一旦是半截的，下游会照算不误，
#     给出基于残缺数据的假分数（静默失败）。所以半成品只住 _progress.json。

import json
import os
import sys
import time
from pathlib import Path

import dashscope
import requests
from dashscope import Generation

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / ".." / "data"

CHUNKS_JSON = DATA_DIR / "chunks.json"
RETRIEVAL_JSON = DATA_DIR / "retrieval_results.json"
ANSWERS_JSON = DATA_DIR / "answers.json"           # 产物①：单次运行（faithfulness 用）
NOISE_JSON = DATA_DIR / "answers_multirun.json"    # 产物②：多次运行（噪声带用）
GEN_CONFIG_JSON = DATA_DIR / "gen_config.json"     # 这批答案是哪套参数生成的
PROGRESS_JSON = DATA_DIR / "_progress.json"        # 断点：没跑完的半成品只住这里

# 瞬时故障的重试等待（秒）。共 3 次重试：1s → 2s → 4s
RETRY_WAITS = (1, 2, 4)

# 可以重试的异常：网络层的抖动。不含 API 报错（key 错/参数错重试也是白等）
RETRYABLE = (
    requests.exceptions.ConnectionError,        # 含 RemoteDisconnected（服务端掐连接）
    requests.exceptions.Timeout,                # 读超时
    requests.exceptions.ChunkedEncodingError,   # 传到一半断了
)

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


def _call_once(prompt):
    """发一次请求，返回答案文本；非 200 就抛错（区分该不该重试）。"""
    resp = Generation.call(
        model=MODEL,
        prompt=prompt,
        temperature=0,   # 最尖的分布，尽量确定性
        seed=42,         # 钉死随机序列（关键，光 temperature=0 不够）
    )
    if resp.status_code != 200:
        # 5xx = 服务端的问题 → 值得再试；4xx = 请求本身不对（key 错/参数错）→ 再试也没用
        if 500 <= resp.status_code < 600:
            raise requests.exceptions.ConnectionError(
                f"服务端 {resp.status_code}: {resp.message}")
        raise RuntimeError(f"API 报错 {resp.status_code}: {resp.message}")
    return resp.output.text


def generate_answer(question, context):
    """调 qwen-plus，返回答案纯文本。瞬时网络故障自动重试。

    ★ 为什么要重试：一个任务要连发 100 个请求，中途被服务端掐连接是常态
      （9/27 就在第 17 题撞上 RemoteDisconnected）。没有重试 → 前面 80 次全白花。"""
    prompt = build_prompt(question, context)
    for i, wait in enumerate(RETRY_WAITS, 1):
        try:
            return _call_once(prompt)
        except RETRYABLE as e:
            print(f"  [!] 第 {i} 次失败（{type(e).__name__}），{wait}s 后重试")
            time.sleep(wait)
    return _call_once(prompt)   # 最后一次不兜住：还失败就把错抛出去，让调用者知道


def _save_progress(top_k, runs, done):
    """把断点写进 _progress.json。

    ★ 为什么不直接写正式产物：正式文件一旦是半截的，下游（keyword_recall）
      会【照算不误】，给出一个基于残缺数据的假分数 —— 那种失败是静默的。
      所以半成品只能住在 _progress.json，正式文件只写【完整的】。

    先写 .tmp 再改名（原子替换）：避免写到一半断电，留下一个坏 json 让断点报废。"""
    tmp = PROGRESS_JSON.with_suffix(".tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump({"top_k": top_k, "runs": runs, "done": done}, f,
                  ensure_ascii=False, indent=2)
    tmp.replace(PROGRESS_JSON)


def _clear_progress():
    """正式产物写完后清掉断点 —— 断点的含义是「有没干完的活」。"""
    PROGRESS_JSON.unlink(missing_ok=True)
    PROGRESS_JSON.with_suffix(".tmp").unlink(missing_ok=True)


def _load_progress(top_k, runs):
    """读断点。配置对不上就作废（换了 top_k/runs 还想接旧进度 = 串味）。"""
    if not PROGRESS_JSON.exists():
        return {}
    try:
        p = json.load(open(PROGRESS_JSON, encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        print("[断点] _progress.json 读不动（可能上次写到一半断了），丢弃重跑")
        return {}

    if p.get("top_k") == top_k and p.get("runs") == runs:
        done = p.get("done", {})
        if done:
            print(f"[断点] 已答完 {len(done)}/20 题（top_k={top_k} runs={runs}），接着跑剩下的")
        return done

    print(f"[断点] 旧进度是 top_k={p.get('top_k')} runs={p.get('runs')}，"
          f"和这次（{top_k}/{runs}）对不上，丢弃重跑")
    return {}


def generate_all(top_k, runs=1):
    """跑 20 题 × runs 轮。返回 [{id, question, context, answers: [第1次, 第2次, ...]}]。

    context 每题只存一份 —— 同一题每一轮给大模型的资料【完全相同】（检索是确定性的），
    所以存一份就够，不用存 N 份。变的是答案，不是材料。

    ★ 断点续跑：每答完一题存一次进度。中途挂了，重跑同一条命令 → 从断点接，
      已经花掉的调用不重花。进度文件带 top_k/runs，换了配置不会串味。"""
    chunks = json.load(open(CHUNKS_JSON, encoding="utf-8"))
    retrieval = json.load(open(RETRIEVAL_JSON, encoding="utf-8"))

    # {块编号: 原文}，方便按 ranked 里的编号取文本
    chunk_map = {c["index"]: c["text"] for c in chunks}

    done = _load_progress(top_k, runs)
    results = []
    for item in retrieval:                                # 每题：id / question / ranked
        qid = item["id"]
        saved = done.get(str(qid))                        # json 的 key 是字符串
        if saved and len(saved.get("answers", [])) == runs:
            results.append(saved)                         # 这题上次已跑完，直接用
            print(f"#{qid:02d} 断点已有 {runs} 轮 → 跳过")
            continue

        question = item["question"]
        ids = item["ranked"][:top_k]                      # 完整榜单切前 k 名
        context = "\n\n".join(chunk_map[i] for i in ids)  # 取原文，空行隔开

        answers = [generate_answer(question, context) for _ in range(runs)]
        record = {
            "id": qid,
            "question": question,
            "context": context,                           # 存下给了哪些资料，第 7 周查忠实度要用
            "answers": answers,                           # N 个答案（runs=1 时就是 1 个）
        }
        results.append(record)
        done[str(qid)] = record
        _save_progress(top_k, runs, done)                 # ← 一题一存，挂了只丢这一题

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

    # 记录生成参数 —— top_k 决定喂几块，直接改变答案，是这批数据的身份之一
    with open(GEN_CONFIG_JSON, "w", encoding="utf-8") as f:
        json.dump({"top_k": top_k, "runs": 1, "model": MODEL}, f,
                  ensure_ascii=False, indent=2)

    print(f"\n20 题答案已存 → {ANSWERS_JSON}")
    print(f"生成参数已记录 → {GEN_CONFIG_JSON}")
    _clear_progress()          # 正式产物已落盘，断点作废
    return flat


def run_noise(top_k, runs):
    """同一配置跑 runs 轮 → answers_multirun.json（给噪声带用）。

    [!] 要花 runs × 20 次生成调用。runs=5 → 100 次。"""
    results = generate_all(top_k, runs=runs)
    with open(NOISE_JSON, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    # 记录生成参数 —— 和 _run() 一样，别漏。
    # runs 是这一批数据的身份之一：1 轮和 5 轮是【两条不同的记录】
    # （见 keyword_recall.py 的 _sweep_key），不写的话两批会撞车互相覆盖。
    with open(GEN_CONFIG_JSON, "w", encoding="utf-8") as f:
        json.dump({"top_k": top_k, "runs": runs, "model": MODEL}, f,
                  ensure_ascii=False, indent=2)

    print(f"\n20 题 × {runs} 轮已存 → {NOISE_JSON}")
    print(f"生成参数已记录 → {GEN_CONFIG_JSON}")
    _clear_progress()          # 正式产物已落盘，断点作废
    print("接着跑：python src/keyword_recall.py --noise --record   看噪声带并存进总账")
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



# python src/generator.py --runs 5  
# python src/keyword_recall.py --noise

