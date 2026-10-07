# verify_prompt_fix.py —— 验证 --prompt 改造：解析、key、断点三处
# 跑法：python verify_prompt_fix.py    （0 次 API 调用，跑完可删）
# 安全：全程不碰真数据（断点那两个测试写进临时沙箱）

import json
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))
import generator as gen
import keyword_recall as kr
import evaluator as ev

RESULTS = []


def ok(cond, label):
    RESULTS.append((bool(cond), label))


# ============================================================
print("=" * 64)
print("A. --prompt 解析：默认 / 两种写法 / 拼错必须响亮退出")
print("=" * 64)

REAL_ARGV = sys.argv


def with_argv(args, fn):
    sys.argv = ["generator.py"] + args
    try:
        return fn()
    finally:
        sys.argv = REAL_ARGV


ok(with_argv([], lambda: gen.prompt_arg(gen.DEFAULT_PROMPT)) == "v1",
   "不给 --prompt          -> 默认 v1")
ok(with_argv(["--prompt", "short"], lambda: gen.prompt_arg(gen.DEFAULT_PROMPT)) == "short",
   "--prompt short         -> short（空格写法）")
ok(with_argv(["--prompt=short"], lambda: gen.prompt_arg(gen.DEFAULT_PROMPT)) == "short",
   "--prompt=short         -> short（等号写法）")


def typo_behavior():
    try:
        with_argv(["--prompt", "shrot"], lambda: gen.prompt_arg(gen.DEFAULT_PROMPT))
        return "没报错，静默跑了默认"
    except SystemExit:
        return "当场退出"


got = typo_behavior()
ok(got == "当场退出", f"--prompt shrot (拼错)  -> {got}")

# 单变量检查：short 去掉新增的那一句，必须和 v1 逐字相同
EXTRA = "用一到两句话直接回答，不要复述资料原文。\n"
ok(gen.PROMPT_VERSIONS["short"].replace(EXTRA, "") == gen.PROMPT_VERSIONS["v1"],
   "short 去掉新增那一句 == v1（单变量：除了这一句没动别的）")

# ============================================================
print()
print("=" * 64)
print("B. _sweep_key：不同 prompt 必须是不同的记录身份")
print("=" * 64)


def gen_entry(prompt, runs=5, cs=400, top_k=2, side="generation"):
    return {"config": {"chunk_size": cs, "chunk_overlap": 160, "top_k": top_k,
                       "runs": runs, "prompt": prompt},
            "side": side, "metrics": {}, "per_question": []}


k_v1 = kr._sweep_key(gen_entry("v1"))
k_short = kr._sweep_key(gen_entry("short"))
k_none = kr._sweep_key(gen_entry(None))

ok(k_v1 != k_short, "v1 和 short 的 key 不同  -> 两条记录不会互相覆盖")
ok(k_none != k_v1, "老记录(无 prompt)的 key 也独立 -> 会和新 v1 共存，不被顶掉")
print(f"      v1    -> {k_v1}")
print(f"      short -> {k_short}")
print(f"      None  -> {k_none}   <- 今天之前那 4 条老记录长这样")

# 形状一致性：同一个 entry，两个文件必须算出【完全相同】的 key
retr_entry = {"config": {"chunk_size": 400, "chunk_overlap": 160, "top_k": 3},
              "side": "retrieval", "metrics": {}, "per_question": []}
ok(kr._sweep_key(retr_entry) == ev._sweep_key(retr_entry),
   "同一个 entry 在两个文件里算出同一个 key（形状一致）")

# ============================================================
print()
print("=" * 64)
print("C. 断点：配置对不上就必须丢弃（prompt 也要比）")
print("=" * 64)

sandbox = Path(tempfile.mkdtemp(prefix="prompt_check_"))
gen.PROGRESS_JSON = sandbox / "_progress.json"


def write_progress(prompt, top_k, runs, done):
    with open(gen.PROGRESS_JSON, "w", encoding="utf-8") as f:
        json.dump({"prompt": prompt, "top_k": top_k, "runs": runs, "done": done},
                  f, ensure_ascii=False)


write_progress("v1", 2, 5, {"1": {"id": 1}})
ok(gen._load_progress("v1", 2, 5) != {}, "配置全同      -> 接着跑")
ok(gen._load_progress("short", 2, 5) == {}, "只 prompt 不同 -> 丢弃（不会半 v1 半 short）")
ok(gen._load_progress("v1", 3, 5) == {}, "只 top_k 不同  -> 丢弃")
ok(gen._load_progress("v1", 2, 1) == {}, "只 runs 不同   -> 丢弃")

# ============================================================
print()
print("=" * 64)
print("结果")
print("=" * 64)
for passed, label in RESULTS:
    print(f"   [{'OK' if passed else 'FAIL'}] {label}")
bad = [r for r in RESULTS if not r[0]]
print(f"\n   {len(RESULTS) - len(bad)}/{len(RESULTS)} 通过")
print(f"\n沙箱目录（可删）：{sandbox}")
