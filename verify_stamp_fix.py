# verify_stamp_fix.py —— 验证 record_sweep 的修复：runs 该由调用者给，不该从戳里读
#
# 安全性：把 keyword_recall 模块里的三个路径全局变量重定向到【临时沙箱】。
#         record_sweep 里读写的 SWEEP_JSON / GEN_CONFIG_JSON / CONFIG_JSON
#         调用时才去查模块全局 —— 改掉它们，函数就只看得见沙箱。
#         真实 data/ 目录它根本碰不到。
#
# 跑法：python verify_stamp_fix.py     （0 次 API 调用，跑完可删）

import json
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))
import keyword_recall as kr


def read_count(p):
    if not Path(p).exists():
        return 0
    with open(p, encoding="utf-8") as f:
        return len(json.load(f))


REAL_SWEEP = kr.SWEEP_JSON                    # 先记住真账本在哪、有几条
real_before = read_count(REAL_SWEEP)

sandbox = Path(tempfile.mkdtemp(prefix="stamp_check_"))
kr.SWEEP_JSON = sandbox / "sweep_results.json"
kr.GEN_CONFIG_JSON = sandbox / "gen_config.json"
kr.CONFIG_JSON = sandbox / "chunk_config.json"

with open(kr.GEN_CONFIG_JSON, "w", encoding="utf-8") as f:
    json.dump({"top_k": 2, "runs": 5, "model": "qwen-plus"}, f)
with open(kr.CONFIG_JSON, "w", encoding="utf-8") as f:
    json.dump({"chunk_size": 400, "chunk_overlap": 160, "n_chunks": 302}, f)

# 沙箱戳故意写 runs=5 —— 这是陷阱：看 record_sweep 会不会照抄它
rows_single = [{"id": 1, "hits": 3, "total": 4, "score": 0.75}]
rows_noise = [{"id": 1, "total": 4, "score_mean": 0.75, "score_lo": 0.5, "score_hi": 1.0}]

print("=" * 64)
print("测试1：先记【单次】(runs=1)，再记【噪声带】(runs=5)")
print("       戳里写着 runs=5 —— 看单次那条会不会被误标成 5，以及会不会覆盖")
print("=" * 64)

kr.record_sweep(0.7428, rows_single, runs=1)
kr.record_sweep(0.7371, rows_noise, runs=5, band=(0.7312, 0.7428))

with open(kr.SWEEP_JSON, encoding="utf-8") as f:
    sweep = json.load(f)

print(f"\n[结果] 沙箱账本 = {len(sweep)} 条")
for e in sweep:
    c = e["config"]
    print(f"   runs={c.get('runs')}  top_k={c.get('top_k')}  "
          f"band={'有' if 'noise_band' in e['metrics'] else '无'}")

single = [e for e in sweep if e["config"].get("runs") == 1]
print(f"\n   ① 两条都在、没互相覆盖？      {'[OK]' if len(sweep) == 2 else '[FAIL]'}")
print(f"   ② 单次那条的 runs 是 1？      {'[OK]' if len(single) == 1 else '[FAIL]'}")
print("      （戳里明明写着 5 —— 记成 1 才说明修复生效）")

print()
print("=" * 64)
print("测试2：忘传 runs —— 该【响亮报错】，不该静默记成 1")
print("=" * 64)
try:
    kr.record_sweep(0.9, rows_single)
    print("   [FAIL] 没报错 —— runs 有默认值，忘传会被静默吞掉")
except TypeError as e:
    print(f"   [OK] TypeError: {e}")

print()
print("=" * 64)
print("测试3：戳和产物对不上 -> 拒绝记账（选了方案①，这道闸新加的）")
print("=" * 64)


def try_check(stamp, records, data_runs, label, should_pass):
    with open(kr.GEN_CONFIG_JSON, "w", encoding="utf-8") as f:
        json.dump(stamp, f)
    try:
        kr._check_stamp(records, data_runs=data_runs)
        how, ok = "放行", should_pass
    except SystemExit:
        how, ok = "拦住", not should_pass
    print(f"   [{'OK' if ok else 'FAIL'}] {how}  <- {label}")


blk2 = [{"context": "A" * 40 + "\n\n" + "B" * 40}]                      # 2 块 -> top_k=2
blk3 = [{"context": "A" * 40 + "\n\n" + "B" * 40 + "\n\n" + "C" * 40}]  # 3 块 -> top_k=3

try_check({"top_k": 2, "runs": 5}, blk2, 1, "戳 runs=5 / 数据 1 轮   (9/27 那种错配)", False)
try_check({"top_k": 3, "runs": 5}, blk2, 5, "戳 top_k=3 / 数据 2 块  (9/27 被错标的字段)", False)
try_check({"top_k": 2, "runs": 5}, blk3, 5, "戳 top_k=2 / 数据 3 块  (只错在 top_k)", False)
try_check({"top_k": 2, "runs": 5}, blk2, 5, "戳和数据完全一致       (该放行)", True)

print()
print("=" * 64)
print("测试4：真实账本动了吗（要证据，不要感觉）")
print("=" * 64)
real_after = read_count(REAL_SWEEP)
print(f"   {real_before} 条 -> {real_after} 条   "
      f"{'[OK] 没动' if real_before == real_after else '[FAIL] 被改了！'}")
print(f"\n沙箱目录（可删）：{sandbox}")
