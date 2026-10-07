# data/experiments/ —— 实验快照归档

> **为什么有这个目录**：账本 `sweep_results.json` 只存**数字**（分数、噪声带、每题明细），
> 不存**原始产物**（那一批答案本身）。想看"当时模型到底写了什么"就得回头重跑 —— 又要花钱。
> 这个目录把**跑过一次的产物**存下来，让每个实验**自带身份**（数字 + 原文 + 当时的配置）。

## 目录结构

```
data/experiments/
├─ README.md              ← 你正在看（索引：每个实验一行）
└─ generation/            ← 生成侧（检索侧没有产物文件，全在账本里）
   ├─ topk6_0927/         answers_multirun.json + config_reconstructed.json
   ├─ run1_0927/          answers.json（runs=1，单次样本）+ config_reconstructed.json
   ├─ v1_0927/            answers_multirun.json + config_reconstructed.json
   ├─ v1_1003/
   └─ short_1003/
```

**检索侧为什么没有子目录**：检索侧的产物是 `retrieval_results.json`（每题的完整块排序），
它跟着 chunk 配置走、且只有一份（chunk 一变就重跑），没有"每个配置留一份"的需求。
检索侧的实验记录全在账本 `sweep_results.json` 里，见下面第二张表。

## 索引一：生成侧（有产物快照的）

> `config_reconstructed.json` 是**从账本重建**的配置，**不是**当时的原始戳
> （原始戳 `gen_config.json` 是共享的可变文件，早被后来的运行覆盖了 —— 这正是 9/29 那个结构病）。
> 文件名故意带 `reconstructed`，免得以后误以为是原件。

| 目录 | 账本# | 日期 | top_k | runs | prompt | 均值 keyword_recall | 结论 |
|---|---|---|---|---|---|---|---|
| `topk6_0927/` | #6 | 9/27 | 6 | 5 | `null`（硬编码 v1 时代） | 0.7755 | k 从 2 抬到 6 只涨 0.038 |
| `run1_0927/` | #7 | 9/27 | 2 | 1 | `null` | 0.7428 | **runs=1，落在噪声带里，只能粗筛** |
| `v1_0927/` | #8 | 9/27 | 2 | 5 | `null`（=v1 硬编码时代） | 0.7371 | 选定 cs=400+k=2 |
| `v1_1003/` | #9 | 10/3 | 2 | 5 | `v1` | 0.7381 | 复跑，验证隔 4 天无漂移 |
| `short_1003/` | #10 | 10/3 | 2 | 5 | `short` | 0.6226 | 掉 0.116，**但答案更好** → 尺子盲区 |

各目录的产物分别是从 `data/` 下同名备份**拷贝**来的（原件未动）：
```
topk6_0927/answers_multirun.json  ← data/answers_multirun_topk6.json
run1_0927/answers.json            ← data/answers.json
v1_0927/answers_multirun.json     ← data/answers_multirun_v1_0927.json
v1_1003/answers_multirun.json     ← data/answers_multirun_v1_1003.json
short_1003/answers_multirun.json  ← data/answers_multirun_short_1003.json
```

> ⚠️ **账本第 5 条（cs=800 / k=3 / runs=5 / 0.7370）没有产物快照** ——
> 它的输出被后面的运行覆盖了，找不回来。这不是笔误，是 9/29 结构病的**现场证据**：
> 手写 `gen_config.json` 当图章 + 固定文件名 `answers_multirun.json`，一起被下一个实验冲掉。
> 这个目录就是在给它"补补丁"。

## 索引二：全部实验（账本 `sweep_results.json`，10 条）

```
—— 检索侧（指标 hit_rate，离散、单次即准）——
#1  cs=400  ov=160  n=302  k=3   hit@1 0.75  hit@3 1.00  hit@5 1.00
#2  cs=600  ov=180  n=173  k=3   hit@1 0.80  hit@3 0.90  hit@5 1.00
#3  cs=800  ov=160  n=114  k=3   hit@1 0.60  hit@3 0.90  hit@5 1.00
#4  cs=1200 ov=240  n=76   k=3   hit@1 0.60  hit@3 0.95  hit@5 1.00
                                 ★ hit@3 顶到 0.90~1.00 → k≥3 上已饱和，区分度在 hit@1

—— 生成侧（指标 keyword_recall，有噪声、要 ≥5 轮）——
#5  cs=800  k=3  runs=5              0.7370
#6  cs=400  k=6  runs=5              0.7755
#7  cs=400  k=2  runs=1              0.7428   ← 单次样本，只能粗筛
#8  cs=400  k=2  runs=5              0.7371   ← 9/27 定稿的一条
#9  cs=400  k=2  runs=5  prompt=v1   0.7381
#10 cs=400  k=2  runs=5  prompt=short 0.6226  ← 尺子判不了"哪个更好"，见 logs/2026-10-03.md
```

**读账本的一条纪律**：靠 `config` 认记录，**别靠"第几条"** ——
因为"覆盖"和"新增"都是"先删同 key 的、再 append 到末尾"，位置看不出来。

## 以后新增实验的快照约定

```
每个新实验建一个目录：  <配置简称>_<日期>/    例：rerank_1012/
里面放：
  · 产物文件（answers_multirun.json / 或重排后的 retrieval_results.json）
  · config_reconstructed.json —— 这次跑的真实配置（top_k / runs / prompt / model）
  · 可选：一行 note.md —— 这次想验什么、结果如何

★ 关键：产物 + 配置【放在同一个目录里】。
  这就是 9/29 log 说的「产物自带身份」—— 在还没改文件格式之前，
  用【目录结构】手动实现它，是成本最低的办法。
```
