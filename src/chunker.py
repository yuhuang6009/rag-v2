# src/chunker.py —— 知识库切块（手写）
# 用法：python src/chunker.py                          → 生成 data/chunks.json
#       python src/chunker.py --chunk-size 400         → 换块大小（第 7 周扫参）
#       python src/chunker.py --chunk-size 400 --overlap 160
#
# 第 7 周调参：两种方式都行 ——
#   ① 改下面 CHUNK_SIZE / CHUNK_OVERLAP 常量（简单，但要开文件）
#   ② 命令行传 --chunk-size / --overlap（能写进 for 循环，且跑完源码里躺着的是默认值）
#
# ⚠️ 换了 chunk_size 必须重跑 retriever.py --rebuild（块变了，向量索引就失效了）

import json
import sys
from pathlib import Path

# 路径跟着本文件走，不管在哪运行都能找到
BASE_DIR = Path(__file__).resolve().parent          # src/
DATA_DIR = BASE_DIR / ".." / "data"                 # data/
SRC_TXT = DATA_DIR / "Understanding_Climate_Change.txt"
OUT_JSON = DATA_DIR / "chunks.json"
CONFIG_JSON = DATA_DIR / "chunk_config.json"        # 记录「这次的块是哪套参数切的」

# ===== 参数区（默认值；第 7 周实验可改，也可命令行覆盖）=====
# 为什么是 800/160：预检发现 500 会把 4 题的答案拦腰切断（答案跨块边界），
# 实验后 800 是让 20 题答案都完整落进某一块的最小 chunk_size。够用最小，块别贪大。
CHUNK_SIZE = 800          # 每块约多少字符
CHUNK_OVERLAP = 160       # 相邻块重叠多少字符（≈ chunk_size 的 20%）


def int_arg(name, default):
    """从命令行取 --name 后面的整数，没给就返回 default。
    两种写法都认：--name 400   或   --name=400"""
    for i, arg in enumerate(sys.argv):
        if arg == name:                          # 空格隔开：值在下一个词
            return int(sys.argv[i + 1])
        if arg.startswith(name + "="):           # 等号连着：值在等号后面
            return int(arg.split("=", 1)[1])
    return default


def load_text(path):
    with open(path, "r", encoding="utf-8") as f:
        return f.read()


def clean_text(text):
    # 去掉 pypdf 提取 PDF 时加的页标记行（"===== 第 N 页 ====="），不是文档内容
    lines = text.splitlines()
    kept = [ln for ln in lines if not ln.strip().startswith("=====")]
    return "\n".join(kept)


def chunk_text(text, chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP):
    # 带重叠切块（逻辑和你 demo 里的 split_text_overlap 一样）
    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunks.append(text[start:end])
        if end >= len(text):
            break
        start = end - chunk_overlap
    return chunks


if __name__ == "__main__":
    # 默认用常量；命令行给了就覆盖。配置成了「输入」，才能写进 for 循环批量跑。
    chunk_size = int_arg("--chunk-size", CHUNK_SIZE)
    chunk_overlap = int_arg("--overlap", CHUNK_OVERLAP)

    raw = load_text(SRC_TXT)
    text = clean_text(raw)
    chunks = chunk_text(text, chunk_size, chunk_overlap)

    records = [{"index": i, "text": c} for i, c in enumerate(chunks)]
    with open(OUT_JSON, "w", encoding="utf-8") as f:
        json.dump(records, f, ensure_ascii=False, indent=2)

    # 把「这批块是哪套参数切的」单独存一份 —— 下游（evaluator/keyword_recall）
    # 记分时能自动带上配置，不用人肉转述，也不会因为覆盖 chunks.json 而认错配置。
    config = {"chunk_size": chunk_size, "chunk_overlap": chunk_overlap,
              "n_chunks": len(chunks)}
    with open(CONFIG_JSON, "w", encoding="utf-8") as f:
        json.dump(config, f, ensure_ascii=False, indent=2)

    print(f"读入原始文本：{len(raw)} 字符")
    print(f"清洗页标记后：{len(text)} 字符（删掉 {len(raw)-len(text)} 字符）")
    print(f"切成 {len(chunks)} 块（chunk_size={chunk_size}, overlap={chunk_overlap}）")
    print(f"已保存 → {OUT_JSON}")
    print(f"配置已记录 → {CONFIG_JSON}")
    print("[!]块变了，向量索引失效 —— 接着跑：python src/retriever.py --rebuild")
