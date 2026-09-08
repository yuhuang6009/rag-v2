# src/chunker.py —— 知识库切块（手写）
# 用法：python src/chunker.py  → 生成 data/chunks.json
# 第 7 周调参：只改 CHUNK_SIZE / CHUNK_OVERLAP，重跑本文件即可

import json
from pathlib import Path

# 路径跟着本文件走，不管在哪运行都能找到
BASE_DIR = Path(__file__).resolve().parent          # src/
DATA_DIR = BASE_DIR / ".." / "data"                 # data/
SRC_TXT = DATA_DIR / "Understanding_Climate_Change.txt"
OUT_JSON = DATA_DIR / "chunks.json"

# ===== 参数区（第 7 周实验改这里）=====
# 为什么是 800/160：预检发现 500 会把 4 题的答案拦腰切断（答案跨块边界），
# 实验后 800 是让 20 题答案都完整落进某一块的最小 chunk_size。够用最小，块别贪大。
CHUNK_SIZE = 800          # 每块约多少字符
CHUNK_OVERLAP = 160       # 相邻块重叠多少字符（≈ chunk_size 的 20%）


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
    raw = load_text(SRC_TXT)
    text = clean_text(raw)
    chunks = chunk_text(text)

    records = [{"index": i, "text": c} for i, c in enumerate(chunks)]
    with open(OUT_JSON, "w", encoding="utf-8") as f:
        json.dump(records, f, ensure_ascii=False, indent=2)

    print(f"读入原始文本：{len(raw)} 字符")
    print(f"清洗页标记后：{len(text)} 字符（删掉 {len(raw)-len(text)} 字符）")
    print(f"切成 {len(chunks)} 块（chunk_size={CHUNK_SIZE}, overlap={CHUNK_OVERLAP}）")
    print(f"已保存 → {OUT_JSON}")
