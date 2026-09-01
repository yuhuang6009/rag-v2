# src/

代码。每个模块一句话说明用途，核心函数手写。

建议模块：
- `chunker.py` —— 文本切块（手写）
- `retriever.py` —— 向量化 + top-k 检索（手写）
- `evaluator.py` —— `hit_rate` / `keyword_recall` 评测指标（手写，阶段1核心）
- `tune.py` —— 调参脚本：换切分大小 / top_k / prompt，记录分数
- `app.py` —— Streamlit 界面（核心逻辑手写，界面 vibe coding）

规则：**核心函数先手写，不查资料，写不出来再看原代码**（回忆练习不是作弊）。
