# 需要导入的包
import os
import dashscope
import numpy as np
from dashscope import TextEmbedding


# RAG demo 改写

# 配置区
dashscope.api_key=os.getenv("DASHSCOPE_API_KEY")


# 1.模拟知识库文本
from pathlib import Path
# 让路径跟着本 .py 文件走：不管你在哪个目录运行都能找到同目录下的文件
base_dir = Path(__file__).parent

with open(base_dir / "RAG 知识库-Building.md", "r", encoding="utf-8") as f:
    knowlege_text = f.read()
    print(f"已读入{len(knowlege_text)}个字符")  # 用于验证
    

# 2.简易文本分块
def split_text(text,chunk_size=200):
    chunks=[]
    start=0
    while start<len(text):
        chunks.append(text[start:start+chunk_size])
        start+=chunk_size
    return chunks    

# 方式2 带重叠切块（overlap）
# 重叠（overlap）切的时候相邻块保留一点重叠，就像 simple_rag 里的 chunk_overlap=200,避免关键句正好落在接缝上。
def split_text_overlap(text,chunk_size=80,chunk_overlap=30):
    chunks=[]
    start=0
    while start<len(text):
        end=start+chunk_size
        chunks.append(text[start:end])
        if end>=len(text):
            break
        start=end-chunk_overlap
    return chunks


# 3.获取文本向量 embedding
def get_embedding(text):
    rsp=TextEmbedding.call(
        model=TextEmbedding.Models.text_embedding_v3,
        input=text
    )
    return rsp.output["embeddings"][0]["embedding"]

# 4.余弦相似度计算，用来匹配相似文本
from math import sqrt
def cosine_similarity(v1, v2):
    cos_up = sum(a * b for a, b in zip(v1, v2))
    cos_down_v1 = sqrt(sum(x * x for x in v1))
    cos_down_v2 = sqrt(sum(x * x for x in v2))
    return cos_up / (cos_down_v1 * cos_down_v2)

# ============ 初始化知识库 ============
# chunk_list=split_text(knowlege_text) #分块放入
chunk_list=split_text_overlap(knowlege_text) #用split_text_overlap 带重叠切块放入
# 保存【文本块+对应向量】
vector_store=[]
for chunk in chunk_list:
    vec=get_embedding(chunk)
    vector_store.append({"text":chunk,"vector":vec})


# ============ RAG查询函数 ===========
def rag_query(user_question,top_k=2):

     # ---- 第1步：问题向量化 ----
     # 将用户自然语言问题转换为高维向量，便于与知识库向量计算相似度
    q_vec=get_embedding(user_question)

     # ---- 第2步：遍历知识库，计算余弦相似度 ----
    scores_list=[] #存储每个文本块的{原文，得分}
    for item in vector_store:
        score=cosine_similarity(q_vec,item["vector"])
        scores_list.append({"text":item["text"],"score":score})
        
     # ---- 第3步：按相似度排序，取 Top-K 条 ----
     # reverse=True 降序排列，最相关的文本排在前面
    scores_list.sort(key=lambda x:x["score"],reverse=True)

    # 将 top_k 条最相关文本用换行拼接成上下文
    related_context="\n".join(x["text"] for x in scores_list[:top_k])

     # ---- 第4步：构造提示词（Prompt）----
     # 将检索到的参考资料与用户问题拼接，引导大模型基于资料回答
    prompt=f"""
参考下面资料回答用户问题，尽量使用资料回答问题，不要编造，
【参考资料】
{related_context}

【用户问题】
{user_question}

    """

    # ---- 第5步：调用通义千问大模型生成答案 ----
    from dashscope import Generation  # 延迟导入，仅在使用时加载
    resp=Generation.call(
        model="qwen-plus",
        prompt=prompt  # 传入构造好的提示词
    )
    return resp.output.text

# ===================== 运行测试 =====================
if __name__ =="__main__":
    answer=rag_query("RAG 索引分为哪几个阶段")
    print(answer)
