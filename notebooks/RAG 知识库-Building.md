RAG 术语	什么意思	你的 demo 里是哪段	simple_rag 里是哪段
Load	读原始文档	knowledge_text（写死字符串）	PyPDFLoader(path).load()
Transform	切块（把长文本变小块）	split_text()	RecursiveCharacterTextSplitter + replace_t_with_space
Embed	每块 → 向量	get_embedding()	FAISS.from_documents 内部
Store	存向量+原文	vector_store.append(...)	FAISS 索引

RAG 索引阶段分 Load / Transform / Embed & Store 四步