from app.retriever import get_vectorstore
from app.reranker import rerank

query = "年假有几天"

# 第一阶段：向量召回（粗筛，此时顺序只由向量相似度决定）
vs = get_vectorstore()
candidates = vs.similarity_search(query, k=5)
print("=== 第一阶段：向量召回 top 5（原始顺序）===")
for i, c in enumerate(candidates):
    print(f"  [{i}] {c.page_content[:40]}...")

# 第二阶段：rerank 精排（对召回结果重新打分排序，顺序会发生变化）
print()
print("=== 第二阶段：rerank 精排（按相关性重排）===")
texts = [c.page_content for c in candidates]
ranked = rerank(query, texts, top_n=5)  # 返回 (score, index, text)
for score, index, text in ranked:
    print(f"  score={score:.4f} 原召回序[{index}] {text[:40]}...")