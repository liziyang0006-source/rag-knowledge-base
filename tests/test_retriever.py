from app.retriever import search

# 问一个 PDF 里确实有的问题
query = "什么是RAG？"

docs = search(query, top_k=3)

print(f"问题：{query}")
print(f"检索到 {len(docs)} 个相关片段")
print("=" * 40)
for i, doc in enumerate(docs, 1):
    print(f"--- 片段 {i} ---")
    print(doc.page_content[:150])  # 只打印前 150 字
    print()
