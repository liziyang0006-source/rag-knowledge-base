from app.retriever import search

print("=== 测试1：相关问题，默认 threshold=0.3（rerank 分数，越高越相关）===")
docs = search("年假有几天", top_k=3)
print(f"捞到 {len(docs)} 个片段")
for d in docs:
    print(f"  - {d.page_content[:30]}...")

print()
print("=== 测试2：无关问题，应被过滤掉（返回空）===")
docs = search("如何做红烧肉", top_k=3)
print(f"捞到 {len(docs)} 个片段（应为 0，因为 rerank 分数低于 0.3 阈值）")

print()
print("=== 测试3：降低阈值到 0.1，放宽过滤（捞更多）===")
docs = search("年假有几天", top_k=3, threshold=0.1)
print(f"threshold=0.1 时捞到 {len(docs)} 个片段")