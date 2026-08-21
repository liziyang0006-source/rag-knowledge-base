from app.retriever import search

print("=== 测试1：相关问题应该能捞到（分数低=相关）===")
docs = search("年假有几天", top_k=3)
print(f"捞到 {len(docs)} 个片段")
for d in docs:
    print(f"  - {d.page_content[:30]}...")

print()
print("=== 测试2：无关问题应该被过滤掉（返回空）===")
docs = search("如何做红烧肉", top_k=3)
print(f"捞到 {len(docs)} 个片段（应为 0，因为库里没有做菜内容）")

print()
print("=== 测试3：提高阈值看效果 ===")
docs = search("年假有几天", top_k=3, threshold=1.4)
print(f"threshold=1.26 时捞到 {len(docs)} 个片段")
