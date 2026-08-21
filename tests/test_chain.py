from app.chain import ask

answer, docs = ask("什么是RAG？")

print("【答案】")
print(answer)
print()
print("【来源】")
for i, doc in enumerate(docs, 1):
    print(f"来源 {i}: {doc.page_content[:60]}...")
