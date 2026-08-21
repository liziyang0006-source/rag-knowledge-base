from app.loader import load_document

docs = load_document("data/samples/sample.txt")

print(f"读取到 {len(docs)} 个文档")
for doc in docs:
    print(doc.page_content)
