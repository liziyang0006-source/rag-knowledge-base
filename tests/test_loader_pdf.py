from app.loader import load_document

docs = load_document("data/samples/sample.pdf")


print(f"这个pdf一共{len(docs)}页")
print("="*40)
for i, doc in enumerate(docs,start=1):
    print(f"---第{i}---")
    print(doc.page_content[:200])
    