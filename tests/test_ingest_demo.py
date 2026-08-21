from app.loader import load_document
from app.splitter import split_documents
from app.vectorstore import create_vectorstore

# 只入 3 个 demo 文档
files = [
    "data/samples/员工手册.pdf",
    "data/samples/产品介绍.md",
    "data/samples/常见问题.txt",
]

all_chunks = []
for path in files:
    docs = load_document(path)
    chunks = split_documents(docs)
    all_chunks.extend(chunks)
    print(f"{path}: {len(docs)} 大块 -> {len(chunks)} 小块")

create_vectorstore(all_chunks)
print(f"\n共入库 {len(all_chunks)} 个小块")
