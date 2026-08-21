from app.loader import load_document
from app.splitter import split_documents
from app.vectorstore import create_vectorstore

# 1. 读取 sample.pdf
docs = load_document("data/samples/sample.pdf")
print(f"读取到 {len(docs)} 个大块")

# 2. 切碎
chunks = split_documents(docs)
print(f"切碎成 {len(chunks)} 个小块")

# 3. 存入 Chroma
vectorstore = create_vectorstore(chunks)
print(f"已存入向量库，共 {vectorstore._collection.count()} 个向量")
print(f"存储位置: data/vectorstore")
