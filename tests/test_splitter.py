from app.loader import load_document
from app.splitter import split_documents

# 先读文档（loader 读成几个大块）
docs = load_document("data/samples/sample.txt")
print(f"切之前：{len(docs)} 个大块")

# 再切碎（splitter 切成很多小块）
chunks = split_documents(docs)
print(f"切之后：{len(chunks)} 个小块")
print("=" * 50)

# 打印前 3 个小块，看切出来长啥样
for i, c in enumerate(chunks[:3], start=1):
    print(f"--- 小块 {i}（{len(c.page_content)} 字）---")
    print(c.page_content)
    print()
