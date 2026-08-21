import hashlib
import os
from dotenv import load_dotenv
from langchain_chroma import Chroma
from app.embedding import get_embeddings

load_dotenv()

# 向量库持久化目录（Chroma 会把数据落盘到这里）
PERSIST_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data", "vectorstore")


def _make_id(doc) -> str:
    """根据片段内容生成固定 ID：内容相同 → ID 相同 → 重复入库时覆盖而不是新增"""
    return hashlib.md5(doc.page_content.encode("utf-8")).hexdigest()


def create_vectorstore(docs):
    """把切好的文档块存进 Chroma（覆盖式入库，不会产生重复片段）"""
    embeddings = get_embeddings()

    # 打开（或创建）持久化向量库，和 retriever 用的是同一个
    vectorstore = Chroma(
        persist_directory=PERSIST_DIR,
        embedding_function=embeddings,
    )

    # 1. 先删除同一来源的旧片段：同一文件再次入库 = 覆盖更新，不会越积越多
    #    （比如文件改过后重新上传，旧版本的过时片段也会被清掉）
    sources = sorted({str(d.metadata.get("source", "")) for d in docs if d.metadata.get("source")})
    for source in sources:
        vectorstore._collection.delete(where={"source": source})

    # 2. 再插入新片段。ID 用内容哈希：不同路径传上来的相同内容也只存一份
    #    同一批里如果出现完全相同的片段，加后缀区分避免 ID 冲突
    ids = []
    seen = {}
    for d in docs:
        base = _make_id(d)
        n = seen.get(base, 0)
        ids.append(base if n == 0 else f"{base}-{n}")
        seen[base] = n + 1

    vectorstore.add_documents(documents=docs, ids=ids)
    return vectorstore