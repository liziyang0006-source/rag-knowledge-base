import hashlib
import json
import os
from dotenv import load_dotenv
from langchain_chroma import Chroma
from app.embedding import get_embeddings

load_dotenv()

# 向量库持久化目录（Chroma 会把数据落盘到这里）
PERSIST_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data", "vectorstore")

# BM25 数据源：入库时把所有片段同步落一份 json，供关键词检索用
# 格式：[{"id": "...", "text": "...", "source": "...", "page": 0}, ...]
CHUNKS_JSON = os.path.join(PERSIST_DIR, "chunks.json")


def _make_id(doc) -> str:
    """根据片段内容生成固定 ID：内容相同 → ID 相同 → 重复入库时覆盖而不是新增"""
    return hashlib.md5(doc.page_content.encode("utf-8")).hexdigest()


def _save_chunks_json(docs, ids):
    """把片段同步写进 chunks.json（与 Chroma 保持同一批 ID，便于两路结果对齐）

    覆盖式：重入库同一来源时，先剔除该来源的旧条目再追加新条目。"""
    # 读取现有数据（文件不存在则从空开始）
    if os.path.exists(CHUNKS_JSON):
        with open(CHUNKS_JSON, "r", encoding="utf-8") as f:
            try:
                records = json.load(f)
            except json.JSONDecodeError:
                records = []
    else:
        records = []

    # 删掉本次入库来源的旧条目（和 Chroma 的删除逻辑保持一致）
    new_sources = {str(d.metadata.get("source", "")) for d in docs if d.metadata.get("source")}
    records = [r for r in records if r.get("source") not in new_sources]

    # 追加新条目
    for d, doc_id in zip(docs, ids):
        records.append({
            "id": doc_id,
            "text": d.page_content,
            "source": str(d.metadata.get("source", "")),
            "page": d.metadata.get("page"),
        })

    os.makedirs(PERSIST_DIR, exist_ok=True)
    with open(CHUNKS_JSON, "w", encoding="utf-8") as f:
        json.dump(records, f, ensure_ascii=False, indent=1)


def load_chunks_json():
    """读取全部片段（BM25 数据源）。文件不存在返回空列表。"""
    if not os.path.exists(CHUNKS_JSON):
        return []
    with open(CHUNKS_JSON, "r", encoding="utf-8") as f:
        return json.load(f)


def create_vectorstore(docs):
    """把切好的文档块存进 Chroma + 落 BM25 数据源（覆盖式入库，不产生重复片段）"""
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

    # 3. 同步落 BM25 数据源（关键词检索用）
    _save_chunks_json(docs, ids)

    return vectorstore