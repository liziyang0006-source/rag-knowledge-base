from dotenv import load_dotenv
from langchain_chroma import Chroma
from app.embedding import get_embeddings
from app.vectorstore import PERSIST_DIR
from functools import lru_cache

def refresh_vectorstore():
    """清掉 get_vectorstore 的缓存
    向量库被 create_vectorstore/run 脚本重建后，先调用本函数，
    再检索才不会读到重建前的旧集合/旧索引。"""
    get_vectorstore.cache_clear()


@lru_cache(maxsize=1)
def get_vectorstore():
    """打开向量库（不重建，只读取）。加缓存复用实例，检索更快"""
    return Chroma(
        persist_directory=PERSIST_DIR,
        embedding_function=get_embeddings(),
    )


def search(query: str, top_k: int = 3, threshold: float = 1.26):
    """把问题变成向量，检索最相关的 top_k 个 chunk；相似度低于阈值的丢弃

    注意：Chroma 的分数是距离，越小越相似。
    threshold=1.26 表示只保留距离 <= 1.26 的片段。
    """
    vectorstore = get_vectorstore()
    # 多捞一些再过滤，防止过滤后不够数
    scored = vectorstore.similarity_search_with_score(query, k=top_k * 3)
    # 过滤：只保留"距离 <= 阈值"的片段。距离越小代表越相关，距离过大说明基本无关，直接丢弃。
    # 因为过滤可能砍掉一部分，所以上面多捞了 top_k*3 个，避免过滤后数量不够 top_k。
    filtered = [(doc, score) for doc, score in scored if score <= threshold]
    return [doc for doc, _ in filtered[:top_k]]
