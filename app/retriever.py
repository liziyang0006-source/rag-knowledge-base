from functools import lru_cache

from dotenv import load_dotenv
from langchain_chroma import Chroma

from app.embedding import get_embeddings
from app.vectorstore import PERSIST_DIR
from app.reranker import rerank

load_dotenv()


def refresh_vectorstore():
    """清掉 get_vectorstore 的缓存
    向量库被重建后先调用本函数，再检索才不会读到旧集合/旧索引。"""
    get_vectorstore.cache_clear()


@lru_cache(maxsize=1)
def get_vectorstore():
    """打开向量库（不重建，只读取）。加缓存复用实例，检索更快"""
    return Chroma(
        persist_directory=PERSIST_DIR,
        embedding_function=get_embeddings(),
    )


def search(query: str, top_k: int = 3, threshold: float = 0.3, recall_k: int = 20):
    """两阶段检索：向量召回 → rerank 精排 → 分数过滤 → 取 top_k

    第一阶段（召回）：向量检索快速捞出 recall_k 个候选，粗筛、追求不漏。
    第二阶段（重排）：rerank 模型对候选重新打分排序，相关性判断更准。
    最后按 rerank 分数过滤（低于 threshold 视为不相关），返回前 top_k 个。

    注意：rerank 分数约在 0-1，越大越相关，方向与向量距离相反。
    """
    vectorstore = get_vectorstore()

    # 第一阶段：向量召回，多捞一些候选，避免漏掉真正相关的片段
    candidates = vectorstore.similarity_search(query, k=recall_k)
    if not candidates:
        return []

    # 第二阶段：rerank 精排，返回 [(score, index, text)]，已按分数降序
    texts = [c.page_content for c in candidates]
    ranked = rerank(query, texts, top_n=min(len(texts), 50))

    # 按 rerank 分数过滤 + 取 top_k；用 index 精确映射回原 Document（保留 metadata）
    docs = []
    for score, idx, _text in ranked:
        if score < threshold:
            continue
        docs.append(candidates[idx])
        if len(docs) >= top_k:
            break

    return docs