from functools import lru_cache

from dotenv import load_dotenv
from langchain_chroma import Chroma
from langchain_core.documents import Document

from app.embedding import get_embeddings
from app.vectorstore import PERSIST_DIR, load_chunks_json
from app.reranker import rerank

load_dotenv()


def refresh_vectorstore():
    """清掉 get_vectorstore / get_bm25 的缓存
    向量库被重建后先调用本函数，再检索才不会读到旧集合/旧索引。"""
    get_vectorstore.cache_clear()
    get_bm25.cache_clear()


@lru_cache(maxsize=1)
def get_vectorstore():
    """打开向量库（不重建，只读取）。加缓存复用实例，检索更快"""
    return Chroma(
        persist_directory=PERSIST_DIR,
        embedding_function=get_embeddings(),
    )


@lru_cache(maxsize=1)
def get_bm25():
    """用 chunks.json 构建 BM25 索引（关键词检索）

    返回 (bm25, records)：bm25 是索引对象，records 是对应片段元数据列表。
    依赖未安装或数据源为空时返回 (None, [])，检索自动退化为纯向量。"""
    records = load_chunks_json()
    if not records:
        return None, []
    try:
        import jieba
        from rank_bm25 import BM25Okapi
    except ImportError:
        # rank-bm25 / jieba 未安装：安静降级，只用向量检索
        return None, []

    # 中文分词后建索引
    corpus = [list(jieba.cut(r["text"])) for r in records]
    return BM25Okapi(corpus), records


def _bm25_search(query: str, k: int) -> list[Document]:
    """BM25 关键词检索：返回带 id 的 Document（id 存在 metadata 供 RRF 对齐）"""
    bm25, records = get_bm25()
    if bm25 is None:
        return []
    import jieba

    scores = bm25.get_scores(list(jieba.cut(query)))
    # 按分数降序取前 k（只保留分数 > 0 的）
    order = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:k]
    docs = []
    for i in order:
        if scores[i] <= 0:
            break
        r = records[i]
        docs.append(Document(
            page_content=r["text"],
            metadata={"source": r["source"], "page": r["page"], "_chunk_id": r["id"]},
        ))
    return docs


def _rrf_merge(list_a: list, list_b: list, k: int = 60) -> list:
    """RRF（Reciprocal Rank Fusion）合并两路检索结果

    每个片段的得分 = Σ 1/(k + 排名)，k=60 是通用默认值。
    不依赖两路分数的量纲，只要排名即可，天然适合 BM25+向量混合。
    按内容去重后返回合并列表。"""
    scores = {}
    items = {}  # 内容 → Document（内容相同的只留一份）

    for docs in (list_a, list_b):
        for rank, doc in enumerate(docs, start=1):
            text = doc.page_content
            if text not in items:
                items[text] = doc
            scores[text] = scores.get(text, 0.0) + 1.0 / (k + rank)

    return sorted(items.values(), key=lambda d: scores[d.page_content], reverse=True)


def search(query: str, top_k: int = 3, threshold: float = 0.3, recall_k: int = 20):
    """混合检索：BM25 + 向量两路召回 → RRF 合并 → rerank 精排 → 过滤 → 取 top_k

    第一阶段（召回）：向量捞语义相关 + BM25 捞关键词命中（数字/型号等），互补不漏。
    第二阶段（合并）：RRF 按排名融合两路结果，去重。
    第三阶段（重排）：rerank 模型对合并候选重新打分，相关性判断更准。
    最后按 rerank 分数过滤（低于 threshold 视为不相关），返回前 top_k 个。

    注意：rerank 分数约在 0-1，越大越相关，方向与向量距离相反。
    """
    # 第一阶段：两路召回（BM25 未就绪时该路为空列表，自动退化为纯向量）
    vector_hits = get_vectorstore().similarity_search(query, k=recall_k)
    bm25_hits = _bm25_search(query, k=recall_k)

    # 第二阶段：RRF 合并去重
    candidates = _rrf_merge(vector_hits, bm25_hits)
    if not candidates:
        return []

    # 第三阶段：rerank 精排，返回 [(score, index, text)]，已按分数降序
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