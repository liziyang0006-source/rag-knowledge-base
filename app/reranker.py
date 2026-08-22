import os
import requests
from dotenv import load_dotenv

load_dotenv()

# 硅基流动 rerank 接口与模型（模型名可通过 .env 覆盖，默认 bge-reranker）
RERANK_URL = "https://api.siliconflow.cn/v1/rerank"
RERANK_MODEL = os.getenv("SILICONFLOW_RERANK_MODEL", "BAAI/bge-reranker-v2-m3")


def rerank(query: str, documents: list[str], top_n: int = 3):
    """调用硅基流动 reranker，对候选文档重新打分排序

    返回: [(score, index, text), ...]，按分数从高到低排列。
    index 是文档在传入 documents 列表中的原始下标，调用方可用它精确映射回原对象
    （避免靠字符串相等反查，从而防止相同内容片段匹配错乱）。
    """
    headers = {
        "Authorization": f"Bearer {os.getenv('SILICONFLOW_API_KEY')}",
        "Content-Type": "application/json",
    }
    payload = {
        "model": RERANK_MODEL,
        "query": query,
        "documents": documents,
        "top_n": top_n,          # 返回前 top_n 个结果（须 <= len(documents)）
        "return_documents": True,  # 让响应里带上文档文本，方便核对
    }

    resp = requests.post(RERANK_URL, json=payload, headers=headers)
    resp.raise_for_status()
    data = resp.json()

    results = []
    for r in data["results"]:
        # relevance_score 越大越相关；index 对应输入 documents 的原始位置
        results.append((r["relevance_score"], r["index"], r["document"]["text"]))

    return results