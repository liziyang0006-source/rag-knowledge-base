import os
from functools import lru_cache
from dotenv import load_dotenv
from langchain_openai import OpenAIEmbeddings

# 加载 .env 里的配置
load_dotenv()

@lru_cache(maxsize=1)
def get_embeddings():
    """创建 embedding 实例（硅基流动 BGE 模型）"""
    return OpenAIEmbeddings(
        model=os.getenv("SILICONFLOW_EMBEDDING_MODEL"),
        api_key=os.getenv("SILICONFLOW_API_KEY"),
        base_url=os.getenv("SILICONFLOW_BASE_URL"),
        check_embedding_ctx_length=False,
    )

def embed_texts(texts: list[str]):
    """把一批文字变成向量列表"""
    embeddings = get_embeddings()
    return embeddings.embed_documents(texts)