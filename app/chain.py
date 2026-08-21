import os
from functools import lru_cache
from dotenv import load_dotenv
from langchain_core.messages import SystemMessage, HumanMessage
from langchain_openai import ChatOpenAI
from app.retriever import search

# 加载 .env 里的配置（API Key、模型名、Base URL 等）
load_dotenv()

# 系统提示词：约束模型只基于资料回答，防止编造（幻觉）
SYSTEM_PROMPT = """你是企业知识库问答助手。请只根据以下资料回答问题，不要编造任何内容。
如果资料中没有答案，请回答"资料中未找到相关信息"。"""


@lru_cache(maxsize=1)
def get_llm():
    """创建大模型实例（硅基流动 Qwen）
    加 lru_cache 保证整个进程只创建一次，复用同一个连接，避免每次提问都重复初始化。"""
    return ChatOpenAI(
        model=os.getenv("SILICONFLOW_LLM_MODEL"),   # 对话模型，例如 Qwen/Qwen2.5-7B-Instruct
        api_key=os.getenv("SILICONFLOW_API_KEY"),   # 硅基流动 API 密钥
        base_url=os.getenv("SILICONFLOW_BASE_URL"), # 硅基流动 OpenAI 兼容接口地址
        temperature=0.1,                            # 温度调低，让回答更稳定、少发散
    )


def build_messages(query: str, docs) -> list:
    """把检索到的文档片段拼成提示词
    系统指令（SystemMessage）与用户内容（HumanMessage）分离，职责更清晰。"""
    # 把多个检索片段用空行连成一块上下文
    context = "\n\n".join([doc.page_content for doc in docs])
    return [
        SystemMessage(content=SYSTEM_PROMPT),
        # 资料放前面，问题放后面，一起作为用户输入
        HumanMessage(content=f"资料：\n{context}\n\n问题：{query}"),
    ]


def ask(query: str, top_k: int = 3):
    """完整问答：先检索相关资料，再调用大模型生成答案
    返回 (答案文本, 命中的文档列表)，方便调用方同时展示答案和出处。"""
    # 1. 检索：从向量库里找最相关的 top_k 个片段
    docs = search(query, top_k=top_k)
    # 2. 拼提示词
    messages = build_messages(query, docs)
    # 3. 生成：拿到 LLM 实例并调用
    llm = get_llm()
    answer = llm.invoke(messages)
    # 返回答案正文 + 命中文档
    return answer.content, docs