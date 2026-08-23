import os
from functools import lru_cache
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI

# 加载 .env 里的配置
load_dotenv()

@lru_cache(maxsize=1)
def get_llm():
    """创建对话模型实例（硅基流动 Qwen）
    加 lru_cache 复用同一个实例，避免每次调用重复初始化。"""
    return ChatOpenAI(
        model=os.getenv("SILICONFLOW_LLM_MODEL"),
        api_key=os.getenv("SILICONFLOW_API_KEY"),
        base_url=os.getenv("SILICONFLOW_BASE_URL"),
        temperature=0.1,
    )

if __name__ == "__main__":
    answer = get_llm().invoke("用一句话介绍你自己")
    print(answer.content)