from langchain_core.messages import HumanMessage
from langchain.agents import create_agent as create_langchain_agent

from app.llm import get_llm
from app.tools import get_tools

def create_agent():
    """创建 ReAct Agent：LLM 自主决定是否调用知识库搜索工具、搜什么、怎么回答

    用法：agent.invoke({"messages": [HumanMessage(content="问题")]})
    结果在 result["messages"] 里，最后一条消息即最终回答。"""
    llm = get_llm()
    tools = get_tools()

    # 系统提示词：约束 Agent 必须先检索、基于资料回答、不许编造
    system_prompt = (
        "你是企业知识库问答助手。"
        "回答公司制度、员工手册、产品资料等问题前，必须先调用 knowledge_search 工具检索资料，"
        "回答只能基于检索到的内容；检索不到相关内容就如实回答不知道，不要编造。"
    )

    # 新版 API：system_prompt 直接传字符串。
    # 注意：这里必须调"别名"——如果导入名也叫 create_agent，
    # 会被下面这个同名函数定义覆盖，变成自己调用自己（TypeError）
    agent = create_langchain_agent(model=llm, tools=tools, system_prompt=system_prompt)
    return agent

if __name__ == "__main__":
    agent = create_agent()
    result = agent.invoke({"messages": [HumanMessage(content="工作满一年有几天年假？")]})
    print(result["messages"][-1].content)
