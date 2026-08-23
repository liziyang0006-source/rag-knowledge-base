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

    # 系统提示词：四条规则约束 Agent 的检索与回答行为
    system_prompt = (
        "你是企业知识库问答助手。"
        "规则1：回答任何问题前，必须调用 knowledge_search 工具检索资料，即使你认为自己知道答案。"
        "规则2：回答只能使用检索到的片段中的信息，禁止补充文档中没有的内容。"
        "规则3：如果检索结果明确表示'无'，必须回答'知识库中未找到相关信息'，禁止编造。"
        "规则4：引用检索结果时，必须使用工具返回的[来源N]编号，编号必须与工具给出的编号一致，禁止编造不存在的来源编号。"
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
