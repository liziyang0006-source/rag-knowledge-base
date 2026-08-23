from langchain_core.tools import tool

from app.retriever import search


@tool
def knowledge_search(query: str) -> str:
    """搜索企业知识库。当用户询问公司制度、员工手册、产品资料、常见问题等知识库内容时使用。
    输入参数 query：要查询的问题或关键词（字符串），例如"年假有几天"。"""
    # 固定 top_k=3：给 Agent 的资料够用且不超上下文
    docs = search(query, top_k=3)
    # 没查到时给出明确提示，方便 Agent 判断继续追问还是直接回答
    if not docs:
        return "知识库中未检索到相关内容。"
    # 拼接成文本：片段之间用分隔线隔开，方便 LLM 阅读
    return "\n\n---\n\n".join(doc.page_content for doc in docs)


def get_tools():
    """返回供 Agent 使用的工具列表"""
    return [knowledge_search]


if __name__ == "__main__":
    tools = get_tools()
    print(f"工具数量: {len(tools)}")
    print(f"工具名: {tools[0].name}")
    print(f"描述: {tools[0].description}")