from langchain_core.tools import tool

from app.retriever import search


@tool
def knowledge_search(query: str) -> str:
    """检索企业知识库。当知识库中没有相关内容时，必须如实告知用户"知识库中未找到相关信息"，禁止编造答案。"""
    docs = search(query, top_k=3)
    if not docs:
        # 关键：返回的不是"没查到"，而是"禁止回答"指令
        return "知识库检索结果：无。你必须回答：'知识库中未找到相关信息'，不得编造任何内容。"
    # 每个片段带上真实编号和出处（文件名 + 页码），LLM 只需照抄编号，没有编造空间
    parts = []
    for i, d in enumerate(docs, start=1):
        source = d.metadata.get("source", "未知来源")
        page = d.metadata.get("page", "?")
        if isinstance(page, int):
            page += 1  # PDF 页码 0 起始，显示时 +1
        parts.append(f"[来源{i}]（{source} 第{page}页）\n{d.page_content}")
    return "\n\n".join(parts)


def get_tools():
    """返回供 Agent 使用的工具列表"""
    return [knowledge_search]


if __name__ == "__main__":
    tools = get_tools()
    print(f"工具数量: {len(tools)}")
    print(f"工具名: {tools[0].name}")
    print(f"描述: {tools[0].description}")