# tests/test_agent.py
import sys
sys.stdout.reconfigure(encoding='utf-8')  # 强制 UTF-8 输出

from app.agent import create_agent
from langchain_core.messages import HumanMessage

agent = create_agent()
result = agent.invoke({"messages": [HumanMessage(content="工作满一年有几天年假？")]})
print("--- 最终答案 ---")
print(result["messages"][-1].content)
