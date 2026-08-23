# tests/test_agent.py
import sys
import time

sys.stdout.reconfigure(encoding='utf-8')

from app.agent import create_agent
from langchain_core.messages import HumanMessage

questions = [
    "工作满一年有几天年假？",
    "如何做红烧肉？",
    "公司团建经费怎么报销？",
]

t_total0 = time.time()          # ① 总计时起点（创建 Agent 前）

agent = create_agent()

t_total1 = time.time()          # ② 创建 Agent 耗时
print(f"⏱️ 创建 Agent 耗时: {t_total1 - t_total0:.1f} 秒")

for q in questions:
    print(f"\n❓ 问: {q}")
    t0 = time.time()            # ③ 单次回答计时起点
    result = agent.invoke({"messages": [HumanMessage(content=q)]})
    dt = time.time() - t0
    print(f"💬 答: {result['messages'][-1].content}")
    print(f"⏱️ 本次回答耗时: {dt:.1f} 秒")

print(f"\n⏱️ 总耗时: {time.time() - t_total1:.1f} 秒")
