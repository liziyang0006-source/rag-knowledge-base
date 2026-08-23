# -*- coding: utf-8 -*-
"""Streamlit 网页前端：企业知识库智能问答助手
左侧边栏负责文档入库，主区域负责提问，两件事分开操作。"""
import os
import sys
import html

import streamlit as st

# 把项目根目录加进 sys.path，保证能 import app 包（无论从哪个目录启动）
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

from app.loader import load_document          # 文档解析（PDF/TXT/MD）
from app.splitter import split_documents      # 切块
from app.vectorstore import create_vectorstore  # 入库（写入 Chroma）
from app.retriever import refresh_vectorstore   # 清检索缓存
from app.agent import create_agent            # Agent 版问答
from langchain_core.messages import HumanMessage, ToolMessage

# 上传文件临时保存目录
UPLOAD_DIR = os.path.join(PROJECT_ROOT, "data", "uploads")

# ---------- 页面基本配置 ----------
st.set_page_config(page_title="企业知识库智能问答助手", page_icon="📚", layout="centered")

# ---------- 缓存 Agent 实例 ----------
# Streamlit 每次交互都会重跑整个脚本，不缓存的话每次提问都要重建 Agent，非常慢
@st.cache_resource
def get_agent():
    return create_agent()

# ---------- 会话状态：保存最近一次问答，避免页面重跑后丢失 ----------
if "last_query" not in st.session_state:
    st.session_state.last_query = ""
if "last_answer" not in st.session_state:
    st.session_state.last_answer = None
if "last_sources" not in st.session_state:
    st.session_state.last_sources = []

# ==================== 页头 ====================
st.title("📚 企业知识库智能问答助手")

# ==================== 侧边栏：上传 + 入库 ====================
with st.sidebar:
    st.header("📥 文档入库")

    # 多选上传：PDF / TXT / MD
    uploaded_files = st.file_uploader(
        "上传文档（可多选）",
        type=["pdf", "txt", "md"],
        accept_multiple_files=True,
        help="支持的格式：PDF、TXT、Markdown",
    )

    # 入库按钮：上传 ≠ 入库，点了按钮才真正写进向量库
    if st.button("📥 入库", type="primary", disabled=not uploaded_files):
        os.makedirs(UPLOAD_DIR, exist_ok=True)  # 确保上传目录存在
        total_chunks = 0
        for f in uploaded_files:
            try:
                # 1. 把上传的文件保存到本地（loader 需要文件路径）
                file_path = os.path.join(UPLOAD_DIR, f.name)
                with open(file_path, "wb") as out:
                    out.write(f.getbuffer())

                # 2. 解析 → 切碎 → 存入向量库
                with st.spinner(f"正在处理：{f.name} ..."):
                    docs = load_document(file_path)
                    chunks = split_documents(docs)
                    create_vectorstore(chunks)
                total_chunks += len(chunks)
                st.success(f"{f.name}：{len(docs)} 个大块 → {len(chunks)} 个小块，已入库")
            except Exception as e:
                st.error(f"{f.name} 入库失败：{e}")

        if total_chunks > 0:
            # 入库后清掉检索缓存，保证下一次提问读到最新入库的内容
            refresh_vectorstore()
            st.success(f"全部完成！本次共入库 {total_chunks} 个小块。")

    st.divider()
    st.caption("入库 = 把文档解析、切块、向量化后存进本地 Chroma 向量库。")

    # 手动刷新检索库缓存：向量库被外部脚本（如 _rebuild_vectorstore.py）重建过，
    # 且进程一直开着时，可点此按钮，让后续提问读到重建后的新数据
    if st.button("🔄 刷新检索库"):
        refresh_vectorstore()
        st.success("已清空检索缓存，下次提问将读到最新入库内容。")

# ==================== 主区域：提问 ====================
st.header("💬 提问")

query = st.text_input(
    "输入你的问题",
    placeholder="例如：什么是RAG？",
    key="query_input",
)

if st.button("🚀 提问", type="primary", disabled=not query.strip()):
    try:
        agent = get_agent()
        # 进度条：随 Agent 消息流推进（决策 → 检索 → 生成），比纯 spinner 直观
        progress = st.progress(0, text="🤔 已收到问题，开始处理...")

        messages = []
        # stream_mode="values"：每产生一条新消息就吐一次完整状态，最后一条即最新消息
        for state in agent.stream(
            {"messages": [HumanMessage(content=query.strip())]},
            stream_mode="values",
        ):
            msg = state["messages"][-1]
            messages.append(msg)
            # 按消息类型推进进度：不同类型对应 Agent 流程的不同阶段
            if isinstance(msg, HumanMessage):
                progress.progress(10, text="🤔 已收到问题...")
            elif isinstance(msg, ToolMessage):
                progress.progress(60, text="📚 已检索到资料，正在生成答案...")
            elif getattr(msg, "tool_calls", None):
                progress.progress(30, text="🔍 正在检索知识库...")
            else:
                progress.progress(90, text="✍️ 正在组织最终答案...")

        progress.progress(100, text="✅ 回答完成")
        # 最终答案 = 最后一条消息的内容
        st.session_state.last_query = query.strip()
        st.session_state.last_answer = messages[-1].content
        # 来源片段 = 所有 ToolMessage 的 content（带编号的检索文本）
        st.session_state.last_sources = [
            m.content for m in messages if isinstance(m, ToolMessage)
        ]
        progress.empty()  # 完成后移除进度条，页面干净
    except Exception as e:
        st.error(f"提问失败：{e}\n\n如果还没入库过任何文档，请先在左侧上传并入库。")

# ---------- 展示最近一次问答结果 ----------
if st.session_state.last_answer is not None:
    st.subheader("答案")
    # 大字显示答案（转义防止特殊字符破坏 HTML）
    st.markdown(
        f'<p style="font-size:22px; line-height:1.8;">{html.escape(st.session_state.last_answer)}</p>',
        unsafe_allow_html=True,
    )

    # 来源：从工具返回的字符串里解析；空/拒答时给出提示
    sources = st.session_state.last_sources
    no_result = (not sources) or any(
        ("检索结果：空" in s) or ("未找到" in s) for s in sources
    )
    if no_result:
        st.info("知识库中没有相关文档，无法回答该问题。")

    if sources and not no_result:
        st.subheader("参考来源")
        for i, s in enumerate(sources, start=1):
            with st.expander(f"来源 {i}"):
                st.write(s)
