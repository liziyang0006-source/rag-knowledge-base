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
from app.chain import ask                     # 完整问答（检索 + 生成）

# 上传文件临时保存目录
UPLOAD_DIR = os.path.join(PROJECT_ROOT, "data", "uploads")

# ---------- 页面基本配置 ----------
st.set_page_config(page_title="企业知识库智能问答助手", page_icon="📚", layout="centered")

# ---------- 会话状态：保存最近一次问答，避免页面重跑后丢失 ----------
if "last_query" not in st.session_state:
    st.session_state.last_query = ""
if "last_answer" not in st.session_state:
    st.session_state.last_answer = None
if "last_docs" not in st.session_state:
    st.session_state.last_docs = []

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
        with st.spinner("正在检索资料并生成答案..."):
            # 完整问答：内部会先检索 top_k 个片段，再调用大模型
            answer, docs = ask(query.strip(), top_k=3)
        # 存进会话状态，页面上其他操作（如入库）触发重跑时答案不丢
        st.session_state.last_query = query.strip()
        st.session_state.last_answer = answer
        st.session_state.last_docs = docs
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

    if not st.session_state.last_docs:
        st.info("没有检索到任何相关片段，可能是向量库还是空的。")

    # 来源：每个片段一个可展开的折叠块
    if st.session_state.last_docs:
        st.subheader("参考来源")
        for i, doc in enumerate(st.session_state.last_docs, start=1):
            source = doc.metadata.get("source", "未知来源")
            page = doc.metadata.get("page", "")
            title = f"来源 {i}：{os.path.basename(str(source))}"
            if page != "":
                title += f"（第 {page + 1 if isinstance(page, int) else page} 页）"
            with st.expander(title):
                st.write(doc.page_content)
