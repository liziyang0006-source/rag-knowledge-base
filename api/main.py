# api/main.py
"""FastAPI 应用：企业知识库问答 API

接口：
- GET  /health  健康检查（Docker 用）
- POST /ask     提问（Agent 检索 + 生成）
- POST /ingest  上传文档入库
"""
import os
import shutil
import sys
import tempfile

sys.stdout.reconfigure(encoding="utf-8")  # Windows 下中文输出不乱码

# 把项目根目录加进 sys.path，保证能 import app 包
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from fastapi import FastAPI, File, UploadFile
from pydantic import BaseModel
from langchain_core.messages import HumanMessage, ToolMessage

from app.agent import create_agent
from app.loader import load_document
from app.splitter import split_documents
from app.vectorstore import create_vectorstore
from app.retriever import refresh_vectorstore

app = FastAPI(title="企业知识库智能问答 API")

# ---------- Agent 懒加载：第一次请求才创建，之后复用 ----------
_agent = None


def get_agent():
    """获取全局 Agent 实例（懒加载：每次请求新建会非常慢）"""
    global _agent
    if _agent is None:
        _agent = create_agent()
    return _agent


# ---------- 请求/响应模型 ----------
class AskRequest(BaseModel):
    """提问请求体"""
    question: str


class AskResponse(BaseModel):
    """提问响应体"""
    answer: str
    sources: list[str]


class IngestResponse(BaseModel):
    """入库响应体"""
    ingested: int
    source: str


# ---------- 接口 ----------
@app.get("/health")
def health():
    """健康检查（Docker 健康探测用）"""
    return {"status": "ok"}


@app.post("/ask", response_model=AskResponse)
def ask(req: AskRequest):
    """提问：Agent 检索知识库后生成答案，附来源片段"""
    agent = get_agent()
    result = agent.invoke({"messages": [HumanMessage(content=req.question)]})
    messages = result["messages"]
    # 最终答案 = 最后一条消息；来源 = 所有 ToolMessage 的内容
    return AskResponse(
        answer=messages[-1].content,
        sources=[m.content for m in messages if isinstance(m, ToolMessage)],
    )


@app.post("/ingest", response_model=IngestResponse)
def ingest(file: UploadFile = File(...)):
    """上传文档（PDF/TXT/MD）解析入库，返回入库片段数"""
    # 保存到临时文件（load_document 需要文件路径，按后缀识别格式）
    suffix = os.path.splitext(file.filename or "")[1] or ".tmp"
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        shutil.copyfileobj(file.file, tmp)
        tmp_path = tmp.name

    try:
        docs = load_document(tmp_path)      # 解析
        chunks = split_documents(docs)      # 切块
        # 临时路径会变，把来源统一改成原始上传文件名（便于展示与覆盖式去重）
        for c in chunks:
            c.metadata["source"] = file.filename
        create_vectorstore(chunks)          # 入库（向量 + BM25 数据源）
    finally:
        os.remove(tmp_path)                 # 用完即删临时文件

    refresh_vectorstore()                   # 清检索缓存，让新内容立即可查
    return IngestResponse(ingested=len(chunks), source=file.filename)


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("api.main:app", host="0.0.0.0", port=8000)