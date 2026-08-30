# 企业知识库智能问答助手

基于 **RAG（检索增强生成）** 的企业私有知识库问答系统。支持上传 PDF / TXT / Markdown 文档，自动解析、切块、向量化后存入本地 Chroma 向量库；提问时先检索相关资料，再交给大语言模型基于资料生成答案，并附上参考来源。

## 功能特性

- 📄 **多格式文档入库**：支持 PDF、TXT、Markdown 三种格式
- ✂️ **自动切块**：递归字符切分，兼顾语义完整性
- 🔍 **语义检索**：基于向量相似度检索，带相似度阈值过滤，避免答非所问
- 🤖 **基于资料的问答**：只依据检索到的资料回答，减少模型幻觉
- 🖥️ **网页前端**：Streamlit 界面，左侧入库、右侧提问，两件事分离
- 🧾 **来源引用**：答案下方展示命中片段及出处（文件名、页码）

## 技术栈

| 组件 | 技术                 |
|------|--------------------|
| 语言 | Python             |
| 框架 | LangChain、FastAPI |
| 向量库 | ChromaDB           |
| 检索 | 混合检索（BM25 + 向量召回 + RRF 融合 + BGE-reranker 两阶段精排） |
| 智能体 | ReAct Agent（LangGraph） |
| 嵌入模型 | BAAI/bge-m3（硅基流动）  |
| 对话模型 | Qwen/Qwen3-8B（硅基流动） |
| 重排模型 | BAAI/bge-reranker-v2-m3（硅基流动） |
| 前端 | Streamlit          |
| 部署 | Docker + docker-compose |

## 项目结构

```
.
├── app/                     # 核心业务模块（11 个文件）
│   ├── __init__.py          # 包标识
│   ├── loader.py            # 文档解析（PDF/TXT/MD）
│   ├── splitter.py          # 文档切块
│   ├── embedding.py         # 文本向量化
│   ├── vectorstore.py       # 向量库入库（覆盖式，附 BM25 数据源 chunks.json）
│   ├── retriever.py         # 混合检索（BM25 + 向量 → RRF 融合 → 重排）
│   ├── reranker.py          # 重排（BGE-reranker 精排）
│   ├── llm.py               # 对话模型实例
│   ├── chain.py             # 问答链（检索 + 生成）
│   ├── tools.py             # 工具封装（知识库检索 Tool）
│   └── agent.py             # ReAct Agent（LangGraph）
├── api/
│   └── main.py              # FastAPI 接口（/ask /ingest /health）
├── frontend/
│   └── streamlit_app.py     # 网页前端
├── tests/                   # 测试脚本
├── data/
│   ├── samples/             # 示例文档
│   ├── uploads/             # 上传文件（本地生成，不入库提交）
│   └── vectorstore/         # 向量库数据（本地生成，不入库提交）
├── Dockerfile               # Docker 镜像构建
├── docker-compose.yml       # Docker 服务编排
├── .env.example             # 环境变量模板
└── requirements.txt         # 依赖清单
```

## 快速开始

### 1. 安装依赖

```powershell
# 建议先创建并激活虚拟环境
python -m venv .venv
.venv\Scripts\activate

# 安装依赖
pip install -r requirements.txt
```

### 2. 配置环境变量

复制模板并填入你的硅基流动 API Key：

```powershell
copy .env.example .env
```

编辑 `.env`：

```ini
SILICONFLOW_API_KEY=你的密钥
SILICONFLOW_BASE_URL=https://api.siliconflow.cn/v1
SILICONFLOW_EMBEDDING_MODEL=BAAI/bge-m3
SILICONFLOW_LLM_MODEL=Qwen/Qwen3-8B
SILICONFLOW_RERANK_MODEL=BAAI/bge-reranker-v2-m3
```

> ⚠️ `.env` 包含敏感密钥，已被 `.gitignore` 排除，**切勿提交到仓库**。

### 3. 启动网页前端

```powershell
.venv\Scripts\streamlit.exe run frontend\streamlit_app.py
```

浏览器打开提示的地址（默认 http://localhost:8501）即可使用。

## Docker 部署

除了本地运行，项目也支持一键容器化部署（只启动 FastAPI 接口服务，端口 8000）。

### 前置条件

- 已安装 [Docker Desktop](https://www.docker.com/products/docker-desktop/)（Windows / Mac），或 Linux 上安装好 `docker` + `docker-compose` 插件。

### 国内拉镜像慢的坑（实测会踩）

构建时需要从 Docker Hub 拉取 `python:3.11-slim` 基础镜像，国内直连经常出现 `failed to copy: httpReadSeeker: failed open...` 这类下载失败。解决办法是给 Docker Engine 配置镜像加速器：

Docker Desktop → **Settings** → **Docker Engine**，在 JSON 中加入：

```json
{
  "registry-mirrors": [
    "https://dockerproxy.net",
    "https://docker.m.daocloud.io"
  ]
}
```

点 **Apply & Restart** 后重新构建即可。Linux 用户修改 `/etc/docker/daemon.json` 后执行 `sudo systemctl restart docker`。

### 部署步骤

**1. 配置环境变量**

```powershell
# Windows（PowerShell / CMD）
copy .env.example .env
```

```bash
# Linux / Mac
cp .env.example .env
```

然后编辑 `.env`，填入真实的 `SILICONFLOW_API_KEY`（`.env` 通过 docker-compose 的 `env_file` 作为环境变量注入容器，不会被打包进镜像）。

**2. 构建并启动**

```powershell
docker compose up -d --build
```

**3. 验证服务**

```powershell
curl http://localhost:8000/health
```

返回 `{"status":"ok"}` 即部署成功。

> 💡 **Docker 部署只暴露 8000 端口**（FastAPI 接口），不包含 Streamlit 网页界面（8501）。如果想用网页前端，请走上面「快速开始」的本地方式，两者不要混用。

### 首次入库

**这一步是必须的**：向量库数据（`data/vectorstore/`）已被 `.gitignore` 排除，克隆仓库后知识库是**空的**，不先入库就无法问答。

通过 `/ingest` 接口上传文档入库（以下示例对应 `data/samples` 里的三个示例文档）：

> ⚠️ 下面的 `curl` 命令在 Linux / Mac 终端直接可用；Windows PowerShell 里的 `curl` 是 `Invoke-WebRequest` 的别名、语法不兼容，建议直接使用下方 Swagger 页面操作，或改用 Git Bash 执行。

```bash
curl -X POST http://localhost:8000/ingest -F "file=@data/samples/常见问题.txt"
curl -X POST http://localhost:8000/ingest -F "file=@data/samples/产品介绍.md"
curl -X POST http://localhost:8000/ingest -F "file=@data/samples/员工手册.pdf"
```

每个文件返回类似：

```json
{"ingested": 4, "source": "常见问题.txt"}
```

其中 `ingested` 表示该文档切分出的片段数。

入库完成后即可提问验证：

```bash
curl -X POST http://localhost:8000/ask -H "Content-Type: application/json" -d '{"question":"年假有几天？"}'
```

> 💡 不熟悉命令行的话，可以打开 **http://localhost:8000/docs** 使用 Swagger 可视化界面操作：点开对应接口 →「Try it out」→ 上传文件或填入问题 →「Execute」，对新手更友好。

## 使用方法

1. **入库**：在左侧边栏上传文档（可多选 PDF/TXT/MD），点击「📥 入库」按钮，文档会被解析、切块、向量化后写入本地 Chroma 向量库。
2. **提问**：在主区域输入问题，点击「🚀 提问」，系统检索相关资料并生成答案，下方展示参考来源。

> 入库和提问是分开的两个操作：上传文件后必须先点「入库」，提问时才能检索到这些文档。

## 运行测试

项目提供了多个测试脚本，验证各模块功能（需先完成 `.env` 配置并保证已入库）。

在 PyCharm 中可直接右键运行；命令行下需指定项目根目录：

```powershell
$env:PYTHONPATH = "项目根目录路径"
.venv\Scripts\python.exe tests\test_retriever.py
```

## 注意事项

- **向量库数据**（`data/vectorstore/`）和**上传文件**（`data/uploads/`）是本地生成的产物，已被 `.gitignore` 排除，克隆后需自行入库。
- 向量库为空时检索不会报错，但提问会提示「资料中未找到相关信息」，请先在侧边栏上传并入库。
- 如需重建向量库，删除 `data/vectorstore/` 后重新入库即可。

## License

本项目采用 [MIT License](LICENSE) 开源协议。