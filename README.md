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

| 组件 | 技术 |
|------|------|
| 语言 | Python |
| 框架 | LangChain |
| 向量库 | ChromaDB |
| 嵌入模型 | BAAI/bge-m3（硅基流动） |
| 对话模型 | Qwen/Qwen2.5-7B-Instruct（硅基流动） |
| 前端 | Streamlit |

## 项目结构

```
.
├── app/                     # 核心业务模块
│   ├── loader.py            # 文档解析（PDF/TXT/MD）
│   ├── splitter.py          # 文档切块
│   ├── embedding.py         # 文本向量化
│   ├── vectorstore.py       # 向量库入库
│   ├── retriever.py         # 语义检索
│   └── chain.py             # 问答链（检索 + 生成）
├── frontend/
│   └── streamlit_app.py     # 网页前端
├── tests/                   # 测试脚本
├── data/
│   ├── samples/             # 示例文档
│   ├── uploads/             # 上传文件（本地生成，不入库提交）
│   └── vectorstore/         # 向量库数据（本地生成，不入库提交）
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
SILICONFLOW_LLM_MODEL=Qwen/Qwen2.5-7B-Instruct
```

> ⚠️ `.env` 包含敏感密钥，已被 `.gitignore` 排除，**切勿提交到仓库**。

### 3. 启动网页前端

```powershell
.venv\Scripts\streamlit.exe run frontend\streamlit_app.py
```

浏览器打开提示的地址（默认 http://localhost:8501）即可使用。

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