

from langchain_community.document_loaders import (

PyPDFLoader,
TextLoader,
)

def load_document(file_path: str):
    """根据文件后缀，加载不同格式的文档，返回 Document 列表"""
    if file_path.endswith('.pdf'):
        loader = PyPDFLoader(file_path)

    elif file_path.endswith('.txt'):
        loader = TextLoader(file_path,encoding='utf-8')

    elif file_path.endswith('.md'):
        loader = TextLoader(file_path,encoding='utf-8')

    else:
        raise ValueError(f"不支持该文件格式：{file_path}")

    return loader.load()

