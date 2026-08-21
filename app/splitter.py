from langchain_text_splitters import RecursiveCharacterTextSplitter

def split_documents(docs,chunk_size=500,chunk_overlap=50):
    """把 Document 列表切成小块，返回新的 Document 列表"""
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
    )
    return splitter.split_documents(docs)