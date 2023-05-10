from typing import List
from langchain.document_loaders import DirectoryLoader
from langchain.embeddings import OpenAIEmbeddings
from langchain.docstore.document import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import FAISS, VectorStore


def get_doc_by_directory(directory: str) -> List[Document]:
    """
    加载目录为文档
    """
    loader = DirectoryLoader(directory)
    raw_doc = loader.load()
    return raw_doc


def split_docs(docs: List[Document], chunk_size: int = 1000, chunk_overlap: int = 200):
    """
    切分文档
    """
    text_splitter = RecursiveCharacterTextSplitter(
        # 每个文档快大小
        chunk_size=chunk_size,
        # 每两个文档重叠的快大小
        chunk_overlap=chunk_overlap,
    )
    docs_split = text_splitter.split_documents(documents=docs)
    return docs_split


async def update_vectorstore(vectorstore: VectorStore, directory: str):
    """
    更新传入的VectorStore对象：
    添加目录中的文档到
    """
    directory_doc = get_doc_by_directory(directory)
    docs_split = split_docs(directory_doc)
    vectorstore.add_documents(docs_split)


def 获得黄海集市资料的Vectorstores():
    # 加载文档
    raw_doc = get_doc_by_directory('HHData')

    doc = split_docs(docs=raw_doc)
    # 获得一个嵌入模型对象，用以将文档低维化，获得token的低维向量，用以对比相关性
    embedding = OpenAIEmbeddings()
    # FAISS是Facebook AI 相似性搜索 (Faiss)，是一个用于高效相似性搜索和密集向量聚类的库
    fs: FAISS = FAISS.from_documents(documents=doc, embedding=embedding)
    fs.save_local("HHFS")


if __name__ == '__main__':
    获得黄海集市资料的Vectorstores()
