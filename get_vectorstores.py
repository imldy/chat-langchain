from langchain.document_loaders import DirectoryLoader
from langchain.embeddings import OpenAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import FAISS


def 获得黄海集市资料的Vectorstores():
    # 加载文档
    loader = DirectoryLoader('黄海集市资料')
    raw_doc = loader.load()

    # 切分文档
    text_splitter = RecursiveCharacterTextSplitter(
        # 每个文档快大小
        chunk_size=1000,
        # 每两个文档重叠的快大小
        chunk_overlap=200,
    )
    doc = text_splitter.split_documents(documents=raw_doc)
    # 获得一个嵌入模型对象，用以将文档低维化，获得token的低维向量，用以对比相关性
    embedding = OpenAIEmbeddings()
    # FAISS是Facebook AI 相似性搜索 (Faiss)，是一个用于高效相似性搜索和密集向量聚类的库
    fs: FAISS = FAISS.from_documents(documents=doc, embedding=embedding)
    fs.save_local("hhjsdoc")


if __name__ == '__main__':
    获得黄海集市资料的Vectorstores()
