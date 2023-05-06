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
        chunk_size=1000,
        chunk_overlap=200,
    )
    doc = text_splitter.split_documents(documents=raw_doc)
    embedding = OpenAIEmbeddings()
    fs: FAISS = FAISS.from_documents(documents=doc, embedding=embedding)
    fs.save_local("hhjsdoc")


if __name__ == '__main__':
    获得黄海集市资料的Vectorstores()
