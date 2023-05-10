"""Create a ChatVectorDBChain for question/answering."""
from langchain.callbacks.manager import AsyncCallbackManager
from langchain.callbacks.tracers import LangChainTracer
from langchain.chains import ConversationalRetrievalChain
from langchain.chains.llm import LLMChain
from langchain.chains.question_answering import load_qa_chain
from langchain.chat_models import ChatOpenAI
from langchain.vectorstores.base import VectorStore
from prompts import HHJS_QA_PROMPT, QA_PROMPT_TEMPLATE, CONDENSE_QUESTION_PROMPT_TEMPLATE

def get_chain(
    vectorstore: VectorStore, question_handler, stream_handler, tracing: bool = False
) -> ConversationalRetrievalChain:
    """为问/回创建一个 ConversationalRetrievalChain"""
    # 使用流式llm结合文档构建ConversationalRetrievalChain
    # 并使用单独的非流式llm生成问题
    # 获取异步回调管理器
    manager = AsyncCallbackManager([])
    # 传入问题生成回调处理器获取问题生成异步回调管理器
    question_manager = AsyncCallbackManager([question_handler])
    # 传入流式LLM回调处理器获取回答的异步流式LLM回调管理器
    stream_manager = AsyncCallbackManager([stream_handler])
    # 如果开启了探测（tracing）
    if tracing:
        tracer = LangChainTracer()
        tracer.load_default_session()
        manager.add_handler(tracer)
        question_manager.add_handler(tracer)
        stream_manager.add_handler(tracer)
    # 创建一个用于问题生成的LLM
    question_gen_llm = ChatOpenAI(
        temperature=0,
        verbose=True,
        callback_manager=question_manager,
    )
    # 创建一个用于回答生成的流式LLM
    streaming_llm = ChatOpenAI(
        streaming=True,
        # 每生成一个token都会使用回调管理器调用回调处理器的方法
        callback_manager=stream_manager,
        verbose=True,
        temperature=0,
    )

    # 创建一个生成问题的链（使用LLM与构造问题的prompt模板）作为问题生成器
    question_generator = LLMChain(
        llm=question_gen_llm, prompt=CONDENSE_QUESTION_PROMPT_TEMPLATE, callback_manager=manager
    )
    # 创建一个使用文档生成回答的链（使用流式LLM与问答prompt模板）
    doc_chain = load_qa_chain(
        streaming_llm, chain_type="stuff", prompt=QA_PROMPT_TEMPLATE, callback_manager=manager
    )

    qa = ConversationalRetrievalChain(
        retriever=vectorstore.as_retriever(),
        # 需要使用可以使用文档与prompt模板生成回答的链
        combine_docs_chain=doc_chain,
        # 需要一个问题生成器
        question_generator=question_generator,
        # 回调
        callback_manager=manager,
    )

    return qa


def get_qa_chain():
    llm = ChatOpenAI()
    doc_chain = load_qa_chain(
        llm=llm, chain_type="stuff", prompt=HHJS_QA_PROMPT
    )
    return doc_chain
