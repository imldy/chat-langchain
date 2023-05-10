"""Main entrypoint for the app."""
import logging
import pickle
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, Request, WebSocket, WebSocketDisconnect
from fastapi.templating import Jinja2Templates
from langchain import FAISS
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import VectorStore

from callback import QuestionGenCallbackHandler, StreamingLLMCallbackHandler
from query_data import get_chain
from query_data import get_qa_chain
from schemas import ChatResponse

app = FastAPI()
templates = Jinja2Templates(directory="templates")
vectorstore: Optional[VectorStore] = None


@app.on_event("startup")
async def startup_event():
    logging.info("loading vectorstore")
    # 获取一个问答链
    global qa_chain
    qa_chain = get_qa_chain()
    # embedding将高维的原始数据映射到低维，用于计算文本相似度
    embedding = OpenAIEmbeddings()
    # 获得一个附属资料的向量存储
    # 此对象用于存储文档和关联的嵌入，并提供通过嵌入查找相关文档的快速方法。
    global fs
    fs = FAISS.load_local("HHFS", embedding)
    global vectorstore
    vectorstore = fs


# 更新vectorstore
@app.post("/update_vectorstore")
async def update_fs_route(directory: str):
    from index import update_vectorstore
    await update_vectorstore(vectorstore=vectorstore, directory=directory)
    return {"message": "vectorstore updated successfully"}


# 关闭
@app.on_event("shutdown")
async def shutdown_event():
    fs.save_local("HHFS")


@app.get("/")
async def get(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})


@app.get("/answer")
async def get_answer(question: str):
    if '' != question:
        # 做向量搜索，查询到相关文档内容（参考资料）
        docs = fs.similarity_search(query=question)
        answer = qa_chain.run(input_documents=docs, question=question)
    else:
        answer = ""
    return {"code": 200, "msg": "OK", "answer": answer}


@app.websocket("/chat")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    # 获得一个问题生成回调处理器
    question_handler = QuestionGenCallbackHandler(websocket)
    # 获得一个流式LLM回调处理器
    stream_handler = StreamingLLMCallbackHandler(websocket)
    chat_history = []
    # 获得用于流式响应用户的QA链
    qa_chain = get_chain(vectorstore, question_handler, stream_handler)
    # 使用以下代码启用跟踪（tracing）
    # 确保 `langchain-server` 正在运行
    # qa_chain = get_chain(vectorstore, question_handler, stream_handler, tracing=True)

    while True:
        try:
            # 接收并返回客户端信息
            question = await websocket.receive_text()
            resp = ChatResponse(sender="you", message=question, type="stream")
            await websocket.send_json(resp.dict())

            # 构造返回的开始响应
            start_resp = ChatResponse(sender="bot", message="", type="start")
            await websocket.send_json(start_resp.dict())
            # 传入本次问题与历史记录，并等待流式QA链将生成的全部结果（token）返回
            result = await qa_chain.acall(
                {"question": question, "chat_history": chat_history}
            )
            # 将本次问答添加至历史记录列表
            chat_history.append((question, result["answer"]))
            # 构造返回的结束响应
            end_resp = ChatResponse(sender="bot", message="", type="end")
            await websocket.send_json(end_resp.dict())
        except WebSocketDisconnect:
            logging.info("websocket 连接断开")
            break
        except Exception as e:
            logging.error(e)
            resp = ChatResponse(
                sender="bot",
                message="抱歉，遇到了一些问题，请重试",
                type="error",
            )
            await websocket.send_json(resp.dict())


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=9000)
