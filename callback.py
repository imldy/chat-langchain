"""Callback handlers used in the app."""
from typing import Any, Dict, List

from langchain.callbacks.base import AsyncCallbackHandler

from schemas import ChatResponse


class StreamingLLMCallbackHandler(AsyncCallbackHandler):
    """用于流式LLM响应的回调处理程序。."""

    def __init__(self, websocket):
        self.websocket = websocket

    async def on_llm_new_token(self, token: str, **kwargs: Any) -> None:
        resp = ChatResponse(sender="bot", message=token, type="stream")
        await self.websocket.send_json(resp.dict())


class QuestionGenCallbackHandler(AsyncCallbackHandler):
    """用于生成问题的回调处理程序"""

    def __init__(self, websocket):
        self.websocket = websocket

    async def on_llm_start(
        self, serialized: Dict[str, Any], prompts: List[str], **kwargs: Any
    ) -> None:
        """Run when LLM starts running."""
        resp = ChatResponse(
            sender="bot", message="思考中...", type="info"
        )
        await self.websocket.send_json(resp.dict())
