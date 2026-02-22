#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Unified Agent Framework v6 — Middleware

================================================================================
Microsoft Agent Framework 1.0.0-rc1 호환 미들웨어 파이프라인

v5에는 없던 새 기능으로, 에이전트 실행 · 채팅 API 호출 · 도구 실행의
전/후에 커스텀 로직을 주입할 수 있습니다.

미들웨어 유형:
    - AgentMiddleware    : Agent.run() 전/후 처리
    - ChatMiddleware     : LLM API 호출 전/후 처리
    - FunctionMiddleware : 도구 함수 호출 전/후 처리

내장 미들웨어:
    - LoggingMiddleware  : 실행 시간 · 입출력 자동 로깅
    - RetryMiddleware    : 실패 시 지수 백오프 재시도

성능 최적화:
    - _bind_middleware() 모듈 수준 함수:
      루프 내 async 클로저 변수 캡처 버그 방지를 위해
      별도 함수로 분리하여 미들웨어와 다음 핸들러를 정확히 바인딩.

사용법:
    >>> class MyMiddleware(AgentMiddleware):
    ...     async def on_run(self, context, next_handler):
    ...         print(f"[LOG] Run started: {context.input}")
    ...         result = await next_handler(context)
    ...         print(f"[LOG] Run completed: {result.text[:50]}")
    ...         return result
================================================================================
"""

from __future__ import annotations

import logging
import time
from typing import Any, Awaitable, Callable

from .types import AgentResponse, Message

__all__ = [
    "AgentMiddleware",
    "ChatMiddleware",
    "FunctionMiddleware",
    "LoggingMiddleware",
    "RetryMiddleware",
    "MiddlewarePipeline",
]

logger = logging.getLogger("agent_framework")


# ─── Middleware Base Classes ─────────────────────────────────

class MiddlewareContext:
    """
    미들웨어 실행 컨텍스트.

    미들웨어 체인에서 공유되는 실행 컨텍스트로,
    입력 메시지 · 옵션 · 메타데이터 · 실행 시간을 포함합니다.

    Attributes:
        input_messages: 입력 메시지 리스트
        options: 채팅 옵션
        metadata: 미들웨어간 데이터 전달용
        elapsed_ms: 경과 시간 (ms)
    """

    def __init__(
        self,
        *,
        input_messages: list[Message] | None = None,
        options: dict[str, Any] | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        self.input_messages = input_messages or []
        self.options = options or {}
        self.metadata = metadata or {}
        self.start_time: float = time.time()

    @property
    def elapsed_ms(self) -> float:
        return (time.time() - self.start_time) * 1000


class AgentMiddleware:
    """
    에이전트 미들웨어 — Agent.run() 전/후 로직 주입

    agent_framework 1.0.0-rc1에서 새로 도입된 패턴입니다.

    사용법:
        >>> class MyMiddleware(AgentMiddleware):
        ...     async def on_run(self, context, next_handler):
        ...         # 전처리
        ...         result = await next_handler(context)
        ...         # 후처리
        ...         return result
    """

    async def on_run(
        self,
        context: MiddlewareContext,
        next_handler: Callable[[MiddlewareContext], Awaitable[AgentResponse]],
    ) -> AgentResponse:
        """Agent.run() 호출 시 실행되는 미들웨어 핸들러."""
        return await next_handler(context)


class ChatMiddleware:
    """
    채팅 미들웨어 — LLM API 호출 전/후 로직 주입

    사용법:
        >>> class TokenCountMiddleware(ChatMiddleware):
        ...     async def on_chat(self, context, next_handler):
        ...         result = await next_handler(context)
        ...         print(f"Tokens used: {result.usage_details}")
        ...         return result
    """

    async def on_chat(
        self,
        context: MiddlewareContext,
        next_handler: Callable[[MiddlewareContext], Awaitable[Any]],
    ) -> Any:
        """채팅 API 호출 시 실행되는 미들웨어 핸들러."""
        return await next_handler(context)


class FunctionMiddleware:
    """
    함수 미들웨어 — 도구 함수 호출 전/후 로직 주입

    사용법:
        >>> class AuditMiddleware(FunctionMiddleware):
        ...     async def on_function(self, context, next_handler):
        ...         print(f"Calling tool: {context.metadata.get('tool_name')}")
        ...         result = await next_handler(context)
        ...         print(f"Tool result: {result}")
        ...         return result
    """

    async def on_function(
        self,
        context: MiddlewareContext,
        next_handler: Callable[[MiddlewareContext], Awaitable[Any]],
    ) -> Any:
        """도구 함수 호출 시 실행되는 미들웨어 핸들러."""
        return await next_handler(context)


# ─── Built-in Middleware ─────────────────────────────────────

class LoggingMiddleware(AgentMiddleware):
    """
    로깅 미들웨어 — 에이전트 실행을 자동 로깅

    사용법:
        >>> agent = Agent(client=client, middleware=[LoggingMiddleware()])
    """

    def __init__(self, *, log_level: int = logging.INFO) -> None:
        self.log_level = log_level

    async def on_run(
        self,
        context: MiddlewareContext,
        next_handler: Callable[[MiddlewareContext], Awaitable[AgentResponse]],
    ) -> AgentResponse:
        input_text = ""
        if context.input_messages:
            input_text = context.input_messages[0].text[:100] if context.input_messages[0].text else ""

        logger.log(self.log_level, "[Agent] Run started — input: %s", input_text)

        try:
            result = await next_handler(context)
            logger.log(
                self.log_level,
                "[Agent] Run completed — output: %s... (%.1fms)",
                result.text[:100] if result.text else "",
                context.elapsed_ms,
            )
            return result
        except Exception as e:
            logger.error("[Agent] Run failed — error: %s (%.1fms)", e, context.elapsed_ms)
            raise


class RetryMiddleware(AgentMiddleware):
    """
    재시도 미들웨어 — 에이전트 실행 실패 시 자동 재시도

    사용법:
        >>> agent = Agent(client=client, middleware=[RetryMiddleware(max_retries=3)])
    """

    def __init__(self, *, max_retries: int = 3, delay_seconds: float = 1.0) -> None:
        self.max_retries = max_retries
        self.delay_seconds = delay_seconds

    async def on_run(
        self,
        context: MiddlewareContext,
        next_handler: Callable[[MiddlewareContext], Awaitable[AgentResponse]],
    ) -> AgentResponse:
        import asyncio

        last_error: Exception | None = None
        for attempt in range(self.max_retries + 1):
            try:
                return await next_handler(context)
            except Exception as e:
                last_error = e
                if attempt < self.max_retries:
                    logger.warning(
                        "[Retry] Attempt %d/%d failed: %s — retrying in %.1fs",
                        attempt + 1,
                        self.max_retries,
                        e,
                        self.delay_seconds,
                    )
                    await asyncio.sleep(self.delay_seconds * (attempt + 1))
        raise last_error  # type: ignore


# ─── Middleware Pipeline ─────────────────────────────────────

class MiddlewarePipeline:
    """
    미들웨어 파이프라인 — 미들웨어 체인 실행 관리

    미들웨어를 역순으로 체이닝하여 execute() 호출 시
    첫 미들웨어부터 순서대로 실행합니다.

    체인 구조: mw1 → mw2 → ... → final_handler

    성능 최적화: _bind_middleware() 함수로 클로저 캡처 버그 방지.
    """

    def __init__(self, middlewares: list[AgentMiddleware] | None = None) -> None:
        self.middlewares = middlewares or []

    async def execute(
        self,
        context: MiddlewareContext,
        final_handler: Callable[[MiddlewareContext], Awaitable[AgentResponse]],
    ) -> AgentResponse:
        """미들웨어 체인을 구성하고 실행."""
        handler = final_handler
        for middleware in reversed(self.middlewares):
            handler = _bind_middleware(middleware, handler)
        return await handler(context)


def _bind_middleware(
    mw: AgentMiddleware,
    next_handler: Callable[[MiddlewareContext], Awaitable[AgentResponse]],
) -> Callable[[MiddlewareContext], Awaitable[AgentResponse]]:
    """
    미들웨어와 다음 핸들러를 바인딩.

    성능 최적화: 루프 내 async 클로저에서 변수 캡처 버그를 방지하기 위해
    별도 모듈 수준 함수로 분리. reversed() 루프에서 호출 시
    mw와 next_handler가 함수 인수로 정확히 바인딩됩니다.
    """
    async def _handler(ctx: MiddlewareContext) -> AgentResponse:
        return await mw.on_run(ctx, next_handler)
    return _handler
