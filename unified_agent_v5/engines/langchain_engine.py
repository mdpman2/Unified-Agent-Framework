#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Unified Agent Framework v5 — LangChain Engine

================================================================================
v4.1 대응: sk_agent_bridge.py + workflow.py 대체
축소 이유: v4.1은 Semantic Kernel을 통해 체인/RAG를 구성했으나,
          LangChain이 범용 체인 구성에서 사실상 표준(ecosystem 최대).
          LangChain 타입 ↔ v5 통합 타입 자동 변환으로 사용자는
          엔진 전환 시 코드 변경 없이 engine="langchain"만 지정.
================================================================================

LangChain/LangGraph 기반 범용 체인/RAG 구성 엔진.
설치: pip install langchain langchain-openai

사용법:
    >>> result = await run_agent("RAG 검색 후 요약", engine="langchain")
"""

from __future__ import annotations

import asyncio
import concurrent.futures
import json
import logging
import time
from typing import Any, AsyncIterator

from ..types import AgentResult, StreamChunk, ToolCall, ToolResult
from ..tools import Tool
from ..callback import CallbackHandler, fire_callbacks
from ..config import Settings

__all__ = ["LangChainEngine"]

logger = logging.getLogger(__name__)

# Import 검증 (설치 안 되어 있으면 이 모듈 로드 시 ImportError)
try:
    from langchain_openai import AzureChatOpenAI, ChatOpenAI
    from langchain_core.messages import (
        HumanMessage, AIMessage, SystemMessage, ToolMessage
    )
    from langchain_core.tools import StructuredTool
    HAS_LANGCHAIN = True
except ImportError:
    HAS_LANGCHAIN = False


def _to_langchain_messages(messages: list[dict[str, Any]]) -> list:
    """OpenAI 형식 메시지를 LangChain 메시지로 변환"""
    lc_messages = []
    for msg in messages:
        role = msg.get("role", "user")
        content = msg.get("content", "")
        if role == "system":
            lc_messages.append(SystemMessage(content=content))
        elif role == "user":
            lc_messages.append(HumanMessage(content=content))
        elif role == "assistant":
            # tool_calls가 있으면 LangChain 형식으로 변환하여 보존
            raw_tc = msg.get("tool_calls")
            if raw_tc:
                lc_tool_calls = []
                for tc in raw_tc:
                    fn = tc.get("function", {})
                    raw_args = fn.get("arguments", "{}")
                    if isinstance(raw_args, str):
                        try:
                            parsed_args = json.loads(raw_args)
                        except (json.JSONDecodeError, ValueError):
                            parsed_args = {}
                    else:
                        parsed_args = raw_args
                    lc_tool_calls.append({
                        "id": tc.get("id", ""),
                        "name": fn.get("name", ""),
                        "args": parsed_args,
                        "type": "tool_call",
                    })
                lc_messages.append(AIMessage(content=content, tool_calls=lc_tool_calls))
            else:
                lc_messages.append(AIMessage(content=content))
        elif role == "tool":
            lc_messages.append(ToolMessage(
                content=content,
                tool_call_id=msg.get("tool_call_id", ""),
            ))
    return lc_messages


# 모듈 레벨 공유 executor (도구별 생성 대신 재사용)
_SYNC_EXECUTOR = concurrent.futures.ThreadPoolExecutor(max_workers=4)


def _tool_to_langchain(tool: Tool) -> Any:
    """Tool을 LangChain StructuredTool로 변환

    LangChain StructuredTool은 sync func와 async coroutine을 동시에 받으며,
    async 컨텍스트에서는 coroutine을, sync 컨텍스트에서는 func를 호출한다.
    sync fallback에서는 공유 스레드풀에서 asyncio.run()을 실행하여
    이미 동작 중인 이벤트 루프와 충돌하지 않도록 한다.
    """
    async def _async_wrapper(**kwargs):
        return await tool.execute(**kwargs)

    def _sync_wrapper(**kwargs):
        """공유 스레드풀에서 async tool을 실행 (running loop 충돌 방지)"""
        future = _SYNC_EXECUTOR.submit(asyncio.run, tool.execute(**kwargs))
        return future.result()

    return StructuredTool.from_function(
        func=_sync_wrapper,
        coroutine=_async_wrapper,
        name=tool.name,
        description=tool.description,
    )


class LangChainEngine:
    """
    LangChain Engine — 범용 체인/RAG 구성

    v4.1 대응: sk_agent_bridge + workflow.py(Graph/Node) 대체
    축소 이유: Semantic Kernel 대신 LangChain 채택.
              범용 체인/RAG 구성에서 사실상 표준. 생태계 가장 넓음.
              LangChain의 ChatModel + ToolCalling을
              v5 통합 인터페이스에 맞춰 실행.
    """

    _llm_cache: dict[tuple, Any] = {}  # (model, endpoint, deployment, temp, max_tokens, api_key) -> LLM
    _LLM_CACHE_MAXSIZE: int = 64  # 캐시 최대 크기 (메모리 누수 방지)

    def __init__(self, **kwargs):
        if not HAS_LANGCHAIN:
            raise ImportError(
                "LangChain is not installed. "
                "Install with: pip install langchain langchain-openai langchain-core"
            )

    def _get_llm(self, model: str, **kwargs) -> Any:
        """LangChain LLM 인스턴스 생성 (캐싱 지원)"""
        azure_endpoint = kwargs.get("azure_endpoint", "")
        azure_api_key = kwargs.get("azure_api_key", "")

        # Reasoning 모델은 temperature 제외
        if Settings.supports_temperature(model):
            temperature = kwargs.get("temperature", 0.7)
        else:
            temperature = None  # type: ignore[assignment]

        cache_key = (
            model, azure_endpoint, kwargs.get("azure_deployment", ""),
            temperature, kwargs.get("max_tokens"),
            kwargs.get("openai_api_key", kwargs.get("api_key", "")),
        )
        cached = self._llm_cache.get(cache_key)
        if cached is not None:
            return cached

        if azure_endpoint and azure_api_key:
            llm = AzureChatOpenAI(
                azure_endpoint=azure_endpoint,
                api_key=azure_api_key,
                azure_deployment=kwargs.get("azure_deployment", model),
                api_version=kwargs.get("azure_api_version", "2025-12-01-preview"),
                temperature=temperature,
                max_tokens=kwargs.get("max_tokens"),
            )
        else:
            llm = ChatOpenAI(
                model=model,
                api_key=kwargs.get("openai_api_key", kwargs.get("api_key", "")),
                temperature=temperature,
                max_tokens=kwargs.get("max_tokens"),
            )

        # 캐시 크기 초과 시 가장 오래된 항목 제거 (LRU-like eviction)
        if len(self._llm_cache) >= self._LLM_CACHE_MAXSIZE:
            oldest_key = next(iter(self._llm_cache))
            del self._llm_cache[oldest_key]

        self._llm_cache[cache_key] = llm
        return llm

    async def run(
        self,
        messages: list[dict[str, Any]],
        model: str,
        tools: list[Tool] | None = None,
        callbacks: list[CallbackHandler] | None = None,
        **kwargs: Any,
    ) -> AgentResult:
        """LangChain으로 실행 (멀티 라운드 도구 호출 지원)"""
        start_time = time.time()
        callbacks = callbacks or []

        await fire_callbacks(callbacks, "on_llm_start", model, messages)

        llm = self._get_llm(model, **kwargs)
        lc_messages = _to_langchain_messages(messages)

        # 도구 바인딩
        tool_map: dict[str, Tool] = {}
        if tools:
            lc_tools = [_tool_to_langchain(t) for t in tools]
            llm_with_tools = llm.bind_tools(lc_tools)
            tool_map = {t.name: t for t in tools}
        else:
            llm_with_tools = llm

        # === 멀티 라운드 도구 호출 루프 (DirectEngine과 동일 동작) ===
        max_rounds = kwargs.get("max_tool_rounds", 10)
        all_tool_calls: list[ToolCall] = []
        current_msgs = list(lc_messages)
        usage: dict[str, int] = {}

        for _round in range(max_rounds):
            response = await llm_with_tools.ainvoke(current_msgs)
            content = response.content or ""

            # 토큰 사용량 추출 (마지막 응답 기준)
            usage_metadata = getattr(response, "usage_metadata", {}) or {}
            usage = {
                "input_tokens": usage_metadata.get("input_tokens", 0),
                "output_tokens": usage_metadata.get("output_tokens", 0),
                "total_tokens": usage_metadata.get("total_tokens", 0),
            }

            # 도구 호출이 없으면 최종 응답
            tool_calls_data = getattr(response, "tool_calls", []) or []
            if not tool_calls_data or not tools:
                await fire_callbacks(callbacks, "on_llm_end", content, usage)
                break

            # 도구 호출 처리
            current_msgs.append(response)

            for tc_data in tool_calls_data:
                tc_name = tc_data.get("name", "")
                tc_args = tc_data.get("args", {})
                tc_id = tc_data.get("id", "")

                tool_call = ToolCall(id=tc_id, name=tc_name, arguments=tc_args)
                all_tool_calls.append(tool_call)

                await fire_callbacks(callbacks, "on_tool_start", tool_call)

                tool = tool_map.get(tc_name)
                if tool:
                    try:
                        result_content = await tool.execute(**tc_args)
                        tool_result = ToolResult(tc_id, tc_name, result_content)
                    except Exception as e:
                        tool_result = ToolResult(tc_id, tc_name, f"Error: {e}", is_error=True)
                else:
                    tool_result = ToolResult(tc_id, tc_name, f"Tool '{tc_name}' not found", is_error=True)

                await fire_callbacks(callbacks, "on_tool_end", tool_result)

                current_msgs.append(ToolMessage(content=tool_result.content, tool_call_id=tc_id))
        else:
            # max_rounds 초과 — on_llm_end 호출 (span 균형 유지)
            content = "[Max tool rounds exceeded]"
            await fire_callbacks(callbacks, "on_llm_end", content, {})

        duration_ms = (time.time() - start_time) * 1000

        return AgentResult(
            content=content,
            tool_calls=all_tool_calls,
            usage=usage,
            model=model,
            engine="langchain",
            duration_ms=duration_ms,
        )

    async def stream(
        self,
        messages: list[dict[str, Any]],
        model: str,
        tools: list[Tool] | None = None,
        callbacks: list[CallbackHandler] | None = None,
        **kwargs: Any,
    ) -> AsyncIterator[StreamChunk]:
        """LangChain 스트리밍"""
        llm = self._get_llm(model, **kwargs)
        lc_messages = _to_langchain_messages(messages)

        async for chunk in llm.astream(lc_messages):
            if chunk.content:
                yield StreamChunk(content=chunk.content)

        yield StreamChunk(content="", is_final=True)
