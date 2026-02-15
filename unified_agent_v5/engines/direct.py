#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Unified Agent Framework v5 — Direct Engine

================================================================================
v4.1 대응: openai_agents_bridge.py + open_weight.py 대체
축소 이유: v4.1의 OpenAI 브릿지는 Semantic Kernel 레이어를 거쳐
          OpenAI를 호출했으나, openai 패키지 직접 호출이 더 가볍고 빠름.
          Azure OpenAI / OpenAI 자동 감지, Tool Calling 루프 내장,
          LiteLLM 스타일의 순수 API 호출.
================================================================================

프레임워크 없이 OpenAI API를 직접 호출하는 가장 가벼운 엔진.

지원:
    - OpenAI API (직접)
    - Azure OpenAI (자동 감지)
    - Tool calling (자동 루프)
    - Streaming
    - 모든 OpenAI 호환 API (Ollama, vLLM 등)

사용법:
    >>> engine = DirectEngine()
    >>> result = await engine.run(
    ...     messages=[{"role": "user", "content": "Hello"}],
    ...     model="gpt-5.2",
    ...     api_key="sk-..."
    ... )
"""

from __future__ import annotations

import json
import logging
import os
import time
from typing import Any, AsyncIterator

from ..types import AgentResult, Message, Role, StreamChunk, ToolCall, ToolResult
from ..tools import Tool
from ..callback import CallbackHandler, fire_callbacks
from ..config import Settings

__all__ = ["DirectEngine"]

logger = logging.getLogger(__name__)


class DirectEngine:
    """
    Direct API Engine — OpenAI / Azure OpenAI 직접 호출

    v4.1 대응: openai_agents_bridge.py + open_weight.py 대체
    축소 이유: Semantic Kernel 레이어 제거. openai 패키지 직접 호출이
              가장 가볍고, 의존성이 없으며, 디버깅이 쉽습니다.
              "프레임워크가 필요없을 때"를 위한 엔진.
    """

    def __init__(self, **kwargs):
        self._client = None
        self._client_key: tuple | None = None

    def _get_client(self, **kwargs) -> Any:
        """OpenAI 클라이언트 생성 (동일 파라미터면 캐시 재사용)"""
        import openai

        azure_endpoint = kwargs.get("azure_endpoint", "")
        azure_api_key = kwargs.get("azure_api_key", "")
        api_key = kwargs.get("api_key", "") or kwargs.get("openai_api_key", "")
        base_url = kwargs.get("base_url", "") or kwargs.get("openai_base_url", "")
        api_version = kwargs.get("azure_api_version", "2025-12-01-preview")

        # 캐시 키: 연결 관련 파라미터로 구성
        cache_key = (azure_endpoint, azure_api_key, api_key, base_url, api_version)
        if self._client is not None and self._client_key == cache_key:
            return self._client

        if azure_endpoint and azure_api_key:
            client = openai.AsyncAzureOpenAI(
                azure_endpoint=azure_endpoint,
                api_key=azure_api_key,
                api_version=api_version,
            )
        elif api_key:
            client = openai.AsyncOpenAI(
                api_key=api_key,
                base_url=base_url or None,
            )
        elif os.getenv("AZURE_OPENAI_ENDPOINT"):
            # 환경변수에서 자동 로드
            client = openai.AsyncAzureOpenAI()
        else:
            client = openai.AsyncOpenAI()

        self._client = client
        self._client_key = cache_key
        return client

    async def run(
        self,
        messages: list[dict[str, Any]],
        model: str,
        tools: list[Tool] | None = None,
        callbacks: list[CallbackHandler] | None = None,
        **kwargs: Any,
    ) -> AgentResult:
        """
        LLM 호출 실행 (도구 자동 루프 포함)

        도구가 있으면 LLM의 tool_calls 응답을 처리하고,
        도구 결과를 다시 LLM에 보내는 루프를 자동으로 실행합니다.
        """
        start_time = time.time()
        client = self._get_client(**kwargs)
        callbacks = callbacks or []

        # 콜백: LLM 시작
        await fire_callbacks(callbacks, "on_llm_start", model, messages)

        # OpenAI API 파라미터 구성
        api_params: dict[str, Any] = {
            "model": kwargs.get("azure_deployment") or model,
        }

        # Reasoning 모델은 temperature 제외
        if Settings.supports_temperature(model):
            temperature = kwargs.get("temperature")
            if temperature is not None:
                api_params["temperature"] = temperature

        max_tokens = kwargs.get("max_tokens")
        if max_tokens:
            api_params["max_tokens"] = max_tokens

        # 도구 스키마 추가
        tool_map: dict[str, Tool] = {}
        if tools:
            api_params["tools"] = [t.to_openai_schema() for t in tools]
            tool_map = {t.name: t for t in tools}

        # === Tool Calling Loop ===
        max_rounds = kwargs.get("max_tool_rounds", 10)
        all_tool_calls: list[ToolCall] = []
        current_messages = list(messages)

        for _round in range(max_rounds):
            api_params["messages"] = current_messages
            response = await client.chat.completions.create(**api_params)
            choice = response.choices[0]

            # 도구 호출이 없으면 최종 응답
            if not choice.message.tool_calls:
                content = choice.message.content or ""
                usage = {
                    "input_tokens": getattr(response.usage, "prompt_tokens", 0),
                    "output_tokens": getattr(response.usage, "completion_tokens", 0),
                    "total_tokens": getattr(response.usage, "total_tokens", 0),
                }

                # 콜백: LLM 완료
                await fire_callbacks(callbacks, "on_llm_end", content, usage)

                duration_ms = (time.time() - start_time) * 1000
                # current_messages에서 전체 대화 히스토리 복원 (도구 상호작용 포함)
                result_messages = []
                for m in current_messages:
                    tc_list = None
                    if m.get("tool_calls"):
                        tc_list = []
                        for tc in m["tool_calls"]:
                            raw_args = tc["function"]["arguments"]
                            if isinstance(raw_args, str):
                                try:
                                    parsed = json.loads(raw_args)
                                except (json.JSONDecodeError, ValueError):
                                    parsed = {}
                            else:
                                parsed = raw_args
                            tc_list.append(ToolCall(
                                id=tc["id"],
                                name=tc["function"]["name"],
                                arguments=parsed,
                            ))
                    result_messages.append(Message(
                        role=Role(m["role"]),
                        content=m.get("content", ""),
                        name=m.get("name"),
                        tool_call_id=m.get("tool_call_id"),
                        tool_calls=tc_list,
                    ))

                return AgentResult(
                    content=content,
                    messages=result_messages,
                    tool_calls=all_tool_calls,
                    usage=usage,
                    model=model,
                    engine="direct",
                    duration_ms=duration_ms,
                )

            # 도구 호출 처리
            assistant_msg = {
                "role": "assistant",
                "content": choice.message.content or "",
                "tool_calls": [
                    {
                        "id": tc.id,
                        "type": "function",
                        "function": {
                            "name": tc.function.name,
                            "arguments": tc.function.arguments,
                        },
                    }
                    for tc in choice.message.tool_calls
                ],
            }
            current_messages.append(assistant_msg)

            for tc in choice.message.tool_calls:
                # JSON 파싱 오류 방어
                try:
                    arguments = json.loads(tc.function.arguments) if tc.function.arguments else {}
                except json.JSONDecodeError:
                    logger.warning(
                        f"Failed to parse tool arguments for {tc.function.name}: "
                        f"{tc.function.arguments}"
                    )
                    arguments = {}

                tool_call = ToolCall(
                    id=tc.id,
                    name=tc.function.name,
                    arguments=arguments,
                )
                all_tool_calls.append(tool_call)

                # 콜백: 도구 시작
                await fire_callbacks(callbacks, "on_tool_start", tool_call)

                # 도구 실행
                tool = tool_map.get(tc.function.name)
                if tool:
                    try:
                        result_content = await tool.execute(**tool_call.arguments)
                        tool_result = ToolResult(
                            tool_call_id=tc.id,
                            name=tc.function.name,
                            content=result_content,
                        )
                    except Exception as e:
                        tool_result = ToolResult(
                            tool_call_id=tc.id,
                            name=tc.function.name,
                            content=f"Error: {e}",
                            is_error=True,
                        )
                else:
                    tool_result = ToolResult(
                        tool_call_id=tc.id,
                        name=tc.function.name,
                        content=f"Error: Tool '{tc.function.name}' not found",
                        is_error=True,
                    )

                # 콜백: 도구 완료
                await fire_callbacks(callbacks, "on_tool_end", tool_result)

                current_messages.append({
                    "role": "tool",
                    "tool_call_id": tc.id,
                    "content": tool_result.content,
                })

        # max_rounds 초과 시 — on_llm_end 호출 (span 균형 유지)
        await fire_callbacks(callbacks, "on_llm_end", "[Max tool rounds exceeded]", {})
        return AgentResult(
            content="[Max tool rounds exceeded]",
            tool_calls=all_tool_calls,
            model=model,
            engine="direct",
            duration_ms=(time.time() - start_time) * 1000,
        )

    async def stream(
        self,
        messages: list[dict[str, Any]],
        model: str,
        tools: list[Tool] | None = None,
        callbacks: list[CallbackHandler] | None = None,
        **kwargs: Any,
    ) -> AsyncIterator[StreamChunk]:
        """스트리밍 실행"""
        client = self._get_client(**kwargs)

        api_params: dict[str, Any] = {
            "model": kwargs.get("azure_deployment") or model,
            "messages": messages,
            "stream": True,
        }

        if Settings.supports_temperature(model):
            temperature = kwargs.get("temperature")
            if temperature is not None:
                api_params["temperature"] = temperature

        max_tokens = kwargs.get("max_tokens")
        if max_tokens:
            api_params["max_tokens"] = max_tokens

        if tools:
            api_params["tools"] = [t.to_openai_schema() for t in tools]

        response = await client.chat.completions.create(**api_params)

        async for chunk in response:
            if not chunk.choices:
                continue
            delta = chunk.choices[0].delta
            # 텍스트 콘텐츠 스트리밍
            if delta.content:
                yield StreamChunk(content=delta.content)
            # tool_call 델타 스트리밍 (function name / arguments)
            if delta.tool_calls:
                for tc_delta in delta.tool_calls:
                    fn = getattr(tc_delta, "function", None)
                    if fn and (fn.name or fn.arguments):
                        yield StreamChunk(
                            content="",
                            metadata={
                                "type": "tool_call_delta",
                                "index": tc_delta.index,
                                "tool_call_id": getattr(tc_delta, "id", None),
                                "function_name": fn.name,
                                "arguments_delta": fn.arguments,
                            },
                        )

        yield StreamChunk(content="", is_final=True)
