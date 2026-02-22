#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Unified Agent Framework v6 — Agent & Session

================================================================================
Microsoft Agent Framework 1.0.0-rc1 호환 Agent / Session / ContextProvider / ChatClient

이 모듈은 프레임워크의 핵심으로, AI 에이전트의 생성 · 실행 · 세션 관리를
담당합니다.

핵심 클래스:
    - Agent             : AI 에이전트 (run / stream / as_tool)
    - AgentSession      : 대화 세션 (session_id + state)
    - BaseContextProvider : 컨텍스트 프로바이더 베이스 (before_run / after_run)
    - InMemoryHistoryProvider : 인메모리 히스토리 + 슬라이딩 윈도우
    - OpenAIChatClient  : OpenAI / Azure OpenAI 클라이언트 (lazy 초기화)
    - run_agent()       : v5 호환 래퍼 함수

v5 → v6 변경 요약:
    - run_agent() 함수 → Agent 클래스 + agent.run() 메서드
    - Memory 클래스 → AgentSession + ContextProvider
    - Runner 클래스 → Agent 클래스가 직접 실행
    - 엔진 선택 → ChatClient 주입 (OpenAI, Azure 등)

성능 최적화:
    - 모듈 레벨 임포트: json, os, re를 상단에서 로드 (함수 호출마다 임포트 오버헤드 제거)
    - 모듈 레벨 _env(): 클래스 내부에서 모듈 수준으로 승격
    - 로컬 복사본 사용: context_providers auto-inject 시 Agent 상태 변경 방지
    - tool_map 사전 계산: 도구 검색 시 dict lookup O(1)
    - author_name 정규화: re.sub으로 OpenAI 명명 규칙 준수
================================================================================
"""

from __future__ import annotations

import inspect
import json
import logging
import os
import re
import time
import uuid
from collections.abc import AsyncIterable, AsyncIterator, Awaitable, Callable, Sequence
from typing import Any, Literal, overload

from .types import (
    AgentResponse,
    AgentResponseUpdate,
    ChatOptions,
    Content,
    Message,
    UsageDetails,
    add_usage_details,
)
from .tools import FunctionTool, ToolTypes, normalize_tools

__all__ = [
    "Agent",
    "AgentSession",
    "BaseContextProvider",
    "InMemoryHistoryProvider",
    "SessionContext",
]

logger = logging.getLogger("agent_framework")


def _env(key: str, default: str = "") -> str:
    """
    환경변수 값 로드 (앞뒤 공백/따옴표 제거)

    성능 최적화: 클래스 내부 메서드에서 모듈 레벨 함수로 승격.
    OpenAIChatClient.__init__() 등에서 반복 호출되므로 모듈 수준이 적합.
    """
    val = os.getenv(key, default)
    return val.strip().strip('"').strip("'") if val else default


# ─── AgentSession ────────────────────────────────────────────

class AgentSession:
    """
    대화 세션 — agent_framework 1.0.0-rc1 호환

    세션 ID 기반으로 대화 상태를 관리합니다.
    state 딕셔너리에 각 ContextProvider의 상태가 저장되며,
    provider.source_id를 키로 사용합니다.

    v5 대응: Memory 클래스
    v6 변경: 세션 ID + state dict 패턴으로 provider 상태 분리.

    Attributes:
        session_id: 고유 세션 식별자 (UUID)
        service_session_id: 외부 서비스 세션 ID (선택)
        state: provider.source_id → 상태 dict 매핑

    사용법:
        >>> session = agent.create_session()
        >>> result = await agent.run("안녕하세요", session=session)
        >>> result2 = await agent.run("내 이름이 뭐였지?", session=session)
    """

    def __init__(
        self,
        *,
        session_id: str | None = None,
        service_session_id: str | None = None,
    ) -> None:
        self.session_id = session_id or str(uuid.uuid4())
        self.service_session_id = service_session_id
        self.state: dict[str, dict[str, Any]] = {}

    def __repr__(self) -> str:
        return f"AgentSession(id={self.session_id[:8]}...)"


# ─── SessionContext ──────────────────────────────────────────

class SessionContext:
    """
    세션 컨텍스트 — 단일 실행 동안의 상태 관리

    Agent.run() 호출 시 생성되며, ContextProvider들이
    before_run / after_run 훅에서 이 컨텍스트를 통해
    메시지 · 지시사항 · 도구를 동적으로 추가합니다.

    주요 메서드:
        - extend_instructions(source_id, text) : 지시사항 추가
        - extend_messages(source_id, msgs)     : 컨텍스트 메시지 추가 (히스토리 등)
        - get_messages(include_input=True)      : 최종 메시지 리스트 반환
    """

    def __init__(
        self,
        *,
        session_id: str | None = None,
        input_messages: list[Message] | None = None,
        options: dict[str, Any] | None = None,
    ) -> None:
        self.session_id = session_id
        self.input_messages = input_messages or []
        self.options = options or {}
        self._context_messages: list[Message] = []
        self._instructions: list[str] = []
        self._tools: list[Any] = []
        self._response: AgentResponse | None = None

    @property
    def response(self) -> AgentResponse | None:
        return self._response

    @property
    def tools(self) -> list[Any]:
        return self._tools

    @property
    def instructions(self) -> list[str]:
        return self._instructions

    def extend_instructions(self, source_id: str, instruction: str) -> None:
        """지시사항 추가 (Context Provider에서 호출)."""
        self._instructions.append(instruction)

    def extend_messages(self, source_id: str, messages: list[Message]) -> None:
        """컨텍스트 메시지 추가 (히스토리 등)."""
        self._context_messages.extend(messages)

    def get_messages(self, *, include_input: bool = True) -> list[Message]:
        """최종 메시지 리스트 반환."""
        result = list(self._context_messages)
        if include_input:
            result.extend(self.input_messages)
        return result


# ─── ContextProvider ─────────────────────────────────────────

class BaseContextProvider:
    """
    컨텍스트 프로바이더 베이스 — agent_framework 1.0.0-rc1 호환

    Agent.run() 실행 전/후에 동적 컨텍스트(instructions, messages, tools)를
    주입하는 훅 기반 프로바이더 패턴입니다.

    v5 대응: Memory 클래스의 기능을 추상화
    v6 변경: before_run / after_run 훅으로 동적 컨텍스트 주입

    확장 방법:
        1. DEFAULT_SOURCE_ID 설정 (상태 저장 키)
        2. before_run() 오버라이드: context.extend_instructions() 등 호출
        3. after_run() 오버라이드: 실행 후 상태 저장 등

    사용법:
        >>> class MyProvider(BaseContextProvider):
        ...     DEFAULT_SOURCE_ID = "my_provider"
        ...
        ...     async def before_run(self, *, agent, session, context, state):
        ...         context.extend_instructions(self.source_id, "Custom instruction")
        ...
        ...     async def after_run(self, *, agent, session, context, state):
        ...         pass  # 실행 후 처리
    """

    DEFAULT_SOURCE_ID: str = "base"

    def __init__(self, *, source_id: str | None = None) -> None:
        self.source_id = source_id or self.DEFAULT_SOURCE_ID

    async def before_run(
        self,
        *,
        agent: Any,
        session: AgentSession | None,
        context: SessionContext,
        state: dict[str, Any],
    ) -> None:
        """실행 전 컨텍스트 주입 (subclass에서 override)."""
        pass

    async def after_run(
        self,
        *,
        agent: Any,
        session: AgentSession | None,
        context: SessionContext,
        state: dict[str, Any],
    ) -> None:
        """실행 후 처리 (subclass에서 override)."""
        pass


class InMemoryHistoryProvider(BaseContextProvider):
    """
    인메모리 히스토리 프로바이더 — 대화 히스토리 자동 관리

    세션 내 대화 메시지를 자동으로 저장하고, 다음 실행 시
    컨텍스트로 주입합니다. 슬라이딩 윈도우로 max_messages를 초과하면
    오래된 메시지를 자동 제거합니다.

    v5 대응: Memory 클래스의 메시지 리스트 + 슬라이딩 윈도우
    v6 변경: ContextProvider 패턴으로 자동 히스토리 주입.

    Args:
        max_messages: 최대 메시지 수 (기본: 100, 초과 시 오래된 메시지 제거)
        source_id: 상태 저장 키 (기본: "in_memory_history")

    사용법:
        >>> agent = Agent(
        ...     client=client,
        ...     context_providers=[InMemoryHistoryProvider(max_messages=50)],
        ... )
        >>> session = agent.create_session()
        >>> await agent.run("안녕!", session=session)  # 히스토리 자동 관리
    """

    DEFAULT_SOURCE_ID = "in_memory_history"

    def __init__(self, *, max_messages: int = 100, source_id: str | None = None) -> None:
        super().__init__(source_id=source_id)
        self.max_messages = max_messages
        self.load_messages = True

    async def before_run(
        self,
        *,
        agent: Any,
        session: AgentSession | None,
        context: SessionContext,
        state: dict[str, Any],
    ) -> None:
        """히스토리 메시지를 컨텍스트에 주입."""
        history = state.get("messages", [])
        if history:
            # 슬라이딩 윈도우
            if len(history) > self.max_messages:
                history = history[-self.max_messages:]
                state["messages"] = history
            messages = [Message.from_dict(m) if isinstance(m, dict) else m for m in history]
            context.extend_messages(self.source_id, messages)

    async def after_run(
        self,
        *,
        agent: Any,
        session: AgentSession | None,
        context: SessionContext,
        state: dict[str, Any],
    ) -> None:
        """실행 결과를 히스토리에 저장."""
        if "messages" not in state:
            state["messages"] = []

        # 입력 메시지 저장
        for msg in context.input_messages:
            state["messages"].append(msg.to_dict())

        # 응답 메시지 저장
        if context.response:
            for msg in context.response.messages:
                state["messages"].append(msg.to_dict())

        # 슬라이딩 윈도우 적용
        if len(state["messages"]) > self.max_messages:
            state["messages"] = state["messages"][-self.max_messages:]


# ─── Chat Client Protocol ───────────────────────────────────

class BaseChatClient:
    """
    채팅 클라이언트 베이스 — agent_framework 1.0.0-rc1 호환

    LLM 프로바이더별 ChatClient를 구현하기 위한 베이스 클래스입니다.
    get_response() 메서드를 오버라이드하여 커스텀 클라이언트를 구현할 수 있습니다.

    v5 대응: engines/ (DirectEngine, LangChainEngine, CrewAIEngine)
    v6 변경: 프로바이더별 ChatClient 주입 패턴.

    내장 구현: OpenAIChatClient (OpenAI + Azure OpenAI 자동 감지)
    """

    model_id: str | None = None

    async def get_response(
        self,
        messages: list[Message],
        *,
        stream: bool = False,
        options: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> Any:
        """LLM에 메시지를 전송하고 응답을 받습니다."""
        raise NotImplementedError("Subclass must implement get_response()")


# ─── Built-in Chat Clients ──────────────────────────────────

class OpenAIChatClient(BaseChatClient):
    """
    OpenAI / Azure OpenAI ChatClient — agent_framework 1.0.0-rc1 호환

    OpenAI API 또는 Azure OpenAI를 자동 감지하여 사용하는 클라이언트입니다.
    환경변수(AZURE_OPENAI_ENDPOINT 등)이 설정되면 Azure OpenAI를,
    아니면 OpenAI를 사용합니다.

    특징:
        - Lazy 초기화: 첫 호출 시에만 openai 클라이언트 생성
        - 자동 도구 호출 루프: tool_calls 응답 시 최대 10회 자동 재호출
        - 스트리밍 지원: stream=True로 AsyncIterable 반환
        - author_name 정규화: 한글 등 비ASCII 문자 자동 변환 (OpenAI 명명 규칙)

    Args:
        model_id: 모델 ID (기본: Azure 배포명 또는 "gpt-5.2")
        api_key: API 키 (환경변수 자동 감지)
        azure_endpoint: Azure OpenAI 엔드포인트 (환경변수 자동 감지)
        default_options: 기본 채팅 옵션 (temperature 등)

    사용법:
        >>> # OpenAI 직접 사용
        >>> client = OpenAIChatClient(model_id="gpt-5.2")
        >>>
        >>> # Azure OpenAI (환경변수 자동 감지)
        >>> client = OpenAIChatClient()  # AZURE_OPENAI_* 환경변수 사용
        >>>
        >>> agent = Agent(client=client, instructions="You are helpful.")
    """

    def __init__(
        self,
        *,
        model_id: str | None = None,
        api_key: str | None = None,
        base_url: str | None = None,
        # Azure 설정
        azure_endpoint: str | None = None,
        azure_deployment: str | None = None,
        azure_api_version: str | None = None,
        # 기타
        default_options: dict[str, Any] | None = None,
    ) -> None:
        self._azure_endpoint = azure_endpoint or _env("AZURE_OPENAI_ENDPOINT") or None
        self._azure_deployment = azure_deployment or _env("AZURE_OPENAI_DEPLOYMENT") or None
        self._azure_api_version = azure_api_version or _env("AZURE_OPENAI_API_VERSION", "2025-01-01-preview")
        self._api_key = api_key or _env("AZURE_OPENAI_API_KEY") or _env("OPENAI_API_KEY") or ""
        self._base_url = base_url or _env("OPENAI_BASE_URL") or None

        # 모델 ID: Azure 배포명 우선 → OPENAI_CHAT_MODEL_ID → 기본값
        self.model_id = model_id or self._azure_deployment or _env("OPENAI_CHAT_MODEL_ID", "gpt-5.2")

        self._default_options = default_options or {}
        self._client = None

    def _get_client(self) -> Any:
        """
        OpenAI 클라이언트 lazy 초기화.

        첫 호출 시에만 AsyncAzureOpenAI 또는 AsyncOpenAI를 생성하여
        인스턴스 변수에 캐시합니다.
        """
        if self._client is None:
            import openai

            if self._azure_endpoint:
                self._client = openai.AsyncAzureOpenAI(
                    api_key=self._api_key,
                    azure_endpoint=self._azure_endpoint,
                    azure_deployment=self._azure_deployment,
                    api_version=self._azure_api_version,
                )
            else:
                kwargs: dict[str, Any] = {"api_key": self._api_key}
                if self._base_url:
                    kwargs["base_url"] = self._base_url
                self._client = openai.AsyncOpenAI(**kwargs)
        return self._client

    def _messages_to_dicts(self, messages: list[Message]) -> list[dict[str, Any]]:
        """
        Message 리스트를 OpenAI API dict 형식으로 변환.

        변환 규칙:
            - role + content 기본 변환
            - author_name: re.sub으로 OpenAI 명명 규칙 준수 (영문·숫자·하이픈만 허용)
            - function_call Content → tool_calls dict 변환
            - function_result Content → tool role 메시지 변환
        """
        result = []
        for msg in messages:
            d: dict[str, Any] = {"role": msg.role, "content": msg.text}
            if msg.author_name:
                safe_name = re.sub(r'[^\w-]', '_', msg.author_name)
                if safe_name:
                    d["name"] = safe_name
            # function_call 콘텐츠 변환
            tool_calls = [
                c for c in msg.contents
                if c.type == "function_call"
            ]
            if tool_calls:
                d["tool_calls"] = [
                    {
                        "id": tc.call_id,
                        "type": "function",
                        "function": {
                            "name": tc.name,
                            "arguments": (
                                tc.arguments if isinstance(tc.arguments, str)
                                else json.dumps(tc.arguments or {})
                            ),
                        },
                    }
                    for tc in tool_calls
                ]
            # function_result → tool role
            tool_results = [c for c in msg.contents if c.type == "function_result"]
            if tool_results:
                d["role"] = "tool"
                d["tool_call_id"] = tool_results[0].call_id
                d["content"] = str(tool_results[0].result or "")
            result.append(d)
        return result

    async def get_response(
        self,
        messages: list[Message],
        *,
        stream: bool = False,
        options: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> Any:
        """
        OpenAI API 호출.

        비스트리밍 모드 (stream=False):
            - 도구 호출 자동 실행 루프 (tool_calls → invoke → 재호출, 최대 10회)
            - tool_map dict lookup O(1)으로 도구 검색
            - 토큰 사용량 누적 합산
            - AgentResponse 반환

        스트리밍 모드 (stream=True):
            - AsyncIterable[AgentResponseUpdate] 리턴 (제너레이터)
        """
        client = self._get_client()
        opts = {**self._default_options, **(options or {})}

        model = opts.pop("model_id", None) or self.model_id
        tools_raw = opts.pop("tools", None)
        instructions = opts.pop("instructions", None)

        api_messages = self._messages_to_dicts(messages)

        # instructions → system 메시지로 삽입
        if instructions:
            api_messages.insert(0, {"role": "system", "content": instructions})

        # OpenAI 도구 스키마 변환 (한 번만 정규화)
        tools_list = normalize_tools(tools_raw) if tools_raw else []
        tool_map = {t.name: t for t in tools_list}
        api_tools = [t.to_openai_schema() for t in tools_list] or None

        # API 호출 파라미터 구성
        call_kwargs: dict[str, Any] = {
            "model": model,
            "messages": api_messages,
        }
        if api_tools:
            call_kwargs["tools"] = api_tools
            call_kwargs["tool_choice"] = opts.pop("tool_choice", "auto")

        # 선택적 파라미터
        for key in ["temperature", "max_tokens", "top_p", "seed", "stop",
                     "frequency_penalty", "presence_penalty"]:
            if key in opts:
                call_kwargs[key] = opts.pop(key)

        if not stream:
            # 비스트리밍
            response = await client.chat.completions.create(**call_kwargs)
            choice = response.choices[0]

            # 도구 호출 처리 (자동 실행 루프)
            max_rounds = kwargs.get("max_tool_rounds", 10)
            round_count = 0
            total_usage = _extract_usage(response)

            while choice.finish_reason == "tool_calls" and tools_raw and round_count < max_rounds:
                round_count += 1
                # 어시스턴트 메시지 추가
                api_messages.append(choice.message.model_dump())

                # 도구 실행
                for tc in choice.message.tool_calls:
                    fn_name = tc.function.name
                    fn_args = json.loads(tc.function.arguments) if tc.function.arguments else {}
                    tool_fn = tool_map.get(fn_name)
                    if tool_fn:
                        try:
                            result = await tool_fn.invoke(arguments=fn_args)
                            api_messages.append({
                                "role": "tool",
                                "tool_call_id": tc.id,
                                "content": str(result),
                            })
                        except Exception as e:
                            api_messages.append({
                                "role": "tool",
                                "tool_call_id": tc.id,
                                "content": f"Error: {e}",
                            })

                # 재호출
                call_kwargs["messages"] = api_messages
                response = await client.chat.completions.create(**call_kwargs)
                choice = response.choices[0]
                total_usage = add_usage_details(total_usage, _extract_usage(response))

            # 응답 변환
            resp_msg = Message(
                "assistant",
                [choice.message.content or ""],
                author_name=None,
            )

            return AgentResponse(
                messages=[resp_msg],
                response_id=response.id,
                usage_details=total_usage,
                raw_representation=response,
            )
        else:
            # 스트리밍
            call_kwargs["stream"] = True
            stream_response = await client.chat.completions.create(**call_kwargs)

            async def _stream_gen() -> AsyncIterable[AgentResponseUpdate]:
                async for chunk in stream_response:
                    if chunk.choices and chunk.choices[0].delta.content:
                        yield AgentResponseUpdate(
                            contents=[Content.from_text(chunk.choices[0].delta.content)],
                            role="assistant",
                            response_id=chunk.id,
                        )

            return _stream_gen()


def _extract_usage(response: Any) -> UsageDetails:
    """OpenAI 응답에서 사용량 추출."""
    usage = getattr(response, "usage", None)
    if usage is None:
        return UsageDetails()
    return UsageDetails(
        input_token_count=getattr(usage, "prompt_tokens", None),
        output_token_count=getattr(usage, "completion_tokens", None),
        total_token_count=getattr(usage, "total_tokens", None),
    )


# ─── Agent ───────────────────────────────────────────────────

class Agent:
    """
    AI 에이전트 — agent_framework 1.0.0-rc1 호환

    v5 대응: run_agent() 함수 + Runner 클래스
    변경: Agent 클래스 기반으로 변경. client 주입 패턴.

    사용법:
        >>> # 기본 생성
        >>> from unified_agent_v6 import Agent, OpenAIChatClient
        >>> client = OpenAIChatClient(model_id="gpt-5.2")
        >>> agent = Agent(client=client, instructions="You are helpful.")
        >>>
        >>> # 실행
        >>> response = await agent.run("안녕하세요!")
        >>> print(response.text)
        >>>
        >>> # 도구 사용
        >>> from unified_agent_v6 import tool
        >>> @tool
        ... def get_weather(city: str) -> str:
        ...     return f"{city}: 맑음 22°C"
        >>> agent = Agent(client=client, tools=[get_weather])
        >>> response = await agent.run("서울 날씨 알려줘")
        >>>
        >>> # 스트리밍
        >>> async for update in agent.run("긴 답변", stream=True):
        ...     print(update.text, end="")
        >>>
        >>> # 멀티턴 대화 (세션)
        >>> session = agent.create_session()
        >>> await agent.run("내 이름은 철수야", session=session)
        >>> response = await agent.run("내 이름이 뭐지?", session=session)
    """

    def __init__(
        self,
        client: BaseChatClient,
        instructions: str | None = None,
        *,
        id: str | None = None,
        name: str | None = None,
        description: str | None = None,
        tools: ToolTypes | Callable[..., Any] | Sequence[ToolTypes | Callable[..., Any]] | None = None,
        context_providers: Sequence[BaseContextProvider] | None = None,
        **kwargs: Any,
    ) -> None:
        self.id = id or str(uuid.uuid4())
        self.name = name
        self.description = description
        self.client = client
        self.instructions = instructions

        # 도구 정규화
        self._tools: list[FunctionTool] = normalize_tools(tools) if tools else []

        # 컨텍스트 프로바이더
        self.context_providers: list[BaseContextProvider] = list(context_providers or [])

        # 추가 옵션
        self.additional_properties = dict(kwargs)

    def create_session(self, *, session_id: str | None = None) -> AgentSession:
        """새 세션 생성."""
        return AgentSession(session_id=session_id)

    def get_session(self, *, service_session_id: str, session_id: str | None = None) -> AgentSession:
        """서비스 세션 ID로 세션 생성/조회."""
        return AgentSession(session_id=session_id, service_session_id=service_session_id)

    @overload
    def run(
        self,
        messages: str | Message | Sequence[str | Message] | None = None,
        *,
        stream: Literal[False] = ...,
        session: AgentSession | None = None,
        tools: Any = None,
        options: ChatOptions | None = None,
        **kwargs: Any,
    ) -> Awaitable[AgentResponse]: ...

    @overload
    def run(
        self,
        messages: str | Message | Sequence[str | Message] | None = None,
        *,
        stream: Literal[True],
        session: AgentSession | None = None,
        tools: Any = None,
        options: ChatOptions | None = None,
        **kwargs: Any,
    ) -> AsyncIterable[AgentResponseUpdate]: ...

    def run(
        self,
        messages: str | Message | Sequence[str | Message] | None = None,
        *,
        stream: bool = False,
        session: AgentSession | None = None,
        tools: Any = None,
        options: ChatOptions | None = None,
        **kwargs: Any,
    ) -> Awaitable[AgentResponse] | AsyncIterable[AgentResponseUpdate]:
        """
        에이전트 실행

        Args:
            messages: 입력 메시지 (문자열, Message, 또는 리스트)
            stream: 스트리밍 모드 (True: AsyncIterable, False: Awaitable)
            session: 대화 세션 (멀티턴 대화용)
            tools: 추가 도구 (기본 도구에 병합)
            options: 채팅 옵션 (ChatOptions TypedDict)

        Returns:
            stream=False: AgentResponse (await 필요)
            stream=True: AsyncIterable[AgentResponseUpdate]
        """
        if not stream:
            return self._run_non_streaming(
                messages=messages, session=session, tools=tools, options=options, **kwargs
            )
        else:
            async def _stream_wrap() -> AsyncIterator[AgentResponseUpdate]:
                gen = await self._run_streaming(
                    messages=messages, session=session, tools=tools, options=options, **kwargs
                )
                async for update in gen:
                    yield update
            return _stream_wrap()

    async def _run_non_streaming(
        self,
        messages: str | Message | Sequence[str | Message] | None = None,
        *,
        session: AgentSession | None = None,
        tools: Any = None,
        options: ChatOptions | None = None,
        **kwargs: Any,
    ) -> AgentResponse:
        """비스트리밍 실행."""
        input_messages = _normalize_messages(messages)

        # Auto-inject InMemoryHistoryProvider (local copy — agent 상태 변경 방지)
        context_providers = list(self.context_providers)
        if session and not context_providers and not session.service_session_id:
            context_providers.append(InMemoryHistoryProvider())

        active_session = session or (AgentSession() if context_providers else None)

        # Context provider pipeline
        session_context = SessionContext(
            session_id=active_session.session_id if active_session else None,
            input_messages=input_messages,
            options=dict(options or {}),
        )

        # before_run
        for provider in context_providers:
            if active_session:
                state = active_session.state.setdefault(provider.source_id, {})
                await provider.before_run(
                    agent=self, session=active_session, context=session_context, state=state
                )

        # Build chat options
        chat_options: dict[str, Any] = {}
        if self.instructions:
            chat_options["instructions"] = self.instructions
        if options:
            chat_options.update(options)

        # Merge tools
        all_tools = list(self._tools)
        if tools:
            extra = normalize_tools(tools)
            all_tools.extend(extra)
        if session_context.tools:
            all_tools.extend(session_context.tools)
        if all_tools:
            chat_options["tools"] = all_tools

        # Merge instructions from providers
        if session_context.instructions:
            existing = chat_options.get("instructions", "")
            combined = "\n".join([existing] + session_context.instructions) if existing else "\n".join(session_context.instructions)
            chat_options["instructions"] = combined

        # Build final messages
        final_messages = session_context.get_messages(include_input=True)

        # Call client
        response = await self.client.get_response(
            messages=final_messages,
            stream=False,
            options=chat_options,
            **kwargs,
        )

        # Ensure it's an AgentResponse
        if not isinstance(response, AgentResponse):
            response = AgentResponse(
                messages=[Message("assistant", [str(response)])],
            )

        # Set author names
        for msg in response.messages:
            if msg.author_name is None:
                msg.author_name = self.name or "Agent"

        # after_run
        session_context._response = response
        for provider in reversed(context_providers):
            if active_session:
                state = active_session.state.setdefault(provider.source_id, {})
                await provider.after_run(
                    agent=self, session=active_session, context=session_context, state=state
                )

        return response

    async def _run_streaming(
        self,
        messages: str | Message | Sequence[str | Message] | None = None,
        *,
        session: AgentSession | None = None,
        tools: Any = None,
        options: ChatOptions | None = None,
        **kwargs: Any,
    ) -> AsyncIterable[AgentResponseUpdate]:
        """스트리밍 실행."""
        input_messages = _normalize_messages(messages)

        chat_options: dict[str, Any] = {}
        if self.instructions:
            chat_options["instructions"] = self.instructions
        if options:
            chat_options.update(options)

        all_tools = list(self._tools)
        if tools:
            all_tools.extend(normalize_tools(tools))
        if all_tools:
            chat_options["tools"] = all_tools

        response = await self.client.get_response(
            messages=input_messages,
            stream=True,
            options=chat_options,
            **kwargs,
        )

        return response

    def as_tool(
        self,
        *,
        name: str | None = None,
        description: str | None = None,
        arg_name: str = "task",
    ) -> FunctionTool:
        """에이전트를 FunctionTool로 변환하여 다른 에이전트의 도구로 사용."""
        tool_name = name or self.name or "agent"
        tool_desc = description or self.description or ""

        async def _agent_wrapper(**tool_kwargs: Any) -> str:
            input_text = tool_kwargs.get(arg_name, "")
            result = await self.run(input_text)
            return result.text

        return FunctionTool(
            name=tool_name,
            description=tool_desc,
            func=_agent_wrapper,
            parameters={arg_name: {"type": "string", "description": f"Task for {tool_name}"}},
        )

    def __repr__(self) -> str:
        return f"Agent(name={self.name!r}, id={self.id[:8]}...)"


# ─── Utilities ───────────────────────────────────────────────

def _normalize_messages(
    messages: str | Message | Content | Sequence[str | Message | Content] | None = None,
) -> list[Message]:
    """다양한 입력 형식을 Message 리스트로 정규화."""
    if messages is None:
        return []
    if isinstance(messages, str):
        return [Message("user", [messages])]
    if isinstance(messages, Content):
        return [Message("user", [messages])]
    if isinstance(messages, Message):
        return [messages]
    result: list[Message] = []
    for msg in messages:
        if isinstance(msg, str):
            result.append(Message("user", [msg]))
        elif isinstance(msg, Content):
            result.append(Message("user", [msg]))
        elif isinstance(msg, Message):
            result.append(msg)
    return result


# ─── run_agent() 호환 함수 ───────────────────────────────────

async def run_agent(
    task: str,
    *,
    model: str | None = None,
    instructions: str = "You are a helpful assistant.",
    tools: Any = None,
    session: AgentSession | None = None,
    stream: bool = False,
    **kwargs: Any,
) -> AgentResponse:
    """
    v5 호환 run_agent() — Agent 클래스 래퍼

    간편하게 한 줄로 에이전트를 실행합니다.
    내부적으로 Agent + OpenAIChatClient를 생성하여 실행합니다.

    사용법:
        >>> result = await run_agent("안녕하세요!")
        >>> print(result.text)
    """
    client = OpenAIChatClient(model_id=model)
    agent = Agent(client=client, instructions=instructions, tools=tools)
    return await agent.run(task, session=session, **kwargs)
