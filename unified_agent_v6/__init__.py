#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Unified Agent Framework v6.0.0

================================================================================
Microsoft Agent Framework 1.0.0-rc1 호환 — Agent 클래스 기반 AI 에이전트 프레임워크

이 패키지는 Microsoft의 공식 agent-framework 1.0.0-rc1 API 패턴을 따르며,
v5(Runner 중심)에서 Agent 클래스 기반 설계로 전면 재설계되었습니다.

주요 구성 모듈:
    - types.py       : Content, Message, AgentResponse, UsageDetails 등 핵심 타입
    - agents.py      : Agent, AgentSession, ContextProvider, OpenAIChatClient
    - tools.py       : @tool 데코레이터, FunctionTool, 스키마 자동 생성
    - middleware.py  : AgentMiddleware, LoggingMiddleware, RetryMiddleware
    - config.py      : AgentConfig TypedDict, 환경변수 기반 설정 로더
    - observability.py: OpenTelemetry 분산 추적, Azure Monitor 연동

핵심 기능:
    1. Agent 클래스 기반 실행 — Agent.run() / Agent.run(stream=True)
    2. ChatClient 주입 패턴 — OpenAI / Azure OpenAI 프로바이더 교체
    3. @tool 데코레이터 — 함수 → FunctionTool 자동 변환 + OpenAI 스키마 생성
    4. AgentSession + ContextProvider — 멀티턴 대화 히스토리 자동 관리
    5. 멀티 에이전트 — agent.as_tool()로 오케스트레이터 패턴 구현
    6. 미들웨어 파이프라인 — 로깅 · 재시도 등 전/후 처리
    7. OpenTelemetry 관측성 — 분산 추적 + Azure Monitor

성능 최적화 (v6.0.0):
    - __slots__ 적용: Content, Message, AgentResponse, AgentResponseUpdate
    - _SERIALIZE_FIELDS 튜플: Content 직렬화 최적화
    - asyncio.to_thread(): 동기 도구 함수의 비동기 실행 (deprecated API 제거)
    - 모듈 레벨 임포트: json, os, re를 agents.py 상단에서 로드
    - 클로저 캡처 수정: middleware.py의 _bind_middleware() 함수

Quick Start:
    >>> from unified_agent_v6 import Agent, OpenAIChatClient, tool
    >>>
    >>> @tool
    ... def get_weather(city: str) -> str:
    ...     \"\"\"도시의 날씨를 반환합니다.\"\"\"
    ...     return f"{city}: 맑음 22°C"
    >>>
    >>> client = OpenAIChatClient(model_id="gpt-5.2")
    >>> agent = Agent(
    ...     client=client,
    ...     instructions="You are a helpful assistant.",
    ...     tools=[get_weather],
    ... )
    >>>
    >>> import asyncio
    >>> response = asyncio.run(agent.run("서울 날씨 알려줘"))
    >>> print(response.text)

v5 호환 (legacy):
    >>> from unified_agent_v6 import run_agent
    >>> result = asyncio.run(run_agent("안녕하세요!"))
    >>> print(result.text)

공식 레퍼런스: https://github.com/microsoft/agent-framework
================================================================================
"""

__version__ = "6.0.0"
__agent_framework_compat__ = "1.0.0-rc1"

# ─── Types ── Content, Message, AgentResponse 등 핵심 타입 ──

from .types import (
    Content,
    Message,
    AgentResponse,
    AgentResponseUpdate,
    UsageDetails,
    ChatOptions,
    ToolMode,
    # Legacy aliases
    AgentResult,
    StreamChunk,
    ToolCall,
    ToolResult,
)

# ─── Tools ── @tool 데코레이터, FunctionTool ────────────────

from .tools import (
    tool,
    FunctionTool,
    normalize_tools,
)

# ─── Agent & Session ── Agent 클래스, 세션, ChatClient ──────

from .agents import (
    Agent,
    AgentSession,
    BaseContextProvider,
    InMemoryHistoryProvider,
    SessionContext,
    BaseChatClient,
    OpenAIChatClient,
    run_agent,
)

# ─── Middleware ── 에이전트 실행 전/후 처리 파이프라인 ───────

from .middleware import (
    AgentMiddleware,
    ChatMiddleware,
    FunctionMiddleware,
    LoggingMiddleware,
    RetryMiddleware,
    MiddlewareContext,
    MiddlewarePipeline,
)

# ─── Config ── 환경변수 기반 설정 관리 ───────────────────────

from .config import (
    AgentConfig,
    load_config,
    get_env,
)

# ─── Observability ── OpenTelemetry 분산 추적 ────────────────

from .observability import (
    configure_tracing,
    get_tracer,
)

# ─── Public API ──────────────────────────────────────────────

__all__ = [
    # Version
    "__version__",
    "__agent_framework_compat__",
    # Types
    "Content",
    "Message",
    "AgentResponse",
    "AgentResponseUpdate",
    "UsageDetails",
    "ChatOptions",
    "ToolMode",
    # Legacy aliases
    "AgentResult",
    "StreamChunk",
    "ToolCall",
    "ToolResult",
    # Tools
    "tool",
    "FunctionTool",
    "normalize_tools",
    # Agent & Session
    "Agent",
    "AgentSession",
    "BaseContextProvider",
    "InMemoryHistoryProvider",
    "SessionContext",
    "BaseChatClient",
    "OpenAIChatClient",
    "run_agent",
    # Middleware
    "AgentMiddleware",
    "ChatMiddleware",
    "FunctionMiddleware",
    "LoggingMiddleware",
    "RetryMiddleware",
    "MiddlewareContext",
    "MiddlewarePipeline",
    # Config
    "AgentConfig",
    "load_config",
    "get_env",
    # Observability
    "configure_tracing",
    "get_tracer",
]
