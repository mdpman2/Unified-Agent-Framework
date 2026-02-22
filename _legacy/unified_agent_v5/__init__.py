#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Unified Agent Framework v5 — Runner-Centric Design

================================================================================
설계 철학: "가장 잘 동작하는 것 하나를 쉽고 안정적으로"
버전: 5.0.0 (2026-02-15)
Python: 3.11+
================================================================================

v4.1 대비 축소 요약:
    - 49개 모듈 → 9개 (82% ↓)  |  380+ API → ~20개 (95% ↓)
    - 16개 프레임워크 브릿지 → 3개 엔진 (Direct, LangChain, CrewAI)
    - 자체 Tracer/Dashboard/DB → OTEL 표준 어댑터 (Export only)
    - 6개 메모리 시스템 → List[Message] + 슬라이딩 윈도우
    - 필수 의존성: semantic-kernel → openai (1개)

모듈 매핑 (v4.1 → v5):
    framework.py + orchestration.py + agents.py  →  runner.py    (Runner 중심)
    models.py + interfaces.py                    →  types.py     (OpenAI 표준 통일)
    config.py (668줄)                            →  config.py    (160줄, 실무 필수만)
    memory.py + persistent_memory.py + 4개       →  memory.py    (List[Message])
    tools.py + mcp_workbench.py                  →  tools.py     (MCP 표준 일원화)
    tracer.py(851줄) + observability.py(721줄)    →  callback.py  (OTEL 어댑터)
    universal_bridge.py + 7개 브릿지              →  engines/     (Top 3 엔진)

축소 이유:
    - 49개 모듈 간 의존성 파악 불가 → 유지보수 한계
    - 380+ API 중 실무 사용은 10% 미만
    - 16개 브릿지 중 실무 사용은 1~2개
    - 자체 모니터링 스택이 프레임워크보다 무거워짐
    → "가장 잘 동작하는 것 하나를 쉽고 안정적으로" 사용하도록 재설계

핵심 원칙:
    1. Top 3 엔진 + Direct API (LangChain, CrewAI, Direct)
    2. 모니터링은 OTEL 표준 어댑터 (자체 구현 최소화)
    3. 핵심 3기능 집중: Unified I/O, Memory, Tool Use
    4. Runner 중심 설계: run_agent(task="...") 한 줄로 끝

빠른 시작:
    >>> from unified_agent_v5 import run_agent
    >>> result = await run_agent("파이썬으로 피보나치 함수 작성해줘")
    >>> print(result.content)

    >>> # 엔진 선택
    >>> result = await run_agent("시장 분석 보고서 작성", engine="crewai")

    >>> # 도구 사용
    >>> from unified_agent_v5 import Tool
    >>> search = Tool(name="web_search", fn=my_search_fn)
    >>> result = await run_agent("최신 AI 뉴스 검색", tools=[search])
"""

# === Core Types ===
from .types import (
    Message,
    Role,
    ToolCall,
    ToolResult,
    AgentResult,
    StreamChunk,
)

# === Configuration ===
from .config import AgentConfig, Settings

# === Memory ===
from .memory import Memory

# === Tools ===
from .tools import Tool, ToolRegistry, mcp_tool

# === Callback / Observability ===
from .callback import (
    CallbackHandler,
    OTelCallbackHandler,
    CompositeCallbackHandler,
    LoggingCallbackHandler,
    fire_callbacks,
)

# === Engines ===
from .engines import EngineProtocol, get_engine

# === Runner (핵심 진입점) ===
from .runner import run_agent, stream_agent, Runner

__all__ = [
    # Runner (가장 중요)
    "run_agent",
    "stream_agent",
    "Runner",
    # Types
    "Message",
    "Role",
    "ToolCall",
    "ToolResult",
    "AgentResult",
    "StreamChunk",
    # Config
    "AgentConfig",
    "Settings",
    # Memory
    "Memory",
    # Tools
    "Tool",
    "ToolRegistry",
    "mcp_tool",
    # Callback
    "CallbackHandler",
    "OTelCallbackHandler",
    "CompositeCallbackHandler",
    "LoggingCallbackHandler",
    "fire_callbacks",
    # Engines
    "EngineProtocol",
    "get_engine",
]

__version__ = "5.0.0"
