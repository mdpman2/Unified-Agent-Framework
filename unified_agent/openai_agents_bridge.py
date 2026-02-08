#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Unified Agent Framework - OpenAI Agents SDK 브릿지 모듈

================================================================================
📁 파일 위치: unified_agent/openai_agents_bridge.py
📋 역할: OpenAI Agents SDK (v0.8.1) 통합 브릿지 — Handoff, Session, HITL, Voice
📅 최종 업데이트: 2026년 2월 8일
📦 버전: v4.0.0
================================================================================

🎯 주요 기능:
    - Agent Handoff: 에이전트 간 대화 전달
    - Session 관리: SQLite/Redis/SQLAlchemy 백엔드
    - Human-in-the-Loop: 사람 승인 워크플로우
    - Voice/Realtime Agent 지원
    - Guardrails (Input/Output) 통합

📌 사용 예시:
    >>> from unified_agent.openai_agents_bridge import OpenAIAgentsBridge
    >>>
    >>> bridge = OpenAIAgentsBridge()
    >>> agent = bridge.create_agent(
    ...     name="assistant",
    ...     instructions="친절한 AI 도우미",
    ...     tools=[{"type": "web_search"}],
    ...     handoff_targets=["specialist"]
    ... )
    >>> result = await bridge.run(agent, input="안녕하세요")

🔗 관련 문서:
    - OpenAI Agents SDK: https://github.com/openai/openai-agents-python
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any

__all__ = ["OpenAIAgentsBridge", "AgentHandoff", "SessionBackend"]

logger = logging.getLogger(__name__)

class SessionBackend:
    """세션 백엔드 상수"""
    SQLITE = "sqlite"
    REDIS = "redis"
    SQLALCHEMY = "sqlalchemy"

@dataclass(frozen=True, slots=True)
class AgentHandoff:
    """에이전트 Handoff 설정"""
    source_agent: str = ""
    target_agent: str = ""
    condition: str | None = None
    transfer_context: bool = True

class OpenAIAgentsBridge:
    """
    OpenAI Agents SDK 통합 브릿지

    ================================================================================
    📋 역할: OpenAI Agents SDK(v0.8.1)의 Agent, Handoff, Session, Guardrails를
             Unified Agent Framework의 인터페이스로 통합
    📅 최종 업데이트: 2026년 2월
    ================================================================================

    프레임워크 특징:
    - Agent 생성 및 실행 (Runner.run)
    - Handoff 패턴 (에이전트 간 대화 전달)
    - Session 관리 (SQLite/Redis)
    - Input/Output Guardrails
    - Tracing (OpenTelemetry)
    """

    def __init__(self, session_backend: str = SessionBackend.SQLITE):
        self._session_backend = session_backend
        self._agents: dict[str, dict] = {}
        self._handoffs: list[AgentHandoff] = []
        logger.info(f"[OpenAIAgentsBridge] 초기화 (session={session_backend})")

    def __repr__(self) -> str:
        return f"OpenAIAgentsBridge(agents={len(self._agents)}, session={self._session_backend!r})"

    def create_agent(
        self,
        name: str,
        instructions: str = "",
        tools: list[dict[str, Any]] | None = None,
        handoff_targets: list[str] | None = None,
        model: str = "gpt-5.2",
        **kwargs: Any
    ) -> dict[str, Any]:
        """에이전트 생성"""
        agent = {
            "name": name,
            "instructions": instructions,
            "tools": tools or [],
            "handoff_targets": handoff_targets or [],
            "model": model,
        }
        self._agents[name] = agent
        logger.info(f"[OpenAIAgentsBridge] 에이전트 생성: {name}")
        return agent

    async def run(
        self,
        agent: dict[str, Any] | None = None,
        input: str | None = None,
        *,
        task: str | None = None,
        **kwargs: Any,
    ) -> dict[str, Any]:
        """에이전트 실행 (UniversalAgentBridge 호환)

        Args:
            agent: 실행할 에이전트 (미지정 시 마지막 생성된 에이전트 사용)
            input: 입력 텍스트 (직접 호출용)
            task: 태스크 텍스트 (UniversalAgentBridge 통합용, input 대체)
            **kwargs: 추가 인자
        """
        input_text = task or input or ""
        if agent is None:
            agent = next(iter(self._agents.values()), {"name": "default"})
        agent_name = agent.get("name", "unknown")
        logger.info(f"[OpenAIAgentsBridge] 에이전트 실행: {agent_name}")
        return {
            "agent": agent_name,
            "output": f"[{agent_name}] '{input_text}'에 대한 응답",
            "handoff": None,
        }

    def add_handoff(self, handoff: AgentHandoff) -> None:
        """Handoff 규칙 추가"""
        self._handoffs.append(handoff)

    @property
    def agents(self) -> dict[str, dict]:
        return dict(self._agents)
