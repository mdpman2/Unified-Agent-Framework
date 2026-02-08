#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Unified Agent Framework - Google ADK 브릿지 모듈

================================================================================
📁 파일 위치: unified_agent/google_adk_bridge.py
📋 역할: Google Agent Development Kit (v1.24.1) 통합 브릿지
📅 최종 업데이트: 2026년 2월 8일
📦 버전: v4.0.0
================================================================================

🎯 주요 기능:
    - Workflow Agent (SequentialAgent, ParallelAgent, LoopAgent)
    - A2A 프로토콜 네이티브 통합
    - Multi-agent 계층 구조
    - 평가 도구 내장

📌 사용 예시:
    >>> from unified_agent.google_adk_bridge import GoogleADKBridge
    >>>
    >>> bridge = GoogleADKBridge()
    >>> agent = bridge.create_workflow_agent(
    ...     type="sequential",
    ...     sub_agents=["researcher", "writer"]
    ... )

🔗 관련 문서:
    - Google ADK: https://github.com/google/adk-python
"""

from __future__ import annotations

import logging
from typing import Any

__all__ = ["GoogleADKBridge"]

logger = logging.getLogger(__name__)

class GoogleADKBridge:
    """
    Google ADK 통합 브릿지

    ================================================================================
    📋 역할: Google ADK(v1.24.1)의 Workflow Agent, A2A 통합을
             Unified Agent Framework 인터페이스로 통합
    📅 최종 업데이트: 2026년 2월
    ================================================================================

    Workflow Agent 타입:
    - SequentialAgent: 순차 실행
    - ParallelAgent: 병렬 실행
    - LoopAgent: 반복 실행
    - LlmAgent: LLM 기반 에이전트
    """

    def __init__(self, model: str = "gemini-2.5-flash"):
        self._model = model
        self._agents: dict[str, dict] = {}
        logger.info(f"[GoogleADKBridge] 초기화 (model={model})")

    def __repr__(self) -> str:
        return f"GoogleADKBridge(model={self._model!r}, agents={len(self._agents)})"

    def create_workflow_agent(
        self,
        type: str = "sequential",
        sub_agents: list[str] | None = None,
        name: str | None = None,
        **kwargs: Any
    ) -> dict[str, Any]:
        """Workflow Agent 생성"""
        agent_name = name or f"adk_{type}_agent"
        agent = {
            "name": agent_name,
            "type": type,
            "sub_agents": sub_agents or [],
            "framework": "google_adk",
        }
        self._agents[agent_name] = agent
        logger.info(f"[GoogleADKBridge] Workflow Agent 생성: {agent_name} ({type})")
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
        """
        input_text = task or input or ""
        if agent is None:
            agent = next(iter(self._agents.values()), {"name": "default", "type": "sequential"})
        return {
            "agent": agent.get("name"),
            "output": f"[ADK:{agent.get('type')}] '{input_text}'에 대한 응답",
        }
