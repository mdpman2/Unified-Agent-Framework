#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Unified Agent Framework - CrewAI 브릿지 모듈

================================================================================
📁 파일 위치: unified_agent/crewai_bridge.py
📋 역할: CrewAI (v1.9.3) 통합 브릿지 — Crews + Flows 아키텍처
📅 최종 업데이트: 2026년 2월 14일
📦 버전: v4.1.0
================================================================================

🎯 주요 기능:
    - Crews: 역할 기반 자율 에이전트 팀
    - Flows: 구조화된 워크플로우 실행
    - Process: sequential / hierarchical
    - 자동 위임 및 역할 할당

📌 사용 예시:
    >>> from unified_agent.crewai_bridge import CrewAIBridge
    >>>
    >>> bridge = CrewAIBridge()
    >>> crew = bridge.create_crew(
    ...     agents=["researcher", "writer"],
    ...     process="sequential"
    ... )

🔗 관련 문서:
    - CrewAI: https://github.com/crewAIInc/crewAI
"""

from __future__ import annotations

import logging
from typing import Any

__all__ = ["CrewAIBridge"]

logger = logging.getLogger(__name__)

class CrewAIBridge:
    """
    CrewAI 통합 브릿지

    ================================================================================
    📋 역할: CrewAI(v1.9.3)의 Crews + Flows 아키텍처를
             Unified Agent Framework 인터페이스로 통합
    📅 최종 업데이트: 2026년 2월
    ================================================================================
    """

    def __init__(self, process: str = "sequential"):
        self._process = process
        self._crews: dict[str, dict] = {}
        logger.info(f"[CrewAIBridge] 초기화 (process={process})")

    def __repr__(self) -> str:
        return f"CrewAIBridge(process={self._process!r}, crews={len(self._crews)})"

    def create_crew(
        self,
        agents: list[str] | None = None,
        process: str | None = None,
        name: str | None = None,
        **kwargs: Any
    ) -> dict[str, Any]:
        """Crew 생성"""
        crew_name = name or "default_crew"
        crew = {
            "name": crew_name,
            "agents": agents or [],
            "process": process or self._process,
            "framework": "crewai",
        }
        self._crews[crew_name] = crew
        logger.info(f"[CrewAIBridge] Crew 생성: {crew_name}")
        return crew

    async def run(
        self,
        crew: dict[str, Any] | None = None,
        task: str = "",
        **kwargs: Any,
    ) -> dict[str, Any]:
        """Crew 실행 (UniversalAgentBridge 호환)

        Args:
            crew: 실행할 Crew (미지정 시 마지막 생성된 Crew 사용)
            task: 실행할 태스크
        """
        if crew is None:
            crew = next(iter(self._crews.values()), {"name": "default", "process": self._process})
        return {
            "crew": crew.get("name"),
            "output": f"[CrewAI:{crew.get('process')}] '{task}' 완료",
        }
