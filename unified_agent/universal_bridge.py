#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Unified Agent Framework - Universal Agent Bridge 모듈

================================================================================
📁 파일 위치: unified_agent/universal_bridge.py
📋 역할: 16개 AI Agent 프레임워크를 하나의 통합 인터페이스로 연결
📅 최종 업데이트: 2026년 2월 8일
📦 버전: v4.0.0
================================================================================

🎯 핵심 혁신 #1: Universal Agent Bridge

16개 AI Agent 프레임워크(OpenAI Agents SDK, Google ADK, CrewAI, LangGraph,
A2A Protocol, SK Agent, MS Agent Framework, AG2 등)를 **하나의 인터페이스**로
통합합니다. 프레임워크 Lock-in 없이, 작업에 최적인 프레임워크를 동적으로
선택할 수 있습니다. 전환 비용 0.

📌 사용 예시:
    >>> from unified_agent.universal_bridge import UniversalAgentBridge
    >>> from unified_agent.openai_agents_bridge import OpenAIAgentsBridge
    >>> from unified_agent.google_adk_bridge import GoogleADKBridge
    >>> from unified_agent.crewai_bridge import CrewAIBridge
    >>>
    >>> bridge = UniversalAgentBridge()
    >>> bridge.register("openai", OpenAIAgentsBridge())
    >>> bridge.register("google", GoogleADKBridge())
    >>> bridge.register("crewai", CrewAIBridge())
    >>>
    >>> # 동일한 인터페이스로 프레임워크 자유 전환
    >>> result = await bridge.run("openai", task="코드 리뷰")
    >>> result = await bridge.run("crewai", task="팀 리서치")  # 코드 변경 없이 전환
    >>>
    >>> # A2A 프로토콜로 외부 에이전트와도 협업
    >>> bridge.enable_a2a_discovery()

💡 아이디어: "어떤 프레임워크를 선택할지 고민하지 마세요. 전부 쓰세요."

🔗 관련 문서:
    - OpenAI Agents SDK: https://github.com/openai/openai-agents-python
    - Google ADK: https://github.com/google/adk-python
    - CrewAI: https://github.com/crewAIInc/crewAI
    - A2A Protocol: https://github.com/a2aproject/A2A
"""

import logging
from typing import Any, Dict, List, Optional, Protocol, runtime_checkable

__all__ = ["UniversalAgentBridge", "BridgeProtocol"]

logger = logging.getLogger(__name__)


# ============================================================================
# Bridge Protocol — 모든 브릿지가 구현해야 하는 인터페이스
# ============================================================================

@runtime_checkable
class BridgeProtocol(Protocol):
    """
    프레임워크 브릿지 프로토콜

    모든 브릿지 모듈이 구현해야 하는 최소 인터페이스입니다.
    """
    async def run(self, *args: Any, **kwargs: Any) -> Dict[str, Any]:
        """에이전트/워크플로우 실행"""
        ...


# ============================================================================
# UniversalAgentBridge — 핵심 혁신 #1
# ============================================================================

class UniversalAgentBridge:
    """
    Universal Agent Bridge — 모든 프레임워크를 하나의 인터페이스로

    ================================================================================
    📋 역할: 16개 AI Agent 프레임워크 통합, 동적 전환, 전환 비용 0
    📅 최종 업데이트: 2026년 2월
    ================================================================================

    지원 프레임워크 (16개):
    1. OpenAI Agents SDK (v0.8.1) — Handoff, Session, HITL, Voice
    2. Google ADK (v1.24.1) — Workflow Agent, A2A
    3. CrewAI (v1.9.3) — Crews + Flows
    4. A2A Protocol (v0.3.0) — Agent Card, JSON-RPC 2.0
    5. Microsoft Agent Framework (Preview) — Graph Workflow
    6. AG2/AutoGen (v0.7.5) — Universal Interop
    7. Semantic Kernel (Py 1.39.3) — Orchestration
    8. LangGraph (v1.0.8) — 상태 그래프
    9-16. 기타 프레임워크 (커스텀 브릿지 등록)

    사용법:
        >>> bridge = UniversalAgentBridge()
        >>> bridge.register("openai", OpenAIAgentsBridge())
        >>> result = await bridge.run("openai", task="코드 리뷰")
    """

    def __init__(self):
        self._bridges: Dict[str, Any] = {}
        self._a2a_enabled: bool = False
        self._default_framework: Optional[str] = None
        logger.info("[UniversalAgentBridge] 초기화")

    def __repr__(self) -> str:
        return f"UniversalAgentBridge(frameworks={self.registered_frameworks})"

    def register(self, name: str, bridge: Any) -> None:
        """
        프레임워크 브릿지 등록

        Args:
            name: 프레임워크 식별자 (예: "openai", "google", "crewai")
            bridge: 브릿지 인스턴스 (run 메서드 필수)
        """
        self._bridges[name] = bridge
        if not self._default_framework:
            self._default_framework = name
        logger.info(f"[UniversalAgentBridge] 브릿지 등록: {name} (총 {len(self._bridges)}개)")

    def unregister(self, name: str) -> None:
        """프레임워크 브릿지 해제"""
        self._bridges.pop(name, None)
        if self._default_framework == name:
            self._default_framework = next(iter(self._bridges), None)
        logger.info(f"[UniversalAgentBridge] 브릿지 해제: {name}")

    async def run(
        self,
        framework: Optional[str] = None,
        task: str = "",
        **kwargs: Any
    ) -> Dict[str, Any]:
        """
        지정된 프레임워크로 태스크 실행

        동일한 인터페이스로 어떤 프레임워크든 전환 가능 — 전환 비용 0

        Args:
            framework: 실행할 프레임워크 (미지정 시 기본값)
            task: 실행할 태스크
            **kwargs: 프레임워크별 추가 인자

        Returns:
            실행 결과 딕셔너리
        """
        fw = framework or self._default_framework
        if not fw or fw not in self._bridges:
            available = list(self._bridges.keys())
            raise ValueError(f"프레임워크 '{fw}'가 등록되지 않음. 사용 가능: {available}")

        bridge = self._bridges[fw]
        logger.info(f"[UniversalAgentBridge] 실행: framework={fw}, task='{task[:50]}...'")

        # 브릿지 run 호출 (각 브릿지의 구현에 따라 다양한 시그니처)
        if hasattr(bridge, 'run'):
            return await bridge.run(task=task, **kwargs)
        else:
            raise AttributeError(f"브릿지 '{fw}'에 run 메서드가 없습니다.")

    def enable_a2a_discovery(self) -> None:
        """A2A 에이전트 자동 발견 활성화"""
        self._a2a_enabled = True
        logger.info("[UniversalAgentBridge] A2A 에이전트 발견 활성화")

    @property
    def registered_frameworks(self) -> List[str]:
        """등록된 프레임워크 목록"""
        return list(self._bridges.keys())

    @property
    def framework_count(self) -> int:
        """등록된 프레임워크 수"""
        return len(self._bridges)

    def get_bridge(self, name: str) -> Optional[Any]:
        """특정 브릿지 인스턴스 반환"""
        return self._bridges.get(name)
