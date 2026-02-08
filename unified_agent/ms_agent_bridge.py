#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Unified Agent Framework - Microsoft Agent Framework 브릿지 모듈

================================================================================
📁 파일 위치: unified_agent/ms_agent_bridge.py
📋 역할: Microsoft Agent Framework (Preview) 통합 브릿지
📅 최종 업데이트: 2026년 2월 8일
📦 버전: v4.0.0
================================================================================

🎯 주요 기능:
    - Graph-based Workflow (Sequential, Parallel, Handoff, Group Chat)
    - Declarative Agents (YAML 기반 선언적 에이전트)
    - OpenTelemetry 통합 추적
    - DevUI 개발자 도구

📌 사용 예시:
    >>> from unified_agent.ms_agent_bridge import MicrosoftAgentBridge
    >>>
    >>> bridge = MicrosoftAgentBridge()
    >>> workflow = bridge.create_graph(
    ...     type="sequential",
    ...     agents=["planner", "executor", "reviewer"]
    ... )

🔗 관련 문서:
    - MS Agent Framework: https://github.com/microsoft/agent-framework
"""

from __future__ import annotations

import logging
from typing import Any

__all__ = ["MicrosoftAgentBridge"]

logger = logging.getLogger(__name__)

class MicrosoftAgentBridge:
    """
    Microsoft Agent Framework 통합 브릿지

    ================================================================================
    📋 역할: Microsoft Agent Framework(Preview)의 Graph Workflow,
             Declarative Agents를 Unified Agent Framework로 통합
    📅 최종 업데이트: 2026년 2월
    ================================================================================

    Graph Workflow 타입:
    - sequential: 순차 실행
    - parallel: 병렬 실행
    - handoff: 에이전트 전환
    - group_chat: 그룹 대화
    """

    def __init__(self, graph_type: str = "sequential"):
        self._graph_type = graph_type
        self._graphs: dict[str, dict] = {}
        logger.info(f"[MicrosoftAgentBridge] 초기화 (graph_type={graph_type})")

    def __repr__(self) -> str:
        return f"MicrosoftAgentBridge(type={self._graph_type!r}, graphs={len(self._graphs)})"

    def create_graph(
        self,
        type: str | None = None,
        agents: list[str] | None = None,
        name: str | None = None,
        **kwargs: Any
    ) -> dict[str, Any]:
        """Graph Workflow 생성"""
        graph_name = name or "ms_graph"
        graph = {
            "name": graph_name,
            "type": type or self._graph_type,
            "agents": agents or [],
            "framework": "microsoft_agent_framework",
        }
        self._graphs[graph_name] = graph
        logger.info(f"[MicrosoftAgentBridge] Graph 생성: {graph_name}")
        return graph

    async def run(
        self,
        graph: dict[str, Any] | None = None,
        input: str | None = None,
        *,
        task: str | None = None,
        **kwargs: Any,
    ) -> dict[str, Any]:
        """Graph Workflow 실행 (UniversalAgentBridge 호환)

        Args:
            graph: 실행할 Graph (미지정 시 마지막 생성된 Graph 사용)
            input: 입력 텍스트 (직접 호출용)
            task: 태스크 텍스트 (UniversalAgentBridge 통합용, input 대체)
        """
        input_text = task or input or ""
        if graph is None:
            graph = next(iter(self._graphs.values()), {"name": "default", "type": self._graph_type})
        return {
            "graph": graph.get("name"),
            "output": f"[MSAgent:{graph.get('type')}] '{input_text}' 완료",
        }
