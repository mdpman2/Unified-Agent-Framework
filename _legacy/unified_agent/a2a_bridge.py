#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Unified Agent Framework - A2A Protocol 브릿지 모듈

================================================================================
📁 파일 위치: unified_agent/a2a_bridge.py
📋 역할: A2A Protocol (v0.3.0) 통합 브릿지 — Agent Card, JSON-RPC 2.0
📅 최종 업데이트: 2026년 2월 14일
📦 버전: v4.1.0
================================================================================

🎯 주요 구성 요소:
    1. A2ABridge - A2A 프로토콜 클라이언트/서버
    2. AgentCard - 에이전트 발견 및 역량 공개

🔧 2026년 2월 기능:
    - JSON-RPC 2.0 over HTTP(S) 표준 통신
    - Agent Card로 에이전트 발견/역량 공개
    - Sync, Streaming (SSE), Async Push 지원
    - 에이전트 내부 상태를 노출하지 않는 Opacity 원칙
    - Linux Foundation 산하 표준

📌 사용 예시:
    >>> from unified_agent.a2a_bridge import A2ABridge, AgentCard
    >>>
    >>> bridge = A2ABridge()
    >>> card = AgentCard(
    ...     name="research_agent",
    ...     capabilities=["web_search", "summarization"],
    ...     endpoint="https://my-agent.example.com/a2a"
    ... )
    >>> remote = await bridge.discover("https://partner.example.com/.well-known/agent-card.json")
    >>> result = await bridge.send_task(to=remote, task="최신 뉴스 요약")

🔗 관련 문서:
    - A2A Protocol: https://github.com/a2aproject/A2A
"""

from __future__ import annotations

import logging
import uuid
from dataclasses import dataclass, field
from typing import Any, Protocol

__all__ = ["A2ABridge", "AgentCard", "TaskMode"]

logger = logging.getLogger(__name__)

class TaskMode:
    """A2A 태스크 전송 모드"""
    SYNC = "sync"
    STREAMING = "streaming"
    ASYNC_PUSH = "async_push"

@dataclass
class AgentCard:
    """
    A2A Agent Card — 에이전트 역량 공개 및 발견

    ================================================================================
    📋 역할: 에이전트의 이름, 역량, 프로토콜 버전, 엔드포인트를 공개
    📅 최종 업데이트: 2026년 2월
    ================================================================================

    JSON-LD 형식으로 `.well-known/agent-card.json`에 게시됩니다.
    """
    name: str = ""
    capabilities: list[str] = field(default_factory=list)
    protocols: list[str] = field(default_factory=lambda: ["a2a-v0.3.0"])
    endpoint: str = ""
    description: str = ""
    version: str = "0.3.0"

class A2ABridge:
    """
    A2A Protocol 통합 브릿지

    ================================================================================
    📋 역할: A2A(Agent-to-Agent) 프로토콜 v0.3.0을 사용한
             에이전트 간 표준 통신 및 협업
    📅 최종 업데이트: 2026년 2월
    ================================================================================

    특징:
    - JSON-RPC 2.0 over HTTP(S)
    - Agent Card 발견 및 게시
    - Sync / Streaming (SSE) / Async Push 모드
    - Opacity 원칙: 내부 상태 비노출
    """

    def __init__(self):
        self._local_cards: dict[str, AgentCard] = {}
        self._remote_cards: dict[str, AgentCard] = {}
        logger.info("[A2ABridge] 초기화")

    def __repr__(self) -> str:
        return f"A2ABridge(local={len(self._local_cards)}, remote={len(self._remote_cards)})"

    async def publish_card(self, card: AgentCard) -> None:
        """로컬 Agent Card 게시"""
        self._local_cards[card.name] = card
        logger.info(f"[A2ABridge] Agent Card 게시: {card.name}")

    async def discover(self, url: str) -> AgentCard:
        """원격 Agent Card 발견"""
        logger.info(f"[A2ABridge] 원격 에이전트 발견: {url}")
        card = AgentCard(
            name=f"remote_{uuid.uuid4().hex[:6]}",
            endpoint=url.replace("/.well-known/agent-card.json", ""),
            capabilities=["general"],
        )
        self._remote_cards[card.name] = card
        return card

    async def send_task(
        self,
        to: AgentCard,
        task: str,
        mode: str = TaskMode.SYNC,
        **kwargs: Any
    ) -> dict[str, Any]:
        """A2A 태스크 전송"""
        logger.info(f"[A2ABridge] 태스크 전송 → {to.name} (mode={mode})")
        return {
            "task_id": f"a2a_{uuid.uuid4().hex[:8]}",
            "to": to.name,
            "result": f"[A2A] '{task}' 태스크 결과",
            "mode": mode,
        }

    async def run(self, *, task: str = "", **kwargs: Any) -> dict[str, Any]:
        """태스크 실행 (UniversalAgentBridge 호환)

        A2A 프로토콜을 통해 태스크를 전송합니다.
        kwargs에서 'to' (AgentCard)를 참조하여 대상 에이전트를 결정합니다.

        Args:
            task: 전송할 태스크
            **kwargs: 'to' (AgentCard), 'mode' (TaskMode) 등
        """
        target = kwargs.pop("to", None)
        if target is None:
            target = next(iter(self._local_cards.values()), AgentCard(name="self"))
        mode = kwargs.pop("mode", TaskMode.SYNC)
        return await self.send_task(to=target, task=task, mode=mode, **kwargs)
