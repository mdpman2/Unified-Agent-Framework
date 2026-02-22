#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Unified Agent Framework v5 — Engine Base Protocol

================================================================================
v4.1 대응: universal_bridge.py + adapter.py 대체
축소 이유: v4.1의 UniversalBridge는 16개 프레임워크를 바론지를 통해
          동적 연결했으나, 애매한 추상화로 디버깅이 어려웠음.
          EngineProtocol 하나로 단순화하고, lazy 레지스트리로
          설치된 엔진만 자동 등록.
================================================================================

모든 엔진은 이 프로토콜을 구현합니다.
엔진 = "만들어진 에이전트를 실행하는 도구"
"""

from __future__ import annotations

import logging
import threading
from typing import Any, AsyncIterator, Protocol, runtime_checkable

from ..types import AgentResult, StreamChunk
from ..tools import Tool
from ..callback import CallbackHandler

__all__ = ["EngineProtocol", "get_engine"]

logger = logging.getLogger(__name__)


@runtime_checkable
class EngineProtocol(Protocol):
    """
    엔진 프로토콜 — 모든 엔진이 구현해야 하는 인터페이스

    v4.1 대응: BridgeInterface + AgentAdapter + FrameworkConnector 통합
    축소 이유: v4.1의 복잡한 어댑터 계층(Bridge → Adapter → Connector)을
              run()/stream() 두 메서드의 단순 프로토콜로 대체.
              엔진은 "에이전트를 만드는 도구"가 아니라
              "만들어진 설정으로 LLM을 실행하는 도구".
    """

    async def run(
        self,
        messages: list[dict[str, Any]],
        model: str,
        tools: list[Tool] | None = None,
        callbacks: list[CallbackHandler] | None = None,
        **kwargs: Any,
    ) -> AgentResult:
        """동기 실행 — 최종 결과 반환"""
        ...

    async def stream(
        self,
        messages: list[dict[str, Any]],
        model: str,
        tools: list[Tool] | None = None,
        callbacks: list[CallbackHandler] | None = None,
        **kwargs: Any,
    ) -> AsyncIterator[StreamChunk]:
        """스트리밍 실행"""
        ...


# === Engine Registry ===

_ENGINES: dict[str, type] = {}
_ENGINES_REGISTERED: bool = False
_ENGINE_CACHE: dict[str, EngineProtocol] = {}
_REGISTRY_LOCK = threading.Lock()  # 스레드 안전한 엔진 등록/해제


def _register_builtin_engines() -> None:
    """빌트인 엔진 등록 (lazy import, 한 번만 실행, 스레드 안전)"""
    global _ENGINES_REGISTERED
    if _ENGINES_REGISTERED:
        return
    with _REGISTRY_LOCK:
        if _ENGINES_REGISTERED:  # double-check locking
            return
        _ENGINES_REGISTERED = True

    from .direct import DirectEngine
    _ENGINES["direct"] = DirectEngine

    # LangChain / CrewAI는 optional — 설치 확인 후 등록
    try:
        from .langchain_engine import LangChainEngine
        _ENGINES["langchain"] = LangChainEngine
    except ImportError:
        logger.debug("LangChain not installed. 'langchain' engine unavailable.")

    try:
        from .crewai_engine import CrewAIEngine
        _ENGINES["crewai"] = CrewAIEngine
    except ImportError:
        logger.debug("CrewAI not installed. 'crewai' engine unavailable.")


def get_engine(name: str = "direct", **kwargs) -> EngineProtocol:
    """
    엔진 인스턴스 가져오기 (기본 인스턴스는 캐시됨)

    Args:
        name: 엔진 이름 (direct, langchain, crewai)
        **kwargs: 엔진 생성 인자 (지정 시 캐시 미사용)

    Returns:
        엔진 인스턴스

    사용법:
        >>> engine = get_engine("direct")
        >>> result = await engine.run(messages, model="gpt-5.2")
    """
    _register_builtin_engines()

    engine_cls = _ENGINES.get(name)
    if engine_cls is None:
        available = list(_ENGINES.keys())
        raise ValueError(
            f"Engine '{name}' not found. Available: {available}. "
            f"Install the required package (e.g., pip install langchain) to enable it."
        )

    # kwargs가 없으면 캐시된 인스턴스 반환 (엔진은 stateless, 스레드 안전)
    if not kwargs:
        if name not in _ENGINE_CACHE:
            with _REGISTRY_LOCK:
                if name not in _ENGINE_CACHE:  # double-check
                    _ENGINE_CACHE[name] = engine_cls()
        return _ENGINE_CACHE[name]

    return engine_cls(**kwargs)
