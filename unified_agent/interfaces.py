#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
인터페이스 모듈 - 순환 의존 해소를 위한 추상 인터페이스

================================================================================
📋 역할: 모듈 간 순환 의존을 방지하기 위한 추상 인터페이스 정의
📅 버전: 3.4.0 (2026년 2월)
================================================================================

🎯 해결하는 문제:
    - orchestration.py ↔ framework.py 순환 참조 문제
    - TYPE_CHECKING 블록 의존성 제거
    - 테스트 가능성 향상 (Mock 주입 용이)

📌 사용 패턴:
    # framework.py에서
    class UnifiedAgentFramework(IFramework):
        ...
    
    # orchestration.py에서
    def __init__(self, framework: IFramework):
        self.framework = framework  # 인터페이스로 받음
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, TYPE_CHECKING

from semantic_kernel import Kernel

if TYPE_CHECKING:
    from .models import AgentState, TeamConfiguration
    from .workflow import Graph
    from .tools import MCPTool

__all__ = [
    "IFramework",
    "IOrchestrator",
    "IMemoryProvider",
    "ICacheProvider",
    "IThinkingProvider",
]

# ============================================================================
# Framework 인터페이스
# ============================================================================

class IFramework(ABC):
    """
    Framework 인터페이스
    
    UnifiedAgentFramework의 추상 인터페이스.
    orchestration.py에서 framework를 참조할 때 이 인터페이스 사용.
    """
    
    @property
    @abstractmethod
    def kernel(self) -> Kernel:
        """Semantic Kernel 인스턴스"""
        pass
    
    @property
    @abstractmethod
    def config(self) -> Any:
        """프레임워크 설정"""
        pass
    
    @property
    @abstractmethod
    def event_bus(self) -> Any | None:
        """이벤트 버스"""
        pass
    
    @abstractmethod
    def create_graph(self, name: str) -> 'Graph':
        """워크플로우 그래프 생성"""
        pass
    
    @abstractmethod
    def register_mcp_tool(self, tool: 'MCPTool'):
        """MCP 도구 등록"""
        pass
    
    @abstractmethod
    async def run(
        self,
        session_id: str,
        workflow_name: str,
        user_message: str = "",
        **kwargs
    ) -> 'AgentState':
        """워크플로우 실행"""
        pass

# ============================================================================
# Orchestrator 인터페이스
# ============================================================================

class IOrchestrator(ABC):
    """
    Orchestrator 인터페이스
    
    다양한 오케스트레이션 전략의 공통 인터페이스.
    ConcurrentOrchestrator, DurableOrchestrator 등이 구현.
    """
    
    @abstractmethod
    async def execute(self, task: str, **kwargs) -> Any:
        """작업 실행"""
        pass
    
    @abstractmethod
    def get_status(self) -> dict[str, Any]:
        """상태 조회"""
        pass

# ============================================================================
# Memory Provider 인터페이스
# ============================================================================

class IMemoryProvider(ABC):
    """
    메모리 제공자 인터페이스
    
    memory.py, persistent_memory.py가 공통으로 구현.
    프레임워크에서 메모리 시스템 교체 가능.
    """
    
    @abstractmethod
    async def store(self, key: str, value: Any, **kwargs) -> bool:
        """데이터 저장"""
        pass
    
    @abstractmethod
    async def retrieve(self, key: str, **kwargs) -> Any | None:
        """데이터 조회"""
        pass
    
    @abstractmethod
    async def search(self, query: str, top_k: int = 5, **kwargs) -> list[Any]:
        """데이터 검색"""
        pass

# ============================================================================
# Cache Provider 인터페이스
# ============================================================================

class ICacheProvider(ABC):
    """
    캐시 제공자 인터페이스
    
    prompt_cache.py가 구현.
    LLM 호출 캐싱을 위한 인터페이스.
    """
    
    @abstractmethod
    async def get(self, key: str) -> Any | None:
        """캐시 조회"""
        pass
    
    @abstractmethod
    async def set(self, key: str, value: Any, ttl: int | None = None) -> bool:
        """캐시 저장"""
        pass
    
    @abstractmethod
    def get_stats(self) -> dict[str, Any]:
        """캐시 통계"""
        pass

# ============================================================================
# Thinking Provider 인터페이스
# ============================================================================

class IThinkingProvider(ABC):
    """
    사고 과정 제공자 인터페이스
    
    extended_thinking.py가 구현.
    tracer.py와 통합 시 사용.
    """
    
    @abstractmethod
    def start_thinking(self, task_id: str) -> Any:
        """사고 과정 시작"""
        pass
    
    @abstractmethod
    def add_step(self, step_type: str, content: str, **kwargs):
        """사고 단계 추가"""
        pass
    
    @abstractmethod
    def end_thinking(self) -> dict[str, Any]:
        """사고 과정 종료 및 결과 반환"""
        pass
