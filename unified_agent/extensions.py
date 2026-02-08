#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
v3.4 확장 모듈 통합 - Extensions Hub

================================================================================
📋 역할: v3.4 신규 모듈들의 통합 진입점 및 팩토리
📅 버전: 3.4.0 (2026년 2월)
================================================================================

🎯 해결하는 문제:
    - v3.4 신규 모듈들이 framework.py와 분리되어 있음
    - 사용자가 개별 모듈을 직접 import해야 함
    - 모듈 간 통합 사용이 번거로움

📌 사용 예시:
    >>> from unified_agent import Extensions
    >>>
    >>> # 확장 모듈 초기화 (프레임워크와 연결)
    >>> ext = Extensions(framework)
    >>>
    >>> # Prompt Caching 사용
    >>> cached_response = await ext.cache.get_or_call(...)
    >>>
    >>> # Durable Workflow 실행
    >>> result = await ext.durable.execute_workflow(my_workflow, data)
    >>>
    >>> # 병렬 실행
    >>> results = await ext.concurrent.fan_out(task, agents)
    >>>
    >>> # MCP 서버 관리
    >>> ext.mcp.register_server(config)
    >>> await ext.mcp.connect_all()
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Callable

from .utils import StructuredLogger
from .interfaces import IFramework

# v3.4 모듈 임포트
from .prompt_cache import PromptCache, CacheConfig
from .durable_agent import DurableOrchestrator, DurableConfig, DurableContext
from .concurrent import ConcurrentOrchestrator, FanOutConfig, AggregationStrategy
from .agent_tool import AgentTool, AgentToolRegistry, DelegationManager
from .extended_thinking import ThinkingTracker, ThinkingConfig, ThinkingMode
from .mcp_workbench import McpWorkbench, McpServerConfig, McpWorkbenchConfig

# Agent 타입 힌트 (런타임에는 필요 없음)
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from .agents import Agent

__all__ = [
    "Extensions",
    "ExtensionsConfig",
]

@dataclass(frozen=True, slots=True)
class ExtensionsConfig:
    """
    확장 모듈 설정
    
    Args:
        enable_cache: Prompt Cache 활성화
        enable_durable: Durable Agent 활성화
        enable_concurrent: Concurrent Orchestration 활성화
        enable_agent_tool: AgentTool 패턴 활성화
        enable_thinking: Extended Thinking 활성화
        enable_mcp: MCP Workbench 활성화
        
        cache_config: 캐시 설정
        durable_config: Durable 설정
        concurrent_config: 병렬 실행 설정
        thinking_config: 사고 추적 설정
        mcp_config: MCP 설정
    """
    # 활성화 플래그
    enable_cache: bool = True
    enable_durable: bool = True
    enable_concurrent: bool = True
    enable_agent_tool: bool = True
    enable_thinking: bool = True
    enable_mcp: bool = True
    
    # 개별 설정
    cache_config: CacheConfig | None = None
    durable_config: DurableConfig | None = None
    concurrent_config: FanOutConfig | None = None
    thinking_config: ThinkingConfig | None = None
    mcp_config: McpWorkbenchConfig | None = None

class Extensions:
    """
    v3.4 확장 모듈 통합 허브
    
    framework.py와 v3.4 신규 모듈들을 연결하는 통합 레이어.
    각 확장 모듈에 대한 편리한 접근과 초기화를 제공.
    
    사용 예시:
        >>> framework = UnifiedAgentFramework.create()
        >>> ext = Extensions(framework)
        >>>
        >>> # 또는 프레임워크에서 직접 접근
        >>> framework.extensions.cache.get_stats()
    """
    
    def __init__(
        self,
        framework: IFramework | None = None,
        config: ExtensionsConfig | None = None,
    ):
        """
        확장 모듈 초기화
        
        Args:
            framework: IFramework 인스턴스 (UnifiedAgentFramework 등)
            config: 확장 모듈 설정
        """
        self._framework = framework
        self._config = config or ExtensionsConfig()
        self._logger = StructuredLogger("extensions")
        
        # 확장 모듈 인스턴스
        self._cache: PromptCache | None = None
        self._durable: DurableOrchestrator | None = None
        self._concurrent: ConcurrentOrchestrator | None = None
        self._agent_tool_registry: AgentToolRegistry | None = None
        self._delegation_manager: DelegationManager | None = None
        self._thinking: ThinkingTracker | None = None
        self._mcp: McpWorkbench | None = None
        
        # 초기화
        self._initialize()
    
    def _initialize(self):
        """확장 모듈 초기화"""
        config = self._config
        
        # 1. Prompt Cache
        if config.enable_cache:
            cache_cfg = config.cache_config or CacheConfig()
            self._cache = PromptCache(cache_cfg)
            self._logger.info("Prompt Cache initialized")
        
        # 2. Durable Orchestrator
        if config.enable_durable:
            durable_cfg = config.durable_config or DurableConfig()
            self._durable = DurableOrchestrator(durable_cfg)
            self._logger.info("Durable Orchestrator initialized")
        
        # 3. AgentTool Registry
        if config.enable_agent_tool:
            self._agent_tool_registry = AgentToolRegistry()
            self._delegation_manager = DelegationManager(self._agent_tool_registry)
            self._logger.info("AgentTool Registry initialized")
        
        # 4. Extended Thinking
        if config.enable_thinking:
            thinking_cfg = config.thinking_config or ThinkingConfig()
            self._thinking = ThinkingTracker(thinking_cfg)
            self._logger.info("Extended Thinking initialized")
        
        # 5. MCP Workbench
        if config.enable_mcp:
            mcp_cfg = config.mcp_config or McpWorkbenchConfig()
            self._mcp = McpWorkbench(mcp_cfg)
            self._logger.info("MCP Workbench initialized")
    
    # =========================================================================
    # 프로퍼티 - 확장 모듈 접근
    # =========================================================================
    
    @property
    def cache(self) -> PromptCache | None:
        """Prompt Cache 인스턴스"""
        return self._cache
    
    @property
    def durable(self) -> DurableOrchestrator | None:
        """Durable Orchestrator 인스턴스"""
        return self._durable
    
    @property
    def concurrent(self) -> ConcurrentOrchestrator | None:
        """Concurrent Orchestrator (lazy initialization)"""
        return self._concurrent
    
    @property
    def agent_tools(self) -> AgentToolRegistry | None:
        """AgentTool Registry 인스턴스"""
        return self._agent_tool_registry
    
    @property
    def delegation(self) -> DelegationManager | None:
        """Delegation Manager 인스턴스"""
        return self._delegation_manager
    
    @property
    def thinking(self) -> ThinkingTracker | None:
        """Extended Thinking Tracker 인스턴스"""
        return self._thinking
    
    @property
    def mcp(self) -> McpWorkbench | None:
        """MCP Workbench 인스턴스"""
        return self._mcp
    
    # =========================================================================
    # 편의 메서드 - Prompt Cache
    # =========================================================================
    
    async def cached_llm_call(
        self,
        model: str,
        messages: list[dict[str, str]],
        call_fn: Callable,
        **kwargs
    ) -> tuple:
        """
        캐시된 LLM 호출
        
        Args:
            model: 모델 이름
            messages: 메시지 목록
            call_fn: 실제 LLM 호출 함수
            
        Returns:
            (response, was_cached) 튜플
        """
        if not self._cache:
            result = await call_fn(model=model, messages=messages, **kwargs)
            return result, False
        
        return await self._cache.get_or_call(
            model=model,
            messages=messages,
            call_fn=call_fn,
            **kwargs
        )
    
    # =========================================================================
    # 편의 메서드 - Concurrent Execution
    # =========================================================================
    
    def create_concurrent_orchestrator(
        self,
        agents: list['Agent'],
        config: FanOutConfig | None = None,
    ) -> ConcurrentOrchestrator:
        """
        병렬 실행 오케스트레이터 생성
        
        Args:
            agents: 에이전트 목록
            config: Fan-out 설정
            
        Returns:
            ConcurrentOrchestrator 인스턴스
        """
        cfg = config or self._config.concurrent_config or FanOutConfig()
        self._concurrent = ConcurrentOrchestrator(agents, cfg)
        return self._concurrent
    
    async def fan_out(
        self,
        task: str,
        agents: list['Agent'] | None = None,
        aggregation: AggregationStrategy = AggregationStrategy.ALL,
    ) -> dict[str, Any]:
        """
        Fan-out 병렬 실행
        
        Args:
            task: 실행할 작업
            agents: 에이전트 목록 (없으면 기존 오케스트레이터 사용)
            aggregation: 결과 집계 전략
            
        Returns:
            집계된 결과
        """
        if agents and not self._concurrent:
            self.create_concurrent_orchestrator(agents)
        
        if not self._concurrent:
            raise ValueError("ConcurrentOrchestrator가 초기화되지 않았습니다.")
        
        return await self._concurrent.fan_out(task, aggregation_strategy=aggregation)
    
    # =========================================================================
    # 편의 메서드 - AgentTool
    # =========================================================================
    
    def register_agent_as_tool(
        self,
        agent: 'Agent',
        name: str | None = None,
        description: str | None = None,
    ) -> AgentTool:
        """
        에이전트를 도구로 등록
        
        Args:
            agent: 등록할 에이전트
            name: 도구 이름 (기본: 에이전트 이름)
            description: 도구 설명
            
        Returns:
            생성된 AgentTool
        """
        if not self._agent_tool_registry:
            raise ValueError("AgentTool Registry가 초기화되지 않았습니다.")
        
        tool = AgentTool.from_agent(
            agent,
            name=name or agent.name,
            description=description or f"Agent: {agent.name}"
        )
        self._agent_tool_registry.register(tool)
        return tool
    
    async def delegate_task(
        self,
        task: str,
        required_capabilities: list[str] | None = None,
    ) -> Any:
        """
        작업 위임
        
        Args:
            task: 위임할 작업
            required_capabilities: 필요한 능력 목록
            
        Returns:
            위임 결과
        """
        if not self._delegation_manager:
            raise ValueError("Delegation Manager가 초기화되지 않았습니다.")
        
        return await self._delegation_manager.delegate(
            task=task,
            required_capabilities=required_capabilities
        )
    
    # =========================================================================
    # 편의 메서드 - Extended Thinking
    # =========================================================================
    
    def track_thinking(self, task_id: str):
        """
        사고 과정 추적 컨텍스트
        
        사용 예시:
            >>> with ext.track_thinking("task-1") as thinking:
            ...     thinking.add_observation("입력 분석...")
            ...     thinking.add_reasoning("추론 수행...")
        """
        if not self._thinking:
            raise ValueError("ThinkingTracker가 초기화되지 않았습니다.")
        
        return self._thinking.track_thinking(task_id)
    
    # =========================================================================
    # 편의 메서드 - MCP Workbench
    # =========================================================================
    
    def register_mcp_server(self, config: McpServerConfig):
        """
        MCP 서버 등록
        
        Args:
            config: MCP 서버 설정
        """
        if not self._mcp:
            raise ValueError("MCP Workbench가 초기화되지 않았습니다.")
        
        self._mcp.register_server(config)
    
    async def connect_mcp_servers(self) -> dict[str, bool]:
        """
        모든 MCP 서버 연결
        
        Returns:
            서버별 연결 결과
        """
        if not self._mcp:
            raise ValueError("MCP Workbench가 초기화되지 않았습니다.")
        
        return await self._mcp.connect_all()
    
    async def call_mcp_tool(
        self,
        tool_name: str,
        server_name: str | None = None,
        **arguments
    ) -> Any:
        """
        MCP 도구 호출
        
        Args:
            tool_name: 도구 이름
            server_name: 서버 이름 (선택적)
            **arguments: 도구 인자
            
        Returns:
            도구 실행 결과
        """
        if not self._mcp:
            raise ValueError("MCP Workbench가 초기화되지 않았습니다.")
        
        return await self._mcp.call_tool(
            tool_name=tool_name,
            server_name=server_name,
            **arguments
        )
    
    # =========================================================================
    # 통계 및 상태
    # =========================================================================
    
    def get_stats(self) -> dict[str, Any]:
        """
        전체 확장 모듈 통계
        
        Returns:
            모듈별 통계 딕셔너리
        """
        stats = {
            "enabled_modules": [],
            "cache": None,
            "durable": None,
            "concurrent": None,
            "agent_tools": None,
            "thinking": None,
            "mcp": None,
        }
        
        if self._cache:
            stats["enabled_modules"].append("cache")
            stats["cache"] = self._cache.get_stats().to_dict()
        
        if self._durable:
            stats["enabled_modules"].append("durable")
            stats["durable"] = {"initialized": True}
        
        if self._concurrent:
            stats["enabled_modules"].append("concurrent")
            stats["concurrent"] = {"initialized": True}
        
        if self._agent_tool_registry:
            stats["enabled_modules"].append("agent_tools")
            stats["agent_tools"] = {
                "registered_tools": len(self._agent_tool_registry),
            }
        
        if self._thinking:
            stats["enabled_modules"].append("thinking")
            stats["thinking"] = {"initialized": True}
        
        if self._mcp:
            stats["enabled_modules"].append("mcp")
            stats["mcp"] = self._mcp.get_status()
        
        return stats
    
    async def cleanup(self):
        """리소스 정리"""
        if self._mcp:
            await self._mcp.disconnect_all()
        
        self._logger.info("Extensions cleanup completed")
