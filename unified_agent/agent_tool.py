#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
AgentTool Pattern - 에이전트를 도구로 사용

================================================================================
📋 역할: 에이전트를 다른 에이전트의 도구로 사용하는 패턴
📅 버전: 3.4.0 (2026년 2월)
📦 영감: Microsoft AutoGen AgentTool, Crew.AI Agent Delegation
================================================================================

🎯 주요 기능:
    - 에이전트를 AIFunction으로 래핑
    - 에이전트 간 위임 (Delegation)
    - 중첩 에이전트 호출
    - 에이전트 체인
    - 동적 에이전트 라우팅

📌 사용 시나리오:
    - 복잡한 작업을 전문 에이전트에 위임
    - 에이전트 계층 구조
    - 동적 능력 확장
    - 전문가 시스템

📌 사용 예시:
    >>> from unified_agent import AgentTool, DelegationManager
    >>>
    >>> # 전문가 에이전트를 도구로 변환
    >>> code_expert_tool = AgentTool.from_agent(
    ...     agent=code_expert,
    ...     name="code_analysis",
    ...     description="코드 분석 전문가에게 위임"
    ... )
    >>>
    >>> # 메인 에이전트의 도구로 등록
    >>> main_agent.add_tool(code_expert_tool)
    >>>
    >>> # 자동 위임
    >>> result = await main_agent.execute("이 코드를 분석해주세요")
"""

from __future__ import annotations

import asyncio
import inspect
import json
import logging
import time
import uuid
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import (
    Any,
    Callable,
    Coroutine,
    Generic,
    Type,
    TypeVar,
)

from .utils import StructuredLogger
from .tools import AIFunction

__all__ = [
    # 설정
    "AgentToolConfig",
    "DelegationPolicy",
    # 도구
    "AgentTool",
    "AgentToolRegistry",
    # 위임
    "DelegationManager",
    "DelegationResult",
    # 체인
    "AgentChain",
    "ChainStep",
]

# ============================================================================
# 설정 및 정책
# ============================================================================

class DelegationPolicy(str, Enum):
    """위임 정책"""
    ALWAYS = "always"           # 항상 위임
    ON_REQUEST = "on_request"   # 요청 시에만
    AUTO = "auto"              # 자동 판단
    NEVER = "never"            # 위임 안함

@dataclass(frozen=True, slots=True)
class AgentToolConfig:
    """
    AgentTool 설정
    
    Args:
        timeout_seconds: 에이전트 호출 타임아웃
        max_retries: 최대 재시도 횟수
        delegation_policy: 위임 정책
        include_context: 컨텍스트 포함 여부
        include_history: 히스토리 포함 여부
    """
    timeout_seconds: float = 120.0
    max_retries: int = 2
    delegation_policy: DelegationPolicy = DelegationPolicy.AUTO
    include_context: bool = True
    include_history: bool = False
    max_history_turns: int = 5

@dataclass(frozen=True, slots=True)
class DelegationResult:
    """위임 결과"""
    success: bool
    agent_id: str
    agent_name: str
    result: Any = None
    error: str | None = None
    duration_ms: float = 0.0
    delegated_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    metadata: dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> dict[str, Any]:
        return {
            "success": self.success,
            "agent_id": self.agent_id,
            "agent_name": self.agent_name,
            "result": self.result,
            "error": self.error,
            "duration_ms": self.duration_ms,
        }

# ============================================================================
# AgentTool - 에이전트를 도구로 래핑
# ============================================================================

class AgentTool(AIFunction):
    """
    에이전트를 도구(Tool)로 변환
    
    다른 에이전트에서 호출 가능한 AIFunction으로 래핑
    
    사용 예시:
        >>> # 기존 에이전트를 도구로 변환
        >>> tool = AgentTool.from_agent(
        ...     agent=researcher_agent,
        ...     name="research",
        ...     description="깊이 있는 리서치 수행"
        ... )
        >>>
        >>> # 도구 스키마
        >>> schema = tool.get_schema()
        >>>
        >>> # 도구 실행
        >>> result = await tool.execute(query="AI 트렌드 분석")
    """
    
    def __init__(
        self,
        name: str,
        description: str,
        agent: Any,
        config: AgentToolConfig | None = None,
        input_schema: dict[str, Any] | None = None,
    ):
        """
        AgentTool 초기화
        
        Args:
            name: 도구 이름
            description: 도구 설명
            agent: 래핑할 에이전트
            config: 설정
            input_schema: 입력 스키마 (선택적)
        """
        self.name = name
        self.description = description
        self._agent = agent
        self._config = config or AgentToolConfig()
        self._input_schema = input_schema
        self._logger = StructuredLogger(f"agent_tool.{name}")
        
        # 에이전트 정보
        self.agent_id = getattr(agent, 'id', None) or getattr(agent, 'name', name)
        self.agent_name = getattr(agent, 'name', name)
    
    @classmethod
    def from_agent(
        cls,
        agent: Any,
        name: str | None = None,
        description: str | None = None,
        config: AgentToolConfig | None = None,
        parameters: dict[str, Any] | None = None,
    ) -> "AgentTool":
        """
        에이전트로부터 AgentTool 생성
        
        Args:
            agent: 에이전트 인스턴스
            name: 도구 이름 (기본: 에이전트 이름)
            description: 도구 설명 (기본: 에이전트 설명)
            config: 설정
            parameters: 입력 파라미터 스키마
            
        Returns:
            AgentTool 인스턴스
        """
        tool_name = name or getattr(agent, 'name', agent.__class__.__name__)
        tool_desc = description or getattr(agent, 'description', f"Delegate to {tool_name} agent")
        
        # 기본 입력 스키마
        default_schema = {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "에이전트에게 전달할 요청"
                },
                "context": {
                    "type": "string",
                    "description": "추가 컨텍스트 (선택적)"
                }
            },
            "required": ["query"]
        }
        
        return cls(
            name=tool_name,
            description=tool_desc,
            agent=agent,
            config=config,
            input_schema=parameters or default_schema,
        )
    
    def get_schema(self) -> dict[str, Any]:
        """OpenAI Function Calling 스키마"""
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": self._input_schema or {
                    "type": "object",
                    "properties": {
                        "query": {
                            "type": "string",
                            "description": "Request to delegate to the agent"
                        }
                    },
                    "required": ["query"]
                }
            }
        }
    
    async def execute(self, **kwargs) -> DelegationResult:
        """
        에이전트 도구 실행
        
        Args:
            **kwargs: 입력 파라미터
            
        Returns:
            위임 결과
        """
        start_time = time.time()
        
        result = DelegationResult(
            success=False,
            agent_id=self.agent_id,
            agent_name=self.agent_name,
        )
        
        self._logger.info(
            "Agent tool execution started",
            agent=self.agent_name,
            kwargs=list(kwargs.keys())
        )
        
        try:
            # 에이전트 호출
            output = await self._call_agent(kwargs)
            
            result.success = True
            result.result = output
            result.duration_ms = (time.time() - start_time) * 1000
            
            self._logger.info(
                "Agent tool execution completed",
                agent=self.agent_name,
                duration_ms=result.duration_ms
            )
            
        except asyncio.TimeoutError:
            result.error = f"Agent {self.agent_name} timed out"
            self._logger.error("Agent tool timeout", agent=self.agent_name)
            
        except Exception as e:
            result.error = str(e)
            self._logger.error("Agent tool failed", agent=self.agent_name, error=str(e))
        
        return result
    
    async def _call_agent(self, kwargs: dict[str, Any]) -> Any:
        """에이전트 호출 (다양한 인터페이스 지원)"""
        query = kwargs.get('query', '')
        context = kwargs.get('context', '')
        
        # 입력 준비
        if self._config.include_context and context:
            input_data = {"query": query, "context": context}
        else:
            input_data = {"query": query}
        
        # 타임아웃 적용
        async def call():
            if asyncio.iscoroutinefunction(self._agent):
                return await self._agent(input_data)
            elif hasattr(self._agent, 'execute'):
                if asyncio.iscoroutinefunction(self._agent.execute):
                    return await self._agent.execute(input_data)
                return self._agent.execute(input_data)
            elif hasattr(self._agent, 'run'):
                if asyncio.iscoroutinefunction(self._agent.run):
                    return await self._agent.run(input_data)
                return self._agent.run(input_data)
            elif hasattr(self._agent, 'invoke'):
                if asyncio.iscoroutinefunction(self._agent.invoke):
                    return await self._agent.invoke(query)
                return self._agent.invoke(query)
            elif callable(self._agent):
                if asyncio.iscoroutinefunction(self._agent):
                    return await self._agent(input_data)
                return await asyncio.to_thread(self._agent, input_data)
            else:
                raise TypeError(f"Agent {self.agent_name} is not callable")
        
        return await asyncio.wait_for(call(), timeout=self._config.timeout_seconds)
    
    def __repr__(self) -> str:
        return f"AgentTool(name={self.name}, agent={self.agent_name})"

# ============================================================================
# AgentToolRegistry - 에이전트 도구 레지스트리
# ============================================================================

class AgentToolRegistry:
    """
    에이전트 도구 레지스트리
    
    여러 에이전트를 도구로 관리하고 동적으로 선택
    
    사용 예시:
        >>> registry = AgentToolRegistry()
        >>> registry.register(code_expert, capabilities=["code", "analysis"])
        >>> registry.register(researcher, capabilities=["research", "web"])
        >>>
        >>> # 능력으로 찾기
        >>> tools = registry.find_by_capability("code")
        >>>
        >>> # 모든 도구 스키마
        >>> schemas = registry.get_all_schemas()
    """
    
    def __init__(self, config: AgentToolConfig | None = None):
        self._config = config or AgentToolConfig()
        self._tools: dict[str, AgentTool] = {}
        self._capabilities: dict[str, set[str]] = {}  # capability -> tool names
        self._logger = StructuredLogger("agent_tool_registry")
    
    def register(
        self,
        agent: Any,
        name: str | None = None,
        description: str | None = None,
        capabilities: list[str] | None = None,
        config: AgentToolConfig | None = None,
    ) -> AgentTool:
        """
        에이전트를 도구로 등록
        
        Args:
            agent: 에이전트 인스턴스
            name: 도구 이름
            description: 도구 설명
            capabilities: 능력 태그
            config: 설정
            
        Returns:
            등록된 AgentTool
        """
        tool = AgentTool.from_agent(
            agent=agent,
            name=name,
            description=description,
            config=config or self._config,
        )
        
        self._tools[tool.name] = tool
        
        # 능력 인덱싱
        for cap in (capabilities or []):
            if cap not in self._capabilities:
                self._capabilities[cap] = set()
            self._capabilities[cap].add(tool.name)
        
        self._logger.info(
            "Agent tool registered",
            name=tool.name,
            capabilities=capabilities
        )
        
        return tool
    
    def unregister(self, name: str) -> bool:
        """도구 등록 해제"""
        if name in self._tools:
            del self._tools[name]
            
            # 능력 인덱스에서 제거
            for cap_set in self._capabilities.values():
                cap_set.discard(name)
            
            return True
        return False
    
    def get(self, name: str) -> AgentTool | None:
        """이름으로 도구 조회"""
        return self._tools.get(name)
    
    def find_by_capability(self, capability: str) -> list[AgentTool]:
        """능력으로 도구 찾기"""
        tool_names = self._capabilities.get(capability, set())
        return [self._tools[name] for name in tool_names if name in self._tools]
    
    def get_all_tools(self) -> list[AgentTool]:
        """모든 도구 조회"""
        return list(self._tools.values())
    
    def get_all_schemas(self) -> list[dict[str, Any]]:
        """모든 도구의 스키마 조회"""
        return [tool.get_schema() for tool in self._tools.values()]
    
    def __len__(self) -> int:
        return len(self._tools)
    
    def __contains__(self, name: str) -> bool:
        return name in self._tools

# ============================================================================
# DelegationManager - 위임 관리자
# ============================================================================

class DelegationManager:
    """
    위임 관리자
    
    에이전트 간 위임을 자동으로 관리
    
    사용 예시:
        >>> manager = DelegationManager(registry)
        >>>
        >>> # 자동 위임 (적절한 에이전트 선택)
        >>> result = await manager.delegate(
        ...     task="이 코드를 분석해주세요",
        ...     hint="code analysis"
        ... )
        >>>
        >>> # 특정 에이전트에 위임
        >>> result = await manager.delegate_to(
        ...     agent_name="code_expert",
        ...     task="버그를 찾아주세요"
        ... )
    """
    
    def __init__(
        self,
        registry: AgentToolRegistry,
        config: AgentToolConfig | None = None,
    ):
        self._registry = registry
        self._config = config or AgentToolConfig()
        self._logger = StructuredLogger("delegation_manager")
        
        # 라우팅 함수 (커스터마이징 가능)
        self._router: Callable[[str], str | None] | None = None
    
    def set_router(self, router: Callable[[str, list[str]], str | None]):
        """
        커스텀 라우터 설정
        
        Args:
            router: (task, available_tools) -> selected_tool_name
        """
        self._router = router
    
    async def delegate(
        self,
        task: str,
        hint: str | None = None,
        context: str | None = None,
        exclude: list[str] | None = None,
    ) -> DelegationResult:
        """
        자동 위임 (적절한 에이전트 선택)
        
        Args:
            task: 위임할 작업
            hint: 능력 힌트 (예: "code", "research")
            context: 추가 컨텍스트
            exclude: 제외할 에이전트
            
        Returns:
            위임 결과
        """
        exclude = exclude or []
        
        # 후보 에이전트 선택
        if hint:
            candidates = [t for t in self._registry.find_by_capability(hint) 
                         if t.name not in exclude]
        else:
            candidates = [t for t in self._registry.get_all_tools() 
                         if t.name not in exclude]
        
        if not candidates:
            return DelegationResult(
                success=False,
                agent_id="none",
                agent_name="none",
                error="No suitable agent found"
            )
        
        # 라우터로 선택 (있으면)
        selected = candidates[0]
        if self._router:
            tool_names = [t.name for t in candidates]
            selected_name = self._router(task, tool_names)
            if selected_name:
                selected = self._registry.get(selected_name) or selected
        
        self._logger.info(
            "Auto delegation",
            task_preview=task[:50],
            selected=selected.name
        )
        
        return await selected.execute(query=task, context=context or "")
    
    async def delegate_to(
        self,
        agent_name: str,
        task: str,
        context: str | None = None,
    ) -> DelegationResult:
        """
        특정 에이전트에 위임
        
        Args:
            agent_name: 에이전트 이름
            task: 위임할 작업
            context: 추가 컨텍스트
            
        Returns:
            위임 결과
        """
        tool = self._registry.get(agent_name)
        
        if not tool:
            return DelegationResult(
                success=False,
                agent_id=agent_name,
                agent_name=agent_name,
                error=f"Agent {agent_name} not found"
            )
        
        return await tool.execute(query=task, context=context or "")
    
    async def delegate_chain(
        self,
        task: str,
        agent_sequence: list[str],
        context: str | None = None,
    ) -> list[DelegationResult]:
        """
        에이전트 체인으로 위임 (순차 실행)
        
        Args:
            task: 초기 작업
            agent_sequence: 에이전트 이름 순서
            context: 초기 컨텍스트
            
        Returns:
            각 단계의 위임 결과 리스트
        """
        results = []
        current_input = task
        current_context = context or ""
        
        for agent_name in agent_sequence:
            result = await self.delegate_to(
                agent_name=agent_name,
                task=current_input,
                context=current_context
            )
            results.append(result)
            
            if not result.success:
                break
            
            # 다음 단계 입력으로 사용
            current_input = str(result.result) if result.result else current_input
            current_context = f"Previous agent ({agent_name}): {current_input}"
        
        return results

# ============================================================================
# AgentChain - 에이전트 체인
# ============================================================================

@dataclass(frozen=True, slots=True)
class ChainStep:
    """체인 단계 정의"""
    agent_name: str
    transform_input: Callable[[Any], dict[str, Any]] | None = None
    transform_output: Callable[[DelegationResult], Any] | None = None
    condition: Callable[[Any], bool] | None = None
    on_error: Callable[[Exception], Any] | None = None

class AgentChain:
    """
    에이전트 체인
    
    여러 에이전트를 순차적으로 실행하는 파이프라인
    
    사용 예시:
        >>> chain = AgentChain(registry)
        >>> chain.add_step("analyzer", transform_input=lambda x: {"code": x})
        >>> chain.add_step("reviewer", condition=lambda x: x.get("has_issues"))
        >>> chain.add_step("fixer")
        >>>
        >>> result = await chain.run(source_code)
    """
    
    def __init__(self, registry: AgentToolRegistry):
        self._registry = registry
        self._steps: list[ChainStep] = []
        self._logger = StructuredLogger("agent_chain")
    
    def add_step(
        self,
        agent_name: str,
        transform_input: Callable[[Any], dict[str, Any]] | None = None,
        transform_output: Callable[[DelegationResult], Any] | None = None,
        condition: Callable[[Any], bool] | None = None,
        on_error: Callable[[Exception], Any] | None = None,
    ) -> "AgentChain":
        """
        체인에 단계 추가
        
        Args:
            agent_name: 에이전트 이름
            transform_input: 입력 변환 함수
            transform_output: 출력 변환 함수
            condition: 실행 조건 (False면 스킵)
            on_error: 에러 핸들러
            
        Returns:
            self (체이닝용)
        """
        self._steps.append(ChainStep(
            agent_name=agent_name,
            transform_input=transform_input,
            transform_output=transform_output,
            condition=condition,
            on_error=on_error,
        ))
        return self
    
    async def run(self, initial_input: Any) -> list[DelegationResult]:
        """
        체인 실행
        
        Args:
            initial_input: 초기 입력
            
        Returns:
            각 단계의 결과 리스트
        """
        results = []
        current_value = initial_input
        
        self._logger.info(
            "Chain started",
            steps=len(self._steps)
        )
        
        for i, step in enumerate(self._steps):
            # 조건 체크
            if step.condition and not step.condition(current_value):
                self._logger.debug(
                    "Step skipped (condition false)",
                    step=i,
                    agent=step.agent_name
                )
                continue
            
            # 입력 변환
            if step.transform_input:
                input_data = step.transform_input(current_value)
            else:
                input_data = {"query": str(current_value)}
            
            # 에이전트 실행
            tool = self._registry.get(step.agent_name)
            if not tool:
                result = DelegationResult(
                    success=False,
                    agent_id=step.agent_name,
                    agent_name=step.agent_name,
                    error=f"Agent {step.agent_name} not found"
                )
            else:
                try:
                    result = await tool.execute(**input_data)
                except Exception as e:
                    if step.on_error:
                        fallback = step.on_error(e)
                        result = DelegationResult(
                            success=True,
                            agent_id=step.agent_name,
                            agent_name=step.agent_name,
                            result=fallback
                        )
                    else:
                        result = DelegationResult(
                            success=False,
                            agent_id=step.agent_name,
                            agent_name=step.agent_name,
                            error=str(e)
                        )
            
            results.append(result)
            
            # 출력 변환
            if result.success:
                if step.transform_output:
                    current_value = step.transform_output(result)
                else:
                    current_value = result.result
            else:
                self._logger.warning(
                    "Chain step failed",
                    step=i,
                    agent=step.agent_name,
                    error=result.error
                )
                break
        
        self._logger.info(
            "Chain completed",
            total_steps=len(self._steps),
            executed=len(results),
            success=all(r.success for r in results)
        )
        
        return results
    
    def __len__(self) -> int:
        return len(self._steps)
