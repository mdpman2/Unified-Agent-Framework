#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Concurrent Orchestration 시스템 - Fan-out/Fan-in 패턴

================================================================================
📋 역할: 병렬 에이전트 실행 및 결과 집계
📅 버전: 3.4.0 (2026년 2월)
📦 영감: LangGraph Fan-out, Azure Durable Functions Fan-out/Fan-in
================================================================================

🎯 주요 기능:
    - Fan-out: 여러 에이전트 병렬 실행
    - Fan-in: 결과 수집 및 집계
    - 조건부 병렬화
    - 동적 에이전트 선택
    - 결과 집계 전략 (first, all, majority, weighted)
    - 타임아웃 및 에러 처리

📌 사용 시나리오:
    - 다중 전문가 의견 수집
    - 병렬 API 호출
    - Map-Reduce 패턴
    - 앙상블 에이전트

📌 사용 예시:
    >>> from unified_agent import (
    ...     ConcurrentOrchestrator, FanOutConfig,
    ...     AggregationStrategy
    ... )
    >>>
    >>> orchestrator = ConcurrentOrchestrator()
    >>>
    >>> # 여러 에이전트 병렬 실행
    >>> results = await orchestrator.fan_out(
    ...     agents=[security_agent, performance_agent, style_agent],
    ...     input_data={"code": source_code},
    ...     strategy=AggregationStrategy.ALL,
    ... )
    >>>
    >>> # 결과 집계
    >>> final_result = await orchestrator.fan_in(
    ...     results,
    ...     aggregator=merge_reviews
    ... )
"""

from __future__ import annotations

import asyncio
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
    Dict,
    Generic,
    List,
    Optional,
    Set,
    Tuple,
    TypeVar,
    Union,
)

from .utils import StructuredLogger
from .models import NodeResult


__all__ = [
    # 설정
    "FanOutConfig",
    "AggregationStrategy",
    # 결과
    "ParallelResult",
    "AggregatedResult",
    # 오케스트레이터
    "ConcurrentOrchestrator",
    # 집계기
    "ResultAggregator",
    "FirstCompleteAggregator",
    "AllCompleteAggregator",
    "MajorityVoteAggregator",
    "WeightedAggregator",
    # 패턴
    "MapReducePattern",
    "ScatterGatherPattern",
]


# ============================================================================
# 설정 및 전략
# ============================================================================

class AggregationStrategy(str, Enum):
    """결과 집계 전략"""
    FIRST = "first"           # 첫 번째 완료 결과
    ALL = "all"               # 모든 결과 (실패 포함)
    ALL_SUCCESS = "all_success"  # 모든 성공 결과만
    MAJORITY = "majority"     # 다수결
    WEIGHTED = "weighted"     # 가중치 기반
    CUSTOM = "custom"         # 커스텀 집계기


@dataclass
class FanOutConfig:
    """
    Fan-out 설정
    
    Args:
        max_concurrency: 최대 동시 실행 수
        timeout_seconds: 전체 타임아웃
        per_agent_timeout: 에이전트별 타임아웃
        fail_fast: 첫 실패 시 전체 중단
        strategy: 집계 전략
        min_success_count: 최소 성공 수 (ALL_SUCCESS용)
    """
    max_concurrency: int = 10
    timeout_seconds: float = 300.0
    per_agent_timeout: float = 60.0
    fail_fast: bool = False
    strategy: AggregationStrategy = AggregationStrategy.ALL
    min_success_count: int = 1


@dataclass
class ParallelResult:
    """병렬 실행 개별 결과"""
    agent_id: str
    agent_name: str
    success: bool
    result: Any = None
    error: Optional[str] = None
    duration_ms: float = 0.0
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "agent_id": self.agent_id,
            "agent_name": self.agent_name,
            "success": self.success,
            "result": self.result,
            "error": self.error,
            "duration_ms": self.duration_ms,
        }


@dataclass
class AggregatedResult:
    """집계된 최종 결과"""
    success: bool
    strategy: AggregationStrategy
    results: List[ParallelResult]
    aggregated_value: Any = None
    total_duration_ms: float = 0.0
    success_count: int = 0
    failure_count: int = 0
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    @property
    def successful_results(self) -> List[ParallelResult]:
        return [r for r in self.results if r.success]
    
    @property
    def failed_results(self) -> List[ParallelResult]:
        return [r for r in self.results if not r.success]
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "success": self.success,
            "strategy": self.strategy.value,
            "aggregated_value": self.aggregated_value,
            "total_duration_ms": self.total_duration_ms,
            "success_count": self.success_count,
            "failure_count": self.failure_count,
            "results": [r.to_dict() for r in self.results],
        }


# ============================================================================
# 결과 집계기
# ============================================================================

T = TypeVar("T")


class ResultAggregator(ABC, Generic[T]):
    """결과 집계기 추상 클래스"""
    
    @abstractmethod
    def aggregate(self, results: List[ParallelResult]) -> T:
        """결과 집계"""
        pass


class FirstCompleteAggregator(ResultAggregator[Any]):
    """첫 번째 완료 결과 반환"""
    
    def aggregate(self, results: List[ParallelResult]) -> Any:
        for result in results:
            if result.success:
                return result.result
        return None


class AllCompleteAggregator(ResultAggregator[List[Any]]):
    """모든 결과 리스트 반환"""
    
    def __init__(self, include_failures: bool = True):
        self.include_failures = include_failures
    
    def aggregate(self, results: List[ParallelResult]) -> List[Any]:
        if self.include_failures:
            return [r.result for r in results]
        return [r.result for r in results if r.success]


class MajorityVoteAggregator(ResultAggregator[Any]):
    """다수결 집계"""
    
    def __init__(self, key_func: Optional[Callable[[Any], Any]] = None):
        self.key_func = key_func or (lambda x: x)
    
    def aggregate(self, results: List[ParallelResult]) -> Any:
        votes: Dict[Any, int] = {}
        
        for result in results:
            if not result.success:
                continue
            
            key = self.key_func(result.result)
            votes[key] = votes.get(key, 0) + 1
        
        if not votes:
            return None
        
        return max(votes.keys(), key=lambda k: votes[k])


class WeightedAggregator(ResultAggregator[Any]):
    """가중치 기반 집계"""
    
    def __init__(self, weights: Dict[str, float]):
        self.weights = weights
    
    def aggregate(self, results: List[ParallelResult]) -> Any:
        weighted_results: List[Tuple[Any, float]] = []
        
        for result in results:
            if not result.success:
                continue
            
            weight = self.weights.get(result.agent_id, 1.0)
            weighted_results.append((result.result, weight))
        
        if not weighted_results:
            return None
        
        # 가장 높은 가중치 결과 반환
        return max(weighted_results, key=lambda x: x[1])[0]


# ============================================================================
# Concurrent Orchestrator
# ============================================================================

class ConcurrentOrchestrator:
    """
    동시 실행 오케스트레이터
    
    Fan-out/Fan-in 패턴을 사용한 병렬 에이전트 실행
    
    사용 예시:
        >>> orchestrator = ConcurrentOrchestrator()
        >>>
        >>> # 단순 병렬 실행
        >>> results = await orchestrator.fan_out(
        ...     agents=my_agents,
        ...     input_data={"query": "분석해주세요"}
        ... )
        >>>
        >>> # Map-Reduce 패턴
        >>> result = await orchestrator.map_reduce(
        ...     items=data_chunks,
        ...     map_func=process_chunk,
        ...     reduce_func=merge_results
        ... )
    """
    
    def __init__(self, config: Optional[FanOutConfig] = None):
        self.config = config or FanOutConfig()
        self._logger = StructuredLogger("concurrent_orchestrator")
        self._active_tasks: Dict[str, asyncio.Task] = {}
    
    async def fan_out(
        self,
        agents: List[Any],
        input_data: Dict[str, Any],
        config: Optional[FanOutConfig] = None,
        agent_configs: Optional[Dict[str, Dict[str, Any]]] = None,
    ) -> List[ParallelResult]:
        """
        Fan-out: 여러 에이전트 병렬 실행
        
        Args:
            agents: 에이전트 리스트 (또는 Callable 리스트)
            input_data: 공통 입력 데이터
            config: Fan-out 설정
            agent_configs: 에이전트별 추가 설정
            
        Returns:
            병렬 실행 결과 리스트
        """
        config = config or self.config
        agent_configs = agent_configs or {}
        
        execution_id = str(uuid.uuid4())[:8]
        start_time = time.time()
        
        self._logger.info(
            "Fan-out started",
            execution_id=execution_id,
            agent_count=len(agents),
            max_concurrency=config.max_concurrency
        )
        
        # 세마포어로 동시성 제어
        semaphore = asyncio.Semaphore(config.max_concurrency)
        
        async def execute_agent(agent, agent_id: str) -> ParallelResult:
            """개별 에이전트 실행"""
            async with semaphore:
                result = ParallelResult(
                    agent_id=agent_id,
                    agent_name=getattr(agent, 'name', agent_id),
                    success=False,
                    started_at=datetime.now(timezone.utc),
                )
                
                agent_config = agent_configs.get(agent_id, {})
                timeout = agent_config.get('timeout', config.per_agent_timeout)
                
                try:
                    agent_start = time.time()
                    
                    # 에이전트 실행 (다양한 인터페이스 지원)
                    if asyncio.iscoroutinefunction(agent):
                        output = await asyncio.wait_for(
                            agent(input_data),
                            timeout=timeout
                        )
                    elif hasattr(agent, 'execute'):
                        output = await asyncio.wait_for(
                            agent.execute(input_data),
                            timeout=timeout
                        )
                    elif hasattr(agent, 'run'):
                        output = await asyncio.wait_for(
                            agent.run(input_data),
                            timeout=timeout
                        )
                    elif callable(agent):
                        output = await asyncio.wait_for(
                            asyncio.to_thread(agent, input_data),
                            timeout=timeout
                        )
                    else:
                        raise TypeError(f"Agent {agent_id} is not callable")
                    
                    result.success = True
                    result.result = output
                    result.duration_ms = (time.time() - agent_start) * 1000
                    
                except asyncio.TimeoutError:
                    result.error = f"Timeout after {timeout}s"
                    
                except Exception as e:
                    result.error = str(e)
                    
                finally:
                    result.completed_at = datetime.now(timezone.utc)
                
                return result
        
        # 모든 에이전트 병렬 실행
        tasks = []
        for i, agent in enumerate(agents):
            agent_id = getattr(agent, 'id', None) or getattr(agent, 'name', None) or f"agent_{i}"
            task = asyncio.create_task(execute_agent(agent, agent_id))
            tasks.append(task)
            self._active_tasks[f"{execution_id}_{agent_id}"] = task
        
        # 전체 타임아웃 적용
        try:
            if config.fail_fast:
                # 첫 실패 시 중단
                results = []
                for task in asyncio.as_completed(tasks):
                    result = await asyncio.wait_for(task, timeout=config.timeout_seconds)
                    results.append(result)
                    if not result.success and config.fail_fast:
                        # 나머지 태스크 취소
                        for t in tasks:
                            if not t.done():
                                t.cancel()
                        break
            else:
                # 모든 태스크 완료 대기
                results = await asyncio.wait_for(
                    asyncio.gather(*tasks, return_exceptions=True),
                    timeout=config.timeout_seconds
                )
                # Exception을 ParallelResult로 변환
                results = [
                    r if isinstance(r, ParallelResult)
                    else ParallelResult(
                        agent_id=f"agent_{i}",
                        agent_name=f"agent_{i}",
                        success=False,
                        error=str(r)
                    )
                    for i, r in enumerate(results)
                ]
                
        except asyncio.TimeoutError:
            self._logger.warning("Fan-out timeout", execution_id=execution_id)
            results = []
            for task in tasks:
                if task.done():
                    try:
                        results.append(task.result())
                    except Exception as e:
                        results.append(ParallelResult(
                            agent_id="unknown",
                            agent_name="unknown",
                            success=False,
                            error=str(e)
                        ))
                else:
                    task.cancel()
        
        # 태스크 정리
        for key in list(self._active_tasks.keys()):
            if key.startswith(execution_id):
                del self._active_tasks[key]
        
        total_duration = (time.time() - start_time) * 1000
        success_count = sum(1 for r in results if r.success)
        
        self._logger.info(
            "Fan-out completed",
            execution_id=execution_id,
            total=len(results),
            success=success_count,
            duration_ms=total_duration
        )
        
        return results
    
    async def fan_in(
        self,
        results: List[ParallelResult],
        strategy: Optional[AggregationStrategy] = None,
        aggregator: Optional[ResultAggregator] = None,
        custom_func: Optional[Callable[[List[ParallelResult]], Any]] = None,
    ) -> AggregatedResult:
        """
        Fan-in: 결과 수집 및 집계
        
        Args:
            results: 병렬 실행 결과 리스트
            strategy: 집계 전략
            aggregator: 커스텀 집계기
            custom_func: 커스텀 집계 함수
            
        Returns:
            집계된 결과
        """
        strategy = strategy or self.config.strategy
        
        success_count = sum(1 for r in results if r.success)
        failure_count = len(results) - success_count
        total_duration = sum(r.duration_ms for r in results)
        
        aggregated_value = None
        
        # 집계 수행
        if aggregator:
            aggregated_value = aggregator.aggregate(results)
        elif custom_func:
            aggregated_value = custom_func(results)
        elif strategy == AggregationStrategy.FIRST:
            aggregator = FirstCompleteAggregator()
            aggregated_value = aggregator.aggregate(results)
        elif strategy == AggregationStrategy.ALL:
            aggregator = AllCompleteAggregator(include_failures=True)
            aggregated_value = aggregator.aggregate(results)
        elif strategy == AggregationStrategy.ALL_SUCCESS:
            aggregator = AllCompleteAggregator(include_failures=False)
            aggregated_value = aggregator.aggregate(results)
        elif strategy == AggregationStrategy.MAJORITY:
            aggregator = MajorityVoteAggregator()
            aggregated_value = aggregator.aggregate(results)
        elif strategy == AggregationStrategy.WEIGHTED:
            # 기본 가중치 (동일)
            weights = {r.agent_id: 1.0 for r in results}
            aggregator = WeightedAggregator(weights)
            aggregated_value = aggregator.aggregate(results)
        
        # 성공 여부 판단
        success = success_count >= self.config.min_success_count
        
        return AggregatedResult(
            success=success,
            strategy=strategy,
            results=results,
            aggregated_value=aggregated_value,
            total_duration_ms=total_duration,
            success_count=success_count,
            failure_count=failure_count,
        )
    
    async def fan_out_fan_in(
        self,
        agents: List[Any],
        input_data: Dict[str, Any],
        config: Optional[FanOutConfig] = None,
        aggregator: Optional[ResultAggregator] = None,
    ) -> AggregatedResult:
        """
        Fan-out/Fan-in 한 번에 실행
        
        Args:
            agents: 에이전트 리스트
            input_data: 입력 데이터
            config: 설정
            aggregator: 집계기
            
        Returns:
            집계된 결과
        """
        config = config or self.config
        results = await self.fan_out(agents, input_data, config)
        return await self.fan_in(results, config.strategy, aggregator)


# ============================================================================
# 고급 패턴
# ============================================================================

class MapReducePattern:
    """
    Map-Reduce 패턴
    
    데이터를 분할하여 병렬 처리 후 결과 합치기
    
    사용 예시:
        >>> pattern = MapReducePattern(orchestrator)
        >>> result = await pattern.execute(
        ...     items=large_dataset,
        ...     map_func=process_item,
        ...     reduce_func=merge_results,
        ...     chunk_size=100
        ... )
    """
    
    def __init__(
        self,
        orchestrator: Optional[ConcurrentOrchestrator] = None,
        config: Optional[FanOutConfig] = None,
    ):
        self.orchestrator = orchestrator or ConcurrentOrchestrator(config)
        self._logger = StructuredLogger("map_reduce")
    
    async def execute(
        self,
        items: List[Any],
        map_func: Callable[[Any], Coroutine[Any, Any, Any]],
        reduce_func: Callable[[List[Any]], Any],
        chunk_size: int = 10,
    ) -> Any:
        """
        Map-Reduce 실행
        
        Args:
            items: 처리할 아이템 리스트
            map_func: Map 함수 (각 아이템 처리)
            reduce_func: Reduce 함수 (결과 합치기)
            chunk_size: 청크 크기
            
        Returns:
            최종 결과
        """
        # 청크 분할
        chunks = [items[i:i + chunk_size] for i in range(0, len(items), chunk_size)]
        
        self._logger.info(
            "Map-Reduce started",
            total_items=len(items),
            chunks=len(chunks),
            chunk_size=chunk_size
        )
        
        # Map 단계: 각 청크 병렬 처리
        async def process_chunk(chunk_data: Dict[str, Any]) -> List[Any]:
            chunk = chunk_data["chunk"]
            results = []
            for item in chunk:
                result = await map_func(item)
                results.append(result)
            return results
        
        chunk_agents = [
            lambda data, c=chunk: process_chunk({"chunk": c})
            for chunk in chunks
        ]
        
        # Fan-out
        parallel_results = await self.orchestrator.fan_out(
            agents=chunk_agents,
            input_data={}
        )
        
        # Reduce 단계: 결과 합치기
        all_mapped = []
        for result in parallel_results:
            if result.success and result.result:
                all_mapped.extend(result.result)
        
        final_result = reduce_func(all_mapped)
        
        self._logger.info(
            "Map-Reduce completed",
            mapped_count=len(all_mapped),
        )
        
        return final_result


class ScatterGatherPattern:
    """
    Scatter-Gather 패턴
    
    요청을 여러 서비스에 분산하고 결과 수집
    
    사용 예시:
        >>> pattern = ScatterGatherPattern(orchestrator)
        >>> results = await pattern.execute(
        ...     request={"query": "검색어"},
        ...     services=[google_search, bing_search, duckduckgo],
        ...     timeout=10
        ... )
    """
    
    def __init__(
        self,
        orchestrator: Optional[ConcurrentOrchestrator] = None,
    ):
        self.orchestrator = orchestrator or ConcurrentOrchestrator()
        self._logger = StructuredLogger("scatter_gather")
    
    async def execute(
        self,
        request: Dict[str, Any],
        services: List[Callable],
        timeout: float = 30.0,
        min_responses: int = 1,
    ) -> AggregatedResult:
        """
        Scatter-Gather 실행
        
        Args:
            request: 요청 데이터
            services: 서비스 리스트 (Callable)
            timeout: 타임아웃
            min_responses: 최소 응답 수
            
        Returns:
            집계된 결과
        """
        config = FanOutConfig(
            timeout_seconds=timeout,
            per_agent_timeout=timeout,
            strategy=AggregationStrategy.ALL_SUCCESS,
            min_success_count=min_responses,
        )
        
        self._logger.info(
            "Scatter-Gather started",
            service_count=len(services),
            timeout=timeout
        )
        
        results = await self.orchestrator.fan_out(
            agents=services,
            input_data=request,
            config=config,
        )
        
        return await self.orchestrator.fan_in(
            results,
            strategy=AggregationStrategy.ALL_SUCCESS
        )


# ============================================================================
# 조건부 분기 Fan-out
# ============================================================================

class ConditionalFanOut:
    """
    조건부 Fan-out
    
    조건에 따라 다른 에이전트 집합 실행
    
    사용 예시:
        >>> fan_out = ConditionalFanOut()
        >>> fan_out.add_branch(
        ...     condition=lambda x: x["type"] == "code",
        ...     agents=[security_agent, performance_agent]
        ... )
        >>> fan_out.add_branch(
        ...     condition=lambda x: x["type"] == "text",
        ...     agents=[grammar_agent, style_agent]
        ... )
        >>> results = await fan_out.execute(input_data)
    """
    
    def __init__(self, orchestrator: Optional[ConcurrentOrchestrator] = None):
        self.orchestrator = orchestrator or ConcurrentOrchestrator()
        self._branches: List[Tuple[Callable, List[Any]]] = []
        self._default_agents: List[Any] = []
        self._logger = StructuredLogger("conditional_fan_out")
    
    def add_branch(
        self,
        condition: Callable[[Dict[str, Any]], bool],
        agents: List[Any],
    ):
        """조건부 브랜치 추가"""
        self._branches.append((condition, agents))
    
    def set_default(self, agents: List[Any]):
        """기본 에이전트 설정"""
        self._default_agents = agents
    
    async def execute(
        self,
        input_data: Dict[str, Any],
        config: Optional[FanOutConfig] = None,
    ) -> AggregatedResult:
        """
        조건부 Fan-out 실행
        
        Args:
            input_data: 입력 데이터
            config: Fan-out 설정
            
        Returns:
            집계된 결과
        """
        # 조건 평가하여 에이전트 선택
        selected_agents = []
        
        for condition, agents in self._branches:
            if condition(input_data):
                selected_agents.extend(agents)
                self._logger.debug(
                    "Branch matched",
                    agent_count=len(agents)
                )
        
        # 선택된 에이전트가 없으면 기본 사용
        if not selected_agents:
            selected_agents = self._default_agents
        
        if not selected_agents:
            return AggregatedResult(
                success=False,
                strategy=AggregationStrategy.ALL,
                results=[],
                aggregated_value=None,
            )
        
        self._logger.info(
            "Conditional fan-out",
            selected_count=len(selected_agents)
        )
        
        return await self.orchestrator.fan_out_fan_in(
            agents=selected_agents,
            input_data=input_data,
            config=config,
        )
