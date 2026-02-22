#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Durable Agent 시스템 - 장기 실행 워크플로우

================================================================================
📋 역할: 장기 실행 워크플로우를 위한 내구성 있는 에이전트 실행
📅 버전: 3.4.0 (2026년 2월)
📦 영감: Microsoft Durable Functions, Temporal.io
================================================================================

🎯 주요 기능:
    - 체크포인트 기반 상태 저장
    - 장애 복구 및 재시작
    - 타임아웃 관리
    - 워크플로우 버전 관리
    - 지속적 타이머
    - 활동(Activity) 재시도

📌 사용 시나리오:
    - 수 시간/일에 걸친 장기 워크플로우
    - 외부 승인 대기 (Human-in-the-loop)
    - 스케줄된 작업
    - 복잡한 다단계 처리

📌 사용 예시:
    >>> from unified_agent import DurableAgent, DurableContext, activity
    >>>
    >>> @activity(retry_count=3, timeout=60)
    >>> async def fetch_data(url: str) -> dict:
    ...     return await http_client.get(url)
    >>>
    >>> class DataPipelineAgent(DurableAgent):
    ...     async def run(self, ctx: DurableContext, input_data: dict):
    ...         # 체크포인트 저장
    ...         await ctx.checkpoint("fetching")
    ...         data = await ctx.call_activity(fetch_data, input_data["url"])
    ...         
    ...         # 타이머 대기
    ...         await ctx.create_timer(minutes=30)
    ...         
    ...         # 외부 이벤트 대기
    ...         approval = await ctx.wait_for_event("approval", timeout=86400)
    ...         
    ...         return {"status": "completed", "data": data}
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import pickle
import time
import uuid
from abc import ABC, abstractmethod
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from datetime import datetime, timezone, timedelta
from enum import Enum
from pathlib import Path
from typing import (
    Any,
    Callable,
    Coroutine,
    Generic,
    TypeVar,
)

from .utils import StructuredLogger

__all__ = [
    # 설정
    "DurableConfig",
    "ActivityConfig",
    "RetryPolicy",
    # 상태
    "WorkflowState",
    "WorkflowStatus",
    "CheckpointData",
    "ActivityResult",
    # 컨텍스트
    "DurableContext",
    # 에이전트
    "DurableAgent",
    "DurableOrchestrator",
    # 데코레이터
    "activity",
    "workflow",
    # 저장소
    "WorkflowStore",
    "FileWorkflowStore",
]

# ============================================================================
# 상태 및 설정
# ============================================================================

class WorkflowStatus(str, Enum):
    """워크플로우 상태"""
    PENDING = "pending"           # 대기 중
    RUNNING = "running"           # 실행 중
    SUSPENDED = "suspended"       # 일시 중단 (타이머/이벤트 대기)
    COMPLETED = "completed"       # 완료
    FAILED = "failed"             # 실패
    CANCELLED = "cancelled"       # 취소됨
    TIMED_OUT = "timed_out"       # 타임아웃

class ActivityStatus(str, Enum):
    """활동 상태"""
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    RETRYING = "retrying"

@dataclass(frozen=True, slots=True)
class RetryPolicy:
    """재시도 정책"""
    max_attempts: int = 3
    initial_delay_seconds: float = 1.0
    max_delay_seconds: float = 60.0
    backoff_multiplier: float = 2.0
    retryable_exceptions: tuple[type, ...] = (Exception,)
    
    def get_delay(self, attempt: int) -> float:
        """지수 백오프 지연 계산"""
        delay = self.initial_delay_seconds * (self.backoff_multiplier ** attempt)
        return min(delay, self.max_delay_seconds)

@dataclass(frozen=True, slots=True)
class ActivityConfig:
    """활동 설정"""
    name: str
    timeout_seconds: float = 300.0  # 5분
    retry_policy: RetryPolicy = field(default_factory=RetryPolicy)
    heartbeat_timeout_seconds: float = 30.0

@dataclass(frozen=True, slots=True)
class DurableConfig:
    """
    Durable Agent 설정
    
    Args:
        workflow_timeout_seconds: 전체 워크플로우 타임아웃
        checkpoint_interval_seconds: 체크포인트 간격
        storage_path: 상태 저장 경로
        enable_versioning: 버전 관리 활성화
        max_concurrent_activities: 최대 동시 활동 수
    """
    workflow_timeout_seconds: float = 86400.0  # 24시간
    checkpoint_interval_seconds: float = 60.0
    storage_path: str = "~/.durable_agent"
    enable_versioning: bool = True
    max_concurrent_activities: int = 10

@dataclass(frozen=True, slots=True)
class CheckpointData:
    """체크포인트 데이터"""
    checkpoint_id: str
    workflow_id: str
    checkpoint_name: str
    state: dict[str, Any]
    created_at: datetime
    metadata: dict[str, Any] = field(default_factory=dict)

@dataclass(slots=True)
class ActivityResult:
    """활동 실행 결과"""
    activity_name: str
    status: ActivityStatus
    result: Any | None = None
    error: str | None = None
    attempts: int = 0
    duration_ms: float = 0.0
    started_at: datetime | None = None
    completed_at: datetime | None = None

@dataclass(slots=True)
class WorkflowState:
    """
    워크플로우 상태
    
    실행 중인 워크플로우의 전체 상태를 저장
    """
    workflow_id: str
    workflow_name: str
    status: WorkflowStatus
    input_data: dict[str, Any]
    output_data: dict[str, Any] | None = None
    error: str | None = None
    
    # 실행 정보
    started_at: datetime | None = None
    completed_at: datetime | None = None
    last_checkpoint: str | None = None
    
    # 체크포인트 및 활동
    checkpoints: list[CheckpointData] = field(default_factory=list)
    activities: dict[str, ActivityResult] = field(default_factory=dict)
    
    # 대기 중인 이벤트/타이머
    pending_events: dict[str, datetime] = field(default_factory=dict)
    pending_timers: dict[str, datetime] = field(default_factory=dict)
    
    # 메타데이터
    version: str = "1.0"
    metadata: dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> dict[str, Any]:
        return {
            "workflow_id": self.workflow_id,
            "workflow_name": self.workflow_name,
            "status": self.status.value,
            "input_data": self.input_data,
            "output_data": self.output_data,
            "error": self.error,
            "started_at": self.started_at.isoformat() if self.started_at else None,
            "completed_at": self.completed_at.isoformat() if self.completed_at else None,
            "last_checkpoint": self.last_checkpoint,
            "checkpoint_count": len(self.checkpoints),
            "activity_count": len(self.activities),
            "version": self.version,
        }

# ============================================================================
# 워크플로우 저장소
# ============================================================================

class WorkflowStore(ABC):
    """워크플로우 저장소 추상 클래스"""
    
    @abstractmethod
    async def save(self, state: WorkflowState) -> None:
        """상태 저장"""
        pass
    
    @abstractmethod
    async def load(self, workflow_id: str) -> WorkflowState | None:
        """상태 로드"""
        pass
    
    @abstractmethod
    async def delete(self, workflow_id: str) -> bool:
        """상태 삭제"""
        pass
    
    @abstractmethod
    async def list_workflows(
        self,
        status: WorkflowStatus | None = None,
        limit: int = 100
    ) -> list[WorkflowState]:
        """워크플로우 목록 조회"""
        pass

class FileWorkflowStore(WorkflowStore):
    """파일 기반 워크플로우 저장소"""
    
    def __init__(self, storage_path: str):
        self._storage_dir = Path(storage_path).expanduser()
        self._storage_dir.mkdir(parents=True, exist_ok=True)
        self._lock = asyncio.Lock()
        self._logger = StructuredLogger("workflow_store")
    
    def _get_file_path(self, workflow_id: str) -> Path:
        return self._storage_dir / f"{workflow_id}.workflow"
    
    async def save(self, state: WorkflowState) -> None:
        file_path = self._get_file_path(state.workflow_id)
        
        async with self._lock:
            with open(file_path, 'wb') as f:
                pickle.dump(state, f)
        
        self._logger.debug("Workflow saved", workflow_id=state.workflow_id)
    
    async def load(self, workflow_id: str) -> WorkflowState | None:
        file_path = self._get_file_path(workflow_id)
        
        if not file_path.exists():
            return None
        
        try:
            async with self._lock:
                with open(file_path, 'rb') as f:
                    return pickle.load(f)
        except Exception as e:
            self._logger.error("Failed to load workflow", workflow_id=workflow_id, error=str(e))
            return None
    
    async def delete(self, workflow_id: str) -> bool:
        file_path = self._get_file_path(workflow_id)
        
        if file_path.exists():
            file_path.unlink()
            return True
        return False
    
    async def list_workflows(
        self,
        status: WorkflowStatus | None = None,
        limit: int = 100
    ) -> list[WorkflowState]:
        workflows = []
        
        for file_path in list(self._storage_dir.glob("*.workflow"))[:limit * 2]:
            state = await self.load(file_path.stem)
            if state:
                if status is None or state.status == status:
                    workflows.append(state)
                    if len(workflows) >= limit:
                        break
        
        return workflows

# ============================================================================
# Durable Context
# ============================================================================

class DurableContext:
    """
    Durable 실행 컨텍스트
    
    워크플로우 내에서 내구성 있는 작업을 수행하기 위한 컨텍스트
    
    주요 기능:
    1. 체크포인트 저장/복구
    2. 활동(Activity) 호출
    3. 타이머 생성
    4. 외부 이벤트 대기
    5. 서브 워크플로우 호출
    """
    
    def __init__(
        self,
        workflow_state: WorkflowState,
        store: WorkflowStore,
        config: DurableConfig,
    ):
        self._state = workflow_state
        self._store = store
        self._config = config
        self._logger = StructuredLogger("durable_context")
        self._activity_semaphore = asyncio.Semaphore(config.max_concurrent_activities)
        
        # 이벤트 큐
        self._event_queue: dict[str, asyncio.Queue] = {}
    
    @property
    def workflow_id(self) -> str:
        return self._state.workflow_id
    
    @property
    def is_replaying(self) -> bool:
        """재실행 중인지 여부 (체크포인트에서 복구 중)"""
        return self._state.last_checkpoint is not None
    
    async def checkpoint(self, name: str, state_data: dict[str, Any] | None = None):
        """
        체크포인트 저장
        
        Args:
            name: 체크포인트 이름
            state_data: 저장할 상태 데이터
        """
        checkpoint = CheckpointData(
            checkpoint_id=str(uuid.uuid4())[:8],
            workflow_id=self.workflow_id,
            checkpoint_name=name,
            state=state_data or {},
            created_at=datetime.now(timezone.utc),
        )
        
        self._state.checkpoints.append(checkpoint)
        self._state.last_checkpoint = name
        await self._store.save(self._state)
        
        self._logger.info("Checkpoint saved", name=name, workflow_id=self.workflow_id)
    
    async def call_activity(
        self,
        activity_func: Callable[..., Coroutine],
        *args,
        config: ActivityConfig | None = None,
        **kwargs,
    ) -> Any:
        """
        활동(Activity) 호출
        
        활동은 재시도 가능하고 타임아웃이 있는 작업 단위
        
        Args:
            activity_func: 활동 함수
            *args: 위치 인자
            config: 활동 설정
            **kwargs: 키워드 인자
            
        Returns:
            활동 결과
        """
        activity_name = getattr(activity_func, '_activity_name', activity_func.__name__)
        config = config or getattr(activity_func, '_activity_config', ActivityConfig(name=activity_name))
        
        # 이미 완료된 활동인지 체크 (재실행 시)
        if activity_name in self._state.activities:
            result = self._state.activities[activity_name]
            if result.status == ActivityStatus.COMPLETED:
                self._logger.debug("Replaying activity result", activity=activity_name)
                return result.result
        
        # 활동 실행
        async with self._activity_semaphore:
            result = await self._execute_activity(activity_func, config, *args, **kwargs)
        
        # 결과 저장
        self._state.activities[activity_name] = result
        await self._store.save(self._state)
        
        if result.status == ActivityStatus.FAILED:
            raise RuntimeError(f"Activity {activity_name} failed: {result.error}")
        
        return result.result
    
    async def _execute_activity(
        self,
        func: Callable[..., Coroutine],
        config: ActivityConfig,
        *args,
        **kwargs,
    ) -> ActivityResult:
        """활동 실행 (재시도 포함)"""
        result = ActivityResult(
            activity_name=config.name,
            status=ActivityStatus.PENDING,
            started_at=datetime.now(timezone.utc),
        )
        
        retry_policy = config.retry_policy
        
        for attempt in range(retry_policy.max_attempts):
            result.attempts = attempt + 1
            result.status = ActivityStatus.RUNNING if attempt == 0 else ActivityStatus.RETRYING
            
            try:
                start_time = time.time()
                
                # 타임아웃 적용
                output = await asyncio.wait_for(
                    func(*args, **kwargs),
                    timeout=config.timeout_seconds
                )
                
                result.result = output
                result.status = ActivityStatus.COMPLETED
                result.duration_ms = (time.time() - start_time) * 1000
                result.completed_at = datetime.now(timezone.utc)
                
                self._logger.info(
                    "Activity completed",
                    activity=config.name,
                    attempts=result.attempts,
                    duration_ms=result.duration_ms
                )
                
                return result
                
            except asyncio.TimeoutError:
                result.error = f"Timeout after {config.timeout_seconds}s"
                self._logger.warning(
                    "Activity timeout",
                    activity=config.name,
                    attempt=attempt + 1
                )
                
            except retry_policy.retryable_exceptions as e:
                result.error = str(e)
                self._logger.warning(
                    "Activity failed, retrying",
                    activity=config.name,
                    attempt=attempt + 1,
                    error=str(e)
                )
                
                if attempt < retry_policy.max_attempts - 1:
                    delay = retry_policy.get_delay(attempt)
                    await asyncio.sleep(delay)
        
        result.status = ActivityStatus.FAILED
        result.completed_at = datetime.now(timezone.utc)
        
        return result
    
    async def create_timer(
        self,
        seconds: float | None = None,
        minutes: float | None = None,
        hours: float | None = None,
        until: datetime | None = None,
    ):
        """
        지속적 타이머 생성
        
        워크플로우가 중단되더라도 지정된 시간에 재개
        
        Args:
            seconds: 대기 시간 (초)
            minutes: 대기 시간 (분)
            hours: 대기 시간 (시간)
            until: 대기할 시간 (절대 시간)
        """
        if until:
            fire_at = until
        else:
            total_seconds = (seconds or 0) + (minutes or 0) * 60 + (hours or 0) * 3600
            fire_at = datetime.now(timezone.utc) + timedelta(seconds=total_seconds)
        
        timer_id = str(uuid.uuid4())[:8]
        self._state.pending_timers[timer_id] = fire_at
        self._state.status = WorkflowStatus.SUSPENDED
        await self._store.save(self._state)
        
        # 실제 대기
        wait_seconds = (fire_at - datetime.now(timezone.utc)).total_seconds()
        if wait_seconds > 0:
            self._logger.info("Timer started", timer_id=timer_id, seconds=wait_seconds)
            await asyncio.sleep(wait_seconds)
        
        # 타이머 완료
        del self._state.pending_timers[timer_id]
        self._state.status = WorkflowStatus.RUNNING
        await self._store.save(self._state)
    
    async def wait_for_event(
        self,
        event_name: str,
        timeout_seconds: float | None = None,
    ) -> Any:
        """
        외부 이벤트 대기
        
        Human-in-the-loop, 외부 시스템 콜백 등에 사용
        
        Args:
            event_name: 이벤트 이름
            timeout_seconds: 타임아웃 (초)
            
        Returns:
            이벤트 데이터
        """
        timeout = timeout_seconds or self._config.workflow_timeout_seconds
        deadline = datetime.now(timezone.utc) + timedelta(seconds=timeout)
        
        self._state.pending_events[event_name] = deadline
        self._state.status = WorkflowStatus.SUSPENDED
        await self._store.save(self._state)
        
        # 이벤트 큐 생성
        if event_name not in self._event_queue:
            self._event_queue[event_name] = asyncio.Queue()
        
        self._logger.info("Waiting for event", event=event_name, timeout=timeout)
        
        try:
            event_data = await asyncio.wait_for(
                self._event_queue[event_name].get(),
                timeout=timeout
            )
            
            del self._state.pending_events[event_name]
            self._state.status = WorkflowStatus.RUNNING
            await self._store.save(self._state)
            
            return event_data
            
        except asyncio.TimeoutError:
            del self._state.pending_events[event_name]
            self._state.status = WorkflowStatus.TIMED_OUT
            await self._store.save(self._state)
            raise TimeoutError(f"Event {event_name} timed out after {timeout}s")
    
    async def raise_event(self, event_name: str, data: Any):
        """외부에서 이벤트 발생"""
        if event_name in self._event_queue:
            await self._event_queue[event_name].put(data)
            self._logger.info("Event raised", event=event_name)
    
    async def call_sub_workflow(
        self,
        workflow_class: type,
        input_data: dict[str, Any],
        workflow_id: str | None = None,
    ) -> Any:
        """
        서브 워크플로우 호출
        
        Args:
            workflow_class: 워크플로우 클래스
            input_data: 입력 데이터
            workflow_id: 워크플로우 ID (선택적)
            
        Returns:
            서브 워크플로우 결과
        """
        sub_id = workflow_id or f"{self.workflow_id}-sub-{uuid.uuid4().hex[:6]}"
        
        orchestrator = DurableOrchestrator(self._config, self._store)
        result = await orchestrator.start_workflow(
            workflow_class,
            input_data,
            workflow_id=sub_id
        )
        
        return result
    
    def get_state(self) -> WorkflowState:
        """현재 상태 조회"""
        return self._state

# ============================================================================
# Durable Agent (추상 클래스)
# ============================================================================

class DurableAgent(ABC):
    """
    Durable Agent 추상 클래스
    
    장기 실행 워크플로우를 구현하기 위한 기반 클래스
    
    사용 예시:
        >>> class MyWorkflow(DurableAgent):
        ...     async def run(self, ctx: DurableContext, input_data: dict):
        ...         await ctx.checkpoint("step1")
        ...         result = await ctx.call_activity(my_activity, input_data)
        ...         return {"result": result}
    """
    
    @property
    def name(self) -> str:
        """워크플로우 이름"""
        return self.__class__.__name__
    
    @property
    def version(self) -> str:
        """워크플로우 버전"""
        return getattr(self, '_version', '1.0')
    
    @abstractmethod
    async def run(self, ctx: DurableContext, input_data: dict[str, Any]) -> Any:
        """
        워크플로우 실행
        
        Args:
            ctx: Durable 컨텍스트
            input_data: 입력 데이터
            
        Returns:
            워크플로우 결과
        """
        pass
    
    async def on_error(self, ctx: DurableContext, error: Exception) -> Any | None:
        """
        에러 핸들러 (선택적 오버라이드)
        
        Args:
            ctx: Durable 컨텍스트
            error: 발생한 예외
            
        Returns:
            대체 결과 또는 None (재발생)
        """
        return None
    
    async def on_complete(self, ctx: DurableContext, result: Any):
        """완료 핸들러 (선택적 오버라이드)"""
        pass

# ============================================================================
# Durable Orchestrator
# ============================================================================

class DurableOrchestrator:
    """
    Durable 워크플로우 오케스트레이터
    
    워크플로우 실행, 재개, 조회 등 관리
    
    사용 예시:
        >>> orchestrator = DurableOrchestrator(config)
        >>> 
        >>> # 워크플로우 시작
        >>> result = await orchestrator.start_workflow(
        ...     MyWorkflow,
        ...     {"input": "data"}
        ... )
        >>>
        >>> # 워크플로우 재개
        >>> result = await orchestrator.resume_workflow(workflow_id)
        >>>
        >>> # 이벤트 발생
        >>> await orchestrator.raise_event(workflow_id, "approval", {"approved": True})
    """
    
    def __init__(
        self,
        config: DurableConfig | None = None,
        store: WorkflowStore | None = None,
    ):
        self.config = config or DurableConfig()
        self._store = store or FileWorkflowStore(self.config.storage_path)
        self._logger = StructuredLogger("durable_orchestrator")
        self._active_contexts: dict[str, DurableContext] = {}
    
    async def start_workflow(
        self,
        workflow_class: type,
        input_data: dict[str, Any],
        workflow_id: str | None = None,
    ) -> Any:
        """
        워크플로우 시작
        
        Args:
            workflow_class: 워크플로우 클래스
            input_data: 입력 데이터
            workflow_id: 워크플로우 ID (선택적)
            
        Returns:
            워크플로우 결과
        """
        workflow_id = workflow_id or str(uuid.uuid4())
        workflow = workflow_class()
        
        state = WorkflowState(
            workflow_id=workflow_id,
            workflow_name=workflow.name,
            status=WorkflowStatus.RUNNING,
            input_data=input_data,
            started_at=datetime.now(timezone.utc),
            version=workflow.version,
        )
        
        await self._store.save(state)
        
        self._logger.info(
            "Workflow started",
            workflow_id=workflow_id,
            workflow_name=workflow.name
        )
        
        return await self._execute_workflow(workflow, state)
    
    async def resume_workflow(self, workflow_id: str) -> Any:
        """
        중단된 워크플로우 재개
        
        Args:
            workflow_id: 워크플로우 ID
            
        Returns:
            워크플로우 결과
        """
        state = await self._store.load(workflow_id)
        
        if not state:
            raise ValueError(f"Workflow {workflow_id} not found")
        
        if state.status not in [WorkflowStatus.SUSPENDED, WorkflowStatus.PENDING]:
            raise ValueError(f"Workflow {workflow_id} cannot be resumed (status: {state.status})")
        
        # 워크플로우 클래스 찾기 (간단한 구현)
        # 실제로는 레지스트리에서 조회해야 함
        self._logger.info("Workflow resumed", workflow_id=workflow_id)
        
        state.status = WorkflowStatus.RUNNING
        await self._store.save(state)
        
        # 체크포인트에서 복구하여 실행
        # 여기서는 단순화를 위해 상태만 반환
        return state
    
    async def _execute_workflow(
        self,
        workflow: DurableAgent,
        state: WorkflowState,
    ) -> Any:
        """워크플로우 실행"""
        ctx = DurableContext(state, self._store, self.config)
        self._active_contexts[state.workflow_id] = ctx
        
        try:
            # 타임아웃 적용
            result = await asyncio.wait_for(
                workflow.run(ctx, state.input_data),
                timeout=self.config.workflow_timeout_seconds
            )
            
            state.status = WorkflowStatus.COMPLETED
            state.output_data = result if isinstance(result, dict) else {"result": result}
            state.completed_at = datetime.now(timezone.utc)
            
            await workflow.on_complete(ctx, result)
            
            self._logger.info(
                "Workflow completed",
                workflow_id=state.workflow_id,
                duration_ms=(state.completed_at - state.started_at).total_seconds() * 1000
            )
            
            return result
            
        except asyncio.TimeoutError:
            state.status = WorkflowStatus.TIMED_OUT
            state.error = f"Workflow timed out after {self.config.workflow_timeout_seconds}s"
            self._logger.error("Workflow timed out", workflow_id=state.workflow_id)
            raise
            
        except Exception as e:
            # 에러 핸들러 호출
            fallback = await workflow.on_error(ctx, e)
            
            if fallback is not None:
                state.status = WorkflowStatus.COMPLETED
                state.output_data = fallback if isinstance(fallback, dict) else {"result": fallback}
                return fallback
            
            state.status = WorkflowStatus.FAILED
            state.error = str(e)
            state.completed_at = datetime.now(timezone.utc)
            
            self._logger.error("Workflow failed", workflow_id=state.workflow_id, error=str(e))
            raise
            
        finally:
            await self._store.save(state)
            del self._active_contexts[state.workflow_id]
    
    async def raise_event(self, workflow_id: str, event_name: str, data: Any):
        """
        워크플로우에 이벤트 발생
        
        Args:
            workflow_id: 워크플로우 ID
            event_name: 이벤트 이름
            data: 이벤트 데이터
        """
        if workflow_id in self._active_contexts:
            ctx = self._active_contexts[workflow_id]
            await ctx.raise_event(event_name, data)
        else:
            self._logger.warning(
                "Workflow not active for event",
                workflow_id=workflow_id,
                event=event_name
            )
    
    async def cancel_workflow(self, workflow_id: str) -> bool:
        """워크플로우 취소"""
        state = await self._store.load(workflow_id)
        
        if not state:
            return False
        
        state.status = WorkflowStatus.CANCELLED
        state.completed_at = datetime.now(timezone.utc)
        await self._store.save(state)
        
        self._logger.info("Workflow cancelled", workflow_id=workflow_id)
        return True
    
    async def get_status(self, workflow_id: str) -> WorkflowState | None:
        """워크플로우 상태 조회"""
        return await self._store.load(workflow_id)
    
    async def list_workflows(
        self,
        status: WorkflowStatus | None = None,
        limit: int = 100
    ) -> list[WorkflowState]:
        """워크플로우 목록 조회"""
        return await self._store.list_workflows(status, limit)

# ============================================================================
# 데코레이터
# ============================================================================

def activity(
    name: str | None = None,
    timeout_seconds: float = 300.0,
    retry_count: int = 3,
    retry_delay: float = 1.0,
):
    """
    활동(Activity) 데코레이터
    
    사용 예시:
        >>> @activity(timeout=60, retry_count=3)
        >>> async def fetch_data(url: str) -> dict:
        ...     return await http_client.get(url)
    """
    def decorator(func):
        config = ActivityConfig(
            name=name or func.__name__,
            timeout_seconds=timeout_seconds,
            retry_policy=RetryPolicy(
                max_attempts=retry_count,
                initial_delay_seconds=retry_delay,
            ),
        )
        func._activity_name = config.name
        func._activity_config = config
        return func
    return decorator

def workflow(version: str = "1.0"):
    """
    워크플로우 데코레이터
    
    사용 예시:
        >>> @workflow(version="2.0")
        >>> class MyWorkflow(DurableAgent):
        ...     pass
    """
    def decorator(cls):
        cls._version = version
        return cls
    return decorator
