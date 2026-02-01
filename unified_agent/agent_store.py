#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Agent Store - 중앙 집중식 태스크/트레이스 저장소

================================================================================
📋 역할: Rollout/Attempt 관리, Span 저장, 리소스 버전 관리
📅 버전: 3.3.0 (2026년 2월)
📦 영감: Microsoft Agent Lightning의 LightningStore
================================================================================

🎯 주요 기능:
    - Rollout (작업 단위) 관리
    - Attempt (시도) 추적
    - Span 저장 및 조회
    - 리소스 (프롬프트, 모델 가중치) 버전 관리
    - 작업 큐잉 (분산 환경 지원)

📌 사용 예시:
    >>> from unified_agent import AgentStore, Rollout
    >>>
    >>> store = AgentStore()
    >>> await store.initialize()
    >>>
    >>> # 롤아웃 생성 및 큐잉
    >>> rollout = Rollout(task={"query": "Hello"})
    >>> await store.enqueue_rollout(rollout)
    >>>
    >>> # 작업 가져오기
    >>> work = await store.dequeue_rollout()
    >>>
    >>> # 스팬 저장
    >>> await store.add_span(span)
    >>>
    >>> # 스팬 조회
    >>> spans = await store.query_spans(rollout_id="...")
"""

from __future__ import annotations

import asyncio
import bisect
import hashlib
import json
import sqlite3
import threading
import time
import uuid
from abc import ABC, abstractmethod
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import (
    Any,
    Callable,
    Dict,
    Generic,
    List,
    Optional,
    Sequence,
    Tuple,
    TypeVar,
    Union,
)

from pydantic import BaseModel, Field

from .tracer import Span, SpanKind, SpanStatus
from .utils import StructuredLogger


# ============================================================================
# Rollout & Attempt 모델
# ============================================================================

class RolloutStatus(str, Enum):
    """롤아웃 상태"""
    PENDING = "pending"          # 대기 중
    QUEUED = "queued"           # 큐에 있음
    IN_PROGRESS = "in_progress" # 진행 중
    COMPLETED = "completed"     # 완료
    FAILED = "failed"           # 실패
    CANCELLED = "cancelled"     # 취소됨


class AttemptStatus(str, Enum):
    """어템프트 상태"""
    STARTED = "started"
    RUNNING = "running"
    IN_PROGRESS = "in_progress"  # 진행 중 (RUNNING과 동의어)
    FINISHED = "finished"
    COMPLETED = "completed"      # 성공 완료
    FAILED = "failed"


@dataclass
class Attempt:
    """
    롤아웃의 개별 시도
    
    하나의 Rollout은 여러 번의 Attempt를 가질 수 있음.
    (예: 실패 후 재시도)
    """
    attempt_id: str
    rollout_id: str
    status: AttemptStatus = AttemptStatus.STARTED
    started_at: float = field(default_factory=time.time)
    finished_at: Optional[float] = None
    worker_id: Optional[str] = None
    error_message: Optional[str] = None
    result: Optional[Dict[str, Any]] = None
    spans: List[Any] = field(default_factory=list)  # Span 리스트
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def start(self) -> None:
        """어템프트 시작"""
        self.status = AttemptStatus.IN_PROGRESS
        self.started_at = time.time()
    
    def complete(self, result: Optional[Dict[str, Any]] = None) -> None:
        """어템프트 성공 완료"""
        self.status = AttemptStatus.COMPLETED
        self.finished_at = time.time()
        self.result = result
    
    def fail(self, error: str) -> None:
        """어템프트 실패"""
        self.status = AttemptStatus.FAILED
        self.finished_at = time.time()
        self.error_message = error
    
    def add_span(self, span: Any) -> None:
        """스팬 추가"""
        self.spans.append(span)
    
    def finish(self, status: AttemptStatus, error: Optional[str] = None) -> None:
        """어템프트 종료 (레거시)"""
        self.status = status
        self.finished_at = time.time()
        if error:
            self.error_message = error
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "attempt_id": self.attempt_id,
            "rollout_id": self.rollout_id,
            "status": self.status.value,
            "started_at": self.started_at,
            "finished_at": self.finished_at,
            "worker_id": self.worker_id,
            "error_message": self.error_message,
            "metadata": self.metadata,
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Attempt":
        return cls(
            attempt_id=data["attempt_id"],
            rollout_id=data["rollout_id"],
            status=AttemptStatus(data["status"]),
            started_at=data["started_at"],
            finished_at=data.get("finished_at"),
            worker_id=data.get("worker_id"),
            error_message=data.get("error_message"),
            metadata=data.get("metadata", {}),
        )


T_task = TypeVar("T_task")


@dataclass
class Rollout(Generic[T_task]):
    """
    작업 단위 (Rollout)
    
    Agent Lightning의 Rollout 개념:
    - 하나의 태스크에 대한 에이전트 실행 단위
    - 여러 Attempt를 포함할 수 있음
    - 리소스 버전과 연결됨
    """
    rollout_id: str = field(default_factory=lambda: uuid.uuid4().hex[:16])
    task: Optional[T_task] = None
    status: RolloutStatus = RolloutStatus.PENDING
    created_at: float = field(default_factory=time.time)
    updated_at: float = field(default_factory=time.time)
    
    # 리소스 연결
    resources_id: Optional[str] = None
    
    # 어템프트 관리
    attempts: List[Attempt] = field(default_factory=list)
    max_attempts: int = 3
    
    # 메타데이터
    metadata: Dict[str, Any] = field(default_factory=dict)
    priority: int = 0  # 높을수록 우선
    tags: List[str] = field(default_factory=list)
    
    @property
    def current_attempt(self) -> Optional[Attempt]:
        """현재 어템프트"""
        return self.attempts[-1] if self.attempts else None
    
    @property
    def attempt_count(self) -> int:
        """시도 횟수"""
        return len(self.attempts)
    
    def create_attempt(self, worker_id: Optional[str] = None) -> Attempt:
        """새 어템프트 생성"""
        attempt = Attempt(
            attempt_id=uuid.uuid4().hex[:16],
            rollout_id=self.rollout_id,
            worker_id=worker_id,
        )
        self.attempts.append(attempt)
        self.status = RolloutStatus.IN_PROGRESS
        self.updated_at = time.time()
        return attempt
    
    def finish_attempt(
        self,
        success: bool,
        error: Optional[str] = None,
    ) -> None:
        """현재 어템프트 종료"""
        if not self.current_attempt:
            return
        
        status = AttemptStatus.FINISHED if success else AttemptStatus.FAILED
        self.current_attempt.finish(status, error)
        
        if success:
            self.status = RolloutStatus.COMPLETED
        elif self.attempt_count >= self.max_attempts:
            self.status = RolloutStatus.FAILED
        else:
            self.status = RolloutStatus.QUEUED  # 재시도 대기
        
        self.updated_at = time.time()
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "rollout_id": self.rollout_id,
            "task": self.task,
            "status": self.status.value,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
            "resources_id": self.resources_id,
            "attempts": [a.to_dict() for a in self.attempts],
            "max_attempts": self.max_attempts,
            "metadata": self.metadata,
            "priority": self.priority,
            "tags": self.tags,
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Rollout":
        rollout = cls(
            rollout_id=data["rollout_id"],
            task=data.get("task"),
            status=RolloutStatus(data["status"]),
            created_at=data["created_at"],
            updated_at=data["updated_at"],
            resources_id=data.get("resources_id"),
            max_attempts=data.get("max_attempts", 3),
            metadata=data.get("metadata", {}),
            priority=data.get("priority", 0),
            tags=data.get("tags", []),
        )
        rollout.attempts = [
            Attempt.from_dict(a) for a in data.get("attempts", [])
        ]
        return rollout


# ============================================================================
# Resource 모델
# ============================================================================

@dataclass
class NamedResource:
    """이름이 있는 리소스 (프롬프트, 모델 등)"""
    name: str
    resource_type: str  # "prompt", "model", "config" 등
    content: Any
    version: str = field(default_factory=lambda: uuid.uuid4().hex[:8])
    created_at: float = field(default_factory=time.time)
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "resource_type": self.resource_type,
            "content": self.content,
            "version": self.version,
            "created_at": self.created_at,
            "metadata": self.metadata,
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "NamedResource":
        return cls(
            name=data["name"],
            resource_type=data["resource_type"],
            content=data["content"],
            version=data["version"],
            created_at=data["created_at"],
            metadata=data.get("metadata", {}),
        )


@dataclass
class ResourceBundle:
    """리소스 번들 (여러 리소스의 스냅샷)"""
    bundle_id: str = field(default_factory=lambda: uuid.uuid4().hex[:16])
    resources: Dict[str, NamedResource] = field(default_factory=dict)
    created_at: float = field(default_factory=time.time)
    
    def get(self, name: str) -> Optional[NamedResource]:
        return self.resources.get(name)
    
    def set(self, resource: NamedResource) -> None:
        self.resources[resource.name] = resource
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "bundle_id": self.bundle_id,
            "resources": {
                name: r.to_dict() for name, r in self.resources.items()
            },
            "created_at": self.created_at,
        }


# ============================================================================
# Store 추상 베이스
# ============================================================================

class AgentStoreBase(ABC):
    """에이전트 스토어 추상 베이스"""
    
    @abstractmethod
    async def initialize(self) -> None:
        """초기화"""
        pass
    
    @abstractmethod
    async def close(self) -> None:
        """종료"""
        pass
    
    # Rollout 관리
    @abstractmethod
    async def enqueue_rollout(self, rollout: Rollout) -> None:
        """롤아웃 큐잉"""
        pass
    
    @abstractmethod
    async def dequeue_rollout(
        self,
        worker_id: Optional[str] = None,
        tags: Optional[List[str]] = None,
    ) -> Optional[Rollout]:
        """롤아웃 가져오기"""
        pass
    
    @abstractmethod
    async def update_rollout(self, rollout: Rollout) -> None:
        """롤아웃 업데이트"""
        pass
    
    @abstractmethod
    async def get_rollout(self, rollout_id: str) -> Optional[Rollout]:
        """롤아웃 조회"""
        pass
    
    # Attempt 관리
    @abstractmethod
    async def update_attempt(
        self,
        rollout_id: str,
        attempt_id: str,
        status: AttemptStatus,
        error: Optional[str] = None,
    ) -> None:
        """어템프트 상태 업데이트"""
        pass
    
    # Span 관리
    @abstractmethod
    async def add_span(self, span: Span) -> None:
        """스팬 추가"""
        pass
    
    @abstractmethod
    async def add_spans(self, spans: Sequence[Span]) -> None:
        """스팬 일괄 추가"""
        pass
    
    @abstractmethod
    async def query_spans(
        self,
        rollout_id: Optional[str] = None,
        attempt_id: Optional[str] = None,
        kind: Optional[SpanKind] = None,
        limit: int = 1000,
    ) -> List[Span]:
        """스팬 조회"""
        pass
    
    # Resource 관리
    @abstractmethod
    async def store_resource(self, resource: NamedResource) -> None:
        """리소스 저장"""
        pass
    
    @abstractmethod
    async def get_resource(
        self,
        name: str,
        version: Optional[str] = None,
    ) -> Optional[NamedResource]:
        """리소스 조회"""
        pass
    
    @abstractmethod
    async def get_latest_resources(self) -> ResourceBundle:
        """최신 리소스 번들"""
        pass


# ============================================================================
# In-Memory Store 구현
# ============================================================================

class InMemoryAgentStore(AgentStoreBase):
    """
    인메모리 에이전트 스토어
    
    개발/테스트용 경량 구현.
    Agent Lightning의 InMemoryLightningStore 참고.
    """
    
    def __init__(self, max_spans_per_rollout: int = 10000):
        """
        Args:
            max_spans_per_rollout: 롤아웃당 최대 스팬 수
        """
        self._max_spans = max_spans_per_rollout
        self._logger = StructuredLogger("agent_store.memory")
        
        # 저장소
        self._rollouts: Dict[str, Rollout] = {}
        self._rollout_queue: List[Tuple[int, str]] = []  # (-priority, rollout_id) for bisect
        self._spans: Dict[str, List[Span]] = defaultdict(list)  # rollout_id -> spans
        self._resources: Dict[str, List[NamedResource]] = defaultdict(list)  # name -> versions
        
        # 동기화
        self._lock = asyncio.Lock()
        self._queue_condition = asyncio.Condition()
        
        self._initialized = False
    
    async def initialize(self) -> None:
        """초기화"""
        self._initialized = True
        self._logger.info("InMemory AgentStore initialized")
    
    async def close(self) -> None:
        """종료"""
        self._rollouts.clear()
        self._rollout_queue.clear()
        self._spans.clear()
        self._resources.clear()
        self._logger.info("InMemory AgentStore closed")
    
    async def enqueue_rollout(self, rollout: Rollout) -> None:
        """롤아웃 큐잉 - O(log n) 삽입"""
        async with self._lock:
            rollout.status = RolloutStatus.QUEUED
            rollout.updated_at = time.time()
            self._rollouts[rollout.rollout_id] = rollout
            
            # bisect를 사용한 O(log n) 우선순위 삽입
            # -priority로 저장하여 높은 우선순위가 먼저 오도록
            bisect.insort(self._rollout_queue, (-rollout.priority, rollout.rollout_id))
        
        # 대기 중인 워커 깨우기
        async with self._queue_condition:
            self._queue_condition.notify()
        
        self._logger.debug(
            "Rollout enqueued",
            rollout_id=rollout.rollout_id,
            queue_size=len(self._rollout_queue),
        )
    
    async def dequeue_rollout(
        self,
        worker_id: Optional[str] = None,
        tags: Optional[List[str]] = None,
        timeout: Optional[float] = None,
    ) -> Optional[Rollout]:
        """롤아웃 가져오기 - 최적화된 버전"""
        start_time = time.time()
        tags_set = frozenset(tags) if tags else None
        
        while True:
            async with self._lock:
                # 조건에 맞는 롤아웃 찾기 (우선순위 높은 순)
                idx_to_remove = None
                for i, (neg_priority, rid) in enumerate(self._rollout_queue):
                    rollout = self._rollouts.get(rid)
                    if not rollout or rollout.status != RolloutStatus.QUEUED:
                        idx_to_remove = i
                        break
                    
                    # 태그 필터 (frozenset으로 O(1) 체크)
                    if tags_set and not tags_set.intersection(rollout.tags):
                        continue
                    
                    # 롤아웃 반환
                    self._rollout_queue.pop(i)
                    attempt = rollout.create_attempt(worker_id)
                    
                    self._logger.debug(
                        "Rollout dequeued",
                        rollout_id=rollout.rollout_id,
                        attempt_id=attempt.attempt_id,
                        worker_id=worker_id,
                    )
                    
                    return rollout
                
                # 유효하지 않은 항목 제거
                if idx_to_remove is not None:
                    self._rollout_queue.pop(idx_to_remove)
                    continue
            
            # 타임아웃 확인
            if timeout is not None:
                elapsed = time.time() - start_time
                if elapsed >= timeout:
                    return None
                remaining = timeout - elapsed
            else:
                remaining = None
            
            # 대기
            try:
                async with self._queue_condition:
                    await asyncio.wait_for(
                        self._queue_condition.wait(),
                        timeout=remaining,
                    )
            except asyncio.TimeoutError:
                return None
    
    async def update_rollout(self, rollout: Rollout) -> None:
        """롤아웃 업데이트"""
        async with self._lock:
            rollout.updated_at = time.time()
            self._rollouts[rollout.rollout_id] = rollout
    
    async def get_rollout(self, rollout_id: str) -> Optional[Rollout]:
        """롤아웃 조회"""
        async with self._lock:
            return self._rollouts.get(rollout_id)
    
    async def update_attempt(
        self,
        rollout_id: str,
        attempt_id: str,
        status: AttemptStatus,
        error: Optional[str] = None,
    ) -> None:
        """어템프트 상태 업데이트"""
        async with self._lock:
            rollout = self._rollouts.get(rollout_id)
            if not rollout:
                return
            
            for attempt in rollout.attempts:
                if attempt.attempt_id == attempt_id:
                    attempt.status = status
                    if status in (AttemptStatus.FINISHED, AttemptStatus.FAILED):
                        attempt.finished_at = time.time()
                    if error:
                        attempt.error_message = error
                    break
            
            rollout.updated_at = time.time()
    
    async def add_span(self, span: Span) -> None:
        """스팬 추가"""
        async with self._lock:
            key = span.rollout_id or "_default"
            spans = self._spans[key]
            
            if len(spans) < self._max_spans:
                spans.append(span)
    
    async def add_spans(self, spans: Sequence[Span]) -> None:
        """스팬 일괄 추가"""
        async with self._lock:
            for span in spans:
                key = span.rollout_id or "_default"
                span_list = self._spans[key]
                
                if len(span_list) < self._max_spans:
                    span_list.append(span)
    
    async def query_spans(
        self,
        rollout_id: Optional[str] = None,
        attempt_id: Optional[str] = None,
        kind: Optional[SpanKind] = None,
        limit: int = 1000,
    ) -> List[Span]:
        """스팬 조회"""
        async with self._lock:
            result: List[Span] = []
            
            if rollout_id:
                spans = self._spans.get(rollout_id, [])
            else:
                spans = []
                for span_list in self._spans.values():
                    spans.extend(span_list)
            
            for span in spans:
                if attempt_id and span.attempt_id != attempt_id:
                    continue
                if kind and span.kind != kind:
                    continue
                
                result.append(span)
                
                if len(result) >= limit:
                    break
            
            # sequence_id로 정렬
            result.sort(key=lambda s: s.sequence_id)
            
            return result
    
    async def get_next_span_sequence_id(
        self,
        rollout_id: str,
        attempt_id: str,
    ) -> int:
        """다음 스팬 시퀀스 ID"""
        async with self._lock:
            spans = self._spans.get(rollout_id, [])
            if not spans:
                return 1
            
            max_seq = max(
                s.sequence_id
                for s in spans
                if s.attempt_id == attempt_id
            ) if spans else 0
            
            return max_seq + 1
    
    async def store_resource(self, resource: NamedResource) -> None:
        """리소스 저장"""
        async with self._lock:
            self._resources[resource.name].append(resource)
    
    async def get_resource(
        self,
        name: str,
        version: Optional[str] = None,
    ) -> Optional[NamedResource]:
        """리소스 조회"""
        async with self._lock:
            versions = self._resources.get(name, [])
            if not versions:
                return None
            
            if version:
                for r in versions:
                    if r.version == version:
                        return r
                return None
            
            # 최신 버전 반환
            return max(versions, key=lambda r: r.created_at)
    
    async def get_latest_resources(self) -> ResourceBundle:
        """최신 리소스 번들"""
        async with self._lock:
            bundle = ResourceBundle()
            
            for name, versions in self._resources.items():
                if versions:
                    latest = max(versions, key=lambda r: r.created_at)
                    bundle.set(latest)
            
            return bundle
    
    # 추가 유틸리티 메서드
    
    async def get_queue_size(self) -> int:
        """큐 크기"""
        async with self._lock:
            return len(self._rollout_queue)
    
    async def get_rollouts_by_status(
        self,
        status: RolloutStatus,
    ) -> List[Rollout]:
        """상태별 롤아웃 조회"""
        async with self._lock:
            return [
                r for r in self._rollouts.values()
                if r.status == status
            ]
    
    async def clear_completed(self, older_than: Optional[float] = None) -> int:
        """완료된 롤아웃 정리"""
        async with self._lock:
            now = time.time()
            to_remove = []
            
            for rid, rollout in self._rollouts.items():
                if rollout.status in (RolloutStatus.COMPLETED, RolloutStatus.FAILED):
                    if older_than is None or (now - rollout.updated_at) > older_than:
                        to_remove.append(rid)
            
            for rid in to_remove:
                del self._rollouts[rid]
                self._spans.pop(rid, None)
            
            return len(to_remove)


# ============================================================================
# SQLite Store 구현
# ============================================================================

class SQLiteAgentStore(AgentStoreBase):
    """
    SQLite 기반 에이전트 스토어
    
    영속적인 저장이 필요한 환경용.
    """
    
    def __init__(
        self,
        db_path: Union[str, Path] = ":memory:",
        max_spans_per_rollout: int = 10000,
    ):
        """
        Args:
            db_path: SQLite 데이터베이스 경로
            max_spans_per_rollout: 롤아웃당 최대 스팬 수
        """
        self._db_path = str(db_path)
        self._max_spans = max_spans_per_rollout
        self._logger = StructuredLogger("agent_store.sqlite")
        
        self._conn: Optional[sqlite3.Connection] = None
        self._lock = asyncio.Lock()
        self._initialized = False
    
    async def initialize(self) -> None:
        """초기화"""
        self._conn = sqlite3.connect(self._db_path, check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        
        # 테이블 생성
        self._conn.executescript("""
            CREATE TABLE IF NOT EXISTS rollouts (
                rollout_id TEXT PRIMARY KEY,
                task_json TEXT,
                status TEXT NOT NULL,
                created_at REAL NOT NULL,
                updated_at REAL NOT NULL,
                resources_id TEXT,
                max_attempts INTEGER DEFAULT 3,
                metadata_json TEXT,
                priority INTEGER DEFAULT 0,
                tags_json TEXT
            );
            
            CREATE TABLE IF NOT EXISTS attempts (
                attempt_id TEXT PRIMARY KEY,
                rollout_id TEXT NOT NULL,
                status TEXT NOT NULL,
                started_at REAL NOT NULL,
                finished_at REAL,
                worker_id TEXT,
                error_message TEXT,
                metadata_json TEXT,
                FOREIGN KEY (rollout_id) REFERENCES rollouts(rollout_id)
            );
            
            CREATE TABLE IF NOT EXISTS spans (
                span_id TEXT PRIMARY KEY,
                rollout_id TEXT,
                attempt_id TEXT,
                name TEXT NOT NULL,
                kind TEXT NOT NULL,
                start_time REAL NOT NULL,
                end_time REAL,
                status TEXT,
                trace_id TEXT,
                parent_span_id TEXT,
                sequence_id INTEGER DEFAULT 0,
                attributes_json TEXT,
                events_json TEXT,
                agent_name TEXT,
                error_message TEXT
            );
            
            CREATE TABLE IF NOT EXISTS resources (
                name TEXT NOT NULL,
                version TEXT NOT NULL,
                resource_type TEXT NOT NULL,
                content_json TEXT,
                created_at REAL NOT NULL,
                metadata_json TEXT,
                PRIMARY KEY (name, version)
            );
            
            CREATE INDEX IF NOT EXISTS idx_rollouts_status ON rollouts(status);
            CREATE INDEX IF NOT EXISTS idx_rollouts_priority ON rollouts(priority DESC);
            CREATE INDEX IF NOT EXISTS idx_spans_rollout ON spans(rollout_id);
            CREATE INDEX IF NOT EXISTS idx_spans_attempt ON spans(attempt_id);
            CREATE INDEX IF NOT EXISTS idx_spans_sequence ON spans(sequence_id);
        """)
        
        self._conn.commit()
        self._initialized = True
        self._logger.info("SQLite AgentStore initialized", db_path=self._db_path)
    
    async def close(self) -> None:
        """종료"""
        if self._conn:
            self._conn.close()
            self._conn = None
        self._logger.info("SQLite AgentStore closed")
    
    async def enqueue_rollout(self, rollout: Rollout) -> None:
        """롤아웃 큐잉"""
        rollout.status = RolloutStatus.QUEUED
        rollout.updated_at = time.time()
        
        async with self._lock:
            self._conn.execute("""
                INSERT OR REPLACE INTO rollouts 
                (rollout_id, task_json, status, created_at, updated_at, 
                 resources_id, max_attempts, metadata_json, priority, tags_json)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                rollout.rollout_id,
                json.dumps(rollout.task),
                rollout.status.value,
                rollout.created_at,
                rollout.updated_at,
                rollout.resources_id,
                rollout.max_attempts,
                json.dumps(rollout.metadata),
                rollout.priority,
                json.dumps(rollout.tags),
            ))
            self._conn.commit()
    
    async def dequeue_rollout(
        self,
        worker_id: Optional[str] = None,
        tags: Optional[List[str]] = None,
        timeout: Optional[float] = None,
    ) -> Optional[Rollout]:
        """롤아웃 가져오기"""
        async with self._lock:
            # 조건에 맞는 롤아웃 찾기
            cursor = self._conn.execute("""
                SELECT * FROM rollouts 
                WHERE status = ?
                ORDER BY priority DESC, created_at ASC
                LIMIT 1
            """, (RolloutStatus.QUEUED.value,))
            
            row = cursor.fetchone()
            if not row:
                return None
            
            # Rollout 재구성
            rollout = self._row_to_rollout(dict(row))
            
            # Attempt 로드
            attempt_cursor = self._conn.execute("""
                SELECT * FROM attempts WHERE rollout_id = ?
                ORDER BY started_at ASC
            """, (rollout.rollout_id,))
            
            rollout.attempts = [
                self._row_to_attempt(dict(r))
                for r in attempt_cursor.fetchall()
            ]
            
            # 새 Attempt 생성
            attempt = rollout.create_attempt(worker_id)
            
            # 업데이트
            self._conn.execute("""
                UPDATE rollouts SET status = ?, updated_at = ?
                WHERE rollout_id = ?
            """, (rollout.status.value, rollout.updated_at, rollout.rollout_id))
            
            self._conn.execute("""
                INSERT INTO attempts
                (attempt_id, rollout_id, status, started_at, worker_id, metadata_json)
                VALUES (?, ?, ?, ?, ?, ?)
            """, (
                attempt.attempt_id,
                attempt.rollout_id,
                attempt.status.value,
                attempt.started_at,
                attempt.worker_id,
                json.dumps(attempt.metadata),
            ))
            
            self._conn.commit()
            
            return rollout
    
    def _row_to_rollout(self, row: Dict[str, Any]) -> Rollout:
        """Row를 Rollout으로 변환"""
        return Rollout(
            rollout_id=row["rollout_id"],
            task=json.loads(row["task_json"]) if row["task_json"] else None,
            status=RolloutStatus(row["status"]),
            created_at=row["created_at"],
            updated_at=row["updated_at"],
            resources_id=row.get("resources_id"),
            max_attempts=row.get("max_attempts", 3),
            metadata=json.loads(row["metadata_json"]) if row.get("metadata_json") else {},
            priority=row.get("priority", 0),
            tags=json.loads(row["tags_json"]) if row.get("tags_json") else [],
        )
    
    def _row_to_attempt(self, row: Dict[str, Any]) -> Attempt:
        """Row를 Attempt로 변환"""
        return Attempt(
            attempt_id=row["attempt_id"],
            rollout_id=row["rollout_id"],
            status=AttemptStatus(row["status"]),
            started_at=row["started_at"],
            finished_at=row.get("finished_at"),
            worker_id=row.get("worker_id"),
            error_message=row.get("error_message"),
            metadata=json.loads(row["metadata_json"]) if row.get("metadata_json") else {},
        )
    
    async def update_rollout(self, rollout: Rollout) -> None:
        """롤아웃 업데이트"""
        rollout.updated_at = time.time()
        
        async with self._lock:
            self._conn.execute("""
                UPDATE rollouts 
                SET task_json = ?, status = ?, updated_at = ?,
                    resources_id = ?, max_attempts = ?, metadata_json = ?,
                    priority = ?, tags_json = ?
                WHERE rollout_id = ?
            """, (
                json.dumps(rollout.task),
                rollout.status.value,
                rollout.updated_at,
                rollout.resources_id,
                rollout.max_attempts,
                json.dumps(rollout.metadata),
                rollout.priority,
                json.dumps(rollout.tags),
                rollout.rollout_id,
            ))
            self._conn.commit()
    
    async def get_rollout(self, rollout_id: str) -> Optional[Rollout]:
        """롤아웃 조회"""
        async with self._lock:
            cursor = self._conn.execute(
                "SELECT * FROM rollouts WHERE rollout_id = ?",
                (rollout_id,)
            )
            row = cursor.fetchone()
            
            if not row:
                return None
            
            rollout = self._row_to_rollout(dict(row))
            
            # Attempt 로드
            attempt_cursor = self._conn.execute("""
                SELECT * FROM attempts WHERE rollout_id = ?
                ORDER BY started_at ASC
            """, (rollout_id,))
            
            rollout.attempts = [
                self._row_to_attempt(dict(r))
                for r in attempt_cursor.fetchall()
            ]
            
            return rollout
    
    async def update_attempt(
        self,
        rollout_id: str,
        attempt_id: str,
        status: AttemptStatus,
        error: Optional[str] = None,
    ) -> None:
        """어템프트 상태 업데이트"""
        async with self._lock:
            finished_at = time.time() if status in (
                AttemptStatus.FINISHED, AttemptStatus.FAILED
            ) else None
            
            self._conn.execute("""
                UPDATE attempts 
                SET status = ?, finished_at = ?, error_message = ?
                WHERE attempt_id = ?
            """, (status.value, finished_at, error, attempt_id))
            
            self._conn.execute("""
                UPDATE rollouts SET updated_at = ? WHERE rollout_id = ?
            """, (time.time(), rollout_id))
            
            self._conn.commit()
    
    async def add_span(self, span: Span) -> None:
        """스팬 추가"""
        async with self._lock:
            self._conn.execute("""
                INSERT OR REPLACE INTO spans
                (span_id, rollout_id, attempt_id, name, kind, start_time, end_time,
                 status, trace_id, parent_span_id, sequence_id, attributes_json,
                 events_json, agent_name, error_message)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                span.span_id,
                span.rollout_id,
                span.attempt_id,
                span.name,
                span.kind.value,
                span.start_time,
                span.end_time,
                span.status.value,
                span.trace_id,
                span.parent_span_id,
                span.sequence_id,
                json.dumps(span.attributes),
                json.dumps(span.events),
                span.agent_name,
                span.error_message,
            ))
            self._conn.commit()
    
    async def add_spans(self, spans: Sequence[Span]) -> None:
        """스팬 일괄 추가"""
        async with self._lock:
            self._conn.executemany("""
                INSERT OR REPLACE INTO spans
                (span_id, rollout_id, attempt_id, name, kind, start_time, end_time,
                 status, trace_id, parent_span_id, sequence_id, attributes_json,
                 events_json, agent_name, error_message)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, [
                (
                    s.span_id, s.rollout_id, s.attempt_id, s.name, s.kind.value,
                    s.start_time, s.end_time, s.status.value, s.trace_id,
                    s.parent_span_id, s.sequence_id, json.dumps(s.attributes),
                    json.dumps(s.events), s.agent_name, s.error_message,
                )
                for s in spans
            ])
            self._conn.commit()
    
    async def query_spans(
        self,
        rollout_id: Optional[str] = None,
        attempt_id: Optional[str] = None,
        kind: Optional[SpanKind] = None,
        limit: int = 1000,
    ) -> List[Span]:
        """스팬 조회"""
        async with self._lock:
            conditions = []
            params = []
            
            if rollout_id:
                conditions.append("rollout_id = ?")
                params.append(rollout_id)
            
            if attempt_id:
                conditions.append("attempt_id = ?")
                params.append(attempt_id)
            
            if kind:
                conditions.append("kind = ?")
                params.append(kind.value)
            
            where_clause = " AND ".join(conditions) if conditions else "1=1"
            
            cursor = self._conn.execute(f"""
                SELECT * FROM spans 
                WHERE {where_clause}
                ORDER BY sequence_id ASC
                LIMIT ?
            """, params + [limit])
            
            return [self._row_to_span(dict(row)) for row in cursor.fetchall()]
    
    def _row_to_span(self, row: Dict[str, Any]) -> Span:
        """Row를 Span으로 변환"""
        return Span(
            span_id=row["span_id"],
            name=row["name"],
            kind=SpanKind(row["kind"]),
            start_time=row["start_time"],
            end_time=row.get("end_time"),
            status=SpanStatus(row["status"]) if row.get("status") else SpanStatus.UNSET,
            trace_id=row.get("trace_id", ""),
            parent_span_id=row.get("parent_span_id"),
            rollout_id=row.get("rollout_id"),
            attempt_id=row.get("attempt_id"),
            sequence_id=row.get("sequence_id", 0),
            attributes=json.loads(row["attributes_json"]) if row.get("attributes_json") else {},
            events=json.loads(row["events_json"]) if row.get("events_json") else [],
            agent_name=row.get("agent_name"),
            error_message=row.get("error_message"),
        )
    
    async def store_resource(self, resource: NamedResource) -> None:
        """리소스 저장"""
        async with self._lock:
            self._conn.execute("""
                INSERT OR REPLACE INTO resources
                (name, version, resource_type, content_json, created_at, metadata_json)
                VALUES (?, ?, ?, ?, ?, ?)
            """, (
                resource.name,
                resource.version,
                resource.resource_type,
                json.dumps(resource.content),
                resource.created_at,
                json.dumps(resource.metadata),
            ))
            self._conn.commit()
    
    async def get_resource(
        self,
        name: str,
        version: Optional[str] = None,
    ) -> Optional[NamedResource]:
        """리소스 조회"""
        async with self._lock:
            if version:
                cursor = self._conn.execute(
                    "SELECT * FROM resources WHERE name = ? AND version = ?",
                    (name, version)
                )
            else:
                cursor = self._conn.execute("""
                    SELECT * FROM resources 
                    WHERE name = ? 
                    ORDER BY created_at DESC 
                    LIMIT 1
                """, (name,))
            
            row = cursor.fetchone()
            if not row:
                return None
            
            return NamedResource(
                name=row["name"],
                version=row["version"],
                resource_type=row["resource_type"],
                content=json.loads(row["content_json"]) if row["content_json"] else None,
                created_at=row["created_at"],
                metadata=json.loads(row["metadata_json"]) if row.get("metadata_json") else {},
            )
    
    async def get_latest_resources(self) -> ResourceBundle:
        """최신 리소스 번들"""
        async with self._lock:
            cursor = self._conn.execute("""
                SELECT r1.* FROM resources r1
                INNER JOIN (
                    SELECT name, MAX(created_at) as max_created
                    FROM resources
                    GROUP BY name
                ) r2 ON r1.name = r2.name AND r1.created_at = r2.max_created
            """)
            
            bundle = ResourceBundle()
            
            for row in cursor.fetchall():
                resource = NamedResource(
                    name=row["name"],
                    version=row["version"],
                    resource_type=row["resource_type"],
                    content=json.loads(row["content_json"]) if row["content_json"] else None,
                    created_at=row["created_at"],
                    metadata=json.loads(row["metadata_json"]) if row.get("metadata_json") else {},
                )
                bundle.set(resource)
            
            return bundle


# ============================================================================
# Store Factory
# ============================================================================

def create_agent_store(
    store_type: str = "memory",
    **kwargs: Any,
) -> AgentStoreBase:
    """
    에이전트 스토어 팩토리
    
    Args:
        store_type: "memory" 또는 "sqlite"
        **kwargs: 스토어 생성자 인자
        
    Returns:
        AgentStoreBase 인스턴스
    """
    if store_type == "memory":
        return InMemoryAgentStore(**kwargs)
    elif store_type == "sqlite":
        return SQLiteAgentStore(**kwargs)
    else:
        raise ValueError(f"Unknown store type: {store_type}")


# 기본 스토어 (싱글톤)
_default_store: Optional[AgentStoreBase] = None


async def get_default_store() -> AgentStoreBase:
    """기본 스토어 가져오기 (비동기)"""
    global _default_store
    
    if _default_store is None:
        _default_store = InMemoryAgentStore()
        await _default_store.initialize()
    
    return _default_store


async def set_default_store(store: AgentStoreBase) -> None:
    """기본 스토어 설정 (비동기)"""
    global _default_store
    _default_store = store


def get_store() -> AgentStoreBase:
    """기본 스토어 가져오기 (동기)"""
    global _default_store
    
    if _default_store is None:
        _default_store = InMemoryAgentStore()
    
    return _default_store


def set_store(store: AgentStoreBase) -> None:
    """기본 스토어 설정 (동기)"""
    global _default_store
    _default_store = store
