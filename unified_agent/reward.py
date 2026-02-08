#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Reward Emitter - 리워드 발행 및 추적 시스템

================================================================================
📋 역할: 에이전트 성능 평가를 위한 리워드 기록 및 추적
📅 버전: 3.3.0 (2026년 2월)
📦 영감: Microsoft Agent Lightning의 emit_reward 시스템
================================================================================

🎯 주요 기능:
    - 명시적 리워드 발행 (emit_reward)
    - 다차원 리워드 지원 (accuracy, latency, quality 등)
    - 리워드 스팬 자동 추적
    - 리워드-LLM 호출 매칭
    - 리워드 집계 및 분석

📌 사용 예시:
    >>> from unified_agent import emit_reward, reward, RewardManager
    >>>
    >>> # 간단한 리워드 발행
    >>> emit_reward(0.85)
    >>>
    >>> # 다차원 리워드
    >>> emit_reward({
    ...     "accuracy": 0.9,
    ...     "latency": 0.7,
    ...     "quality": 0.85
    ... })
    >>>
    >>> # 데코레이터로 함수 결과를 리워드로
    >>> @reward
    >>> def evaluate(response):
    ...     return calculate_score(response)
"""

from __future__ import annotations

import functools
import time
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Callable, Sequence, TypeVar

from pydantic import BaseModel, Field

from .tracer import Span, SpanKind, SpanStatus, AgentTracer, get_tracer
from .utils import StructuredLogger

# ============================================================================
# Reward 모델
# ============================================================================

class RewardType(str, Enum):
    """리워드 타입"""
    SCALAR = "scalar"           # 단일 수치
    MULTI_DIM = "multi_dim"     # 다차원
    BINARY = "binary"           # 0 또는 1
    RANKING = "ranking"         # 순위 기반

@dataclass(frozen=True, slots=True)
class RewardDimension:
    """리워드 차원 (다차원 리워드용)"""
    name: str
    value: float
    weight: float = 1.0
    description: str | None = None
    
    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "value": self.value,
            "weight": self.weight,
            "description": self.description,
        }

@dataclass(slots=True)
class RewardRecord:
    """
    리워드 기록
    
    하나의 리워드 발행에 대한 전체 정보.
    """
    reward_id: str = field(default_factory=lambda: uuid.uuid4().hex[:16])
    
    # 기본 값
    value: float = 0.0
    reward_type: RewardType = RewardType.SCALAR
    
    # 다차원 리워드
    dimensions: list[RewardDimension] = field(default_factory=list)
    
    # 컨텍스트
    rollout_id: str | None = None
    attempt_id: str | None = None
    span_id: str | None = None
    
    # 타임스탬프
    timestamp: float = field(default_factory=time.time)
    
    # 메타데이터
    metadata: dict[str, Any] = field(default_factory=dict)
    tags: list[str] = field(default_factory=list)
    
    @property
    def weighted_value(self) -> float:
        """가중 평균 값 (다차원인 경우)"""
        if not self.dimensions:
            return self.value
        
        total_weight = sum(d.weight for d in self.dimensions)
        if total_weight == 0:
            return 0.0
        
        return sum(d.value * d.weight for d in self.dimensions) / total_weight
    
    def to_dict(self) -> dict[str, Any]:
        return {
            "reward_id": self.reward_id,
            "value": self.value,
            "reward_type": self.reward_type.value,
            "dimensions": [d.to_dict() for d in self.dimensions],
            "rollout_id": self.rollout_id,
            "attempt_id": self.attempt_id,
            "span_id": self.span_id,
            "timestamp": self.timestamp,
            "metadata": self.metadata,
            "tags": self.tags,
            "weighted_value": self.weighted_value,
        }
    
    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "RewardRecord":
        record = cls(
            reward_id=data.get("reward_id", uuid.uuid4().hex[:16]),
            value=data.get("value", 0.0),
            reward_type=RewardType(data.get("reward_type", "scalar")),
            rollout_id=data.get("rollout_id"),
            attempt_id=data.get("attempt_id"),
            span_id=data.get("span_id"),
            timestamp=data.get("timestamp", time.time()),
            metadata=data.get("metadata", {}),
            tags=data.get("tags", []),
        )
        
        for dim_data in data.get("dimensions", []):
            record.dimensions.append(RewardDimension(
                name=dim_data["name"],
                value=dim_data["value"],
                weight=dim_data.get("weight", 1.0),
                description=dim_data.get("description"),
            ))
        
        return record

# ============================================================================
# Span Core Fields (emit 결과)
# ============================================================================

@dataclass(frozen=True, slots=True)
class SpanCoreFields:
    """스팬 핵심 필드 (emit 결과로 반환)"""
    span_id: str
    name: str
    kind: SpanKind
    start_time: float
    attributes: dict[str, Any] = field(default_factory=dict)
    
    def to_span(
        self,
        trace_id: str = "",
        rollout_id: str | None = None,
        attempt_id: str | None = None,
        sequence_id: int = 0,
    ) -> Span:
        """Span으로 변환"""
        span = Span(
            span_id=self.span_id,
            name=self.name,
            kind=self.kind,
            start_time=self.start_time,
            trace_id=trace_id,
            rollout_id=rollout_id,
            attempt_id=attempt_id,
            sequence_id=sequence_id,
            attributes=self.attributes,
        )
        span.end()
        return span

# ============================================================================
# Reward Emitter 함수들
# ============================================================================

def emit_reward(
    reward: float | dict[str, Any],
    *,
    primary_key: str | None = None,
    attributes: dict[str, Any] | None = None,
    propagate: bool = True,
) -> SpanCoreFields:
    """
    리워드 발행
    
    Agent Lightning의 emit_reward 함수 참고.
    
    Args:
        reward: 리워드 값 (float) 또는 다차원 리워드 (dict)
        primary_key: 주요 차원 키 (다차원인 경우)
        attributes: 추가 속성
        propagate: 트레이서에 전파 여부
        
    Returns:
        SpanCoreFields
        
    Examples:
        >>> # 단순 리워드
        >>> emit_reward(0.85)
        
        >>> # 다차원 리워드
        >>> emit_reward({
        ...     "accuracy": 0.9,
        ...     "latency": 0.7,
        ...     "quality": 0.85
        ... }, primary_key="accuracy")
    """
    span_id = uuid.uuid4().hex[:16]
    timestamp = time.time()
    
    # 속성 구성
    span_attrs: dict[str, Any] = {
        "reward.type": "reward",
        "reward.timestamp": timestamp,
    }
    
    if attributes:
        span_attrs.update(attributes)
    
    # 리워드 값 처리
    if isinstance(reward, (int, float)):
        span_attrs["reward.value"] = float(reward)
        span_attrs["reward.kind"] = "scalar"
    elif isinstance(reward, dict):
        # 다차원 리워드
        span_attrs["reward.kind"] = "multi_dim"
        span_attrs["reward.dimensions"] = list(reward.keys())
        
        for key, value in reward.items():
            span_attrs[f"reward.dim.{key}"] = float(value)
        
        # 주요 값 설정
        if primary_key and primary_key in reward:
            span_attrs["reward.value"] = float(reward[primary_key])
            span_attrs["reward.primary_key"] = primary_key
        else:
            # 평균값
            span_attrs["reward.value"] = sum(reward.values()) / len(reward)
    else:
        span_attrs["reward.value"] = 0.0
        span_attrs["reward.kind"] = "unknown"
    
    # SpanCoreFields 생성
    core_fields = SpanCoreFields(
        span_id=span_id,
        name="reward",
        kind=SpanKind.REWARD,
        start_time=timestamp,
        attributes=span_attrs,
    )
    
    # 트레이서에 전파
    if propagate:
        tracer = get_tracer()
        span = core_fields.to_span(
            trace_id=tracer.current_trace_id or "",
            rollout_id=tracer.current_rollout_id,
            attempt_id=tracer.current_attempt_id,
        )
        tracer.record_span(span)
    
    return core_fields

def emit_annotation(
    name: str,
    content: Any,
    *,
    attributes: dict[str, Any] | None = None,
    propagate: bool = True,
) -> SpanCoreFields:
    """
    주석/메타데이터 발행
    
    Args:
        name: 주석 이름
        content: 주석 내용
        attributes: 추가 속성
        propagate: 트레이서에 전파 여부
        
    Returns:
        SpanCoreFields
    """
    span_id = uuid.uuid4().hex[:16]
    timestamp = time.time()
    
    span_attrs: dict[str, Any] = {
        "annotation.name": name,
        "annotation.content": str(content)[:10000],  # 길이 제한
        "annotation.timestamp": timestamp,
    }
    
    if attributes:
        span_attrs.update(attributes)
    
    core_fields = SpanCoreFields(
        span_id=span_id,
        name=f"annotation:{name}",
        kind=SpanKind.ANNOTATION,
        start_time=timestamp,
        attributes=span_attrs,
    )
    
    if propagate:
        tracer = get_tracer()
        span = core_fields.to_span(
            trace_id=tracer.current_trace_id or "",
            rollout_id=tracer.current_rollout_id,
            attempt_id=tracer.current_attempt_id,
        )
        tracer.record_span(span)
    
    return core_fields

# ============================================================================
# Reward 데코레이터
# ============================================================================

F = TypeVar("F", bound=Callable[..., Any])

def reward(fn: F) -> F:
    """
    함수 결과를 리워드로 기록하는 데코레이터
    
    함수가 float 또는 dict를 반환하면 자동으로 emit_reward 호출.
    
    Args:
        fn: 데코레이팅할 함수
        
    Returns:
        래핑된 함수
        
    Examples:
        >>> @reward
        >>> def evaluate_response(response: str) -> float:
        ...     return calculate_score(response)
        >>>
        >>> score = evaluate_response("Hello")  # 자동으로 리워드 기록
    """
    @functools.wraps(fn)
    def wrapper(*args: Any, **kwargs: Any) -> Any:
        result = fn(*args, **kwargs)
        
        # 결과가 리워드로 사용 가능한 경우
        if isinstance(result, (int, float)):
            emit_reward(float(result), attributes={
                "reward.source": fn.__name__,
                "reward.decorated": True,
            })
        elif isinstance(result, dict) and all(
            isinstance(v, (int, float)) for v in result.values()
        ):
            emit_reward(result, attributes={
                "reward.source": fn.__name__,
                "reward.decorated": True,
            })
        
        return result
    
    return wrapper  # type: ignore

def reward_async(fn: F) -> F:
    """
    비동기 함수용 리워드 데코레이터
    """
    @functools.wraps(fn)
    async def wrapper(*args: Any, **kwargs: Any) -> Any:
        result = await fn(*args, **kwargs)
        
        if isinstance(result, (int, float)):
            emit_reward(float(result), attributes={
                "reward.source": fn.__name__,
                "reward.decorated": True,
            })
        elif isinstance(result, dict) and all(
            isinstance(v, (int, float)) for v in result.values()
        ):
            emit_reward(result, attributes={
                "reward.source": fn.__name__,
                "reward.decorated": True,
            })
        
        return result
    
    return wrapper  # type: ignore

# ============================================================================
# Reward Span 유틸리티
# ============================================================================

def is_reward_span(span: Span) -> bool:
    """스팬이 리워드 스팬인지 확인"""
    if span.kind == SpanKind.REWARD:
        return True
    
    if span.attributes:
        return span.attributes.get("reward.type") == "reward"
    
    return False

def get_reward_value(span: Span) -> float | None:
    """스팬에서 리워드 값 추출"""
    if not is_reward_span(span):
        return None
    
    if span.attributes:
        value = span.attributes.get("reward.value")
        if value is not None:
            return float(value)
    
    return None

def find_reward_spans(spans: Sequence[Span]) -> list[Span]:
    """리워드 스팬들 찾기"""
    return [s for s in spans if is_reward_span(s)]

def find_final_reward(spans: Sequence[Span]) -> float | None:
    """마지막 리워드 값 찾기"""
    reward_spans = find_reward_spans(spans)
    
    if not reward_spans:
        return None
    
    # 시퀀스 ID로 정렬하여 마지막 리워드
    reward_spans.sort(key=lambda s: s.sequence_id)
    last_span = reward_spans[-1]
    
    return get_reward_value(last_span)

def calculate_cumulative_reward(
    spans: Sequence[Span],
    discount_factor: float = 1.0,
) -> float:
    """누적 리워드 계산"""
    reward_spans = find_reward_spans(spans)
    
    if not reward_spans:
        return 0.0
    
    reward_spans.sort(key=lambda s: s.sequence_id)
    
    total = 0.0
    factor = 1.0
    
    for span in reversed(reward_spans):
        value = get_reward_value(span)
        if value is not None:
            total += value * factor
            factor *= discount_factor
    
    return total

# ============================================================================
# Reward Manager
# ============================================================================

class RewardManager:
    """
    리워드 관리자
    
    리워드 기록, 집계, 분석 기능 제공.
    """
    
    def __init__(
        self,
        tracer: AgentTracer | None = None,
    ):
        """
        Args:
            tracer: 사용할 트레이서 (없으면 전역 트레이서)
        """
        self._tracer = tracer
        self._logger = StructuredLogger("reward_manager")
        
        # 리워드 기록
        self._records: list[RewardRecord] = []
        
        # 집계
        self._total_rewards: float = 0.0
        self._reward_count: int = 0
        self._dimension_totals: dict[str, float] = {}
        self._dimension_counts: dict[str, int] = {}
    
    @property
    def tracer(self) -> AgentTracer:
        """트레이서"""
        return self._tracer or get_tracer()
    
    def emit(
        self,
        value: float | dict[str, float],
        *,
        tags: list[str] | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> RewardRecord:
        """
        리워드 발행 및 기록
        
        Args:
            value: 리워드 값
            tags: 태그
            metadata: 메타데이터
            
        Returns:
            RewardRecord
        """
        # 기록 생성
        record = RewardRecord(
            rollout_id=self.tracer.current_rollout_id,
            attempt_id=self.tracer.current_attempt_id,
            tags=tags or [],
            metadata=metadata or {},
        )
        
        # 값 처리
        if isinstance(value, (int, float)):
            record.value = float(value)
            record.reward_type = RewardType.SCALAR
        elif isinstance(value, dict):
            record.reward_type = RewardType.MULTI_DIM
            
            total = 0.0
            for name, v in value.items():
                dim = RewardDimension(name=name, value=float(v))
                record.dimensions.append(dim)
                total += float(v)
                
                # 차원별 집계
                self._dimension_totals[name] = (
                    self._dimension_totals.get(name, 0.0) + float(v)
                )
                self._dimension_counts[name] = (
                    self._dimension_counts.get(name, 0) + 1
                )
            
            record.value = total / len(value) if value else 0.0
        
        # 기록 저장
        self._records.append(record)
        
        # 집계 업데이트
        self._total_rewards += record.value
        self._reward_count += 1
        
        # 스팬 발행
        span_core = emit_reward(
            value,
            attributes={
                "reward.record_id": record.reward_id,
                **(metadata or {}),
            },
        )
        record.span_id = span_core.span_id
        
        self._logger.debug(
            "Reward emitted",
            reward_id=record.reward_id,
            value=record.value,
        )
        
        return record
    
    def get_records(
        self,
        rollout_id: str | None = None,
        tags: list[str] | None = None,
        limit: int = 1000,
    ) -> list[RewardRecord]:
        """
        리워드 기록 조회
        
        Args:
            rollout_id: 롤아웃 ID 필터
            tags: 태그 필터
            limit: 최대 개수
            
        Returns:
            RewardRecord 리스트
        """
        result = []
        
        for record in self._records:
            if rollout_id and record.rollout_id != rollout_id:
                continue
            
            if tags and not any(t in record.tags for t in tags):
                continue
            
            result.append(record)
            
            if len(result) >= limit:
                break
        
        return result
    
    @property
    def average_reward(self) -> float:
        """평균 리워드"""
        if self._reward_count == 0:
            return 0.0
        return self._total_rewards / self._reward_count
    
    @property
    def total_reward(self) -> float:
        """총 리워드"""
        return self._total_rewards
    
    @property
    def reward_count(self) -> int:
        """리워드 수"""
        return self._reward_count
    
    def get_dimension_average(self, dimension: str) -> float:
        """차원별 평균"""
        count = self._dimension_counts.get(dimension, 0)
        if count == 0:
            return 0.0
        return self._dimension_totals.get(dimension, 0.0) / count
    
    def get_statistics(self) -> dict[str, Any]:
        """통계 반환"""
        return {
            "total_reward": self._total_rewards,
            "reward_count": self._reward_count,
            "average_reward": self.average_reward,
            "dimensions": {
                name: {
                    "total": self._dimension_totals.get(name, 0.0),
                    "count": self._dimension_counts.get(name, 0),
                    "average": self.get_dimension_average(name),
                }
                for name in self._dimension_totals.keys()
            },
        }
    
    def reset(self) -> None:
        """집계 초기화"""
        self._records.clear()
        self._total_rewards = 0.0
        self._reward_count = 0
        self._dimension_totals.clear()
        self._dimension_counts.clear()

# ============================================================================
# 전역 RewardManager
# ============================================================================

_global_reward_manager: RewardManager | None = None

def get_reward_manager() -> RewardManager:
    """전역 RewardManager 가져오기"""
    global _global_reward_manager
    
    if _global_reward_manager is None:
        _global_reward_manager = RewardManager()
    
    return _global_reward_manager

def set_reward_manager(manager: RewardManager) -> None:
    """전역 RewardManager 설정"""
    global _global_reward_manager
    _global_reward_manager = manager
