#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Tracer 시스템 - OpenTelemetry 기반 Span 추적

================================================================================
📋 역할: 에이전트 실행 추적, LLM 호출 캡처, 도구 실행 모니터링
📅 버전: 3.3.0 (2026년 2월)
📦 영감: Microsoft Agent Lightning의 Tracer 시스템
================================================================================

🎯 주요 기능:
    - OpenTelemetry 표준 기반 Span 수집
    - LLM 호출 자동 캡처 (prompt, response, tokens)
    - 도구/함수 실행 추적
    - Rollout/Attempt 기반 트레이스 관리
    - 비동기 span 제출

📌 사용 예시:
    >>> from unified_agent import AgentTracer, SpanKind
    >>>
    >>> tracer = AgentTracer("my-agent")
    >>> await tracer.initialize()
    >>>
    >>> # 트레이스 컨텍스트 시작
    >>> async with tracer.trace_context("task-001", "attempt-1"):
    ...     # LLM 호출 추적
    ...     with tracer.span("llm_call", SpanKind.LLM):
    ...         response = await llm.chat(prompt)
    ...         tracer.set_attribute("tokens", response.usage.total_tokens)
    ...
    >>> # 스팬 조회
    >>> spans = tracer.get_last_trace()
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import threading
import time
import uuid
import weakref
from abc import ABC, abstractmethod
from contextlib import asynccontextmanager, contextmanager
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import (
    Any,
    Callable,
    ContextManager,
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

from .utils import StructuredLogger


# ============================================================================
# Span 관련 타입 정의
# ============================================================================

class SpanKind(str, Enum):
    """스팬 종류"""
    INTERNAL = "internal"      # 내부 처리
    LLM = "llm"               # LLM 호출
    TOOL = "tool"             # 도구 실행
    AGENT = "agent"           # 에이전트 실행
    WORKFLOW = "workflow"     # 워크플로우 단계
    REWARD = "reward"         # 리워드 기록
    ANNOTATION = "annotation" # 주석/메타데이터


class SpanStatus(str, Enum):
    """스팬 상태"""
    UNSET = "unset"
    OK = "ok"
    ERROR = "error"


class TraceStatus(str, Enum):
    """트레이스 상태"""
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


@dataclass
class SpanContext:
    """스팬 컨텍스트 (트레이스 연결 정보)"""
    trace_id: str
    span_id: str
    parent_span_id: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "trace_id": self.trace_id,
            "span_id": self.span_id,
            "parent_span_id": self.parent_span_id,
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "SpanContext":
        return cls(
            trace_id=data["trace_id"],
            span_id=data["span_id"],
            parent_span_id=data.get("parent_span_id"),
        )


@dataclass
class Span:
    """
    에이전트 실행 추적을 위한 Span
    
    OpenTelemetry의 Span 컨셉을 경량화하여 구현.
    Agent Lightning의 Span 구조 참고.
    """
    span_id: str
    name: str
    kind: SpanKind
    start_time: float
    end_time: Optional[float] = None
    status: SpanStatus = SpanStatus.UNSET
    
    # 컨텍스트
    trace_id: str = ""
    parent_span_id: Optional[str] = None
    rollout_id: Optional[str] = None
    attempt_id: Optional[str] = None
    sequence_id: int = 0
    
    # 속성 및 이벤트
    attributes: Dict[str, Any] = field(default_factory=dict)
    events: List[Dict[str, Any]] = field(default_factory=list)
    
    # 메타데이터
    agent_name: Optional[str] = None
    error_message: Optional[str] = None
    
    @property
    def duration_ms(self) -> Optional[float]:
        """실행 시간 (밀리초)"""
        if self.end_time is None:
            return None
        return (self.end_time - self.start_time) * 1000
    
    def set_attribute(self, key: str, value: Any) -> None:
        """속성 설정"""
        self.attributes[key] = value
    
    def add_event(self, name: str, attributes: Optional[Dict[str, Any]] = None) -> None:
        """이벤트 추가"""
        self.events.append({
            "name": name,
            "timestamp": time.time(),
            "attributes": attributes or {},
        })
    
    def set_status(self, status: SpanStatus, message: Optional[str] = None) -> None:
        """상태 설정"""
        self.status = status
        if message:
            self.error_message = message
    
    def end(self, end_time: Optional[float] = None) -> None:
        """스팬 종료"""
        self.end_time = end_time or time.time()
        if self.status == SpanStatus.UNSET:
            self.status = SpanStatus.OK
    
    def to_dict(self) -> Dict[str, Any]:
        """딕셔너리 변환"""
        return {
            "span_id": self.span_id,
            "name": self.name,
            "kind": self.kind.value,
            "start_time": self.start_time,
            "end_time": self.end_time,
            "duration_ms": self.duration_ms,
            "status": self.status.value,
            "trace_id": self.trace_id,
            "parent_span_id": self.parent_span_id,
            "rollout_id": self.rollout_id,
            "attempt_id": self.attempt_id,
            "sequence_id": self.sequence_id,
            "attributes": self.attributes,
            "events": self.events,
            "agent_name": self.agent_name,
            "error_message": self.error_message,
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Span":
        """딕셔너리에서 생성"""
        return cls(
            span_id=data["span_id"],
            name=data["name"],
            kind=SpanKind(data["kind"]),
            start_time=data["start_time"],
            end_time=data.get("end_time"),
            status=SpanStatus(data.get("status", "unset")),
            trace_id=data.get("trace_id", ""),
            parent_span_id=data.get("parent_span_id"),
            rollout_id=data.get("rollout_id"),
            attempt_id=data.get("attempt_id"),
            sequence_id=data.get("sequence_id", 0),
            attributes=data.get("attributes", {}),
            events=data.get("events", []),
            agent_name=data.get("agent_name"),
            error_message=data.get("error_message"),
        )


# ============================================================================
# Span Recording Context
# ============================================================================

class SpanRecordingContext:
    """스팬 기록 컨텍스트 (컨텍스트 매니저용)"""
    
    def __init__(
        self,
        tracer: "AgentTracer",
        span: Span,
    ):
        self._tracer = tracer
        self._span = span
        self._token: Optional[Any] = None
    
    @property
    def span(self) -> Span:
        return self._span
    
    def set_attribute(self, key: str, value: Any) -> None:
        self._span.set_attribute(key, value)
    
    def add_event(self, name: str, attributes: Optional[Dict[str, Any]] = None) -> None:
        self._span.add_event(name, attributes)
    
    def set_status(self, status: SpanStatus, message: Optional[str] = None) -> None:
        self._span.set_status(status, message)


# ============================================================================
# Tracer 베이스 클래스
# ============================================================================

class Tracer(ABC):
    """트레이서 추상 베이스 클래스"""
    
    @abstractmethod
    async def initialize(self) -> None:
        """초기화"""
        pass
    
    @abstractmethod
    def trace_context(
        self,
        rollout_id: str,
        attempt_id: str,
        **kwargs: Any,
    ) -> AsyncContextManager[Any]:
        """트레이스 컨텍스트 시작"""
        pass
    
    @abstractmethod
    def span(
        self,
        name: str,
        kind: SpanKind = SpanKind.INTERNAL,
        attributes: Optional[Dict[str, Any]] = None,
    ) -> ContextManager[SpanRecordingContext]:
        """스팬 시작"""
        pass
    
    @abstractmethod
    def get_last_trace(self) -> List[Span]:
        """마지막 트레이스의 스팬들 반환"""
        pass
    
    @abstractmethod
    def close(self) -> None:
        """리소스 정리"""
        pass


# ============================================================================
# Agent Tracer 구현
# ============================================================================

class AgentTracer(Tracer):
    """
    에이전트 트레이서
    
    Agent Lightning의 OtelTracer를 참고하여 구현.
    경량화된 OpenTelemetry 호환 트레이서.
    
    특징:
        - Rollout/Attempt 기반 트레이스 관리
        - 자동 시퀀스 ID 할당
        - 부모-자식 스팬 관계 추적
        - 스레드 안전 스팬 버퍼
    """
    
    def __init__(
        self,
        name: str = "agent-tracer",
        max_spans_per_trace: int = 10000,
        auto_flush: bool = True,
    ):
        """
        Args:
            name: 트레이서 이름
            max_spans_per_trace: 트레이스당 최대 스팬 수
            auto_flush: 자동 플러시 활성화
        """
        self._name = name
        self._max_spans = max_spans_per_trace
        self._auto_flush = auto_flush
        
        self._logger = StructuredLogger(f"tracer.{name}")
        
        # 현재 트레이스 상태
        self._current_trace_id: Optional[str] = None
        self._current_rollout_id: Optional[str] = None
        self._current_attempt_id: Optional[str] = None
        
        # 스팬 버퍼 (스레드 안전)
        self._spans: List[Span] = []
        self._span_stack: List[Span] = []  # 활성 스팬 스택
        self._sequence_counter: int = 0
        self._lock = threading.RLock()
        
        # 마지막 완료된 트레이스
        self._last_trace: List[Span] = []
        
        # 이벤트 루프 (비동기 제출용)
        self._loop: Optional[asyncio.AbstractEventLoop] = None
        self._initialized = False
        
        # 콜백
        self._on_span_end_callbacks: List[Callable[[Span], None]] = []
        self._on_trace_end_callbacks: List[Callable[[List[Span]], None]] = []
    
    @property
    def name(self) -> str:
        return self._name
    
    @property
    def current_trace_id(self) -> Optional[str]:
        return self._current_trace_id
    
    @property
    def current_rollout_id(self) -> Optional[str]:
        return self._current_rollout_id
    
    @property
    def current_attempt_id(self) -> Optional[str]:
        return self._current_attempt_id
    
    async def initialize(self) -> None:
        """트레이서 초기화"""
        if self._initialized:
            return
        
        try:
            self._loop = asyncio.get_running_loop()
        except RuntimeError:
            self._loop = None
        
        self._initialized = True
        self._logger.info("Tracer initialized", name=self._name)
    
    def _generate_id(self) -> str:
        """고유 ID 생성"""
        return uuid.uuid4().hex[:16]
    
    def _get_next_sequence(self) -> int:
        """다음 시퀀스 ID"""
        with self._lock:
            self._sequence_counter += 1
            return self._sequence_counter
    
    @asynccontextmanager
    async def trace_context(
        self,
        rollout_id: str,
        attempt_id: str,
        trace_id: Optional[str] = None,
        **kwargs: Any,
    ) -> AsyncGenerator[None, None]:
        """
        트레이스 컨텍스트 시작
        
        Args:
            rollout_id: 롤아웃 ID
            attempt_id: 어템프트 ID
            trace_id: 트레이스 ID (없으면 자동 생성)
            
        Yields:
            None (컨텍스트 내에서 span() 사용)
        """
        # 이전 트레이스 저장
        if self._spans:
            self._last_trace = list(self._spans)
            self._trigger_trace_end_callbacks(self._last_trace)
        
        # 새 트레이스 시작
        with self._lock:
            self._current_trace_id = trace_id or self._generate_id()
            self._current_rollout_id = rollout_id
            self._current_attempt_id = attempt_id
            self._spans = []
            self._span_stack = []
            self._sequence_counter = 0
        
        self._logger.debug(
            "Trace context started",
            trace_id=self._current_trace_id,
            rollout_id=rollout_id,
            attempt_id=attempt_id,
        )
        
        try:
            yield
        finally:
            # 트레이스 종료
            with self._lock:
                self._last_trace = list(self._spans)
                self._trigger_trace_end_callbacks(self._last_trace)
                
                self._logger.debug(
                    "Trace context ended",
                    trace_id=self._current_trace_id,
                    span_count=len(self._spans),
                )
                
                self._current_trace_id = None
                self._current_rollout_id = None
                self._current_attempt_id = None
    
    @contextmanager
    def span(
        self,
        name: str,
        kind: SpanKind = SpanKind.INTERNAL,
        attributes: Optional[Dict[str, Any]] = None,
    ) -> ContextManager[SpanRecordingContext]:
        """
        새 스팬 시작
        
        Args:
            name: 스팬 이름
            kind: 스팬 종류
            attributes: 초기 속성
            
        Yields:
            SpanRecordingContext
        """
        # 스팬 생성
        span = Span(
            span_id=self._generate_id(),
            name=name,
            kind=kind,
            start_time=time.time(),
            trace_id=self._current_trace_id or "",
            rollout_id=self._current_rollout_id,
            attempt_id=self._current_attempt_id,
            sequence_id=self._get_next_sequence(),
            attributes=attributes or {},
        )
        
        # 부모 스팬 연결
        with self._lock:
            if self._span_stack:
                span.parent_span_id = self._span_stack[-1].span_id
            self._span_stack.append(span)
        
        ctx = SpanRecordingContext(self, span)
        
        try:
            yield ctx
        except Exception as e:
            span.set_status(SpanStatus.ERROR, str(e))
            raise
        finally:
            # 스팬 종료
            span.end()
            
            with self._lock:
                if self._span_stack and self._span_stack[-1] is span:
                    self._span_stack.pop()
                
                if len(self._spans) < self._max_spans:
                    self._spans.append(span)
            
            # 콜백 호출
            self._trigger_span_end_callbacks(span)
    
    def create_span(
        self,
        name: str,
        kind: SpanKind = SpanKind.INTERNAL,
        attributes: Optional[Dict[str, Any]] = None,
        start_time: Optional[float] = None,
    ) -> Span:
        """
        스팬 직접 생성 (수동 관리용)
        
        Args:
            name: 스팬 이름
            kind: 스팬 종류
            attributes: 초기 속성
            start_time: 시작 시간
            
        Returns:
            생성된 Span
        """
        span = Span(
            span_id=self._generate_id(),
            name=name,
            kind=kind,
            start_time=start_time or time.time(),
            trace_id=self._current_trace_id or "",
            rollout_id=self._current_rollout_id,
            attempt_id=self._current_attempt_id,
            sequence_id=self._get_next_sequence(),
            attributes=attributes or {},
        )
        
        with self._lock:
            if self._span_stack:
                span.parent_span_id = self._span_stack[-1].span_id
        
        return span
    
    def record_span(self, span: Span) -> None:
        """
        스팬 기록 (수동 관리용)
        
        Args:
            span: 기록할 스팬
        """
        with self._lock:
            if len(self._spans) < self._max_spans:
                self._spans.append(span)
        
        self._trigger_span_end_callbacks(span)
    
    def get_last_trace(self) -> List[Span]:
        """마지막 완료된 트레이스 반환"""
        with self._lock:
            return list(self._last_trace)
    
    def get_current_spans(self) -> List[Span]:
        """현재 트레이스의 스팬들 반환"""
        with self._lock:
            return list(self._spans)
    
    def get_active_span(self) -> Optional[Span]:
        """현재 활성 스팬 반환"""
        with self._lock:
            return self._span_stack[-1] if self._span_stack else None
    
    def add_callback_on_span_end(self, callback: Callable[[Span], None]) -> None:
        """스팬 종료 콜백 등록"""
        self._on_span_end_callbacks.append(callback)
    
    def add_callback_on_trace_end(self, callback: Callable[[List[Span]], None]) -> None:
        """트레이스 종료 콜백 등록"""
        self._on_trace_end_callbacks.append(callback)
    
    def _trigger_span_end_callbacks(self, span: Span) -> None:
        """스팬 종료 콜백 실행"""
        for callback in self._on_span_end_callbacks:
            try:
                callback(span)
            except Exception as e:
                self._logger.error("Span end callback error", error=str(e))
    
    def _trigger_trace_end_callbacks(self, spans: List[Span]) -> None:
        """트레이스 종료 콜백 실행"""
        for callback in self._on_trace_end_callbacks:
            try:
                callback(spans)
            except Exception as e:
                self._logger.error("Trace end callback error", error=str(e))
    
    def close(self) -> None:
        """리소스 정리"""
        with self._lock:
            if self._spans:
                self._last_trace = list(self._spans)
            self._spans = []
            self._span_stack = []
        
        self._logger.info("Tracer closed", name=self._name)


# ============================================================================
# LLM Tracer (LLM 호출 전용 추적)
# ============================================================================

class LLMCallTracer:
    """
    LLM 호출 전용 트레이서
    
    LLM 호출에 특화된 추적 기능 제공.
    """
    
    def __init__(self, tracer: AgentTracer):
        """
        Args:
            tracer: 베이스 트레이서
        """
        self._tracer = tracer
    
    @contextmanager
    def trace_llm_call(
        self,
        model: str,
        prompt: Optional[str] = None,
        messages: Optional[List[Dict[str, str]]] = None,
        **kwargs: Any,
    ) -> ContextManager[SpanRecordingContext]:
        """
        LLM 호출 추적
        
        Args:
            model: 모델 이름
            prompt: 프롬프트 (단일)
            messages: 메시지 목록 (채팅)
            
        Yields:
            SpanRecordingContext
        """
        attributes = {
            "llm.model": model,
            "llm.request.type": "chat" if messages else "completion",
            **kwargs,
        }
        
        if prompt:
            # 프롬프트 해시 (개인정보 보호)
            attributes["llm.prompt.hash"] = hashlib.sha256(
                prompt.encode()
            ).hexdigest()[:16]
            attributes["llm.prompt.length"] = len(prompt)
        
        if messages:
            attributes["llm.messages.count"] = len(messages)
        
        with self._tracer.span("llm_call", SpanKind.LLM, attributes) as ctx:
            yield ctx
    
    def record_response(
        self,
        ctx: SpanRecordingContext,
        response: str,
        tokens: Optional[Dict[str, int]] = None,
        finish_reason: Optional[str] = None,
    ) -> None:
        """
        LLM 응답 기록
        
        Args:
            ctx: 스팬 컨텍스트
            response: 응답 텍스트
            tokens: 토큰 사용량 {"prompt": N, "completion": M, "total": K}
            finish_reason: 완료 이유
        """
        ctx.set_attribute("llm.response.length", len(response))
        ctx.set_attribute("llm.response.hash", hashlib.sha256(
            response.encode()
        ).hexdigest()[:16])
        
        if tokens:
            ctx.set_attribute("llm.tokens.prompt", tokens.get("prompt", 0))
            ctx.set_attribute("llm.tokens.completion", tokens.get("completion", 0))
            ctx.set_attribute("llm.tokens.total", tokens.get("total", 0))
        
        if finish_reason:
            ctx.set_attribute("llm.finish_reason", finish_reason)


# ============================================================================
# Tool Tracer (도구 실행 추적)
# ============================================================================

class ToolCallTracer:
    """
    도구 호출 전용 트레이서
    """
    
    def __init__(self, tracer: AgentTracer):
        self._tracer = tracer
    
    @contextmanager
    def trace_tool_call(
        self,
        tool_name: str,
        input_args: Optional[Dict[str, Any]] = None,
        **kwargs: Any,
    ) -> ContextManager[SpanRecordingContext]:
        """
        도구 호출 추적
        
        Args:
            tool_name: 도구 이름
            input_args: 입력 인자
            
        Yields:
            SpanRecordingContext
        """
        attributes = {
            "tool.name": tool_name,
            **kwargs,
        }
        
        if input_args:
            # 인자 요약 (크기 제한)
            args_str = json.dumps(input_args, ensure_ascii=False, default=str)
            if len(args_str) > 1000:
                args_str = args_str[:1000] + "..."
            attributes["tool.input.summary"] = args_str
            attributes["tool.input.keys"] = list(input_args.keys())
        
        with self._tracer.span(f"tool:{tool_name}", SpanKind.TOOL, attributes) as ctx:
            yield ctx
    
    def record_result(
        self,
        ctx: SpanRecordingContext,
        result: Any,
        success: bool = True,
    ) -> None:
        """
        도구 결과 기록
        
        Args:
            ctx: 스팬 컨텍스트
            result: 실행 결과
            success: 성공 여부
        """
        result_str = str(result)
        ctx.set_attribute("tool.output.length", len(result_str))
        ctx.set_attribute("tool.success", success)
        
        if not success:
            ctx.set_status(SpanStatus.ERROR, result_str[:500])


# ============================================================================
# Tracer Factory
# ============================================================================

def create_tracer(
    name: str = "default",
    max_spans: int = 10000,
) -> AgentTracer:
    """
    트레이서 팩토리
    
    Args:
        name: 트레이서 이름
        max_spans: 최대 스팬 수
        
    Returns:
        AgentTracer 인스턴스
    """
    return AgentTracer(name=name, max_spans_per_trace=max_spans)


# ============================================================================
# 전역 트레이서 관리
# ============================================================================

_global_tracer: Optional[AgentTracer] = None
_tracer_lock = threading.Lock()


def get_tracer(name: str = "global") -> AgentTracer:
    """전역 트레이서 가져오기 또는 생성"""
    global _global_tracer
    
    with _tracer_lock:
        if _global_tracer is None:
            _global_tracer = create_tracer(name)
        return _global_tracer


def set_tracer(tracer: AgentTracer) -> None:
    """전역 트레이서 설정"""
    global _global_tracer
    
    with _tracer_lock:
        _global_tracer = tracer


@asynccontextmanager
async def trace_context(
    tracer: Optional[AgentTracer] = None,
    name: str = "trace",
    rollout_id: Optional[str] = None,
    attempt_id: Optional[str] = None,
    **kwargs: Any,
) -> AsyncGenerator[SpanRecordingContext, None]:
    """
    전역 트레이스 컨텍스트 헬퍼
    
    Args:
        tracer: 사용할 트레이서 (없으면 전역 트레이서 사용)
        name: 트레이스 이름
        rollout_id: 롤아웃 ID
        attempt_id: 어템프트 ID
        **kwargs: 추가 속성
        
    Yields:
        SpanRecordingContext
        
    Example:
        >>> async with trace_context(tracer, "my_trace") as ctx:
        ...     print(f"Trace ID: {ctx.trace_id}")
    """
    _tracer = tracer or get_tracer()
    
    # 트레이스 ID 생성
    import uuid
    trace_id = uuid.uuid4().hex[:16]
    
    # 루트 스팬으로 트레이스 컨텍스트 시뮬레이션
    with _tracer.span(name, SpanKind.WORKFLOW, trace_id=trace_id) as root_span:
        # SpanRecordingContext에 trace_id 속성 추가
        root_span.trace_id = trace_id
        
        if rollout_id:
            root_span.set_attribute("rollout.id", rollout_id)
        if attempt_id:
            root_span.set_attribute("attempt.id", attempt_id)
        
        for key, value in kwargs.items():
            root_span.set_attribute(key, value)
        
        yield root_span


def current_span() -> Optional[Span]:
    """
    현재 활성 스팬 반환
    
    Returns:
        현재 스팬 또는 None
    """
    tracer = get_tracer()
    if tracer._span_stack:
        return tracer._span_stack[-1]
    return None