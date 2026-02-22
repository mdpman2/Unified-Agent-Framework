#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Unified Agent Framework - 미들웨어 파이프라인 모듈 (Middleware Pipeline Module)

================================================================================
📁 파일 위치: unified_agent/middleware.py
📋 역할: 에이전트 요청/응답 처리 미들웨어 파이프라인
📅 최종 업데이트: 2026년 2월 13일
📦 버전: v4.1.0
✅ 테스트: test_v41_scenarios.py
================================================================================

🎯 주요 구성 요소:
    1. MiddlewareManager - 미들웨어 등록/실행 관리
    2. MiddlewareChain - 미들웨어 체인 (순차 실행)
    3. RequestMiddleware - 요청 전처리 미들웨어 기본 클래스
    4. ResponseMiddleware - 응답 후처리 미들웨어 기본 클래스
    5. BuiltinMiddlewares - 내장 미들웨어 (로깅, 인증, 레이트 리밋 등)

🔧 2026년 2월 기능:
    - Microsoft Agent Framework의 미들웨어 패턴 호환
    - 요청 전처리 / 응답 후처리 분리 파이프라인
    - 미들웨어 우선순위 및 조건부 실행
    - 에러 핸들링 미들웨어 (자동 재시도, 폴백)
    - 컨텍스트 기반 미들웨어 라우팅
    - 비동기 미들웨어 체인 (async pipeline)
    - 미들웨어 메트릭 (실행 시간, 호출 횟수 추적)

📌 사용 예시:
    >>> from unified_agent.middleware import (
    ...     MiddlewareManager, MiddlewareChain,
    ...     LoggingMiddleware, AuthMiddleware, RateLimitMiddleware,
    ...     RetryMiddleware, MiddlewareConfig
    ... )
    >>>
    >>> # 미들웨어 파이프라인 구성
    >>> manager = MiddlewareManager(MiddlewareConfig(
    ...     enable_metrics=True,
    ...     max_middleware_timeout=30.0
    ... ))
    >>>
    >>> # 내장 미들웨어 추가
    >>> manager.add(LoggingMiddleware(log_level="DEBUG"))
    >>> manager.add(AuthMiddleware(provider="entra_id"))
    >>> manager.add(RateLimitMiddleware(max_rpm=60, max_tpm=100000))
    >>> manager.add(RetryMiddleware(max_retries=3, backoff_factor=2.0))
    >>>
    >>> # 요청 처리
    >>> context = MiddlewareContext(agent_id="agent-1", request=user_request)
    >>> result = await manager.process(context)

⚠️ 주의사항:
    - 미들웨어 순서가 실행 결과에 영향을 줌 (순서 주의)
    - 무한 루프 방지를 위해 timeout을 설정하세요
    - 미들웨어 내에서 예외 발생 시 체인이 중단됩니다

🔗 관련 문서:
    - Microsoft Agent Framework Middleware: https://github.com/microsoft/agent-framework
    - ASP.NET Core Middleware Pattern: https://learn.microsoft.com/aspnet/core/fundamentals/middleware
"""

from __future__ import annotations

import asyncio
import logging
import time
import uuid
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime, timezone
from collections.abc import Callable
from enum import Enum, unique
from typing import Any

logger = logging.getLogger(__name__)

# ──────────────────────────────────────────────
# Public API
# ──────────────────────────────────────────────
__all__ = [
    # Enums
    "MiddlewarePhase",
    "MiddlewarePriority",
    "MiddlewareStatus",
    # Data Models
    "MiddlewareConfig",
    "MiddlewareContext",
    "MiddlewareResult",
    "MiddlewareMetrics",
    # Core Components
    "BaseMiddleware",
    "RequestMiddleware",
    "ResponseMiddleware",
    "MiddlewareChain",
    "MiddlewareManager",
    # Built-in Middlewares
    "LoggingMiddleware",
    "AuthMiddleware",
    "RateLimitMiddleware",
    "RetryMiddleware",
    "ContentFilterMiddleware",
    "CacheMiddleware",
]


# ══════════════════════════════════════════════
# Enums
# ══════════════════════════════════════════════


@unique
class MiddlewarePhase(str, Enum):
    """미들웨어 실행 단계 (Middleware execution phase)"""

    PRE_REQUEST = "pre_request"  # 요청 전처리
    POST_REQUEST = "post_request"  # 요청 후처리
    PRE_RESPONSE = "pre_response"  # 응답 전처리
    POST_RESPONSE = "post_response"  # 응답 후처리
    ON_ERROR = "on_error"  # 에러 발생 시


@unique
class MiddlewarePriority(int, Enum):
    """미들웨어 실행 우선순위 (낮은 값 = 높은 우선순위)"""

    CRITICAL = 0  # 인증, 보안 (가장 먼저 실행)
    HIGH = 10  # 레이트 리밋, 필터링
    NORMAL = 50  # 일반 처리
    LOW = 90  # 로깅, 메트릭
    LAST = 100  # 최종 처리 (캐싱 등)


@unique
class MiddlewareStatus(str, Enum):
    """미들웨어 실행 상태"""

    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    SKIPPED = "skipped"
    FAILED = "failed"
    TIMED_OUT = "timed_out"


# ══════════════════════════════════════════════
# Data Models
# ══════════════════════════════════════════════


@dataclass(slots=True)
class MiddlewareConfig:
    """미들웨어 파이프라인 설정"""

    enable_metrics: bool = True  # 미들웨어 메트릭 수집 활성화
    max_middleware_timeout: float = 30.0  # 개별 미들웨어 타임아웃 (초)
    pipeline_timeout: float = 120.0  # 전체 파이프라인 타임아웃 (초)
    stop_on_error: bool = False  # 에러 시 체인 중단 여부
    enable_retry_on_error: bool = True  # 에러 발생 시 에러 미들웨어 실행
    max_context_size_mb: float = 10.0  # 컨텍스트 최대 크기 (MB)


@dataclass(slots=True)
class MiddlewareContext:
    """미들웨어 파이프라인 컨텍스트 (요청/응답 데이터 전달용)"""

    request_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    agent_id: str = ""
    request: Any = None  # 원본 요청
    response: Any = None  # 처리된 응답
    metadata: dict[str, Any] = field(default_factory=dict)
    errors: list[Exception] = field(default_factory=list)
    timestamps: dict[str, float] = field(default_factory=dict)
    cancelled: bool = False

    # 미들웨어 간 공유 상태
    shared_state: dict[str, Any] = field(default_factory=dict)

    def set(self, key: str, value: Any) -> None:
        """공유 상태에 값 설정"""
        self.shared_state[key] = value

    def get(self, key: str, default: Any = None) -> Any:
        """공유 상태에서 값 조회"""
        return self.shared_state.get(key, default)

    def cancel(self) -> None:
        """파이프라인 실행 취소"""
        self.cancelled = True

    @property
    def has_errors(self) -> bool:
        return len(self.errors) > 0


@dataclass(slots=True)
class MiddlewareResult:
    """미들웨어 실행 결과"""

    middleware_name: str
    phase: MiddlewarePhase
    status: MiddlewareStatus
    duration_ms: float = 0.0
    error: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class MiddlewareMetrics:
    """미들웨어 실행 메트릭"""

    middleware_name: str
    total_calls: int = 0
    successful_calls: int = 0
    failed_calls: int = 0
    skipped_calls: int = 0
    total_duration_ms: float = 0.0
    avg_duration_ms: float = 0.0
    max_duration_ms: float = 0.0
    min_duration_ms: float = float("inf")
    last_called: datetime | None = None

    def record(self, duration_ms: float, success: bool) -> None:
        """메트릭 기록"""
        self.total_calls += 1
        if success:
            self.successful_calls += 1
        else:
            self.failed_calls += 1
        self.total_duration_ms += duration_ms
        self.avg_duration_ms = self.total_duration_ms / self.total_calls
        self.max_duration_ms = max(self.max_duration_ms, duration_ms)
        self.min_duration_ms = min(self.min_duration_ms, duration_ms)
        self.last_called = datetime.now(timezone.utc)


# ══════════════════════════════════════════════
# Core Components - 미들웨어 기본 클래스
# ══════════════════════════════════════════════


class BaseMiddleware(ABC):
    """미들웨어 기본 추상 클래스

    모든 미들웨어는 이 클래스를 상속해야 합니다.
    """

    def __init__(
        self,
        name: str | None = None,
        priority: MiddlewarePriority = MiddlewarePriority.NORMAL,
        phases: list[MiddlewarePhase] | None = None,
        condition: Callable[[MiddlewareContext], bool] | None = None,
    ):
        self.name = name or self.__class__.__name__
        self.priority = priority
        self.phases = phases or [
            MiddlewarePhase.PRE_REQUEST,
            MiddlewarePhase.POST_RESPONSE,
        ]
        self.condition = condition  # 조건부 실행 함수
        self.enabled = True

    @abstractmethod
    async def process(
        self, context: MiddlewareContext, next_fn: Callable
    ) -> MiddlewareContext:
        """미들웨어 처리 로직 (구현 필수)

        Args:
            context: 미들웨어 컨텍스트
            next_fn: 다음 미들웨어 호출 함수

        Returns:
            처리된 컨텍스트
        """
        ...

    def should_execute(self, context: MiddlewareContext) -> bool:
        """미들웨어 실행 조건 확인"""
        if not self.enabled:
            return False
        if self.condition and not self.condition(context):
            return False
        return True


class RequestMiddleware(BaseMiddleware):
    """요청 전처리 전용 미들웨어"""

    def __init__(self, **kwargs):
        super().__init__(
            phases=[MiddlewarePhase.PRE_REQUEST, MiddlewarePhase.POST_REQUEST],
            **kwargs,
        )


class ResponseMiddleware(BaseMiddleware):
    """응답 후처리 전용 미들웨어"""

    def __init__(self, **kwargs):
        super().__init__(
            phases=[MiddlewarePhase.PRE_RESPONSE, MiddlewarePhase.POST_RESPONSE],
            **kwargs,
        )


# ══════════════════════════════════════════════
# Core Components - 미들웨어 체인
# ══════════════════════════════════════════════


class MiddlewareChain:
    """미들웨어 체인 (순차 실행 파이프라인)

    등록된 미들웨어를 우선순위 순으로 체인 형태로 실행합니다.
    각 미들웨어는 next_fn을 호출하여 다음 미들웨어로 전달합니다.
    """

    def __init__(self, config: MiddlewareConfig | None = None):
        self.config = config or MiddlewareConfig()
        self._middlewares: list[BaseMiddleware] = []
        self._metrics: dict[str, MiddlewareMetrics] = {}
        self._results: list[MiddlewareResult] = []

    def add(self, middleware: BaseMiddleware) -> MiddlewareChain:
        """미들웨어 추가 (우선순위 순으로 자동 정렬)"""
        self._middlewares.append(middleware)
        self._middlewares.sort(key=lambda m: m.priority.value)
        if self.config.enable_metrics:
            self._metrics[middleware.name] = MiddlewareMetrics(
                middleware_name=middleware.name
            )
        logger.info(
            "미들웨어 추가: %s (우선순위: %s)", middleware.name, middleware.priority.name
        )
        return self

    def remove(self, name: str) -> bool:
        """미들웨어 제거"""
        before = len(self._middlewares)
        self._middlewares = [m for m in self._middlewares if m.name != name]
        removed = len(self._middlewares) < before
        if removed:
            self._metrics.pop(name, None)
            logger.info("미들웨어 제거: %s", name)
        return removed

    async def execute(
        self,
        context: MiddlewareContext,
        phase: MiddlewarePhase = MiddlewarePhase.PRE_REQUEST,
    ) -> MiddlewareContext:
        """미들웨어 체인 실행

        Args:
            context: 미들웨어 컨텍스트
            phase: 실행할 미들웨어 단계

        Returns:
            처리된 컨텍스트
        """
        context.timestamps[f"{phase.value}_start"] = time.time()
        self._results.clear()

        # 해당 단계에 해당하는 미들웨어만 필터링
        phase_middlewares = [
            m for m in self._middlewares if phase in m.phases and m.should_execute(context)
        ]

        if not phase_middlewares:
            logger.debug("실행할 미들웨어 없음 (phase=%s)", phase.value)
            return context

        # 체인 실행 (역순으로 next_fn 구성)
        async def _terminal(ctx: MiddlewareContext) -> MiddlewareContext:
            return ctx

        chain_fn = _terminal

        for middleware in reversed(phase_middlewares):
            chain_fn = self._wrap_middleware(middleware, phase, chain_fn)

        try:
            context = await asyncio.wait_for(
                chain_fn(context), timeout=self.config.pipeline_timeout
            )
        except asyncio.TimeoutError:
            logger.error("미들웨어 파이프라인 타임아웃 (%.1fs)", self.config.pipeline_timeout)
            context.errors.append(
                TimeoutError(
                    f"Pipeline timeout after {self.config.pipeline_timeout}s"
                )
            )

        context.timestamps[f"{phase.value}_end"] = time.time()
        return context

    def _wrap_middleware(
        self,
        middleware: BaseMiddleware,
        phase: MiddlewarePhase,
        next_fn: Callable,
    ) -> Callable:
        """미들웨어를 체인으로 래핑"""

        async def _wrapped(context: MiddlewareContext) -> MiddlewareContext:
            if context.cancelled:
                self._results.append(
                    MiddlewareResult(
                        middleware_name=middleware.name,
                        phase=phase,
                        status=MiddlewareStatus.SKIPPED,
                    )
                )
                return context

            start = time.time()
            try:
                context = await asyncio.wait_for(
                    middleware.process(context, next_fn),
                    timeout=self.config.max_middleware_timeout,
                )
                duration_ms = (time.time() - start) * 1000

                self._results.append(
                    MiddlewareResult(
                        middleware_name=middleware.name,
                        phase=phase,
                        status=MiddlewareStatus.COMPLETED,
                        duration_ms=duration_ms,
                    )
                )

                if self.config.enable_metrics and middleware.name in self._metrics:
                    self._metrics[middleware.name].record(duration_ms, success=True)

            except asyncio.TimeoutError:
                duration_ms = (time.time() - start) * 1000
                logger.warning(
                    "미들웨어 타임아웃: %s (%.1fms)", middleware.name, duration_ms
                )
                self._results.append(
                    MiddlewareResult(
                        middleware_name=middleware.name,
                        phase=phase,
                        status=MiddlewareStatus.TIMED_OUT,
                        duration_ms=duration_ms,
                    )
                )
                if self.config.enable_metrics and middleware.name in self._metrics:
                    self._metrics[middleware.name].record(duration_ms, success=False)

                if self.config.stop_on_error:
                    context.errors.append(
                        TimeoutError(f"Middleware '{middleware.name}' timed out")
                    )
                    return context

            except Exception as e:
                duration_ms = (time.time() - start) * 1000
                logger.error(
                    "미들웨어 에러: %s - %s", middleware.name, str(e)
                )
                self._results.append(
                    MiddlewareResult(
                        middleware_name=middleware.name,
                        phase=phase,
                        status=MiddlewareStatus.FAILED,
                        duration_ms=duration_ms,
                        error=str(e),
                    )
                )
                if self.config.enable_metrics and middleware.name in self._metrics:
                    self._metrics[middleware.name].record(duration_ms, success=False)

                context.errors.append(e)
                if self.config.stop_on_error:
                    return context

                # 에러 발생 시에도 다음 미들웨어 계속 실행
                return await next_fn(context)

            return context

        return _wrapped

    @property
    def results(self) -> list[MiddlewareResult]:
        """최근 실행 결과 조회"""
        return list(self._results)

    @property
    def metrics(self) -> dict[str, MiddlewareMetrics]:
        """미들웨어 메트릭 조회"""
        return dict(self._metrics)

    def get_registered_middlewares(self) -> list[dict[str, Any]]:
        """등록된 미들웨어 목록 조회"""
        return [
            {
                "name": m.name,
                "priority": m.priority.name,
                "phases": [p.value for p in m.phases],
                "enabled": m.enabled,
                "has_condition": m.condition is not None,
            }
            for m in self._middlewares
        ]


# ══════════════════════════════════════════════
# Core Components - 미들웨어 매니저
# ══════════════════════════════════════════════


class MiddlewareManager:
    """미들웨어 매니저 (전체 파이프라인 관리)

    요청 → 전처리 체인 → [에이전트 실행] → 후처리 체인 → 응답
    의 전체 흐름을 관리합니다.
    """

    def __init__(self, config: MiddlewareConfig | None = None):
        self.config = config or MiddlewareConfig()
        self._request_chain = MiddlewareChain(config=self.config)
        self._response_chain = MiddlewareChain(config=self.config)
        self._error_chain = MiddlewareChain(config=self.config)
        self._initialized = False
        logger.info("MiddlewareManager 초기화 완료")

    def add(self, middleware: BaseMiddleware) -> MiddlewareManager:
        """미들웨어 추가 (단계에 따라 적절한 체인에 자동 배치)"""
        has_request = any(
            p
            in (MiddlewarePhase.PRE_REQUEST, MiddlewarePhase.POST_REQUEST)
            for p in middleware.phases
        )
        has_response = any(
            p
            in (MiddlewarePhase.PRE_RESPONSE, MiddlewarePhase.POST_RESPONSE)
            for p in middleware.phases
        )
        has_error = MiddlewarePhase.ON_ERROR in middleware.phases

        if has_request:
            self._request_chain.add(middleware)
        if has_response:
            self._response_chain.add(middleware)
        if has_error:
            self._error_chain.add(middleware)

        return self

    def remove(self, name: str) -> bool:
        """미들웨어 제거"""
        r1 = self._request_chain.remove(name)
        r2 = self._response_chain.remove(name)
        r3 = self._error_chain.remove(name)
        return r1 or r2 or r3

    async def process(
        self,
        context: MiddlewareContext,
        agent_fn: Callable | None = None,
    ) -> MiddlewareContext:
        """전체 미들웨어 파이프라인 실행

        Args:
            context: 미들웨어 컨텍스트
            agent_fn: 에이전트 실행 함수 (선택적)

        Returns:
            처리된 컨텍스트
        """
        pipeline_start = time.time()
        context.timestamps["pipeline_start"] = pipeline_start

        try:
            # 1) 요청 전처리
            logger.debug("[Pipeline] PRE_REQUEST 시작")
            context = await self._request_chain.execute(
                context, MiddlewarePhase.PRE_REQUEST
            )

            if context.cancelled:
                logger.info("[Pipeline] 요청이 취소됨 (PRE_REQUEST)")
                return context

            # 2) 요청 후처리
            logger.debug("[Pipeline] POST_REQUEST 시작")
            context = await self._request_chain.execute(
                context, MiddlewarePhase.POST_REQUEST
            )

            if context.cancelled:
                logger.info("[Pipeline] 요청이 취소됨 (POST_REQUEST)")
                return context

            # 3) 에이전트 실행
            if agent_fn:
                logger.debug("[Pipeline] 에이전트 실행 시작")
                context.timestamps["agent_start"] = time.time()
                try:
                    context.response = await agent_fn(context)
                except Exception as e:
                    context.errors.append(e)
                    logger.error("[Pipeline] 에이전트 실행 에러: %s", str(e))
                context.timestamps["agent_end"] = time.time()

            # 4) 응답 전처리
            logger.debug("[Pipeline] PRE_RESPONSE 시작")
            context = await self._response_chain.execute(
                context, MiddlewarePhase.PRE_RESPONSE
            )

            # 5) 응답 후처리
            logger.debug("[Pipeline] POST_RESPONSE 시작")
            context = await self._response_chain.execute(
                context, MiddlewarePhase.POST_RESPONSE
            )

        except Exception as e:
            context.errors.append(e)
            logger.error("[Pipeline] 파이프라인 에러: %s", str(e))

            # 에러 미들웨어 실행
            if self.config.enable_retry_on_error:
                logger.debug("[Pipeline] ON_ERROR 시작")
                context = await self._error_chain.execute(
                    context, MiddlewarePhase.ON_ERROR
                )

        context.timestamps["pipeline_end"] = time.time()
        pipeline_duration = (time.time() - pipeline_start) * 1000
        logger.info("[Pipeline] 완료 (%.1fms)", pipeline_duration)

        return context

    def get_all_metrics(self) -> dict[str, dict[str, MiddlewareMetrics]]:
        """전체 미들웨어 메트릭 조회"""
        return {
            "request_chain": self._request_chain.metrics,
            "response_chain": self._response_chain.metrics,
            "error_chain": self._error_chain.metrics,
        }

    def get_pipeline_info(self) -> dict[str, Any]:
        """파이프라인 정보 조회"""
        return {
            "request_middlewares": self._request_chain.get_registered_middlewares(),
            "response_middlewares": self._response_chain.get_registered_middlewares(),
            "error_middlewares": self._error_chain.get_registered_middlewares(),
            "config": {
                "enable_metrics": self.config.enable_metrics,
                "max_middleware_timeout": self.config.max_middleware_timeout,
                "pipeline_timeout": self.config.pipeline_timeout,
                "stop_on_error": self.config.stop_on_error,
            },
        }


# ══════════════════════════════════════════════
# Built-in Middlewares - 내장 미들웨어
# ══════════════════════════════════════════════


class LoggingMiddleware(BaseMiddleware):
    """로깅 미들웨어 - 요청/응답 로깅

    모든 요청과 응답을 로깅하여 디버깅 및 감사를 지원합니다.
    """

    def __init__(
        self,
        log_level: str = "INFO",
        log_request: bool = True,
        log_response: bool = True,
        log_metadata: bool = False,
    ):
        super().__init__(
            name="LoggingMiddleware",
            priority=MiddlewarePriority.LOW,
            phases=[
                MiddlewarePhase.PRE_REQUEST,
                MiddlewarePhase.POST_RESPONSE,
            ],
        )
        self.log_level = getattr(logging, log_level.upper(), logging.INFO)
        self.log_request = log_request
        self.log_response = log_response
        self.log_metadata = log_metadata

    async def process(
        self, context: MiddlewareContext, next_fn: Callable
    ) -> MiddlewareContext:
        """요청/응답 로깅 처리"""
        if self.log_request and context.request is not None:
            logger.log(
                self.log_level,
                "[Logging] 요청 - agent=%s, request_id=%s, type=%s",
                context.agent_id,
                context.request_id,
                type(context.request).__name__,
            )

        context = await next_fn(context)

        if self.log_response and context.response is not None:
            logger.log(
                self.log_level,
                "[Logging] 응답 - agent=%s, request_id=%s, type=%s, errors=%d",
                context.agent_id,
                context.request_id,
                type(context.response).__name__,
                len(context.errors),
            )

        if self.log_metadata:
            logger.log(
                self.log_level,
                "[Logging] 메타데이터 - %s",
                context.metadata,
            )

        return context


class AuthMiddleware(RequestMiddleware):
    """인증 미들웨어 - 에이전트 인증 확인

    Microsoft Entra ID 또는 API 키 기반 인증을 처리합니다.
    """

    def __init__(
        self,
        provider: str = "entra_id",
        required_scopes: list[str] | None = None,
        allow_anonymous: bool = False,
    ):
        super().__init__(
            name="AuthMiddleware",
            priority=MiddlewarePriority.CRITICAL,
        )
        self.provider = provider
        self.required_scopes = required_scopes or []
        self.allow_anonymous = allow_anonymous
        self._validated_tokens: dict[str, float] = {}  # token -> expiry

    async def process(
        self, context: MiddlewareContext, next_fn: Callable
    ) -> MiddlewareContext:
        """인증 처리"""
        auth_token = context.get("auth_token")

        if not auth_token and not self.allow_anonymous:
            logger.warning(
                "[Auth] 인증 토큰 없음 - agent=%s", context.agent_id
            )
            context.errors.append(
                PermissionError("Authentication required: no auth_token provided")
            )
            context.cancel()
            return context

        if auth_token:
            # 토큰 유효성 검증 (캐시 확인)
            cached_expiry = self._validated_tokens.get(auth_token)
            if cached_expiry and cached_expiry > time.time():
                context.set("auth_validated", True)
                context.set("auth_provider", self.provider)
                logger.debug("[Auth] 캐시된 토큰 유효 - agent=%s", context.agent_id)
            else:
                # 실제 프로덕션에서는 Entra ID 토큰 검증 API 호출
                is_valid = await self._validate_token(auth_token)
                if is_valid:
                    self._validated_tokens[auth_token] = time.time() + 3600
                    context.set("auth_validated", True)
                    context.set("auth_provider", self.provider)
                    logger.info("[Auth] 토큰 검증 성공 - agent=%s", context.agent_id)
                else:
                    context.errors.append(
                        PermissionError("Invalid authentication token")
                    )
                    context.cancel()
                    return context

            # 스코프 확인
            if self.required_scopes:
                token_scopes = context.get("token_scopes", [])
                missing = set(self.required_scopes) - set(token_scopes)
                if missing:
                    logger.warning(
                        "[Auth] 스코프 부족 - 필요: %s, 누락: %s",
                        self.required_scopes,
                        list(missing),
                    )
                    context.errors.append(
                        PermissionError(f"Missing scopes: {missing}")
                    )
                    context.cancel()
                    return context

        return await next_fn(context)

    async def _validate_token(self, token: str) -> bool:
        """토큰 유효성 검증 (시뮬레이션)"""
        # 실제 프로덕션에서는 Microsoft Entra ID와 통합
        await asyncio.sleep(0.01)  # 네트워크 호출 시뮬레이션
        return len(token) > 10  # 시뮬레이션: 10자 이상이면 유효


class RateLimitMiddleware(RequestMiddleware):
    """레이트 리밋 미들웨어 - API 호출 제한

    분당 요청 수(RPM)와 분당 토큰 수(TPM)를 제한합니다.
    Token Bucket 알고리즘을 사용합니다.
    """

    def __init__(
        self,
        max_rpm: int = 60,
        max_tpm: int = 100_000,
        burst_multiplier: float = 1.5,
    ):
        super().__init__(
            name="RateLimitMiddleware",
            priority=MiddlewarePriority.HIGH,
        )
        self.max_rpm = max_rpm
        self.max_tpm = max_tpm
        self.burst_multiplier = burst_multiplier
        self._request_counts: dict[str, list[float]] = {}  # agent_id -> timestamps
        self._token_counts: dict[str, list[tuple[float, int]]] = {}  # agent_id -> (ts, tokens)

    async def process(
        self, context: MiddlewareContext, next_fn: Callable
    ) -> MiddlewareContext:
        """레이트 리밋 확인"""
        agent_id = context.agent_id or "global"
        now = time.time()
        window = 60.0  # 1분 윈도우

        # RPM 확인
        if agent_id not in self._request_counts:
            self._request_counts[agent_id] = []

        # 윈도우 밖의 요청 제거
        self._request_counts[agent_id] = [
            ts for ts in self._request_counts[agent_id] if now - ts < window
        ]

        current_rpm = len(self._request_counts[agent_id])
        max_burst = int(self.max_rpm * self.burst_multiplier)

        if current_rpm >= max_burst:
            wait_time = self._request_counts[agent_id][0] + window - now
            logger.warning(
                "[RateLimit] RPM 제한 초과 - agent=%s, current=%d, max=%d, wait=%.1fs",
                agent_id,
                current_rpm,
                self.max_rpm,
                wait_time,
            )
            context.set("rate_limited", True)
            context.set("retry_after", wait_time)
            context.errors.append(
                RuntimeError(
                    f"Rate limit exceeded: {current_rpm}/{self.max_rpm} RPM. "
                    f"Retry after {wait_time:.1f}s"
                )
            )
            context.cancel()
            return context

        # TPM 확인
        estimated_tokens = context.get("estimated_tokens", 0)
        if estimated_tokens > 0:
            if agent_id not in self._token_counts:
                self._token_counts[agent_id] = []

            self._token_counts[agent_id] = [
                (ts, t)
                for ts, t in self._token_counts[agent_id]
                if now - ts < window
            ]

            current_tpm = sum(t for _, t in self._token_counts[agent_id])
            if current_tpm + estimated_tokens > self.max_tpm:
                logger.warning(
                    "[RateLimit] TPM 제한 초과 - agent=%s, current=%d, estimated=%d, max=%d",
                    agent_id,
                    current_tpm,
                    estimated_tokens,
                    self.max_tpm,
                )
                context.set("rate_limited", True)
                context.errors.append(
                    RuntimeError(
                        f"Token rate limit exceeded: {current_tpm}/{self.max_tpm} TPM"
                    )
                )
                context.cancel()
                return context

            self._token_counts[agent_id].append((now, estimated_tokens))

        self._request_counts[agent_id].append(now)
        context.set("rate_limited", False)

        return await next_fn(context)


class RetryMiddleware(BaseMiddleware):
    """재시도 미들웨어 - 에러 발생 시 자동 재시도

    지수 백오프(Exponential Backoff)를 사용한 재시도 로직입니다.
    """

    def __init__(
        self,
        max_retries: int = 3,
        backoff_factor: float = 2.0,
        initial_delay: float = 1.0,
        retryable_errors: list[type] | None = None,
    ):
        super().__init__(
            name="RetryMiddleware",
            priority=MiddlewarePriority.NORMAL,
            phases=[MiddlewarePhase.ON_ERROR],
        )
        self.max_retries = max_retries
        self.backoff_factor = backoff_factor
        self.initial_delay = initial_delay
        self.retryable_errors = retryable_errors or [
            TimeoutError,
            ConnectionError,
            RuntimeError,
        ]

    async def process(
        self, context: MiddlewareContext, next_fn: Callable
    ) -> MiddlewareContext:
        """에러 발생 시 재시도 처리"""
        if not context.errors:
            return await next_fn(context)

        last_error = context.errors[-1]

        # 재시도 가능한 에러인지 확인
        if not any(isinstance(last_error, t) for t in self.retryable_errors):
            logger.debug(
                "[Retry] 재시도 불가능한 에러 유형: %s", type(last_error).__name__
            )
            return await next_fn(context)

        retry_count = context.get("retry_count", 0)

        if retry_count >= self.max_retries:
            logger.warning(
                "[Retry] 최대 재시도 횟수 초과: %d/%d",
                retry_count,
                self.max_retries,
            )
            return await next_fn(context)

        # 지수 백오프 대기
        delay = self.initial_delay * (self.backoff_factor ** retry_count)
        logger.info(
            "[Retry] 재시도 %d/%d - %.1fs 후 재시도",
            retry_count + 1,
            self.max_retries,
            delay,
        )
        await asyncio.sleep(delay)

        # 에러 초기화 및 재시도 카운트 증가
        context.errors.clear()
        context.cancelled = False
        context.set("retry_count", retry_count + 1)
        context.set("last_retry_error", str(last_error))

        return await next_fn(context)


class ContentFilterMiddleware(BaseMiddleware):
    """컨텐츠 필터 미들웨어 - 유해 컨텐츠 차단

    Azure AI Content Safety 연동하여 요청/응답의 유해 컨텐츠를 필터링합니다.
    """

    def __init__(
        self,
        block_categories: list[str] | None = None,
        severity_threshold: int = 4,
        enable_pii_detection: bool = True,
    ):
        super().__init__(
            name="ContentFilterMiddleware",
            priority=MiddlewarePriority.HIGH,
            phases=[
                MiddlewarePhase.PRE_REQUEST,
                MiddlewarePhase.POST_RESPONSE,
            ],
        )
        self.block_categories = block_categories or [
            "hate",
            "violence",
            "self_harm",
            "sexual",
        ]
        self.severity_threshold = severity_threshold
        self.enable_pii_detection = enable_pii_detection
        self._blocked_count = 0

    async def process(
        self, context: MiddlewareContext, next_fn: Callable
    ) -> MiddlewareContext:
        """컨텐츠 필터링 처리"""
        content = None

        # 요청 또는 응답에서 컨텐츠 추출
        if context.request and isinstance(context.request, str):
            content = context.request
        elif context.response and isinstance(context.response, str):
            content = context.response

        if content:
            # 컨텐츠 안전성 분석 (시뮬레이션)
            analysis = await self._analyze_content(content)

            if analysis.get("blocked"):
                self._blocked_count += 1
                logger.warning(
                    "[ContentFilter] 컨텐츠 차단됨 - agent=%s, category=%s, severity=%d",
                    context.agent_id,
                    analysis.get("category"),
                    analysis.get("severity", 0),
                )
                context.set("content_blocked", True)
                context.set("block_reason", analysis.get("category"))
                context.errors.append(
                    ValueError(
                        f"Content blocked: {analysis.get('category')} "
                        f"(severity={analysis.get('severity')})"
                    )
                )
                context.cancel()
                return context

            context.set("content_safe", True)

            # PII 감지
            if self.enable_pii_detection and analysis.get("pii_detected"):
                context.set("pii_detected", True)
                context.set("pii_types", analysis.get("pii_types", []))
                logger.info(
                    "[ContentFilter] PII 감지됨 - types=%s",
                    analysis.get("pii_types"),
                )

        return await next_fn(context)

    async def _analyze_content(self, content: str) -> dict[str, Any]:
        """컨텐츠 안전성 분석 (시뮬레이션)

        프로덕션에서는 Azure AI Content Safety API를 호출합니다.
        """
        await asyncio.sleep(0.005)  # API 호출 시뮬레이션
        # 시뮬레이션: 간단한 키워드 기반 분석
        result: dict[str, Any] = {
            "blocked": False,
            "category": None,
            "severity": 0,
            "pii_detected": False,
            "pii_types": [],
        }
        return result

    @property
    def blocked_count(self) -> int:
        """차단된 컨텐츠 수"""
        return self._blocked_count


class CacheMiddleware(BaseMiddleware):
    """캐시 미들웨어 - 응답 캐싱

    동일한 요청에 대한 응답을 캐싱하여 중복 LLM 호출을 방지합니다.
    TTL 기반 만료를 지원합니다.
    """

    def __init__(
        self,
        ttl_seconds: float = 300.0,
        max_cache_size: int = 1000,
        cache_key_fn: Callable[[MiddlewareContext], str] | None = None,
    ):
        super().__init__(
            name="CacheMiddleware",
            priority=MiddlewarePriority.LAST,
            phases=[
                MiddlewarePhase.PRE_REQUEST,
                MiddlewarePhase.POST_RESPONSE,
            ],
        )
        self.ttl_seconds = ttl_seconds
        self.max_cache_size = max_cache_size
        self.cache_key_fn = cache_key_fn
        self._cache: dict[str, tuple[Any, float]] = {}  # key -> (value, expiry)
        self._hits = 0
        self._misses = 0

    async def process(
        self, context: MiddlewareContext, next_fn: Callable
    ) -> MiddlewareContext:
        """캐시 확인 및 저장"""
        cache_key = self._get_cache_key(context)

        # 캐시 히트 확인 (요청 단계)
        if context.response is None:  # 아직 응답이 없으면 캐시 조회
            cached = self._cache.get(cache_key)
            if cached:
                value, expiry = cached
                if time.time() < expiry:
                    self._hits += 1
                    context.response = value
                    context.set("cache_hit", True)
                    logger.debug("[Cache] 캐시 히트 - key=%s", cache_key[:20])
                    return context  # 캐시 히트 시 나머지 체인 스킵
                else:
                    del self._cache[cache_key]  # 만료된 캐시 제거

            self._misses += 1
            context.set("cache_hit", False)

        context = await next_fn(context)

        # 응답 캐싱 (응답 단계)
        if context.response is not None and not context.has_errors:
            if not context.get("cache_hit"):
                self._set_cache(cache_key, context.response)
                logger.debug("[Cache] 캐시 저장 - key=%s", cache_key[:20])

        return context

    def _get_cache_key(self, context: MiddlewareContext) -> str:
        """캐시 키 생성"""
        if self.cache_key_fn:
            return self.cache_key_fn(context)
        # 기본 키: agent_id + request hash
        request_str = str(context.request) if context.request else ""
        return f"{context.agent_id}:{hash(request_str)}"

    def _set_cache(self, key: str, value: Any) -> None:
        """캐시 저장 (LRU 방식으로 크기 제한)"""
        if len(self._cache) >= self.max_cache_size:
            # 가장 오래된 항목 제거
            oldest_key = min(self._cache, key=lambda k: self._cache[k][1])
            del self._cache[oldest_key]
        self._cache[key] = (value, time.time() + self.ttl_seconds)

    def clear(self) -> None:
        """캐시 초기화"""
        self._cache.clear()
        self._hits = 0
        self._misses = 0

    @property
    def hit_rate(self) -> float:
        """캐시 적중률"""
        total = self._hits + self._misses
        return self._hits / total if total > 0 else 0.0

    @property
    def stats(self) -> dict[str, Any]:
        """캐시 통계"""
        return {
            "hits": self._hits,
            "misses": self._misses,
            "hit_rate": f"{self.hit_rate:.1%}",
            "cache_size": len(self._cache),
            "max_size": self.max_cache_size,
            "ttl_seconds": self.ttl_seconds,
        }
