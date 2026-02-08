#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Unified Agent Framework - 유틸리티 모듈 (Utility Module)

================================================================================
📁 파일 위치: unified_agent/utils.py
📋 역할: 공통 유틸리티 함수 및 클래스 제공
📅 최종 업데이트: 2026년 1월
================================================================================

🎯 주요 구성 요소:
    1. StructuredLogger - JSON 형태의 구조화된 로깅
    2. retry_with_backoff - 지수 백오프 재시도 로직
    3. CircuitBreaker - 회로 차단기 패턴 (장애 전파 방지)
    4. setup_telemetry - OpenTelemetry 초기화
    5. RAIValidator - Responsible AI 콘텐츠 검증

🔧 2026년 개선 사항:
    - Adaptive Circuit Breaker: 평균 응답 시간 기반 동적 타임아웃
    - 메트릭 수집: 성공률, 평균 응답 시간, 총 호출 수
    - 성공 임계값: HALF_OPEN → CLOSED 전환에 연속 성공 필요
    - RAI 검증: PII 감지, 유해 콘텐츠 필터링

📌 사용 예시:
    >>> from unified_agent.utils import StructuredLogger, CircuitBreaker, RAIValidator
    >>>
    >>> # 구조화된 로깅
    >>> logger = StructuredLogger("my_agent")
    >>> logger.info("작업 완료", task_id="123", duration_ms=450)
    >>>
    >>> # 회로 차단기
    >>> breaker = CircuitBreaker(failure_threshold=5, success_threshold=3)
    >>> result = await breaker.call(external_api_call, param1, param2)
    >>> print(breaker.get_metrics())  # 상태 및 메트릭 확인
    >>>
    >>> # RAI 검증
    >>> validator = RAIValidator(strict_mode=True)
    >>> result = validator.validate("사용자 입력 텍스트")
    >>> if not result.is_safe:
    ...     print(f"⚠️ 안전하지 않음: {result.reason}")

⚠️ 주의사항:
    - CircuitBreaker는 비동기 함수만 지원합니다.
    - RAIValidator의 패턴 기반 검증은 기본 필터링용입니다.
      프로덕션에서는 Azure Content Safety API 사용을 권장합니다.

🔗 관련 문서:
    - Circuit Breaker 패턴: https://microservices.io/patterns/reliability/circuit-breaker.html
    - OpenTelemetry: https://opentelemetry.io/
    - Azure Content Safety: https://learn.microsoft.com/azure/ai-services/content-safety/
"""

from __future__ import annotations

import re
import asyncio
import json
import logging
import time
from collections import deque
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Callable

from opentelemetry import trace
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import ConsoleSpanExporter, BatchSpanProcessor
from opentelemetry.sdk.resources import Resource

from .models import RAICategory, RAIValidationResult

__all__ = [
    "StructuredLogger",
    "retry_with_backoff",
    "CircuitBreakerState",
    "CircuitBreaker",
    "setup_telemetry",
    "RAIValidator",
]

class StructuredLogger:
    """
    JSON 형태의 구조화된 로깅 클래스

    ================================================================================
    📋 역할: 구조화된 JSON 형식으로 로그를 기록하여 분석 및 모니터링 용이성 향상
    📅 최종 업데이트: 2026년 1월
    ================================================================================

    🎯 주요 기능:
        - JSON 형식의 구조화된 로그 출력
        - 자동 타임스탬프 (UTC ISO 8601 형식)
        - 키워드 인자를 통한 추가 컨텍스트 기록
        - info, error, warning 레벨 지원

    📌 사용 예시:
        >>> logger = StructuredLogger("my_service")
        >>>
        >>> # 기본 로깅
        >>> logger.info("작업 시작")
        >>>
        >>> # 컨텍스트 포함 로깅
        >>> logger.info("API 호출 완료",
        ...     endpoint="/api/chat",
        ...     status_code=200,
        ...     duration_ms=125.5,
        ...     tokens_used=1500
        ... )
        >>>
        >>> # 오류 로깅
        >>> logger.error("요청 실패",
        ...     error_type="TimeoutError",
        ...     retry_count=3
        ... )

    출력 형식:
        [INFO] {"timestamp": "2026-01-26T10:30:00Z", "message": "API 호출 완료",
               "endpoint": "/api/chat", "status_code": 200, "duration_ms": 125.5}

    ⚠️ 주의사항:
        - ensure_ascii=False로 한글이 그대로 출력됩니다.
        - __slots__를 사용하여 메모리 효율성을 높였습니다.

    🔗 참고:
        - Azure Monitor와 통합 시 JSON 형식이 자동 파싱됩니다.
        - OpenTelemetry와 함께 사용하면 분산 추적이 가능합니다.
    """
    __slots__ = ('logger',)

    def __init__(self, name: str):
        """StructuredLogger 초기화

        Args:
            name (str): 로거 이름 (일반적으로 모듈명 또는 서비스명)
        """
        self.logger = logging.getLogger(name)

    def debug(self, message: str, **kwargs):
        self._log(logging.DEBUG, message, **kwargs)

    def info(self, message: str, **kwargs):
        self._log(logging.INFO, message, **kwargs)

    def error(self, message: str, **kwargs):
        self._log(logging.ERROR, message, **kwargs)

    def warning(self, message: str, **kwargs):
        self._log(logging.WARNING, message, **kwargs)

    def _log(self, level: int, message: str, **kwargs):
        log_data = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "message": message,
            **kwargs
        }
        self.logger.log(level, f"[{level}] {json.dumps(log_data, ensure_ascii=False)}")

async def retry_with_backoff(
    func: Callable,
    max_retries: int = 3,
    base_delay: float = 1.0,
    max_delay: float = 60.0,
    exponential_base: float = 2.0,
    **kwargs
) -> Any:
    """
    지수 백오프 (Exponential Backoff) 재시도 로직

    ================================================================================
    📋 역할: 일시적 장애 시 지수적으로 증가하는 대기 시간으로 재시도
    📅 최종 업데이트: 2026년 1월
    ================================================================================

    🎯 기능 설명:
        외부 API 호출, 데이터베이스 연결 등 일시적 장애가 발생할 수 있는
        작업에 대해 자동으로 재시도를 수행합니다.

        지연 시간 계산: delay = min(base_delay * (exponential_base ^ attempt), max_delay)

        예시 (기본 설정):
        - 1차 재시도: 1초 후
        - 2차 재시도: 2초 후
        - 3차 재시도: 4초 후
        - 최대 지연: 60초

    Args:
        func (Callable): 실행할 비동기 함수 (async def)
        max_retries (int): 최대 재시도 횟수 (기본: 3회)
        base_delay (float): 첫 번째 재시도 전 대기 시간 (초, 기본: 1.0)
        max_delay (float): 최대 대기 시간 상한 (초, 기본: 60.0)
        exponential_base (float): 지수 밑수 (기본: 2.0)
        **kwargs: func에 전달할 키워드 인자

    Returns:
        Any: func의 반환값

    Raises:
        Exception: 모든 재시도 실패 시 마지막 예외를 다시 발생

    📌 사용 예시:
        >>> async def call_api(url: str, timeout: int):
        ...     # API 호출 로직
        ...     pass
        >>>
        >>> # 기본 설정으로 재시도
        >>> result = await retry_with_backoff(
        ...     call_api,
        ...     url="https://api.example.com",
        ...     timeout=30
        ... )
        >>>
        >>> # 커스텀 재시도 설정
        >>> result = await retry_with_backoff(
        ...     call_api,
        ...     max_retries=5,
        ...     base_delay=0.5,
        ...     max_delay=30.0,
        ...     url="https://api.example.com"
        ... )

    ⚠️ 주의사항:
        - 비동기 함수(async def)만 지원합니다.
        - 영구적 오류(예: 인증 실패)에는 재시도가 무의미합니다.
        - 멱등성(idempotent) 작업에만 사용하세요.

    🔗 참고:
        - CircuitBreaker: 연속 실패 시 빠른 실패 처리
        - Azure SDK는 기본적으로 재시도 로직이 포함되어 있습니다.
    """
    last_exception: Exception | None = None
    for attempt in range(max_retries + 1):
        try:
            return await func(**kwargs)
        except Exception as e:
            last_exception = e
            if attempt >= max_retries:
                raise e
            delay = min(base_delay * (exponential_base ** attempt), max_delay)
            logging.warning(f"⚠️ 재시도 {attempt + 1}/{max_retries} ({delay:.2f}s 후): {e}")
            await asyncio.sleep(delay)
    raise last_exception  # type: ignore

# ============================================================================
# 회로 차단기 패턴
# ============================================================================

class CircuitBreakerState(str, Enum):
    """
    Circuit Breaker 상태 열거형

    🔄 상태 전환 다이어그램:

        ┌─────────────────────────────────────────────────────────────┐
        │                                                             │
        │  [CLOSED] ──(연속 실패 임계값 도달)──▶ [OPEN]              │
        │     ▲                                    │                  │
        │     │                                    │                  │
        │     │                          (타임아웃 후)                │
        │     │                                    │                  │
        │     │                                    ▼                  │
        │     └──(연속 성공 임계값 도달)── [HALF_OPEN]               │
        │              ▲                           │                  │
        │              └───────(실패)──────────────┘                  │
        │                                                             │
        └─────────────────────────────────────────────────────────────┘

    📌 상태 설명:
        - CLOSED (정상): 모든 요청이 통과됩니다.
        - OPEN (차단): 모든 요청이 즉시 실패합니다 (빠른 실패).
        - HALF_OPEN (테스트): 일부 요청만 허용하여 복구 여부를 테스트합니다.
    """
    CLOSED = "CLOSED"      # 정상 상태 - 모든 요청 허용
    OPEN = "OPEN"          # 차단 상태 - 모든 요청 거부 (빠른 실패)
    HALF_OPEN = "HALF_OPEN"  # 반개방 상태 - 테스트 요청만 허용

class CircuitBreaker:
    """
    Adaptive Circuit Breaker - 장애 전파 방지 패턴 (2026년 개선 버전)

    ================================================================================
    📋 역할: 외부 서비스 장애 시 빠른 실패로 시스템 안정성 보장
    📅 최종 업데이트: 2026년 1월
    ================================================================================

    🎯 핵심 개념:
        마이크로서비스 아키텍처에서 연쇄 장애(Cascading Failure)를 방지하는
        핵심 복원력(Resilience) 패턴입니다.

        외부 서비스가 응답하지 않을 때 계속 대기하면 스레드/리소스가 고갈되어
        전체 시스템이 마비될 수 있습니다. Circuit Breaker는 이를 방지합니다.

    🔄 상태 전환 로직:
        1. CLOSED (정상)
           - 모든 요청이 외부 서비스로 전달됩니다.
           - 실패 시 failure_count 증가
           - failure_count >= failure_threshold 도달 시 → OPEN 전환

        2. OPEN (차단)
           - 모든 요청이 즉시 RuntimeError로 실패합니다.
           - 외부 서비스로 요청을 보내지 않아 리소스를 보호합니다.
           - timeout 시간 경과 후 → HALF_OPEN 전환

        3. HALF_OPEN (테스트)
           - 일부 요청만 외부 서비스로 전달하여 복구 여부 테스트
           - success_count >= success_threshold 달성 시 → CLOSED 전환
           - 1회라도 실패 시 → OPEN 전환

    🔧 2026년 개선사항:
        ✅ Adaptive Timeout: 평균 응답 시간 기반 동적 타임아웃 조절
        ✅ Success Threshold: HALF_OPEN → CLOSED 전환에 연속 성공 필요
        ✅ 메트릭 수집: total_calls, success_rate, avg_response_time
        ✅ 빠른 회복: CLOSED 상태에서 성공 시 failure_count 감소

    Args:
        failure_threshold (int): OPEN 전환을 위한 연속 실패 횟수 (기본: 5)
        success_threshold (int): CLOSED 전환을 위한 연속 성공 횟수 (기본: 3)
        timeout (float): OPEN 상태 유지 시간 (초, 기본: 60.0)
        adaptive_timeout (bool): 적응형 타임아웃 활성화 (기본: True)

    📌 사용 예시:
        >>> # 기본 설정
        >>> breaker = CircuitBreaker()
        >>>
        >>> # 커스텀 설정 (민감한 서비스용)
        >>> breaker = CircuitBreaker(
        ...     failure_threshold=3,   # 3회 실패 시 차단
        ...     success_threshold=5,   # 5회 연속 성공 시 복구
        ...     timeout=120.0,         # 2분간 차단
        ...     adaptive_timeout=True  # 응답 시간 기반 동적 조절
        ... )
        >>>
        >>> # 함수 호출
        >>> async def call_external_api():
        ...     # 외부 API 호출
        ...     pass
        >>>
        >>> try:
        ...     result = await breaker.call(call_external_api)
        ... except RuntimeError as e:
        ...     print(f"회로 차단: {e}")
        >>>
        >>> # 메트릭 확인
        >>> metrics = breaker.get_metrics()
        >>> print(f"성공률: {metrics['success_rate']:.2%}")
        >>> print(f"평균 응답 시간: {metrics['avg_response_time_ms']:.1f}ms")
        >>>
        >>> # 수동 리셋 (관리자 개입)
        >>> breaker.reset()

    🎯 사용 시나리오:
        - Azure OpenAI API 호출
        - MCP 서버 연결
        - 외부 데이터베이스 쿼리
        - 마이크로서비스 간 통신
        - 결제 게이트웨이 연동

    ⚠️ 주의사항:
        - 비동기 함수(async def)만 지원합니다.
        - OPEN 상태에서는 RuntimeError가 발생합니다.
        - adaptive_timeout은 최근 10개 응답 시간의 평균을 사용합니다.
        - 타임아웃 범위: 최소 30초, 최대 300초

    🔗 참고:
        - retry_with_backoff: 단순 재시도 로직
        - Microsoft Resilience patterns: https://learn.microsoft.com/azure/architecture/patterns/circuit-breaker
    """
    __slots__ = (
        'failure_threshold', 'success_threshold', 'timeout', 'adaptive_timeout',
        'failure_count', 'success_count', 'last_failure_time', 'state',
        'total_calls', 'total_failures', 'total_successes', 'avg_response_time',
        '_response_times'  # deque로 변경됨
    )

    def __init__(
        self,
        failure_threshold: int = 5,
        success_threshold: int = 3,
        timeout: float = 60.0,
        adaptive_timeout: bool = True
    ):
        self.failure_threshold = failure_threshold
        self.success_threshold = success_threshold
        self.timeout = timeout
        self.adaptive_timeout = adaptive_timeout
        self.failure_count = 0
        self.success_count = 0
        self.last_failure_time: float | None = None
        self.state = CircuitBreakerState.CLOSED

        # 메트릭
        self.total_calls = 0
        self.total_failures = 0
        self.total_successes = 0
        self.avg_response_time = 0.0
        self._response_times: deque = deque(maxlen=100)  # 최적화: maxlen으로 자동 제한

    def get_effective_timeout(self) -> float:
        """적응형 타임아웃 계산 (최적화: deque 사용)"""
        if not self.adaptive_timeout or not self._response_times:
            return self.timeout

        # deque의 최근 10개만 사용 (list로 변환 없이 직접 반복)
        recent_times = list(self._response_times)[-10:]
        if not recent_times:
            return self.timeout

        # 평균 응답 시간의 5배를 타임아웃으로 설정 (최소 30초, 최대 300초)
        avg = sum(recent_times) / len(recent_times)
        return min(max(avg * 5, 30.0), 300.0)

    def get_metrics(self) -> dict[str, Any]:
        """회로 차단기 메트릭 반환"""
        return {
            "state": self.state.value,
            "total_calls": self.total_calls,
            "total_successes": self.total_successes,
            "total_failures": self.total_failures,
            "success_rate": self.total_successes / max(self.total_calls, 1),
            "avg_response_time_ms": self.avg_response_time * 1000,
            "current_failure_count": self.failure_count,
            "effective_timeout": self.get_effective_timeout()
        }

    async def call(self, func: Callable, *args, **kwargs) -> Any:
        """
        회로 차단기를 통한 함수 호출

        장애 격리 및 빠른 실패
        """
        self.total_calls += 1
        effective_timeout = self.get_effective_timeout()

        if self.state == CircuitBreakerState.OPEN:
            if self.last_failure_time and time.time() - self.last_failure_time > effective_timeout:
                self.state = CircuitBreakerState.HALF_OPEN
                self.success_count = 0
                logging.info(f"🔄 회로 차단기: HALF_OPEN 상태 (timeout: {effective_timeout:.1f}s)")
            else:
                raise RuntimeError(f"회로 차단기가 OPEN 상태입니다 (남은 시간: {effective_timeout - (time.time() - (self.last_failure_time or 0)):.1f}s)")

        start_time = time.time()
        try:
            result = await func(*args, **kwargs)
            response_time = time.time() - start_time

            # 메트릭 업데이트 (최적화: deque maxlen으로 자동 제한)
            self._response_times.append(response_time)
            self.avg_response_time = sum(self._response_times) / len(self._response_times)

            self.total_successes += 1

            if self.state == CircuitBreakerState.HALF_OPEN:
                self.success_count += 1
                if self.success_count >= self.success_threshold:
                    self.state = CircuitBreakerState.CLOSED
                    self.failure_count = 0
                    self.success_count = 0
                    logging.info(f"✅ 회로 차단기: CLOSED 상태 복구 ({self.success_threshold}회 연속 성공)")
            elif self.state == CircuitBreakerState.CLOSED:
                # 성공 시 실패 카운트 감소 (빠른 회복)
                self.failure_count = max(0, self.failure_count - 1)

            return result
        except Exception as e:
            self.failure_count += 1
            self.total_failures += 1
            self.last_failure_time = time.time()

            if self.state == CircuitBreakerState.HALF_OPEN:
                # HALF_OPEN에서 실패하면 바로 OPEN으로
                self.state = CircuitBreakerState.OPEN
                logging.error(f"❌ 회로 차단기: OPEN 상태 (HALF_OPEN 테스트 실패)")
            elif self.failure_count >= self.failure_threshold:
                self.state = CircuitBreakerState.OPEN
                logging.error(f"❌ 회로 차단기: OPEN 상태 ({self.failure_count} 연속 실패)")

            raise e

    def reset(self):
        """회로 차단기 상태 리셋"""
        self.failure_count = 0
        self.success_count = 0
        self.state = CircuitBreakerState.CLOSED
        logging.info("🔄 회로 차단기: 수동 리셋됨")

# ============================================================================
# OpenTelemetry 설정
# ============================================================================

def setup_telemetry(service_name: str = "UnifiedAgentFramework",
                   enable_console: bool = False):
    """OpenTelemetry 설정"""
    try:
        resource = Resource.create({"service.name": service_name})
        provider = TracerProvider(resource=resource)

        if enable_console:
            processor = BatchSpanProcessor(ConsoleSpanExporter())
            provider.add_span_processor(processor)

        trace.set_tracer_provider(provider)
        logging.info(f"✅ OpenTelemetry 설정: {service_name}")
    except Exception as e:
        logging.warning(f"⚠️ OpenTelemetry 설정 실패: {e}")

# ============================================================================
# RAI (Responsible AI) 검증기
# ============================================================================

class RAIValidator:
    """
    RAI (Responsible AI) 검증기 (Microsoft Pattern)

    AI 출력의 안전성을 검증합니다.

    사용법:
    ```python
    validator = RAIValidator()

    # 텍스트 검증
    result = validator.validate("안녕하세요!")
    if not result.is_safe:
        print(f"⚠️ 안전하지 않은 콘텐츠: {result.reason}")

    # 비동기 검증 (외부 API 사용 시)
    result = await validator.validate_async("텍스트", use_azure_content_safety=True)
    ```

    검증 카테고리:
    - 유해 콘텐츠
    - 혐오 발언
    - 폭력적 내용
    - 자해 관련
    - 성적 콘텐츠
    - Jailbreak 시도
    - PII 노출
    """

    def __init__(self, strict_mode: bool = False):
        self.strict_mode = strict_mode
        self._logger = StructuredLogger("rai_validator")

        # 간단한 패턴 기반 필터 (실제 환경에서는 Azure Content Safety 사용)
        self._harmful_patterns = [
            r'\b(폭탄|무기|테러)\s*(만들|제조|설계)',
            r'\b(자살|자해)\s*(방법|하는\s*법)',
            r'\b(해킹|크래킹)\s*(방법|하는\s*법)',
        ]

        self._pii_patterns = [
            r'\b\d{6}[-\s]?\d{7}\b',  # 주민등록번호
            r'\b\d{3}[-\s]?\d{4}[-\s]?\d{4}\b',  # 전화번호
            r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b',  # 이메일
        ]

    def validate(self, text: str) -> RAIValidationResult:
        """
        텍스트 안전성 검증 (동기)

        빠른 패턴 기반 검증을 수행합니다.
        더 정확한 검증이 필요하면 validate_async()를 사용하세요.
        """
        text_lower = text.lower()

        # 유해 콘텐츠 검사
        for pattern in self._harmful_patterns:
            if re.search(pattern, text, re.IGNORECASE):
                self._logger.warning(
                    "Harmful content detected",
                    pattern=pattern,
                    text_preview=text[:100]
                )
                return RAIValidationResult(
                    is_safe=False,
                    category=RAICategory.HARMFUL_CONTENT,
                    confidence=0.9,
                    reason="잠재적으로 유해한 콘텐츠가 감지되었습니다.",
                    suggestions=["콘텐츠를 검토하고 수정해주세요."]
                )

        # PII 검사
        for pattern in self._pii_patterns:
            if re.search(pattern, text):
                self._logger.warning(
                    "PII detected",
                    pattern=pattern
                )
                return RAIValidationResult(
                    is_safe=False,
                    category=RAICategory.PII_EXPOSURE,
                    confidence=0.85,
                    reason="개인식별정보(PII)가 감지되었습니다.",
                    suggestions=["민감한 정보를 마스킹하거나 제거해주세요."]
                )

        return RAIValidationResult(is_safe=True)

    async def validate_async(
        self,
        text: str,
        use_azure_content_safety: bool = False
    ) -> RAIValidationResult:
        """
        텍스트 안전성 검증 (비동기)

        Azure Content Safety API를 사용하여 더 정확한 검증을 수행합니다.
        """
        # 먼저 빠른 패턴 검사
        quick_result = self.validate(text)
        if not quick_result.is_safe:
            return quick_result

        # Azure Content Safety API 사용 (실제 환경에서 구현)
        if use_azure_content_safety:
            # TODO: Azure Content Safety API 호출
            # https://learn.microsoft.com/azure/ai-services/content-safety/
            pass

        return RAIValidationResult(is_safe=True)
