#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Unified Agent Framework - 유틸리티 모듈

로깅, 재시도 로직, 회로 차단기, OpenTelemetry 설정 등 유틸리티 함수들
"""

import re
import asyncio
import json
import logging
import time
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Callable, Dict, List, Optional

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
    JSON 형태의 구조화된 로깅
    """
    __slots__ = ('logger',)

    def __init__(self, name: str):
        self.logger = logging.getLogger(name)

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
    지수 백오프 재시도 로직

    Args:
        func: 실행할 비동기 함수
        max_retries: 최대 재시도 횟수
        base_delay: 기본 지연 시간 (초)
        max_delay: 최대 지연 시간 (초)
        exponential_base: 지수 기반 값
        **kwargs: func에 전달할 키워드 인자
    """
    last_exception: Optional[Exception] = None
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
    """Circuit Breaker 상태"""
    CLOSED = "CLOSED"
    OPEN = "OPEN"
    HALF_OPEN = "HALF_OPEN"


class CircuitBreaker:
    """
    회로 차단기 - 장애 전파 방지

    마이크로서비스 아키텍처의 핵심 패턴

    상태 전환:
    1. CLOSED (정상): 모든 요청 허용
    2. OPEN (차단): 실패 임계값 도달, 모든 요청 차단
    3. HALF_OPEN (반개방): 타임아웃 후 일부 요청 허용하여 테스트

    주요 파라미터:
    - failure_threshold: 연속 실패 임계값 (기본 5회)
    - timeout: OPEN 상태 유지 시간 (기본 60초)

    사용 시나리오:
    - 외부 API 호출
    - 데이터베이스 쿼리
    - LLM API 호출
    """
    __slots__ = ('failure_threshold', 'timeout', 'failure_count', 'last_failure_time', 'state')

    def __init__(self, failure_threshold: int = 5, timeout: float = 60.0):
        self.failure_threshold = failure_threshold
        self.timeout = timeout
        self.failure_count = 0
        self.last_failure_time: Optional[float] = None
        self.state = CircuitBreakerState.CLOSED

    async def call(self, func: Callable, *args, **kwargs) -> Any:
        """
        회로 차단기를 통한 함수 호출

        장애 격리 및 빠른 실패
        """
        if self.state == CircuitBreakerState.OPEN:
            if self.last_failure_time and time.time() - self.last_failure_time > self.timeout:
                self.state = CircuitBreakerState.HALF_OPEN
                logging.info("🔄 회로 차단기: HALF_OPEN 상태")
            else:
                raise RuntimeError("회로 차단기가 OPEN 상태입니다")

        try:
            result = await func(*args, **kwargs)
            if self.state == CircuitBreakerState.HALF_OPEN:
                self.state = CircuitBreakerState.CLOSED
                self.failure_count = 0
                logging.info("✅ 회로 차단기: CLOSED 상태 복구")
            return result
        except Exception as e:
            self.failure_count += 1
            self.last_failure_time = time.time()

            if self.failure_count >= self.failure_threshold:
                self.state = CircuitBreakerState.OPEN
                logging.error(f"❌ 회로 차단기: OPEN 상태 ({self.failure_count} 실패)")

            raise e


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
