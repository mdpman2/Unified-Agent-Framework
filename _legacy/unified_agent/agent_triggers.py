#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Unified Agent Framework - 에이전트 트리거 모듈 (Agent Triggers Module)

================================================================================
📁 파일 위치: unified_agent/agent_triggers.py
📋 역할: 이벤트 기반 에이전트 자동 호출 (Event-Driven Agent Invocation)
📅 최종 업데이트: 2026년 2월 13일
📦 버전: v4.1.0
✅ 테스트: test_v41_scenarios.py
================================================================================

🎯 주요 구성 요소:
    1. TriggerManager - 트리거 등록/해제/실행 관리
    2. EventTrigger - 이벤트 기반 트리거
    3. ScheduleTrigger - 스케줄(Cron) 기반 트리거
    4. WebhookTrigger - HTTP 웹훅 기반 트리거
    5. QueueTrigger - Azure Queue/Service Bus 메시지 트리거
    6. FileChangeTrigger - 파일/Blob 변경 감지 트리거

🔧 2026년 2월 기능:
    - Azure Logic Apps 트리거 패턴 호환
    - Azure Functions 트리거 바인딩과 통합
    - 이벤트 기반 에이전트 자동 호출 (Event Grid, Service Bus)
    - Cron 기반 정기 에이전트 실행 스케줄링
    - 웹훅 기반 외부 시스템 연동
    - 파일/Blob 변경 감지 자동 처리
    - 트리거 체이닝 (하나의 에이전트 완료 → 다음 에이전트 트리거)
    - 트리거 필터링 및 조건부 실행

📌 사용 예시:
    >>> from unified_agent.agent_triggers import (
    ...     TriggerManager, EventTrigger, ScheduleTrigger,
    ...     WebhookTrigger, TriggerConfig, TriggerCondition
    ... )
    >>>
    >>> # 트리거 매니저 초기화
    >>> manager = TriggerManager(TriggerConfig(
    ...     enable_logging=True,
    ...     max_concurrent_triggers=10
    ... ))
    >>>
    >>> # 이벤트 트리거: 새 문서 업로드 시 에이전트 실행
    >>> @manager.on_event("document.uploaded")
    ... async def handle_document(event):
    ...     agent = create_research_agent()
    ...     return await agent.run(event.data)
    >>>
    >>> # 스케줄 트리거: 매일 오전 9시에 보고서 생성
    >>> @manager.on_schedule("0 9 * * *")
    ... async def daily_report():
    ...     agent = create_report_agent()
    ...     return await agent.run("일일 보고서 생성")
    >>>
    >>> # 웹훅 트리거: GitHub 이벤트 수신
    >>> @manager.on_webhook("/github/events", methods=["POST"])
    ... async def handle_github(payload):
    ...     agent = create_devops_agent()
    ...     return await agent.run(payload)

⚠️ 주의사항:
    - 트리거 무한 루프를 방지하세요 (에이전트 → 이벤트 → 트리거 → 에이전트)
    - 스케줄 트리거는 시스템 시간대(timezone)에 유의하세요
    - 프로덕션에서는 분산 락을 사용하여 중복 트리거를 방지하세요

🔗 관련 문서:
    - Azure Logic Apps Triggers: https://learn.microsoft.com/azure/logic-apps/logic-apps-overview
    - Azure Functions Triggers: https://learn.microsoft.com/azure/azure-functions/functions-triggers-bindings
    - Azure Event Grid: https://learn.microsoft.com/azure/event-grid/overview
"""

from __future__ import annotations

import asyncio
import hashlib
import hmac
import logging
import time
import uuid
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum, unique
from typing import Any

logger = logging.getLogger(__name__)

# ──────────────────────────────────────────────
# Public API
# ──────────────────────────────────────────────
__all__ = [
    # Enums
    "TriggerType",
    "TriggerStatus",
    "TriggerPriority",
    # Data Models
    "TriggerConfig",
    "TriggerEvent",
    "TriggerCondition",
    "TriggerResult",
    "TriggerMetrics",
    # Core Components
    "BaseTrigger",
    "EventTrigger",
    "ScheduleTrigger",
    "WebhookTrigger",
    "QueueTrigger",
    "FileChangeTrigger",
    "AgentCompletionTrigger",
    "TriggerManager",
]


# ══════════════════════════════════════════════
# Enums
# ══════════════════════════════════════════════


@unique
class TriggerType(str, Enum):
    """트리거 유형"""

    EVENT = "event"  # 이벤트 기반
    SCHEDULE = "schedule"  # 스케줄(Cron) 기반
    WEBHOOK = "webhook"  # HTTP 웹훅
    QUEUE = "queue"  # 메시지 큐
    FILE_CHANGE = "file_change"  # 파일 변경 감지
    AGENT_COMPLETION = "agent_completion"  # 에이전트 완료 트리거
    MANUAL = "manual"  # 수동 실행


@unique
class TriggerStatus(str, Enum):
    """트리거 상태"""

    ACTIVE = "active"  # 활성 상태
    PAUSED = "paused"  # 일시 정지
    DISABLED = "disabled"  # 비활성화
    FIRING = "firing"  # 실행 중
    ERROR = "error"  # 에러 상태
    COOLDOWN = "cooldown"  # 쿨다운 (재실행 대기)


@unique
class TriggerPriority(int, Enum):
    """트리거 우선순위"""

    CRITICAL = 0  # 즉시 실행 (장애 대응 등)
    HIGH = 10  # 높은 우선순위
    NORMAL = 50  # 일반
    LOW = 90  # 낮은 우선순위
    BACKGROUND = 100  # 백그라운드 실행


# ══════════════════════════════════════════════
# Data Models
# ══════════════════════════════════════════════


@dataclass(slots=True)
class TriggerConfig:
    """트리거 매니저 설정"""

    enable_logging: bool = True  # 트리거 이벤트 로깅
    max_concurrent_triggers: int = 10  # 동시 트리거 실행 수 제한
    default_timeout: float = 300.0  # 기본 트리거 실행 타임아웃 (초)
    cooldown_seconds: float = 5.0  # 트리거 재실행 쿨다운 (초)
    enable_dead_letter: bool = True  # 실패 이벤트 Dead Letter 큐 활성화
    max_retry_count: int = 3  # 트리거 실행 실패 시 최대 재시도 횟수
    enable_metrics: bool = True  # 트리거 메트릭 수집


@dataclass(slots=True)
class TriggerEvent:
    """트리거 이벤트 데이터"""

    event_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    event_type: str = ""  # 이벤트 유형 (e.g., "document.uploaded")
    source: str = ""  # 이벤트 소스
    data: Any = None  # 이벤트 페이로드
    metadata: dict[str, Any] = field(default_factory=dict)
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    correlation_id: str | None = None  # 관련 이벤트 추적용 ID

    def to_dict(self) -> dict[str, Any]:
        """이벤트를 딕셔너리로 변환"""
        return {
            "event_id": self.event_id,
            "event_type": self.event_type,
            "source": self.source,
            "data": self.data,
            "metadata": self.metadata,
            "timestamp": self.timestamp.isoformat(),
            "correlation_id": self.correlation_id,
        }


@dataclass(slots=True)
class TriggerCondition:
    """트리거 실행 조건"""

    field: str = ""  # 이벤트 데이터 필드 (e.g., "data.type")
    operator: str = "eq"  # 비교 연산자 (eq, ne, gt, lt, contains, regex)
    value: Any = None  # 비교 값
    negate: bool = False  # 조건 반전

    def evaluate(self, event: TriggerEvent) -> bool:
        """조건 평가"""
        # 필드 값 추출
        actual = self._get_field_value(event, self.field)
        if actual is None:
            result = False
        elif self.operator == "eq":
            result = actual == self.value
        elif self.operator == "ne":
            result = actual != self.value
        elif self.operator == "gt":
            result = actual > self.value
        elif self.operator == "lt":
            result = actual < self.value
        elif self.operator == "gte":
            result = actual >= self.value
        elif self.operator == "lte":
            result = actual <= self.value
        elif self.operator == "contains":
            result = self.value in str(actual) if actual else False
        elif self.operator == "in":
            result = actual in (self.value if isinstance(self.value, (list, set)) else [self.value])
        elif self.operator == "exists":
            result = actual is not None
        else:
            result = False

        return not result if self.negate else result

    def _get_field_value(self, event: TriggerEvent, field_path: str) -> Any:
        """이벤트에서 필드 값 추출 (점 표기법 지원)"""
        parts = field_path.split(".")
        current: Any = event

        for part in parts:
            if isinstance(current, dict):
                current = current.get(part)
            elif hasattr(current, part):
                current = getattr(current, part)
            else:
                return None

        return current


@dataclass(slots=True)
class TriggerResult:
    """트리거 실행 결과"""

    trigger_id: str
    trigger_name: str
    trigger_type: TriggerType
    event: TriggerEvent | None = None
    success: bool = False
    result: Any = None
    error: str | None = None
    duration_ms: float = 0.0
    retry_count: int = 0
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))


@dataclass(slots=True)
class TriggerMetrics:
    """트리거별 메트릭"""

    trigger_name: str
    total_fires: int = 0
    successful_fires: int = 0
    failed_fires: int = 0
    skipped_fires: int = 0
    total_duration_ms: float = 0.0
    avg_duration_ms: float = 0.0
    max_duration_ms: float = 0.0
    last_fired: datetime | None = None
    last_error: str | None = None

    def record(self, duration_ms: float, success: bool, error: str | None = None) -> None:
        """메트릭 기록"""
        self.total_fires += 1
        if success:
            self.successful_fires += 1
        else:
            self.failed_fires += 1
            self.last_error = error
        self.total_duration_ms += duration_ms
        self.avg_duration_ms = self.total_duration_ms / self.total_fires
        self.max_duration_ms = max(self.max_duration_ms, duration_ms)
        self.last_fired = datetime.now(timezone.utc)


# ══════════════════════════════════════════════
# Core Components - 트리거 기본 클래스
# ══════════════════════════════════════════════


class BaseTrigger:
    """트리거 기본 클래스

    모든 트리거는 이 클래스를 상속합니다.
    """

    def __init__(
        self,
        name: str,
        trigger_type: TriggerType,
        handler: Callable | None = None,
        conditions: list[TriggerCondition] | None = None,
        priority: TriggerPriority = TriggerPriority.NORMAL,
        timeout: float = 300.0,
        max_retries: int = 0,
        cooldown: float = 0.0,
        metadata: dict[str, Any] | None = None,
    ):
        self.trigger_id = str(uuid.uuid4())
        self.name = name
        self.trigger_type = trigger_type
        self.handler = handler
        self.conditions = conditions or []
        self.priority = priority
        self.timeout = timeout
        self.max_retries = max_retries
        self.cooldown = cooldown
        self.metadata = metadata or {}
        self.status = TriggerStatus.ACTIVE
        self._last_fired: float | None = None
        self._fire_count = 0

    def should_fire(self, event: TriggerEvent) -> bool:
        """트리거 실행 여부 확인"""
        if self.status != TriggerStatus.ACTIVE:
            return False

        # 쿨다운 확인
        if self._last_fired and self.cooldown > 0:
            elapsed = time.time() - self._last_fired
            if elapsed < self.cooldown:
                logger.debug(
                    "[Trigger] 쿨다운 중: %s (%.1f/%.1fs)",
                    self.name,
                    elapsed,
                    self.cooldown,
                )
                return False

        # 조건 확인 (모든 조건 AND)
        if self.conditions:
            return all(c.evaluate(event) for c in self.conditions)

        return True

    async def fire(self, event: TriggerEvent) -> TriggerResult:
        """트리거 실행"""
        start = time.time()
        self.status = TriggerStatus.FIRING
        self._fire_count += 1

        result = TriggerResult(
            trigger_id=self.trigger_id,
            trigger_name=self.name,
            trigger_type=self.trigger_type,
            event=event,
        )

        try:
            if self.handler:
                handler_result = await asyncio.wait_for(
                    self._execute_handler(event), timeout=self.timeout
                )
                result.result = handler_result
                result.success = True
            else:
                logger.warning("[Trigger] 핸들러 없음: %s", self.name)
                result.success = False
                result.error = "No handler registered"

        except asyncio.TimeoutError:
            result.error = f"Trigger timeout after {self.timeout}s"
            logger.error("[Trigger] 타임아웃: %s", self.name)

        except Exception as e:
            result.error = str(e)
            logger.error("[Trigger] 실행 에러: %s - %s", self.name, str(e))

        finally:
            self._last_fired = time.time()
            self.status = TriggerStatus.ACTIVE
            result.duration_ms = (time.time() - start) * 1000

        return result

    async def _execute_handler(self, event: TriggerEvent) -> Any:
        """핸들러 실행"""
        if asyncio.iscoroutinefunction(self.handler):
            return await self.handler(event)
        else:
            return self.handler(event)

    def pause(self) -> None:
        """트리거 일시 정지"""
        self.status = TriggerStatus.PAUSED
        logger.info("[Trigger] 일시 정지: %s", self.name)

    def resume(self) -> None:
        """트리거 재개"""
        self.status = TriggerStatus.ACTIVE
        logger.info("[Trigger] 재개: %s", self.name)

    def disable(self) -> None:
        """트리거 비활성화"""
        self.status = TriggerStatus.DISABLED
        logger.info("[Trigger] 비활성화: %s", self.name)

    def to_dict(self) -> dict[str, Any]:
        """트리거 정보를 딕셔너리로 변환"""
        return {
            "trigger_id": self.trigger_id,
            "name": self.name,
            "type": self.trigger_type.value,
            "status": self.status.value,
            "priority": self.priority.name,
            "timeout": self.timeout,
            "max_retries": self.max_retries,
            "cooldown": self.cooldown,
            "fire_count": self._fire_count,
            "conditions_count": len(self.conditions),
            "metadata": self.metadata,
        }


# ══════════════════════════════════════════════
# Core Components - 구체 트리거 구현
# ══════════════════════════════════════════════


class EventTrigger(BaseTrigger):
    """이벤트 기반 트리거

    특정 이벤트 유형이 발생하면 핸들러를 실행합니다.
    Azure Event Grid, Service Bus 이벤트와 호환됩니다.
    """

    def __init__(
        self,
        name: str,
        event_types: list[str] | None = None,
        source_filter: str | None = None,
        **kwargs,
    ):
        super().__init__(
            name=name, trigger_type=TriggerType.EVENT, **kwargs
        )
        self.event_types = event_types or ["*"]  # 구독할 이벤트 유형
        self.source_filter = source_filter  # 이벤트 소스 필터

    def should_fire(self, event: TriggerEvent) -> bool:
        """이벤트 유형 및 소스 필터 확인"""
        if not super().should_fire(event):
            return False

        # 이벤트 유형 확인
        if "*" not in self.event_types:
            if event.event_type not in self.event_types:
                return False

        # 소스 필터 확인
        if self.source_filter and self.source_filter not in event.source:
            return False

        return True


class ScheduleTrigger(BaseTrigger):
    """스케줄(Cron) 기반 트리거

    Cron 표현식을 사용하여 정기적으로 에이전트를 실행합니다.
    """

    def __init__(
        self,
        name: str,
        cron_expression: str = "*/5 * * * *",  # 기본: 5분마다
        timezone_name: str = "UTC",
        **kwargs,
    ):
        super().__init__(
            name=name, trigger_type=TriggerType.SCHEDULE, **kwargs
        )
        self.cron_expression = cron_expression
        self.timezone_name = timezone_name
        self._next_run: datetime | None = None
        self._running = False
        self._task: asyncio.Task | None = None

    def parse_cron(self) -> dict[str, Any]:
        """Cron 표현식 파싱 (간단한 파서)"""
        parts = self.cron_expression.split()
        if len(parts) != 5:
            raise ValueError(
                f"Invalid cron expression: '{self.cron_expression}'. "
                "Expected 5 fields: minute hour day month weekday"
            )

        return {
            "minute": parts[0],
            "hour": parts[1],
            "day": parts[2],
            "month": parts[3],
            "weekday": parts[4],
        }

    def _matches_cron_field(self, field_expr: str, value: int, max_value: int) -> bool:
        """Cron 필드가 현재 값과 일치하는지 확인"""
        if field_expr == "*":
            return True

        # */n 형태 (매 n번째)
        if field_expr.startswith("*/"):
            interval = int(field_expr[2:])
            return value % interval == 0

        # 쉼표로 구분된 값
        if "," in field_expr:
            return value in [int(v) for v in field_expr.split(",")]

        # 범위 (a-b)
        if "-" in field_expr:
            start, end = field_expr.split("-")
            return int(start) <= value <= int(end)

        # 단일 값
        return value == int(field_expr)

    def should_fire_now(self) -> bool:
        """현재 시각이 Cron 표현식과 일치하는지 확인"""
        if self.status != TriggerStatus.ACTIVE:
            return False

        now = datetime.now(timezone.utc)
        cron = self.parse_cron()

        return (
            self._matches_cron_field(cron["minute"], now.minute, 59)
            and self._matches_cron_field(cron["hour"], now.hour, 23)
            and self._matches_cron_field(cron["day"], now.day, 31)
            and self._matches_cron_field(cron["month"], now.month, 12)
            and self._matches_cron_field(cron["weekday"], now.weekday(), 6)
        )

    async def start_scheduler(self, check_interval: float = 30.0) -> None:
        """스케줄러 시작 (백그라운드에서 Cron 확인)"""
        self._running = True
        logger.info(
            "[ScheduleTrigger] 스케줄러 시작: %s (cron=%s)",
            self.name,
            self.cron_expression,
        )

        while self._running:
            try:
                if self.should_fire_now():
                    event = TriggerEvent(
                        event_type="schedule.fired",
                        source=f"schedule:{self.name}",
                        data={"cron": self.cron_expression, "time": datetime.now(timezone.utc).isoformat()},
                    )
                    await self.fire(event)
            except Exception as e:
                logger.error("[ScheduleTrigger] 스케줄 에러: %s - %s", self.name, str(e))

            await asyncio.sleep(check_interval)

    async def stop_scheduler(self) -> None:
        """스케줄러 중지"""
        self._running = False
        if self._task:
            self._task.cancel()
        logger.info("[ScheduleTrigger] 스케줄러 중지: %s", self.name)


class WebhookTrigger(BaseTrigger):
    """HTTP 웹훅 기반 트리거

    외부 시스템에서 HTTP 요청을 받아 에이전트를 트리거합니다.
    GitHub, Slack, Azure DevOps 등의 웹훅과 연동됩니다.
    """

    def __init__(
        self,
        name: str,
        path: str = "/webhook",
        methods: list[str] | None = None,
        secret: str | None = None,
        validate_signature: bool = True,
        **kwargs,
    ):
        super().__init__(
            name=name, trigger_type=TriggerType.WEBHOOK, **kwargs
        )
        self.path = path
        self.methods = methods or ["POST"]
        self.secret = secret
        self.validate_signature = validate_signature
        self._received_count = 0

    async def handle_request(
        self,
        method: str,
        headers: dict[str, str],
        body: Any,
        query_params: dict[str, str] | None = None,
    ) -> TriggerResult:
        """HTTP 요청 처리"""
        self._received_count += 1

        # HTTP 메서드 확인
        if method.upper() not in [m.upper() for m in self.methods]:
            logger.warning(
                "[Webhook] 허용되지 않는 메서드: %s (allowed=%s)",
                method,
                self.methods,
            )
            return TriggerResult(
                trigger_id=self.trigger_id,
                trigger_name=self.name,
                trigger_type=TriggerType.WEBHOOK,
                success=False,
                error=f"Method {method} not allowed",
            )

        # 시그니처 검증
        if self.validate_signature and self.secret:
            signature = headers.get("X-Hub-Signature-256") or headers.get(
                "X-Signature"
            )
            if not self._verify_signature(body, signature):
                logger.warning("[Webhook] 시그니처 검증 실패: %s", self.name)
                return TriggerResult(
                    trigger_id=self.trigger_id,
                    trigger_name=self.name,
                    trigger_type=TriggerType.WEBHOOK,
                    success=False,
                    error="Invalid signature",
                )

        # 이벤트 생성 및 트리거 실행
        event = TriggerEvent(
            event_type=f"webhook.{self.path.strip('/').replace('/', '.')}",
            source=f"webhook:{self.path}",
            data=body,
            metadata={
                "method": method,
                "headers": {k: v for k, v in headers.items() if k.lower() != "authorization"},
                "query_params": query_params or {},
            },
        )

        if not self.should_fire(event):
            return TriggerResult(
                trigger_id=self.trigger_id,
                trigger_name=self.name,
                trigger_type=TriggerType.WEBHOOK,
                success=False,
                error="Trigger conditions not met",
            )

        return await self.fire(event)

    def _verify_signature(self, body: Any, signature: str | None) -> bool:
        """웹훅 시그니처 검증 (HMAC-SHA256)"""
        if not signature:
            return False

        body_bytes = str(body).encode("utf-8") if not isinstance(body, bytes) else body
        expected = hmac.new(
            self.secret.encode("utf-8"), body_bytes, hashlib.sha256
        ).hexdigest()

        return hmac.compare_digest(f"sha256={expected}", signature)

    @property
    def received_count(self) -> int:
        """수신된 요청 수"""
        return self._received_count


class QueueTrigger(BaseTrigger):
    """메시지 큐 트리거

    Azure Queue Storage / Service Bus 메시지를 감지하여 에이전트를 트리거합니다.
    """

    def __init__(
        self,
        name: str,
        queue_name: str = "agent-tasks",
        connection_string: str | None = None,
        batch_size: int = 1,
        visibility_timeout: float = 30.0,
        polling_interval: float = 5.0,
        **kwargs,
    ):
        super().__init__(
            name=name, trigger_type=TriggerType.QUEUE, **kwargs
        )
        self.queue_name = queue_name
        self.connection_string = connection_string
        self.batch_size = batch_size
        self.visibility_timeout = visibility_timeout
        self.polling_interval = polling_interval
        self._running = False
        self._processed_count = 0

    async def start_polling(self) -> None:
        """큐 폴링 시작"""
        self._running = True
        logger.info(
            "[QueueTrigger] 폴링 시작: %s (queue=%s, interval=%.1fs)",
            self.name,
            self.queue_name,
            self.polling_interval,
        )

        while self._running:
            try:
                messages = await self._receive_messages()
                for msg in messages:
                    event = TriggerEvent(
                        event_type="queue.message_received",
                        source=f"queue:{self.queue_name}",
                        data=msg.get("body"),
                        metadata={
                            "message_id": msg.get("message_id"),
                            "queue_name": self.queue_name,
                            "dequeue_count": msg.get("dequeue_count", 1),
                        },
                    )

                    if self.should_fire(event):
                        result = await self.fire(event)
                        if result.success:
                            await self._delete_message(msg.get("message_id", ""))
                            self._processed_count += 1
                        else:
                            logger.warning(
                                "[QueueTrigger] 메시지 처리 실패: %s",
                                msg.get("message_id"),
                            )

            except Exception as e:
                logger.error(
                    "[QueueTrigger] 폴링 에러: %s - %s", self.name, str(e)
                )

            await asyncio.sleep(self.polling_interval)

    async def stop_polling(self) -> None:
        """큐 폴링 중지"""
        self._running = False
        logger.info("[QueueTrigger] 폴링 중지: %s", self.name)

    async def _receive_messages(self) -> list[dict[str, Any]]:
        """큐에서 메시지 수신 (시뮬레이션)

        프로덕션에서는 Azure Queue Storage SDK를 사용합니다.
        """
        await asyncio.sleep(0.01)
        return []  # 시뮬레이션: 빈 응답

    async def _delete_message(self, message_id: str) -> None:
        """처리된 메시지 삭제 (시뮬레이션)"""
        await asyncio.sleep(0.01)
        logger.debug("[QueueTrigger] 메시지 삭제: %s", message_id)

    @property
    def processed_count(self) -> int:
        """처리된 메시지 수"""
        return self._processed_count


class FileChangeTrigger(BaseTrigger):
    """파일 변경 감지 트리거

    Azure Blob Storage / 로컬 파일시스템의 변경을 감지하여 에이전트를 트리거합니다.
    """

    def __init__(
        self,
        name: str,
        watch_path: str = "",
        patterns: list[str] | None = None,
        change_types: list[str] | None = None,
        polling_interval: float = 10.0,
        **kwargs,
    ):
        super().__init__(
            name=name, trigger_type=TriggerType.FILE_CHANGE, **kwargs
        )
        self.watch_path = watch_path
        self.patterns = patterns or ["*"]  # 파일 패턴 (e.g., *.pdf, *.docx)
        self.change_types = change_types or ["created", "modified", "deleted"]
        self.polling_interval = polling_interval
        self._running = False
        self._known_files: dict[str, float] = {}  # path -> mtime

    async def start_watching(self) -> None:
        """파일 감시 시작"""
        self._running = True
        logger.info(
            "[FileChangeTrigger] 감시 시작: %s (path=%s, patterns=%s)",
            self.name,
            self.watch_path,
            self.patterns,
        )

        # 초기 파일 목록 스캔
        self._known_files = await self._scan_files()

        while self._running:
            try:
                current_files = await self._scan_files()
                changes = self._detect_changes(current_files)

                for change_type, file_path, mtime in changes:
                    if change_type in self.change_types:
                        event = TriggerEvent(
                            event_type=f"file.{change_type}",
                            source=f"file:{self.watch_path}",
                            data={
                                "file_path": file_path,
                                "change_type": change_type,
                                "modified_time": mtime,
                            },
                        )

                        if self.should_fire(event):
                            await self.fire(event)

                self._known_files = current_files

            except Exception as e:
                logger.error(
                    "[FileChangeTrigger] 감시 에러: %s - %s", self.name, str(e)
                )

            await asyncio.sleep(self.polling_interval)

    async def stop_watching(self) -> None:
        """파일 감시 중지"""
        self._running = False
        logger.info("[FileChangeTrigger] 감시 중지: %s", self.name)

    async def _scan_files(self) -> dict[str, float]:
        """파일 스캔 (시뮬레이션)

        프로덕션에서는 Azure Blob Storage SDK 또는 os.scandir을 사용합니다.
        """
        await asyncio.sleep(0.01)
        return {}

    def _detect_changes(
        self, current: dict[str, float]
    ) -> list[tuple[str, str, float]]:
        """파일 변경 감지"""
        changes: list[tuple[str, str, float]] = []

        # 새 파일 또는 수정된 파일
        for path, mtime in current.items():
            if path not in self._known_files:
                changes.append(("created", path, mtime))
            elif self._known_files[path] != mtime:
                changes.append(("modified", path, mtime))

        # 삭제된 파일
        for path, mtime in self._known_files.items():
            if path not in current:
                changes.append(("deleted", path, mtime))

        return changes


class AgentCompletionTrigger(BaseTrigger):
    """에이전트 완료 트리거

    다른 에이전트의 실행이 완료되면 차기 에이전트를 트리거합니다.
    에이전트 체이닝(Chaining) 패턴을 구현합니다.
    """

    def __init__(
        self,
        name: str,
        source_agent_ids: list[str] | None = None,
        require_success: bool = True,
        transform_fn: Callable | None = None,
        **kwargs,
    ):
        super().__init__(
            name=name, trigger_type=TriggerType.AGENT_COMPLETION, **kwargs
        )
        self.source_agent_ids = source_agent_ids or []
        self.require_success = require_success
        self.transform_fn = transform_fn  # 이전 에이전트 결과 변환 함수

    def should_fire(self, event: TriggerEvent) -> bool:
        """소스 에이전트 확인"""
        if not super().should_fire(event):
            return False

        # 소스 에이전트 ID 확인
        if self.source_agent_ids:
            source_id = event.metadata.get("agent_id", "")
            if source_id not in self.source_agent_ids:
                return False

        # 성공 여부 확인
        if self.require_success:
            if not event.metadata.get("success", False):
                return False

        return True

    async def _execute_handler(self, event: TriggerEvent) -> Any:
        """핸들러 실행 (결과 변환 적용)"""
        if self.transform_fn:
            transformed_data = self.transform_fn(event.data)
            event.data = transformed_data

        return await super()._execute_handler(event)


# ══════════════════════════════════════════════
# Core Components - 트리거 매니저
# ══════════════════════════════════════════════


class TriggerManager:
    """트리거 매니저 - 모든 트리거의 등록/해제/실행을 관리

    Azure Logic Apps 트리거 패턴을 에이전트 프레임워크에 적용합니다.
    """

    def __init__(self, config: TriggerConfig | None = None):
        self.config = config or TriggerConfig()
        self._triggers: dict[str, BaseTrigger] = {}
        self._event_handlers: dict[str, list[str]] = {}  # event_type -> [trigger_ids]
        self._metrics: dict[str, TriggerMetrics] = {}
        self._dead_letter: list[TriggerEvent] = []
        self._active_fires: int = 0
        self._semaphore = asyncio.Semaphore(self.config.max_concurrent_triggers)
        logger.info("TriggerManager 초기화 완료")

    # ── 등록 메서드 ──

    def register(self, trigger: BaseTrigger) -> str:
        """트리거 등록"""
        self._triggers[trigger.trigger_id] = trigger
        if self.config.enable_metrics:
            self._metrics[trigger.trigger_id] = TriggerMetrics(
                trigger_name=trigger.name
            )
        logger.info(
            "[TriggerManager] 트리거 등록: %s (type=%s, id=%s)",
            trigger.name,
            trigger.trigger_type.value,
            trigger.trigger_id,
        )
        return trigger.trigger_id

    def unregister(self, trigger_id: str) -> bool:
        """트리거 해제"""
        trigger = self._triggers.pop(trigger_id, None)
        if trigger:
            self._metrics.pop(trigger_id, None)
            logger.info("[TriggerManager] 트리거 해제: %s", trigger.name)
            return True
        return False

    # ── 데코레이터 메서드 ──

    def on_event(
        self,
        event_type: str,
        conditions: list[TriggerCondition] | None = None,
        **kwargs,
    ) -> Callable:
        """이벤트 트리거 데코레이터

        Usage:
            @manager.on_event("document.uploaded")
            async def handle_doc(event):
                ...
        """

        def decorator(fn: Callable) -> Callable:
            trigger = EventTrigger(
                name=fn.__name__,
                event_types=[event_type],
                handler=fn,
                conditions=conditions,
                **kwargs,
            )
            trigger_id = self.register(trigger)

            # 이벤트 유형별 매핑
            if event_type not in self._event_handlers:
                self._event_handlers[event_type] = []
            self._event_handlers[event_type].append(trigger_id)

            return fn

        return decorator

    def on_schedule(
        self,
        cron_expression: str,
        timezone_name: str = "UTC",
        **kwargs,
    ) -> Callable:
        """스케줄 트리거 데코레이터

        Usage:
            @manager.on_schedule("0 9 * * *")
            async def daily_task():
                ...
        """

        def decorator(fn: Callable) -> Callable:
            trigger = ScheduleTrigger(
                name=fn.__name__,
                cron_expression=cron_expression,
                timezone_name=timezone_name,
                handler=fn,
                **kwargs,
            )
            self.register(trigger)
            return fn

        return decorator

    def on_webhook(
        self,
        path: str,
        methods: list[str] | None = None,
        secret: str | None = None,
        **kwargs,
    ) -> Callable:
        """웹훅 트리거 데코레이터

        Usage:
            @manager.on_webhook("/github/events", methods=["POST"])
            async def handle_github(payload):
                ...
        """

        def decorator(fn: Callable) -> Callable:
            trigger = WebhookTrigger(
                name=fn.__name__,
                path=path,
                methods=methods,
                secret=secret,
                handler=fn,
                **kwargs,
            )
            self.register(trigger)
            return fn

        return decorator

    # ── 이벤트 디스패치 ──

    async def dispatch_event(self, event: TriggerEvent) -> list[TriggerResult]:
        """이벤트 디스패치 - 매칭되는 모든 트리거 실행"""
        results: list[TriggerResult] = []
        matching_triggers: list[BaseTrigger] = []

        # 매칭되는 트리거 찾기
        for trigger in self._triggers.values():
            if trigger.should_fire(event):
                matching_triggers.append(trigger)

        if not matching_triggers:
            logger.debug(
                "[TriggerManager] 매칭 트리거 없음: event_type=%s",
                event.event_type,
            )
            if self.config.enable_dead_letter:
                self._dead_letter.append(event)
            return results

        # 우선순위 순으로 정렬
        matching_triggers.sort(key=lambda t: t.priority.value)

        logger.info(
            "[TriggerManager] %d개 트리거 매칭됨 (event=%s)",
            len(matching_triggers),
            event.event_type,
        )

        # 동시 실행 (세마포어로 제한)
        tasks = [
            self._fire_with_semaphore(trigger, event)
            for trigger in matching_triggers
        ]
        results = await asyncio.gather(*tasks, return_exceptions=False)

        return results

    async def _fire_with_semaphore(
        self, trigger: BaseTrigger, event: TriggerEvent
    ) -> TriggerResult:
        """세마포어로 동시 실행 제한"""
        async with self._semaphore:
            self._active_fires += 1
            try:
                result = await trigger.fire(event)

                # 메트릭 기록
                if self.config.enable_metrics and trigger.trigger_id in self._metrics:
                    self._metrics[trigger.trigger_id].record(
                        duration_ms=result.duration_ms,
                        success=result.success,
                        error=result.error,
                    )

                # 재시도 처리
                if not result.success and trigger.max_retries > 0:
                    result = await self._retry_trigger(
                        trigger, event, trigger.max_retries
                    )

                return result

            finally:
                self._active_fires -= 1

    async def _retry_trigger(
        self, trigger: BaseTrigger, event: TriggerEvent, max_retries: int
    ) -> TriggerResult:
        """트리거 재시도"""
        result = TriggerResult(
            trigger_id=trigger.trigger_id,
            trigger_name=trigger.name,
            trigger_type=trigger.trigger_type,
            event=event,
        )

        for attempt in range(1, max_retries + 1):
            delay = 2.0 ** attempt  # 지수 백오프
            logger.info(
                "[TriggerManager] 재시도 %d/%d - %s (%.1fs 후)",
                attempt,
                max_retries,
                trigger.name,
                delay,
            )
            await asyncio.sleep(delay)

            result = await trigger.fire(event)
            result.retry_count = attempt

            if result.success:
                break

        return result

    # ── 조회 메서드 ──

    def get_trigger(self, trigger_id: str) -> BaseTrigger | None:
        """트리거 조회"""
        return self._triggers.get(trigger_id)

    def get_all_triggers(self) -> list[dict[str, Any]]:
        """모든 트리거 목록 조회"""
        return [t.to_dict() for t in self._triggers.values()]

    def get_active_triggers(self) -> list[BaseTrigger]:
        """활성 트리거 목록"""
        return [
            t
            for t in self._triggers.values()
            if t.status == TriggerStatus.ACTIVE
        ]

    def get_metrics(self) -> dict[str, TriggerMetrics]:
        """트리거 메트릭 조회"""
        return dict(self._metrics)

    def get_dead_letter_queue(self) -> list[TriggerEvent]:
        """Dead Letter 큐 조회"""
        return list(self._dead_letter)

    def clear_dead_letter(self) -> int:
        """Dead Letter 큐 초기화"""
        count = len(self._dead_letter)
        self._dead_letter.clear()
        return count

    # ── 관리 메서드 ──

    def pause_all(self) -> int:
        """모든 트리거 일시 정지"""
        count = 0
        for trigger in self._triggers.values():
            if trigger.status == TriggerStatus.ACTIVE:
                trigger.pause()
                count += 1
        logger.info("[TriggerManager] %d개 트리거 일시 정지", count)
        return count

    def resume_all(self) -> int:
        """모든 트리거 재개"""
        count = 0
        for trigger in self._triggers.values():
            if trigger.status == TriggerStatus.PAUSED:
                trigger.resume()
                count += 1
        logger.info("[TriggerManager] %d개 트리거 재개", count)
        return count

    async def start_all_schedulers(self) -> None:
        """모든 스케줄 트리거의 스케줄러 시작"""
        tasks = []
        for trigger in self._triggers.values():
            if isinstance(trigger, ScheduleTrigger):
                task = asyncio.create_task(trigger.start_scheduler())
                trigger._task = task
                tasks.append(task)
        if tasks:
            logger.info("[TriggerManager] %d개 스케줄러 시작", len(tasks))

    async def stop_all_schedulers(self) -> None:
        """모든 스케줄 트리거의 스케줄러 중지"""
        for trigger in self._triggers.values():
            if isinstance(trigger, ScheduleTrigger):
                await trigger.stop_scheduler()

    def get_summary(self) -> dict[str, Any]:
        """트리거 매니저 요약 정보"""
        type_counts: dict[str, int] = {}
        status_counts: dict[str, int] = {}
        for trigger in self._triggers.values():
            t_type = trigger.trigger_type.value
            t_status = trigger.status.value
            type_counts[t_type] = type_counts.get(t_type, 0) + 1
            status_counts[t_status] = status_counts.get(t_status, 0) + 1

        return {
            "total_triggers": len(self._triggers),
            "active_fires": self._active_fires,
            "by_type": type_counts,
            "by_status": status_counts,
            "dead_letter_count": len(self._dead_letter),
            "config": {
                "max_concurrent": self.config.max_concurrent_triggers,
                "default_timeout": self.config.default_timeout,
                "max_retries": self.config.max_retry_count,
            },
        }
