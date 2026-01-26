#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Unified Agent Framework - 이벤트 시스템 모듈 (Events Module)

================================================================================
📁 파일 위치: unified_agent/events.py
📋 역할: Pub-Sub 패턴 기반 이벤트 버스 및 이벤트 모델 제공
📅 최종 업데이트: 2026년 1월
================================================================================

🎯 주요 구성 요소:

    📌 EventType (Enum):
        에이전트 및 워크플로우 생명주기 이벤트 타입 정의
        - Agent 생명주기: STARTED, COMPLETED, FAILED
        - Node 생명주기: NODE_STARTED, NODE_COMPLETED
        - 승인 관련: APPROVAL_REQUESTED, APPROVAL_GRANTED, APPROVAL_DENIED
        - 메시지: MESSAGE_RECEIVED, MESSAGE_SENT

    📌 AgentEvent (Pydantic Model):
        이벤트 데이터 모델
        - event_type: 이벤트 타입
        - timestamp: 발생 시간 (UTC ISO 8601)
        - agent_name: 에이전트 이름
        - node_name: 노드 이름
        - data: 추가 데이터 (Dict)

    📌 EventBus:
        Pub-Sub 패턴 구현 클래스
        - subscribe(): 이벤트 구독
        - unsubscribe(): 구독 해제
        - publish(): 이벤트 발행 (비동기)
        - get_event_history(): 히스토리 조회

🔧 사용 시나리오:
    - 로깅 및 모니터링: 모든 에이전트 활동 기록
    - 알림 전송: Slack, Teams, 이메일 알림
    - 메트릭 수집: Application Insights, Prometheus
    - 워크플로우 조율: 이벤트 기반 상태 머신
    - 승인 워크플로우: Human-in-the-loop 통지

📌 사용 예시:

    예제 1: 이벤트 구독 및 발행
    ----------------------------------------
    >>> from unified_agent.events import EventBus, EventType, AgentEvent
    >>>
    >>> # 이벤트 버스 생성
    >>> event_bus = EventBus()
    >>>
    >>> # 이벤트 핸들러 정의
    >>> async def on_agent_completed(event: AgentEvent):
    ...     print(f"✅ 에이전트 완료: {event.agent_name}")
    ...     print(f"   결과: {event.data.get('result')}")
    >>>
    >>> # 이벤트 구독
    >>> event_bus.subscribe(EventType.AGENT_COMPLETED, on_agent_completed)
    >>>
    >>> # 이벤트 발행
    >>> await event_bus.publish(AgentEvent(
    ...     event_type=EventType.AGENT_COMPLETED,
    ...     agent_name="assistant",
    ...     data={"result": "작업 완료!"}
    ... ))

    예제 2: 승인 요청 알림
    ----------------------------------------
    >>> async def send_slack_notification(event: AgentEvent):
    ...     # Slack으로 승인 요청 알림 전송
    ...     await slack_client.send(
    ...         channel="#approvals",
    ...         text=f"승인 필요: {event.data['action']}"
    ...     )
    >>>
    >>> event_bus.subscribe(EventType.APPROVAL_REQUESTED, send_slack_notification)

⚠️ 주의사항:
    - 모든 핸들러는 비동기(async)로 정의해야 합니다.
    - publish()는 모든 핸들러를 병렬로 실행합니다.
    - event_history는 메모리에 저장되므로 장기 실행 시 주의가 필요합니다.

🔗 참고:
    - Pub-Sub 패턴: https://en.wikipedia.org/wiki/Publish%E2%80%93subscribe_pattern
    - asyncio: https://docs.python.org/3/library/asyncio.html
"""

import asyncio
import logging
from collections import defaultdict
from datetime import datetime, timezone
from enum import Enum
from typing import Callable, Dict, List, Optional, Any

from pydantic import BaseModel, Field

__all__ = [
    "EventType",
    "AgentEvent",
    "EventBus",
]


class EventType(str, Enum):
    """
    이벤트 타입 열거형 (Event Type Enum)

    ================================================================================
    📋 역할: 프레임워크에서 발생하는 모든 이벤트 타입 정의
    📅 최종 업데이트: 2026년 1월
    ================================================================================

    📌 이벤트 카테고리:

        Agent 생명주기 (3개):
        - AGENT_STARTED: 에이전트 실행 시작
        - AGENT_COMPLETED: 에이전트 실행 완료 (성공)
        - AGENT_FAILED: 에이전트 실행 실패 (오류)

        Node 생명주기 (2개):
        - NODE_STARTED: 워크플로우 노드 실행 시작
        - NODE_COMPLETED: 워크플로우 노드 실행 완료

        승인 관련 (3개) - Human-in-the-loop:
        - APPROVAL_REQUESTED: 사용자 승인 요청
        - APPROVAL_GRANTED: 승인 완료
        - APPROVAL_DENIED: 승인 거부

        메시지 (2개):
        - MESSAGE_RECEIVED: 사용자 메시지 수신
        - MESSAGE_SENT: 에이전트 메시지 전송

    📌 사용 예시:
        >>> from unified_agent.events import EventType
        >>>
        >>> # 이벤트 타입 확인
        >>> event_type = EventType.AGENT_COMPLETED
        >>> print(event_type.value)  # "agent_completed"
        >>>
        >>> # 문자열로 변환 (자동)
        >>> print(str(EventType.APPROVAL_REQUESTED))  # "approval_requested"

    ⚠️ 주의사항:
        - str을 상속하여 JSON 직렬화 시 자동 문자열 변환
    """
    AGENT_STARTED = "agent_started"
    AGENT_COMPLETED = "agent_completed"
    AGENT_FAILED = "agent_failed"
    NODE_STARTED = "node_started"
    NODE_COMPLETED = "node_completed"
    APPROVAL_REQUESTED = "approval_requested"
    APPROVAL_GRANTED = "approval_granted"
    APPROVAL_DENIED = "approval_denied"
    MESSAGE_RECEIVED = "message_received"
    MESSAGE_SENT = "message_sent"


class AgentEvent(BaseModel):
    """Agent 이벤트 모델"""
    event_type: EventType
    timestamp: str = Field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    agent_name: Optional[str] = None
    node_name: Optional[str] = None
    data: Dict[str, Any] = Field(default_factory=dict)


class EventBus:
    """
    이벤트 버스 - Pub-Sub (Publisher-Subscriber) 패턴 구현

    ================================================================================
    📋 역할: 에이전트와 워크플로우 간의 느슨한 결합(Loose Coupling) 통신
    📅 최종 업데이트: 2026년 1월
    ================================================================================

    🎯 Pub-Sub 패턴 설명:
        발행자(Publisher)와 구독자(Subscriber)가 직접 통신하지 않고
        이벤트 버스를 통해 간접적으로 통신하는 패턴입니다.

        장점:
        - 구성 요소 간 결합도 감소
        - 새로운 구독자 추가 용이
        - 비동기 처리 지원
        - 테스트 용이성 향상

    🔧 주요 메서드:
        - subscribe(event_type, handler): 이벤트 구독
        - unsubscribe(event_type, handler): 구독 해제
        - publish(event): 이벤트 발행 (모든 핸들러 병렬 실행)
        - get_event_history(): 이벤트 히스토리 조회

    📌 사용 예시:

        >>> event_bus = EventBus()
        >>>
        >>> # 1. 이벤트 핸들러 정의 (비동기 함수)
        >>> async def on_approval_requested(event: AgentEvent):
        ...     # Slack으로 승인 알림 전송
        ...     await slack_client.post_message(
        ...         channel="#approvals",
        ...         text=f"승인 필요: {event.data.get('action')}",
        ...         blocks=[...]
        ...     )
        >>>
        >>> # 2. 이벤트 구독
        >>> event_bus.subscribe(EventType.APPROVAL_REQUESTED, on_approval_requested)
        >>>
        >>> # 3. 에이전트에서 이벤트 발행
        >>> await event_bus.publish(AgentEvent(
        ...     event_type=EventType.APPROVAL_REQUESTED,
        ...     agent_name="payment_agent",
        ...     data={"action": "process_payment", "amount": 1000000}
        ... ))
        >>>
        >>> # 4. 히스토리 조회
        >>> history = event_bus.get_event_history()
        >>> print(f"총 {len(history)}개 이벤트 발생")

    🎯 사용 시나리오:
        - 로깅 및 모니터링: 모든 에이전트 활동 기록
        - 알림 전송: Slack, Teams, 이메일 통지
        - 메트릭 수집: Application Insights, Prometheus, Datadog
        - 워크플로우 조율: 이벤트 기반 상태 머신
        - 감사 로그: 보안 및 컴플라이언스

    ⚠️ 주의사항:
        - 모든 핸들러는 async def로 정의해야 합니다.
        - publish()는 asyncio.gather로 모든 핸들러를 병렬 실행합니다.
        - event_history는 인메모리 저장이므로 장기 실행 시 주기적 정리 필요
        - 핸들러에서 발생한 예외는 로깅되지만 다른 핸들러 실행을 차단하지 않음

    🔗 참고:
        - Pub-Sub 패턴: https://en.wikipedia.org/wiki/Publish-subscribe_pattern
        - asyncio.gather: https://docs.python.org/3/library/asyncio-task.html#asyncio.gather
    """
    __slots__ = ('subscribers', 'event_history')

    def __init__(self):
        self.subscribers: Dict[EventType, List[Callable]] = defaultdict(list)
        self.event_history: List[AgentEvent] = []

    def subscribe(self, event_type: EventType, handler: Callable):
        """이벤트 구독"""
        self.subscribers[event_type].append(handler)
        logging.info(f"📢 이벤트 구독: {event_type}")

    def unsubscribe(self, event_type: EventType, handler: Callable):
        """이벤트 구독 해제"""
        if handler in self.subscribers[event_type]:
            self.subscribers[event_type].remove(handler)
            logging.info(f"🔕 이벤트 구독 해제: {event_type}")

    async def publish(self, event: AgentEvent):
        """이벤트 발행 (최적화: 병렬 실행)"""
        self.event_history.append(event)

        handlers = self.subscribers.get(event.event_type, [])
        if not handlers:
            return

        # 병렬 실행을 위한 태스크 수집
        tasks = []
        sync_handlers = []

        for handler in handlers:
            if asyncio.iscoroutinefunction(handler):
                tasks.append(handler(event))
            else:
                sync_handlers.append(handler)

        # 동기 핸들러 먼저 실행
        for handler in sync_handlers:
            try:
                handler(event)
            except Exception as e:
                logging.error(f"❌ 이벤트 핸들러 오류: {e}")

        # 비동기 핸들러 병렬 실행 (return_exceptions=True로 예외 격리)
        if tasks:
            results = await asyncio.gather(*tasks, return_exceptions=True)
            for i, result in enumerate(results):
                if isinstance(result, Exception):
                    logging.error(f"❌ 이벤트 핸들러 오류: {result}")

    def get_event_history(self, event_type: Optional[EventType] = None,
                         limit: int = 100) -> List[AgentEvent]:
        """이벤트 히스토리 조회 (최적화: 역순 반복)"""
        if event_type is None:
            return self.event_history[-limit:] if len(self.event_history) > limit else list(self.event_history)

        # 역순으로 검색하여 limit개만 수집 (효율적)
        result = []
        for event in reversed(self.event_history):
            if event.event_type == event_type:
                result.append(event)
                if len(result) >= limit:
                    break
        result.reverse()
        return result

    def clear_history(self):
        """이벤트 히스토리 초기화"""
        self.event_history.clear()
