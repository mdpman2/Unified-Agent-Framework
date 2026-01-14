#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Unified Agent Framework - 이벤트 시스템 모듈

Pub-Sub 패턴을 위한 이벤트 버스 및 이벤트 모델
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
    이벤트 타입

    10가지 이벤트 타입:
    - Agent 생명주기: STARTED, COMPLETED, FAILED
    - Node 생명주기: NODE_STARTED, NODE_COMPLETED
    - 승인 관련: APPROVAL_REQUESTED, APPROVAL_GRANTED, APPROVAL_DENIED
    - 메시지: MESSAGE_RECEIVED, MESSAGE_SENT
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
    이벤트 버스 - Pub-Sub 패턴 구현

    주요 기능:
    - subscribe(): 이벤트 구독
    - publish(): 이벤트 발행
    - get_event_history(): 이벤트 히스토리 조회

    사용 시나리오:
    - 로깅 및 모니터링
    - 알림 전송
    - 메트릭 수집
    - 워크플로우 조율

    예시:
    ```python
    async def on_approval_requested(event):
        await send_slack_notification(event.data)

    event_bus.subscribe(EventType.APPROVAL_REQUESTED, on_approval_requested)
    ```
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
        """이벤트 발행"""
        self.event_history.append(event)

        handlers = self.subscribers.get(event.event_type, [])
        for handler in handlers:
            try:
                if asyncio.iscoroutinefunction(handler):
                    await handler(event)
                else:
                    handler(event)
            except Exception as e:
                logging.error(f"❌ 이벤트 핸들러 오류: {e}")

    def get_event_history(self, event_type: Optional[EventType] = None,
                         limit: int = 100) -> List[AgentEvent]:
        """이벤트 히스토리 조회"""
        if event_type:
            filtered = [e for e in self.event_history if e.event_type == event_type]
            return filtered[-limit:]
        return self.event_history[-limit:]

    def clear_history(self):
        """이벤트 히스토리 초기화"""
        self.event_history.clear()
