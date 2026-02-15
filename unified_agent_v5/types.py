#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Unified Agent Framework v5 — Core Types

================================================================================
v4.1 대응: models.py + interfaces.py 통합
축소 이유: v4.1의 30+ 모델 타입(AgentRole, AgentState, PlanStep, AgentConfig,
          ConversationMessage, SkillResult, TraceEvent, MetricData, AlertRule
          등)을 OpenAI ChatCompletion 표준 6개 타입으로 단순화.
          프레임워크마다 다른 메시지 형식을 쓰던 문제를 OpenAI API
          표준으로 통일하여 엔진 간 전환 비용을 0으로 만듬.
================================================================================

v5 핵심 타입 (6개):
    - Role: 메시지 역할 (system/user/assistant/tool)
    - Message: 통합 메시지 — OpenAI ChatCompletion 형식
    - ToolCall: 도구 호출 요청
    - ToolResult: 도구 실행 결과
    - AgentResult: run_agent() 반환값 (content + usage + metadata)
    - StreamChunk: 스트리밍 응답 청크

모든 엔진(Direct, LangChain, CrewAI)이 동일한 입출력 타입을 사용합니다.
"""

from __future__ import annotations

import json
import uuid
from dataclasses import dataclass, field
from enum import Enum
from typing import Any


class Role(str, Enum):
    """
    메시지 역할 (OpenAI ChatCompletion 표준)

    v4.1 대응: AgentRole(18개 역할) → 4개로 축소.
    이유: OpenAI API 표준 4개 역할만으로 모든 시나리오 커버 가능.
    """
    SYSTEM = "system"
    USER = "user"
    ASSISTANT = "assistant"
    TOOL = "tool"


@dataclass(slots=True)
class Message:
    """
    통합 메시지 타입 — OpenAI ChatCompletion 형식

    v4.1 대응: ConversationMessage + SkillResult + TraceEvent 통합
    축소 이유: 프레임워크별 다른 메시지 형식 → OpenAI 표준 하나로 단일화.
              모든 엔진(Direct, LangChain, CrewAI)이 동일한 Message 구조 사용.

    사용법:
        >>> msg = Message(role=Role.USER, content="안녕하세요")
        >>> msg.to_dict()
        {'role': 'user', 'content': '안녕하세요'}
    """
    role: Role
    content: str
    name: str | None = None
    tool_call_id: str | None = None
    tool_calls: list[ToolCall] | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """OpenAI API 호환 딕셔너리 변환"""
        d: dict[str, Any] = {"role": self.role.value, "content": self.content}
        if self.name:
            d["name"] = self.name
        if self.tool_call_id:
            d["tool_call_id"] = self.tool_call_id
        if self.tool_calls:
            d["tool_calls"] = [tc.to_dict() for tc in self.tool_calls]
        return d

    @classmethod
    def system(cls, content: str) -> Message:
        return cls(role=Role.SYSTEM, content=content)

    @classmethod
    def user(cls, content: str) -> Message:
        return cls(role=Role.USER, content=content)

    @classmethod
    def assistant(cls, content: str, tool_calls: list[ToolCall] | None = None) -> Message:
        return cls(role=Role.ASSISTANT, content=content, tool_calls=tool_calls)

    @classmethod
    def tool(cls, content: str, tool_call_id: str) -> Message:
        return cls(role=Role.TOOL, content=content, tool_call_id=tool_call_id)


@dataclass(slots=True)
class ToolCall:
    """
    도구 호출 요청 — OpenAI Function Calling 표준

    v4.1 대응: FunctionCallRequest + MCPToolInvocation 통합
    축소 이유: 프레임워크별 다른 도구 호출 형식 → OpenAI 표준으로 단일화.
    """
    id: str
    name: str
    arguments: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "type": "function",
            "function": {
                "name": self.name,
                "arguments": json.dumps(self.arguments, ensure_ascii=False),
            },
        }


@dataclass(slots=True)
class ToolResult:
    """
    도구 실행 결과

    v4.1 대응: SkillResult + MCPToolOutput 통합
    축소 이유: 도구 결과도 단일 형식으로 통일 → 엔진 간 전환 비용 0.
    """
    tool_call_id: str
    name: str
    content: str
    is_error: bool = False

    def to_message(self) -> Message:
        return Message.tool(content=self.content, tool_call_id=self.tool_call_id)


@dataclass(slots=True)
class AgentResult:
    """
    에이전트 실행 결과 — run_agent()의 반환값

    v4.1 대응: AgentState + WorkflowResult + TraceData + MetricData 통합
    축소 이유: 실행 결과를 하나의 데이터클래스로 통일.
              content(응답) + usage(토큰) + duration_ms(성능) + tool_calls(도구)를
              한 번에 확인할 수 있어 추가 API 호출 불필요.

    핵심 필드:
        content: 최종 응답 텍스트
        messages: 전체 대화 히스토리
        tool_calls: 실행된 도구 호출 내역
        usage: 토큰 사용량 (input_tokens, output_tokens, total_tokens)
        duration_ms: 실행 시간 (ms)
        metadata: 엔진별 추가 정보
    """
    content: str
    messages: list[Message] = field(default_factory=list)
    tool_calls: list[ToolCall] = field(default_factory=list)
    usage: dict[str, int] = field(default_factory=dict)
    model: str = ""
    engine: str = ""
    duration_ms: float = 0.0
    metadata: dict[str, Any] = field(default_factory=dict)
    request_id: str = field(default_factory=lambda: str(uuid.uuid4()))


@dataclass(slots=True)
class StreamChunk:
    """
    스트리밍 응답 청크

    엔진의 stream() 메서드가 반환하는 개별 청크.
    is_final=True이면 스트리밍 종료.
    """
    content: str = ""
    is_final: bool = False
    tool_call: ToolCall | None = None
    metadata: dict[str, Any] = field(default_factory=dict)
