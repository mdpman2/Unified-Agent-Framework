#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Unified Agent Framework v6 — Core Types

================================================================================
Microsoft Agent Framework 1.0.0-rc1 호환 타입 시스템

모든 에이전트 입출력의 기반이 되는 핵심 타입들을 정의합니다.
Content 기반 Message 시스템으로 text, error, function_call 등 다양한
콘텐츠 유형을 통일된 인터페이스로 처리합니다.

타입 계층 구조:
    Content       ─ 통합 콘텐츠 컨테이너 (text, error, function_call 등)
    Message       ─ Content 기반 메시지 (role + contents[])
    AgentResponse ─ Agent.run() 반환값 (messages + usage_details)
    AgentResponseUpdate ─ 스트리밍 청크 (contents + role)
    UsageDetails  ─ 토큰 사용량 TypedDict
    ChatOptions   ─ 채팅 요청 옵션 TypedDict

v5 → v6 변경 요약:
    - Message: OpenAI dict 기반 → Content 기반 (agent_framework 호환)
    - AgentResult → AgentResponse (공식 API 명칭)
    - StreamChunk → AgentResponseUpdate (공식 스트리밍 타입)
    - Role: Enum → str (agent_framework 패턴)

성능 최적화:
    - __slots__ 적용: Content, Message, AgentResponse, AgentResponseUpdate
      → 메모리 사용량 감소 + 속성 접근 속도 향상
    - _SERIALIZE_FIELDS 튜플: Content 직렬화 시 동적 필드 탐색 제거
    - {*usage1, *usage2} 셋 언패킹: add_usage_details() 키 합산 최적화
================================================================================
"""

from __future__ import annotations

import json
import uuid
from dataclasses import dataclass, field
from typing import (
    Any,
    Literal,
    Mapping,
    MutableMapping,
    Sequence,
    TypedDict,
)

__all__ = [
    "Content",
    "ContentType",
    "Message",
    "AgentResponse",
    "AgentResponseUpdate",
    "UsageDetails",
    "ChatOptions",
    "ToolMode",
    "FinishReason",
    "Role",
    "Annotation",
    # legacy compat
    "AgentResult",
    "StreamChunk",
    "ToolCall",
    "ToolResult",
]


# ─── Role ────────────────────────────────────────────────────

RoleLiteral = Literal["system", "user", "assistant", "tool"]
"""역할 리터럴 타입. agent_framework 1.0 호환."""

Role = str
"""메시지 역할. 문자열로 사용 (예: "user", "assistant", "system", "tool")."""


# ─── Content ─────────────────────────────────────────────────

ContentType = Literal[
    "text",
    "text_reasoning",
    "data",
    "uri",
    "error",
    "function_call",
    "function_result",
    "usage",
]


class Annotation(TypedDict, total=False):
    """콘텐츠 어노테이션 (인용, 참조 등)."""
    type: str
    title: str
    url: str
    file_id: str
    snippet: str


class Content:
    """
    통합 콘텐츠 컨테이너 — agent_framework 1.0.0-rc1 호환

    모든 콘텐츠 유형(text, data, uri, error, function_call 등)을 통일된
    인터페이스로 다룹니다. v5에서는 Message.content가 str이었으나
    v6에서는 Content 객체 리스트로 변경되었습니다.

    성능 최적화:
        - __slots__: 메모리 사용량 감소 + 속성 접근 속도 향상 (__dict__ 제거)
        - _SERIALIZE_FIELDS 튜플: to_dict() 시 동적 필드 탐색 제거

    사용법:
        >>> content = Content.from_text("안녕하세요")
        >>> print(content.text)  # "안녕하세요"
        >>> print(content.type)  # "text"

    팩토리 메서드:
        - from_text(text)            : 텍스트 콘텐츠
        - from_error(message, code)  : 에러 콘텐츠
        - from_function_call(...)    : 함수 호출 콘텐츠
        - from_function_result(...)  : 함수 결과 콘텐츠
        - from_usage(usage_details)  : 사용량 콘텐츠
        - from_uri(uri, media_type)  : URI 콘텐츠
    """

    # 성능 최적화: __slots__로 __dict__ 제거 → 메모리 절약 + 불필요 속성 접근 방지
    __slots__ = (
        "type", "text", "uri", "media_type", "message", "error_code",
        "usage_details", "call_id", "name", "arguments", "result",
        "exception", "annotations", "additional_properties", "raw_representation",
    )

    # 성능 최적화: 직렬화 대상 필드를 튜플로 사전 정의 → to_dict()에서 동적 탐색 제거
    _SERIALIZE_FIELDS = (
        "text", "uri", "media_type", "message", "error_code",
        "usage_details", "call_id", "name", "arguments", "result",
        "exception",
    )

    def __init__(
        self,
        type: ContentType,
        *,
        text: str | None = None,
        uri: str | None = None,
        media_type: str | None = None,
        message: str | None = None,
        error_code: str | None = None,
        usage_details: dict[str, Any] | None = None,
        call_id: str | None = None,
        name: str | None = None,
        arguments: str | Mapping[str, Any] | None = None,
        result: Any = None,
        exception: str | None = None,
        annotations: Sequence[Annotation] | None = None,
        additional_properties: MutableMapping[str, Any] | None = None,
        raw_representation: Any | None = None,
    ) -> None:
        self.type = type
        self.text = text
        self.uri = uri
        self.media_type = media_type
        self.message = message
        self.error_code = error_code
        self.usage_details = usage_details
        self.call_id = call_id
        self.name = name
        self.arguments = arguments
        self.result = result
        self.exception = exception
        self.annotations = annotations
        self.additional_properties: dict[str, Any] = dict(additional_properties or {})
        self.raw_representation = raw_representation

    # ── Factory methods ──

    @classmethod
    def from_text(cls, text: str, **kwargs: Any) -> Content:
        """텍스트 콘텐츠 생성."""
        return cls("text", text=text, **kwargs)

    @classmethod
    def from_error(cls, *, message: str | None = None, error_code: str | None = None, **kwargs: Any) -> Content:
        """에러 콘텐츠 생성."""
        return cls("error", message=message, error_code=error_code, **kwargs)

    @classmethod
    def from_function_call(
        cls,
        call_id: str,
        name: str,
        *,
        arguments: str | Mapping[str, Any] | None = None,
        **kwargs: Any,
    ) -> Content:
        """함수 호출 콘텐츠 생성."""
        return cls("function_call", call_id=call_id, name=name, arguments=arguments, **kwargs)

    @classmethod
    def from_function_result(
        cls,
        call_id: str,
        *,
        result: Any = None,
        exception: str | None = None,
        **kwargs: Any,
    ) -> Content:
        """함수 결과 콘텐츠 생성."""
        return cls("function_result", call_id=call_id, result=result, exception=exception, **kwargs)

    @classmethod
    def from_usage(cls, usage_details: UsageDetails, **kwargs: Any) -> Content:
        """사용량 콘텐츠 생성."""
        return cls("usage", usage_details=dict(usage_details), **kwargs)

    @classmethod
    def from_uri(cls, uri: str, *, media_type: str | None = None, **kwargs: Any) -> Content:
        """URI 콘텐츠 생성."""
        content_type: ContentType = "data" if uri.startswith("data:") else "uri"
        return cls(content_type, uri=uri, media_type=media_type, **kwargs)

    def to_dict(self, *, exclude_none: bool = True) -> dict[str, Any]:
        """딕셔너리 직렬화."""
        d: dict[str, Any] = {"type": self.type}
        for f in self._SERIALIZE_FIELDS:
            val = getattr(self, f, None)
            if exclude_none and val is None:
                continue
            d[f] = val
        if self.annotations is not None:
            d["annotations"] = [dict(a) for a in self.annotations]
        return d

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> Content:
        """딕셔너리에서 Content 생성."""
        remaining = dict(data)
        content_type = remaining.pop("type", "text")
        annotations = remaining.pop("annotations", None)
        return cls(type=content_type, annotations=annotations, **remaining)

    def parse_arguments(self) -> dict[str, Any] | None:
        """function_call arguments 파싱."""
        if self.arguments is None:
            return None
        if isinstance(self.arguments, str):
            try:
                loaded = json.loads(self.arguments)
                return loaded if isinstance(loaded, dict) else {"raw": loaded}
            except (json.JSONDecodeError, TypeError):
                return {"raw": self.arguments}
        return dict(self.arguments)

    def __str__(self) -> str:
        if self.type == "text":
            return self.text or ""
        if self.type == "error":
            return f"Error: {self.message or ''}"
        return f"Content(type={self.type})"

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, Content):
            return False
        return self.to_dict(exclude_none=False) == other.to_dict(exclude_none=False)


# ─── Usage ───────────────────────────────────────────────────

class UsageDetails(TypedDict, total=False):
    """
    토큰 사용량 — agent_framework 1.0 호환

    v5 대응: AgentResult.usage (dict[str, int])
    변경: input_tokens → input_token_count 등 공식 명칭 사용.
    """
    input_token_count: int | None
    output_token_count: int | None
    total_token_count: int | None


def add_usage_details(usage1: UsageDetails | None, usage2: UsageDetails | None) -> UsageDetails:
    """
    두 UsageDetails를 합산합니다.

    성능 최적화: {*usage1, *usage2} 셋 언패킹으로 키 합산을 에 최적화.
    set(list() + list()) 대비 메모리 할당 감소.

    Args:
        usage1: 첫 번째 사용량 (또는 None)
        usage2: 두 번째 사용량 (또는 None)

    Returns:
        합산된 UsageDetails
    """
    if usage1 is None:
        return usage2 or UsageDetails()
    if usage2 is None:
        return usage1
    result = UsageDetails()
    for key in {*usage1, *usage2}:
        v1 = usage1.get(key)  # type: ignore[arg-type]
        v2 = usage2.get(key)  # type: ignore[arg-type]
        if v1 is not None and v2 is not None:
            result[key] = v1 + v2  # type: ignore[literal-required]
        elif v1 is not None:
            result[key] = v1  # type: ignore[literal-required]
        elif v2 is not None:
            result[key] = v2  # type: ignore[literal-required]
    return result


# ─── Message ─────────────────────────────────────────────────

class Message:
    """
    통합 메시지 — agent_framework 1.0.0-rc1 호환

    role + contents[Content] 구조로 다양한 콘텐츠 유형을 하나의
    메시지에 담을 수 있습니다.

    v5 대응: Message (role + content: str)
    v6 변경: contents: list[Content] 기반. text 프로퍼티로 텍스트 접근.

    성능 최적화: __slots__ 적용으로 메모리 절약 + 속성 접근 속도 향상.

    사용법:
        >>> msg = Message("user", ["안녕하세요"])
        >>> print(msg.text)  # "안녕하세요"
        >>> print(msg.role)  # "user"
    """

    # 성능 최적화: __slots__로 __dict__ 제거 → 메모리 절약
    __slots__ = (
        "role", "contents", "author_name", "message_id",
        "additional_properties", "raw_representation",
    )

    def __init__(
        self,
        role: RoleLiteral | str,
        contents: Sequence[Content | str | Mapping[str, Any]] | None = None,
        *,
        author_name: str | None = None,
        message_id: str | None = None,
        additional_properties: MutableMapping[str, Any] | None = None,
        raw_representation: Any | None = None,
    ) -> None:
        self.role: str = role
        self.contents: list[Content] = []
        if contents is not None:
            for c in contents:
                if isinstance(c, str):
                    self.contents.append(Content.from_text(c))
                elif isinstance(c, Content):
                    self.contents.append(c)
                elif isinstance(c, dict):
                    self.contents.append(Content.from_dict(c))
        self.author_name = author_name
        self.message_id = message_id
        self.additional_properties = dict(additional_properties or {})
        self.raw_representation = raw_representation

    @property
    def text(self) -> str:
        """모든 텍스트 콘텐츠의 결합 문자열."""
        return " ".join(c.text for c in self.contents if c.type == "text" and c.text)

    def to_dict(self) -> dict[str, Any]:
        """딕셔너리 직렬화."""
        d: dict[str, Any] = {
            "type": "chat_message",
            "role": self.role,
            "contents": [c.to_dict() for c in self.contents],
        }
        if self.author_name:
            d["author_name"] = self.author_name
        if self.message_id:
            d["message_id"] = self.message_id
        return d

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> Message:
        """딕셔너리에서 Message 복원."""
        role = data.get("role", "user")
        contents_raw = data.get("contents", [])
        contents = []
        for c in contents_raw:
            if isinstance(c, dict):
                contents.append(Content.from_dict(c))
            elif isinstance(c, str):
                contents.append(Content.from_text(c))
            elif isinstance(c, Content):
                contents.append(c)
        return cls(
            role=role,
            contents=contents,
            author_name=data.get("author_name"),
            message_id=data.get("message_id"),
        )

    def __str__(self) -> str:
        return self.text

    def __repr__(self) -> str:
        return f"Message(role={self.role!r}, text={self.text[:50]!r})"


# ─── Tool Mode / Chat Options ───────────────────────────────

class ToolMode(TypedDict, total=False):
    """도구 선택 모드."""
    mode: Literal["auto", "required", "none"]
    required_function_name: str


FinishReason = str
"""응답 완료 사유 ("stop", "length", "tool_calls", "content_filter")."""


class ChatOptions(TypedDict, total=False):
    """
    채팅 요청 옵션 — agent_framework 1.0 호환

    v5 대응: AgentConfig (dataclass)
    변경: TypedDict로 변경하여 IDE 자동완성 및 타입 안전성 향상.
    """
    model_id: str
    temperature: float
    top_p: float
    max_tokens: int
    stop: str | Sequence[str]
    seed: int
    frequency_penalty: float
    presence_penalty: float
    tools: Any
    tool_choice: ToolMode | Literal["auto", "required", "none"]
    instructions: str
    response_format: Any
    metadata: dict[str, Any]
    user: str
    store: bool
    conversation_id: str


# ─── Agent Response ──────────────────────────────────────────

class AgentResponse:
    """
    에이전트 실행 결과 — agent_framework 1.0.0-rc1 호환

    Agent.run()의 반환값으로, messages 리스트와 토큰 사용량을 포함합니다.
    text 프로퍼티로 모든 메시지의 텍스트를 결합하여 접근합니다.

    v5 대응: AgentResult (content + usage + duration_ms)
    v6 변경: messages[] 기반. text 프로퍼티로 텍스트 접근.

    성능 최적화: __slots__ 적용으로 메모리 절약.

    사용법:
        >>> response = await agent.run("안녕하세요")
        >>> print(response.text)
        >>> print(response.usage_details)
    """

    __slots__ = (
        "messages", "response_id", "agent_id", "created_at",
        "usage_details", "additional_properties", "raw_representation",
    )

    def __init__(
        self,
        *,
        messages: Message | Sequence[Message] | None = None,
        response_id: str | None = None,
        agent_id: str | None = None,
        created_at: str | None = None,
        usage_details: UsageDetails | None = None,
        additional_properties: dict[str, Any] | None = None,
        raw_representation: Any | None = None,
    ) -> None:
        if messages is None:
            self.messages: list[Message] = []
        elif isinstance(messages, Message):
            self.messages = [messages]
        else:
            self.messages = list(messages)
        self.response_id = response_id
        self.agent_id = agent_id
        self.created_at = created_at
        self.usage_details = usage_details
        self.additional_properties = additional_properties or {}
        self.raw_representation = raw_representation

    @property
    def text(self) -> str:
        """모든 메시지의 텍스트 결합."""
        return "".join(msg.text for msg in self.messages) if self.messages else ""

    def __str__(self) -> str:
        return self.text

    def __repr__(self) -> str:
        return f"AgentResponse(text={self.text[:60]!r}, messages={len(self.messages)})"


class AgentResponseUpdate:
    """
    스트리밍 응답 청크 — agent_framework 1.0.0-rc1 호환

    agent.run(stream=True) 시 비동기 이터레이터로 반환되는 각 청크입니다.
    contents 기반으로 text 프로퍼티로 텍스트에 접근합니다.

    v5 대응: StreamChunk
    v6 변경: contents[Content] 기반.

    성능 최적화: __slots__ 적용.
    """

    __slots__ = (
        "contents", "role", "author_name", "response_id",
        "message_id", "raw_representation",
    )

    def __init__(
        self,
        *,
        contents: Sequence[Content] | None = None,
        role: str | None = None,
        author_name: str | None = None,
        response_id: str | None = None,
        message_id: str | None = None,
        raw_representation: Any | None = None,
    ) -> None:
        self.contents: list[Content] = list(contents) if contents else []
        self.role = role
        self.author_name = author_name
        self.response_id = response_id
        self.message_id = message_id
        self.raw_representation = raw_representation

    @property
    def text(self) -> str:
        """텍스트 콘텐츠 결합."""
        return "".join(c.text for c in self.contents if c.type == "text" and c.text)

    def __str__(self) -> str:
        return self.text


# ─── Legacy Compatibility (v5 호환) ─────────────────────────

@dataclass(slots=True)
class ToolCall:
    """도구 호출 요청 (v5 호환 + agent_framework function_call 매핑)."""
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

    def to_content(self) -> Content:
        """Content.from_function_call()로 변환."""
        return Content.from_function_call(
            call_id=self.id,
            name=self.name,
            arguments=self.arguments,
        )


@dataclass(slots=True)
class ToolResult:
    """도구 실행 결과 (v5 호환)."""
    tool_call_id: str
    name: str
    content: str
    is_error: bool = False

    def to_content(self) -> Content:
        """Content.from_function_result()로 변환."""
        return Content.from_function_result(
            call_id=self.tool_call_id,
            result=self.content,
            exception=self.content if self.is_error else None,
        )


# v5 → v6 호환 별칭
AgentResult = AgentResponse
StreamChunk = AgentResponseUpdate
