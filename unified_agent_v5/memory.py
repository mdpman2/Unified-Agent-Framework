#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Unified Agent Framework v5 — Memory

================================================================================
v4.1 대응: memory.py(586줄) + persistent_memory.py + compaction.py
          + session_tree.py + memory_layer.py + semantic_memory.py 통합
축소 이유: v4.1의 6개 메모리 시스템(MemoryLayer, PersistentMemory,
          CompactionEngine, SessionTree, SemanticMemory, EpisodicMemory)은
          실무에서 거의 채 사용되지 않았음.
          대부분 사용 사례는 List[Message] + 슬라이딩 윈도우로 충분.
          영속화 필요 시 to_json() → Redis/CosmosDB 직접 저장.
================================================================================

설계 원칙:
    - 메모리 = List[Message] (이것으로 충분)
    - 토큰 제한에 맞춤 자동 슬라이딩 윈도우
    - 시스템 프롬프트는 항상 보존
    - 영속화는 to_json()/from_json()로 외부에 위임

사용법:
    >>> memory = Memory(system_prompt="You are helpful.")
    >>> memory.add_user("안녕하세요")
    >>> memory.add_assistant("안녕하세요! 무엇을 도와드릴까요?")
    >>> messages = memory.get_messages()  # OpenAI API용 딕셔너리 리스트
"""

from __future__ import annotations

import json
from typing import Any

from .types import Message, Role, ToolCall

__all__ = ["Memory"]


class Memory:
    """
    대화 메모리 — 단순한 List[Message] 기반

    v4.1 대응: MemoryManager + PersistentMemory + CompactionEngine
              + SessionTree + SemanticMemory + EpisodicMemory (6개 클래스)
    축소 이유: 실무 95%는 "시스템 프롬프트 + 메시지 리스트 + 슬라이딩 윈도우"로 충분.
              복잡한 레이어/네임스페이스/TTL/압축은 실무 미사용.
              영속화 필요 시 to_json() + 외부 스토어로 대체.
    """

    __slots__ = ('_system', '_messages', 'max_messages')

    def __init__(
        self,
        system_prompt: str = "You are a helpful assistant.",
        max_messages: int = 100,
    ):
        """
        Args:
            system_prompt: 시스템 프롬프트 (항상 첫 번째로 포함)
            max_messages: 최대 보관 메시지 수 (시스템 프롬프트 제외)
        """
        self._system = Message.system(system_prompt)
        self._messages: list[Message] = []
        self.max_messages = max_messages

    @property
    def system_prompt(self) -> str:
        return self._system.content

    @system_prompt.setter
    def system_prompt(self, value: str) -> None:
        self._system = Message.system(value)

    @property
    def messages(self) -> list[Message]:
        """시스템 프롬프트를 포함한 전체 메시지 리스트"""
        return [self._system] + self._messages

    @property
    def history(self) -> list[Message]:
        """시스템 프롬프트를 제외한 대화 히스토리"""
        return list(self._messages)

    def add(self, message: Message) -> None:
        """메시지 추가 (슬라이딩 윈도우 자동 적용)"""
        self._messages.append(message)
        self._trim()

    def add_user(self, content: str) -> None:
        """사용자 메시지 추가"""
        self.add(Message.user(content))

    def add_assistant(self, content: str) -> None:
        """어시스턴트 메시지 추가"""
        self.add(Message.assistant(content))

    def add_tool_result(self, content: str, tool_call_id: str) -> None:
        """도구 결과 메시지 추가"""
        self.add(Message.tool(content, tool_call_id))

    def get_messages(self, max_messages: int | None = None) -> list[dict[str, Any]]:
        """
        OpenAI API 호환 메시지 리스트 반환

        Args:
            max_messages: 반환할 최대 메시지 수 (None이면 전체)

        Returns:
            [{"role": "system", "content": "..."}, {"role": "user", ...}, ...]
        """
        msgs = self._messages
        if max_messages is not None and len(msgs) > max_messages:
            msgs = msgs[-max_messages:]
        return [self._system.to_dict()] + [m.to_dict() for m in msgs]

    def clear(self) -> None:
        """대화 히스토리 초기화 (시스템 프롬프트 유지)"""
        self._messages.clear()

    def _trim(self) -> None:
        """슬라이딩 윈도우 적용 (인플레이스 삭제)"""
        if len(self._messages) > self.max_messages:
            overflow = len(self._messages) - self.max_messages
            del self._messages[:overflow]

    def __len__(self) -> int:
        """총 메시지 수 (시스템 프롬프트 포함)"""
        return len(self._messages) + 1

    # ── 직렬화 (외부 스토어 연동용) ──

    def to_json(self) -> str:
        """JSON 직렬화"""
        return json.dumps({
            "system_prompt": self._system.content,
            "messages": [m.to_dict() for m in self._messages],
            "max_messages": self.max_messages,
        }, ensure_ascii=False, indent=2)

    @classmethod
    def from_json(cls, data: str) -> Memory:
        """JSON에서 복원 (tool_calls 포함)"""
        parsed = json.loads(data)
        memory = cls(
            system_prompt=parsed["system_prompt"],
            max_messages=parsed.get("max_messages", 100),
        )
        for msg_dict in parsed.get("messages", []):
            # tool_calls 복원
            tool_calls = None
            if msg_dict.get("tool_calls"):
                tool_calls = [
                    ToolCall(
                        id=tc["id"],
                        name=tc["function"]["name"],
                        arguments=(
                            json.loads(tc["function"]["arguments"])
                            if isinstance(tc["function"]["arguments"], str)
                            else tc["function"]["arguments"]
                        ),
                    )
                    for tc in msg_dict["tool_calls"]
                ]
            memory.add(Message(
                role=Role(msg_dict["role"]),
                content=msg_dict["content"],
                name=msg_dict.get("name"),
                tool_call_id=msg_dict.get("tool_call_id"),
                tool_calls=tool_calls,
            ))
        return memory

    def __repr__(self) -> str:
        return f"Memory(messages={len(self._messages)}, max={self.max_messages})"
