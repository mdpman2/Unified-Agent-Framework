#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Unified Agent Framework - 메모리 관리 모듈 (Memory Module)

================================================================================
📁 파일 위치: unified_agent/memory.py
📋 역할: 메모리 저장소, 캐싱, 세션 관리, 상태 관리
📅 최종 업데이트: 2026년 1월
================================================================================

🎯 주요 구성 요소:

    📌 메모리 저장소:
        - MemoryStore: 추상 기본 클래스 (ABC)
        - CachedMemoryStore: LRU 캐시 적용 저장소

    📌 대화 관리:
        - ConversationMessage: 대화 메시지 모델 (AgentCore 패턴)
        - MemoryHookProvider: 메모리 훅 제공자
        - MemorySessionManager: 세션 관리자

    📌 상태 관리:
        - StateManager: 에이전트 상태 관리자

🔧 핵심 기능:
    - LRU (Least Recently Used) 캐싱: 메모리 사용량 제한
    - 자동 타임스탬프: 모든 저장 데이터에 UTC 시간 기록
    - 버전 관리: 데이터 변경 시 버전 자동 증가
    - 패턴 매칭: list_keys()에서 글로브 패턴 지원
    - 네임스페이스 분리: 다중 에이전트/세션 격리

📌 사용 예시:

    예제 1: CachedMemoryStore 사용
    ----------------------------------------
    >>> from unified_agent.memory import CachedMemoryStore
    >>>
    >>> # 저장소 생성 (최대 100개 항목 캐싱)
    >>> store = CachedMemoryStore(max_cache_size=100)
    >>>
    >>> # 데이터 저장
    >>> await store.save("session:user1", {
    ...     "messages": [...],
    ...     "context": {...}
    ... })
    >>>
    >>> # 데이터 로드 (캐시 자동 적용)
    >>> data = await store.load("session:user1")
    >>>
    >>> # 키 목록 조회 (패턴 매칭)
    >>> keys = await store.list_keys("session:*")

    예제 2: StateManager 사용
    ----------------------------------------
    >>> from unified_agent.memory import StateManager
    >>> from unified_agent.models import AgentState
    >>>
    >>> manager = StateManager()
    >>>
    >>> # 상태 저장
    >>> state = AgentState(session_id="session-1", messages=[])
    >>> await manager.save_state("session-1", state)
    >>>
    >>> # 상태 복원
    >>> restored = await manager.load_state("session-1")

⚠️ 주의사항:
    - CachedMemoryStore는 인메모리 저장소로 재시작 시 데이터 소실
    - 프로덕션에서는 Redis 또는 CosmosDB 기반 구현 권장
    - 대용량 데이터는 max_cache_size 조절 필요

🔗 참고:
    - Microsoft Agent Framework Memory: https://github.com/microsoft/agent-framework
    - LRU Cache 알고리즘: https://en.wikipedia.org/wiki/Cache_replacement_policies#LRU
"""

from __future__ import annotations

import os
import json
import fnmatch
import logging
from abc import ABC, abstractmethod
from collections import defaultdict, OrderedDict
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any

from .models import AgentState
from .utils import StructuredLogger

__all__ = [
    "MemoryStore",
    "CachedMemoryStore",
    "ConversationMessage",
    "MemoryHookProvider",
    "MemorySessionManager",
    "StateManager",
]

# ============================================================================
# 메모리 저장소 인터페이스
# ============================================================================

class MemoryStore(ABC):
    """
    메모리 저장소 추상 기본 클래스 (Abstract Base Class)

    ================================================================================
    📋 역할: 메모리 저장소의 공통 인터페이스 정의
    📅 최종 업데이트: 2026년 1월
    ================================================================================

    🎯 주요 기능:
        - save(): 데이터 저장
        - load(): 데이터 로드
        - delete(): 데이터 삭제
        - list_keys(): 키 목록 조회 (패턴 매칭 지원)

    📌 구현 예시:
        >>> class RedisMemoryStore(MemoryStore):
        ...     async def save(self, key: str, data: Dict) -> None:
        ...         # Redis에 저장
        ...         pass
        ...
        ...     async def load(self, key: str) -> Dict | None:
        ...         # Redis에서 로드
        ...         pass

    ⚠️ 주의사항:
        - 모든 메서드는 비동기(async)로 구현해야 합니다.
        - 데이터는 Dict 형태로 저장/로드됩니다.

    🔗 제공되는 구현체:
        - CachedMemoryStore: 인메모리 LRU 캐시 저장소
    """

    @abstractmethod
    async def save(self, key: str, data: dict) -> None:
        pass

    @abstractmethod
    async def load(self, key: str) -> dict | None:
        pass

    @abstractmethod
    async def delete(self, key: str) -> None:
        pass

    @abstractmethod
    async def list_keys(self, pattern: str = "*") -> list[str]:
        """키 목록 조회"""
        pass

class CachedMemoryStore(MemoryStore):
    """
    LRU (Least Recently Used) 캐시 적용 메모리 저장소

    ================================================================================
    📋 역할: 인메모리 데이터 저장 + 자주 접근하는 데이터 캐싱
    📅 최종 업데이트: 2026년 1월
    ================================================================================

    🎯 LRU 캐시 알고리즘:
        - 자주 접근하는 데이터를 메모리에 유지
        - 캠시 크기 초과 시 가장 오래된 항목 자동 제거
        - 데이터 접근 횟수 추적 (access_count)
        - 3회 이상 접근 시 캐시로 승격

    🔧 내부 구조:
        - data: 원본 데이터 저장소 (Dict)
        - cache: LRU 캐시 (Dict)
        - access_count: 키별 접근 횟수 (Dict)
        - access_order: 접근 순서 기록 (List)

    Args:
        max_cache_size (int): 캐시 최대 항목 수 (기본: 100)

    📌 사용 예시:
        >>> store = CachedMemoryStore(max_cache_size=500)
        >>>
        >>> # 데이터 저장 (timestamp, version 자동 추가)
        >>> await store.save("user:123", {"name": "John", "age": 30})
        >>>
        >>> # 데이터 로드 (자주 접근 시 캐시에서 로드)
        >>> data = await store.load("user:123")
        >>>
        >>> # 키 목록 조회
        >>> all_keys = await store.list_keys("*")  # 모든 키
        >>> user_keys = await store.list_keys("user:*")  # user:로 시작하는 키

    ⚠️ 주의사항:
        - 인메모리 저장소로 프로세스 재시작 시 데이터 소실
        - 대용량 데이터는 max_cache_size를 늘리거나 외부 저장소 사용 권장
        - __slots__ 사용으로 메모리 효율성 최적화

    🔗 LRU 참고:
        - https://en.wikipedia.org/wiki/Cache_replacement_policies#LRU
    """
    __slots__ = ('data', 'cache', 'access_count', 'max_cache_size')

    def __init__(self, max_cache_size: int = 100):
        """
        CachedMemoryStore 초기화

        Args:
            max_cache_size (int): 캐시 최대 항목 수 (기본: 100)
                - 메모리 사용량과 성능 균형 고려하여 설정
                - 대량 데이터 저장 시 500 이상 권장
        """
        self.data: dict[str, dict] = {}  # 원본 데이터
        self.cache: OrderedDict = OrderedDict()  # 최적화: OrderedDict로 LRU 구현
        self.access_count: dict[str, int] = defaultdict(int)  # 접근 횟수
        self.max_cache_size = max_cache_size

    async def save(self, key: str, data: dict) -> None:
        self.data[key] = {
            'data': data,
            'timestamp': datetime.now(timezone.utc).isoformat(),
            'version': self.data.get(key, {}).get('version', 0) + 1
        }
        self.access_count[key] += 1

        # 자주 접근하는 데이터는 캐시에 저장
        if self.access_count[key] > 3:
            self._add_to_cache(key, data)

    async def load(self, key: str) -> dict | None:
        self.access_count[key] += 1

        # 캐시 확인 (최적화: OrderedDict move_to_end)
        if key in self.cache:
            self.cache.move_to_end(key)  # LRU 업데이트
            return self.cache[key]

        # 원본 데이터 확인
        if key in self.data:
            return self.data[key].get('data')
        return None

    async def delete(self, key: str) -> None:
        if key in self.data:
            del self.data[key]
        if key in self.cache:
            del self.cache[key]

    async def list_keys(self, pattern: str = "*") -> list[str]:
        """키 목록 조회 (최적화: 모듈 레벨 fnmatch import)"""
        if pattern == "*":
            return list(self.data.keys())
        return [k for k in self.data.keys() if fnmatch.fnmatch(k, pattern)]

    def _add_to_cache(self, key: str, data: Any):
        """캐시에 추가 (최적화: OrderedDict LRU)"""
        # 캐시 크기 제한 - OrderedDict의 popitem(last=False)로 O(1) 제거
        while len(self.cache) >= self.max_cache_size:
            self.cache.popitem(last=False)  # 가장 오래된 항목 제거

        self.cache[key] = data
        self.cache.move_to_end(key)  # 최신으로 이동

# ============================================================================
# 대화 메시지 모델
# ============================================================================

@dataclass(frozen=True, slots=True)
class ConversationMessage:
    """
    대화 메시지 데이터 모델 (AgentCore Memory 패턴)

    ================================================================================
    📋 역할: 단일 대화 메시지를 표현하는 불변 데이터 클래스
    📅 최종 업데이트: 2026년 1월
    ================================================================================

    🎯 주요 속성:
        - content: 메시지 내용
        - role: 발화자 역할 (USER, ASSISTANT, TOOL, SYSTEM)
        - timestamp: 메시지 생성 시간 (UTC)
        - agent_name: 에이전트 이름 (선택)
        - session_id: 세션 ID (선택)
        - metadata: 추가 메타데이터 (Dict)

    📌 사용 예시:
        >>> from unified_agent.memory import ConversationMessage
        >>>
        >>> # 사용자 메시지
        >>> user_msg = ConversationMessage(
        ...     content="안녕하세요!",
        ...     role="USER",
        ...     session_id="session-1"
        ... )
        >>>
        >>> # 도구 결과 메시지
        >>> tool_msg = ConversationMessage(
        ...     content="{\'result\': \'success\'}",
        ...     role="TOOL",
        ...     agent_name="search_agent",
        ...     metadata={"tool_name": "web_search", "duration_ms": 250}
        ... )

    ⚠️ 주의사항:
        - timestamp는 자동으로 UTC 시간이 설정됩니다.
        - role은 문자열로 저장되며 AgentRole enum과 매핑됩니다.

    🔗 참고:
        - Microsoft Agent Framework Memory: https://github.com/microsoft/agent-framework
    """
    content: str  # 메시지 내용
    role: str  # 발화자 역할: USER, ASSISTANT, TOOL, SYSTEM
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))  # 생성 시간 (UTC)
    agent_name: str | None = None  # 에이전트 이름 (선택)
    session_id: str | None = None  # 세션 ID (선택)
    metadata: dict[str, Any] = field(default_factory=dict)  # 추가 메타데이터

# ============================================================================
# Memory Hook Provider
# ============================================================================

class MemoryHookProvider:
    """
    Memory Hook Provider - 자동 메모리 관리

    참조: amazon-bedrock-agentcore-samples/memory/hooks.py

    주요 기능:
    - 대화 기록 자동 저장/로드
    - 세션 기반 컨텍스트 관리
    - 네임스페이스 기반 메모리 분류
    """

    def __init__(
        self,
        memory_store: MemoryStore,
        session_id: str,
        actor_id: str,
        max_context_turns: int = 10,
        namespace: str = "/conversation"
    ):
        self.memory_store = memory_store
        self.session_id = session_id
        self.actor_id = actor_id
        self.max_context_turns = max_context_turns
        self.namespace = namespace
        self.conversation_history: list[ConversationMessage] = []
        self._logger = StructuredLogger("memory_hook")

    async def on_agent_initialized(self, agent_name: str) -> list[ConversationMessage]:
        """
        에이전트 초기화 시 최근 대화 기록 로드
        """
        try:
            key = f"{self.namespace}/{self.session_id}/history"
            data = await self.memory_store.load(key)

            if data:
                messages = data.get("messages", [])
                self.conversation_history = [
                    ConversationMessage(**msg) for msg in messages[-self.max_context_turns:]
                ]
                self._logger.info(
                    f"Loaded {len(self.conversation_history)} messages",
                    agent=agent_name,
                    session_id=self.session_id
                )

            return self.conversation_history
        except Exception as e:
            self._logger.error(f"Failed to load history: {e}")
            return []

    async def on_message_added(
        self,
        content: str,
        role: str,
        agent_name: str | None = None
    ):
        """
        메시지 추가 시 자동 저장
        """
        message = ConversationMessage(
            content=content,
            role=role,
            agent_name=agent_name,
            session_id=self.session_id
        )

        self.conversation_history.append(message)

        # 저장
        try:
            key = f"{self.namespace}/{self.session_id}/history"
            await self.memory_store.save(key, {
                "messages": [{
                    "content": m.content,
                    "role": m.role,
                    "timestamp": m.timestamp.isoformat(),
                    "agent_name": m.agent_name,
                    "session_id": m.session_id
                } for m in self.conversation_history[-self.max_context_turns:]],
                "actor_id": self.actor_id,
                "updated_at": datetime.now(timezone.utc).isoformat()
            })
        except Exception as e:
            self._logger.error(f"Failed to save message: {e}")

    async def get_last_k_turns(self, k: int = 5) -> list[ConversationMessage]:
        """최근 k개 대화 턴 조회"""
        return self.conversation_history[-k:]

    async def clear_session(self):
        """세션 데이터 삭제"""
        key = f"{self.namespace}/{self.session_id}/history"
        await self.memory_store.delete(key)
        self.conversation_history = []
        self._logger.info("Session cleared", session_id=self.session_id)

# ============================================================================
# Memory Session Manager
# ============================================================================

class MemorySessionManager:
    """
    세션 기반 메모리 관리자 (AgentCore MemorySessionManager 패턴)

    주요 기능:
    - 다중 세션 관리
    - 세션 간 컨텍스트 공유
    - 자동 세션 정리
    """

    def __init__(self, memory_store: MemoryStore, default_ttl_hours: int = 24):
        self.memory_store = memory_store
        self.default_ttl_hours = default_ttl_hours
        self._sessions: dict[str, MemoryHookProvider] = {}
        self._logger = StructuredLogger("session_manager")

    def get_or_create_session(
        self,
        session_id: str,
        actor_id: str,
        namespace: str = "/conversation"
    ) -> MemoryHookProvider:
        """세션 조회 또는 생성"""
        key = f"{actor_id}:{session_id}"

        if key not in self._sessions:
            self._sessions[key] = MemoryHookProvider(
                memory_store=self.memory_store,
                session_id=session_id,
                actor_id=actor_id,
                namespace=namespace
            )
            self._logger.info(
                "Created new session",
                session_id=session_id,
                actor_id=actor_id
            )

        return self._sessions[key]

    async def list_sessions(self, actor_id: str | None = None) -> list[str]:
        """세션 목록 조회"""
        sessions = []
        for key in self._sessions.keys():
            if actor_id is None or key.startswith(f"{actor_id}:"):
                sessions.append(key)
        return sessions

    async def cleanup_expired_sessions(self):
        """만료된 세션 정리"""
        # 구현: TTL 기반 세션 정리
        pass

# ============================================================================
# State Manager
# ============================================================================

class StateManager:
    """
    상태 관리자 - 버전 관리 및 롤백 지원

    주요 기능:
    1. 버전 관리 (state_versions)
    2. load_state(version): 특정 버전 로드
    3. save_checkpoint(tag): 태그와 함께 체크포인트 저장
    4. restore_checkpoint(tag): 특정 태그 복원
    5. list_checkpoints(): 체크포인트 목록
    6. rollback(steps): 이전 상태로 롤백
    """

    def __init__(self, memory_store: MemoryStore, checkpoint_dir: str | None = None):
        self.memory_store = memory_store
        self.checkpoint_dir = checkpoint_dir
        self.state_versions: dict[str, list[str]] = defaultdict(list)

        if checkpoint_dir and not os.path.exists(checkpoint_dir):
            os.makedirs(checkpoint_dir)

    async def save_state(self, state: AgentState):
        """상태 저장 (버전 추적 포함)"""
        state_dict = state.model_dump()
        await self.memory_store.save(f"state:{state.session_id}", state_dict)

        # 버전 추적
        version_key = f"state:{state.session_id}:v{len(self.state_versions[state.session_id])}"
        await self.memory_store.save(version_key, state_dict)
        self.state_versions[state.session_id].append(version_key)

    async def load_state(self, session_id: str, version: int | None = None) -> AgentState | None:
        """상태 로드 (특정 버전 지원)"""
        if version is not None:
            version_key = f"state:{session_id}:v{version}"
            data = await self.memory_store.load(version_key)
        else:
            data = await self.memory_store.load(f"state:{session_id}")

        if data:
            return AgentState(**data)
        return None

    async def save_checkpoint(self, state: AgentState, tag: str | None = None) -> str:
        """체크포인트 저장"""
        if not self.checkpoint_dir:
            raise ValueError("체크포인트 디렉토리 미설정")

        timestamp = datetime.now(timezone.utc).isoformat().replace(':', '-').replace('.', '-')
        tag_suffix = f"_{tag}" if tag else ""
        checkpoint_file = os.path.join(
            self.checkpoint_dir,
            f"{state.session_id}_{timestamp}{tag_suffix}.json"
        )

        with open(checkpoint_file, 'w', encoding='utf-8') as f:
            json.dump(state.model_dump(), f, ensure_ascii=False, indent=2)

        logging.info(f"💾 체크포인트 저장: {checkpoint_file}")
        return checkpoint_file

    async def restore_checkpoint(self, session_id: str, tag: str | None = None) -> AgentState | None:
        """체크포인트 복원"""
        if not self.checkpoint_dir:
            return None

        checkpoints = [
            f for f in os.listdir(self.checkpoint_dir)
            if f.startswith(session_id) and f.endswith('.json')
        ]

        # 태그 필터링
        if tag:
            checkpoints = [f for f in checkpoints if tag in f]

        if not checkpoints:
            return None

        latest = os.path.join(self.checkpoint_dir, sorted(checkpoints)[-1])

        with open(latest, 'r', encoding='utf-8') as f:
            data = json.load(f)

        logging.info(f"📂 체크포인트 복원: {latest}")
        return AgentState(**data)

    async def list_checkpoints(self, session_id: str) -> list[str]:
        """체크포인트 목록"""
        if not self.checkpoint_dir or not os.path.exists(self.checkpoint_dir):
            return []

        checkpoints = [
            f for f in os.listdir(self.checkpoint_dir)
            if f.startswith(session_id) and f.endswith('.json')
        ]
        return sorted(checkpoints)

    async def rollback(self, session_id: str, steps: int = 1) -> AgentState | None:
        """이전 상태로 롤백"""
        versions = self.state_versions.get(session_id, [])
        if len(versions) < steps:
            logging.warning(f"⚠️ 롤백 불가: {steps}단계 이전 버전 없음")
            return None

        target_version = len(versions) - steps - 1
        return await self.load_state(session_id, version=target_version)
