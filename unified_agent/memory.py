#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Unified Agent Framework - 메모리 관리 모듈

메모리 저장소, 캐싱, 세션 관리 등을 담당합니다.
"""

import os
import json
import logging
from abc import ABC, abstractmethod
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Dict, List, Optional, Any

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
    메모리 저장소 인터페이스
    """

    @abstractmethod
    async def save(self, key: str, data: Dict) -> None:
        pass

    @abstractmethod
    async def load(self, key: str) -> Optional[Dict]:
        pass

    @abstractmethod
    async def delete(self, key: str) -> None:
        pass

    @abstractmethod
    async def list_keys(self, pattern: str = "*") -> List[str]:
        """키 목록 조회"""
        pass


class CachedMemoryStore(MemoryStore):
    """
    캐싱 메모리 저장소 - LRU 캐시

    LRU (Least Recently Used) 캐시 알고리즘 적용

    LRU 캐시 장점:
    - 메모리 사용량 제한 (max_cache_size)
    - 최근 사용 데이터 우선 유지
    - 오래된 데이터 자동 제거
    """
    __slots__ = ('data', 'cache', 'access_count', 'max_cache_size', 'access_order')

    def __init__(self, max_cache_size: int = 100):
        self.data: Dict[str, Dict] = {}
        self.cache: Dict[str, Any] = {}
        self.access_count: Dict[str, int] = defaultdict(int)
        self.max_cache_size = max_cache_size
        self.access_order: List[str] = []

    async def save(self, key: str, data: Dict) -> None:
        self.data[key] = {
            'data': data,
            'timestamp': datetime.now(timezone.utc).isoformat(),
            'version': self.data.get(key, {}).get('version', 0) + 1
        }
        self.access_count[key] += 1

        # 자주 접근하는 데이터는 캐시에 저장
        if self.access_count[key] > 3:
            self._add_to_cache(key, data)

    async def load(self, key: str) -> Optional[Dict]:
        self.access_count[key] += 1

        # 캐시 확인
        if key in self.cache:
            self._update_access_order(key)
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
        if key in self.access_order:
            self.access_order.remove(key)

    async def list_keys(self, pattern: str = "*") -> List[str]:
        """키 목록 조회"""
        import fnmatch
        if pattern == "*":
            return list(self.data.keys())
        return [k for k in self.data.keys() if fnmatch.fnmatch(k, pattern)]

    def _add_to_cache(self, key: str, data: Any):
        """캐시에 추가 (LRU 정책)"""
        # 캐시 크기 제한
        while len(self.cache) >= self.max_cache_size and self.access_order:
            oldest_key = self.access_order.pop(0)
            if oldest_key in self.cache:
                del self.cache[oldest_key]

        self.cache[key] = data
        self._update_access_order(key)

    def _update_access_order(self, key: str):
        """접근 순서 업데이트"""
        if key in self.access_order:
            self.access_order.remove(key)
        self.access_order.append(key)


# ============================================================================
# 대화 메시지 모델
# ============================================================================

@dataclass
class ConversationMessage:
    """
    대화 메시지 모델 (AgentCore Memory 패턴)
    """
    content: str
    role: str  # USER, ASSISTANT, TOOL
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    agent_name: Optional[str] = None
    session_id: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


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
        self.conversation_history: List[ConversationMessage] = []
        self._logger = StructuredLogger("memory_hook")

    async def on_agent_initialized(self, agent_name: str) -> List[ConversationMessage]:
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
        agent_name: Optional[str] = None
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

    async def get_last_k_turns(self, k: int = 5) -> List[ConversationMessage]:
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
        self._sessions: Dict[str, MemoryHookProvider] = {}
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

    async def list_sessions(self, actor_id: Optional[str] = None) -> List[str]:
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

    def __init__(self, memory_store: MemoryStore, checkpoint_dir: Optional[str] = None):
        self.memory_store = memory_store
        self.checkpoint_dir = checkpoint_dir
        self.state_versions: Dict[str, List[str]] = defaultdict(list)

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

    async def load_state(self, session_id: str, version: Optional[int] = None) -> Optional[AgentState]:
        """상태 로드 (특정 버전 지원)"""
        if version is not None:
            version_key = f"state:{session_id}:v{version}"
            data = await self.memory_store.load(version_key)
        else:
            data = await self.memory_store.load(f"state:{session_id}")

        if data:
            return AgentState(**data)
        return None

    async def save_checkpoint(self, state: AgentState, tag: Optional[str] = None) -> str:
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

    async def restore_checkpoint(self, session_id: str, tag: Optional[str] = None) -> Optional[AgentState]:
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

    async def list_checkpoints(self, session_id: str) -> List[str]:
        """체크포인트 목록"""
        if not self.checkpoint_dir or not os.path.exists(self.checkpoint_dir):
            return []

        checkpoints = [
            f for f in os.listdir(self.checkpoint_dir)
            if f.startswith(session_id) and f.endswith('.json')
        ]
        return sorted(checkpoints)

    async def rollback(self, session_id: str, steps: int = 1) -> Optional[AgentState]:
        """이전 상태로 롤백"""
        versions = self.state_versions.get(session_id, [])
        if len(versions) < steps:
            logging.warning(f"⚠️ 롤백 불가: {steps}단계 이전 버전 없음")
            return None

        target_version = len(versions) - steps - 1
        return await self.load_state(session_id, version=target_version)
