#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Unified Agent Framework - 세션 트리 시스템 (Session Tree Module)

================================================================================
📁 파일 위치: unified_agent/session_tree.py
📋 역할: Pi 스타일 세션 트리 - 대화 브랜칭, 리와인드, 요약
📅 최종 업데이트: 2026년 2월
================================================================================

🎯 주요 구성 요소:

    📌 세션 트리 (Session Tree):
        - 대화를 트리 구조로 관리
        - 브랜치 생성으로 사이드 퀘스트 지원
        - 메인 세션으로 복귀 가능

    📌 리와인드 (Rewind):
        - 특정 시점으로 되돌아가기
        - 브랜치에서 작업 후 복귀
        - 실험적 대화 후 롤백

    📌 브랜치 요약 (Branch Summary):
        - 브랜치 종료 시 자동 요약
        - 메인 세션에 요약 주입

🔧 핵심 기능:
    - 트리 기반 대화 관리
    - 브랜치 분기/병합
    - 상태 스냅샷
    - Hot Reloading 지원

📌 참고:
    - Pi Agent: https://lucumr.pocoo.org/2026/1/31/pi/
    - Session Branching: Pi sessions are trees
"""

from __future__ import annotations

import os
import json
import uuid
import copy
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable
from enum import Enum

from .utils import StructuredLogger

__all__ = [
    # 설정
    "SessionTreeConfig",
    # 노드
    "SessionNode",
    "NodeType",
    # 트리
    "SessionTree",
    "BranchInfo",
    # 매니저
    "SessionTreeManager",
    # 스냅샷
    "SessionSnapshot",
]

# ============================================================================
# Enums & Constants
# ============================================================================

class NodeType(Enum):
    """세션 노드 유형"""
    ROOT = "root"              # 루트 노드
    USER = "user"              # 사용자 메시지
    ASSISTANT = "assistant"    # 어시스턴트 응답
    TOOL = "tool"              # 도구 호출/결과
    SYSTEM = "system"          # 시스템 메시지
    BRANCH_POINT = "branch"    # 브랜치 분기점
    SUMMARY = "summary"        # 브랜치 요약

# ============================================================================
# Configuration
# ============================================================================

@dataclass(frozen=True, slots=True)
class SessionTreeConfig:
    """
    세션 트리 설정
    
    Args:
        max_depth: 최대 트리 깊이
        auto_summarize_on_merge: 브랜치 병합 시 자동 요약
        snapshot_interval: 스냅샷 간격 (노드 수)
        persist_to_disk: 디스크 영속화 여부
        session_dir: 세션 저장 디렉토리
    """
    max_depth: int = 100
    auto_summarize_on_merge: bool = True
    snapshot_interval: int = 10
    persist_to_disk: bool = True
    session_dir: str = field(default_factory=lambda: os.path.expanduser("~/.agent_sessions"))

# ============================================================================
# Data Models
# ============================================================================

@dataclass(slots=True)
class SessionNode:
    """
    세션 트리의 노드
    
    각 노드는 하나의 대화 턴 또는 특수 이벤트를 나타냄
    """
    id: str
    type: NodeType
    content: str
    parent_id: str | None = None
    children_ids: list[str] = field(default_factory=list)
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    metadata: dict[str, Any] = field(default_factory=dict)
    branch_name: str | None = None  # 이 노드가 속한 브랜치
    
    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "type": self.type.value,
            "content": self.content,
            "parent_id": self.parent_id,
            "children_ids": self.children_ids,
            "timestamp": self.timestamp.isoformat(),
            "metadata": self.metadata,
            "branch_name": self.branch_name
        }
    
    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "SessionNode":
        return cls(
            id=data["id"],
            type=NodeType(data["type"]),
            content=data["content"],
            parent_id=data.get("parent_id"),
            children_ids=data.get("children_ids", []),
            timestamp=datetime.fromisoformat(data["timestamp"]) if "timestamp" in data else datetime.now(timezone.utc),
            metadata=data.get("metadata", {}),
            branch_name=data.get("branch_name")
        )

@dataclass(slots=True)
class BranchInfo:
    """브랜치 정보"""
    name: str
    branch_point_id: str  # 분기 시작점
    head_id: str          # 현재 헤드
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    description: str | None = None
    is_active: bool = True
    summary: str | None = None  # 병합 시 요약
    
    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "branch_point_id": self.branch_point_id,
            "head_id": self.head_id,
            "created_at": self.created_at.isoformat(),
            "description": self.description,
            "is_active": self.is_active,
            "summary": self.summary
        }
    
    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "BranchInfo":
        return cls(
            name=data["name"],
            branch_point_id=data["branch_point_id"],
            head_id=data["head_id"],
            created_at=datetime.fromisoformat(data["created_at"]) if "created_at" in data else datetime.now(timezone.utc),
            description=data.get("description"),
            is_active=data.get("is_active", True),
            summary=data.get("summary")
        )

@dataclass(frozen=True, slots=True)
class SessionSnapshot:
    """세션 스냅샷 (특정 시점의 전체 상태)"""
    id: str
    session_id: str
    node_id: str  # 스냅샷 시점의 노드
    branch_name: str
    nodes: dict[str, dict]  # 노드 ID -> 노드 데이터
    branches: dict[str, dict]  # 브랜치 이름 -> 브랜치 데이터
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    metadata: dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "session_id": self.session_id,
            "node_id": self.node_id,
            "branch_name": self.branch_name,
            "nodes": self.nodes,
            "branches": self.branches,
            "created_at": self.created_at.isoformat(),
            "metadata": self.metadata
        }

# ============================================================================
# Session Tree
# ============================================================================

class SessionTree:
    """
    세션 트리 - Pi 스타일 대화 관리
    
    대화를 트리 구조로 관리하여:
    - 브랜치 분기: 사이드 퀘스트, 실험적 대화
    - 리와인드: 특정 시점으로 되돌아가기
    - 병합: 브랜치 작업을 메인에 반영
    
    사용 예시:
        >>> tree = SessionTree(session_id="main-session")
        >>> 
        >>> # 대화 추가
        >>> tree.add_message("user", "API 설계 도와줘")
        >>> tree.add_message("assistant", "어떤 종류의 API인가요?")
        >>> 
        >>> # 브랜치 생성 (사이드 퀘스트)
        >>> tree.create_branch("fix-bug", "버그 수정용 브랜치")
        >>> tree.add_message("user", "여기서 버그 좀 고쳐줘")
        >>> 
        >>> # 메인으로 복귀
        >>> tree.switch_branch("main")
        >>> 
        >>> # 브랜치 요약 주입
        >>> summary = tree.get_branch_summary("fix-bug")
    """
    
    MAIN_BRANCH = "main"
    
    def __init__(
        self,
        session_id: str,
        config: SessionTreeConfig | None = None
    ):
        self.session_id = session_id
        self.config = config or SessionTreeConfig()
        
        # 노드 저장소
        self._nodes: dict[str, SessionNode] = {}
        
        # 브랜치 관리
        self._branches: dict[str, BranchInfo] = {}
        self._current_branch = self.MAIN_BRANCH
        
        # 루트 노드 생성
        self._root_id = self._create_root()
        
        # 현재 헤드 (가장 최근 노드)
        self._head_id = self._root_id
        
        # 스냅샷
        self._snapshots: list[SessionSnapshot] = []
        self._nodes_since_snapshot = 0
        
        self._logger = StructuredLogger("session_tree")
    
    def _generate_id(self) -> str:
        """고유 ID 생성"""
        return str(uuid.uuid4())[:8]
    
    def _create_root(self) -> str:
        """루트 노드 생성"""
        root_id = f"root_{self._generate_id()}"
        root_node = SessionNode(
            id=root_id,
            type=NodeType.ROOT,
            content="Session started",
            branch_name=self.MAIN_BRANCH
        )
        self._nodes[root_id] = root_node
        
        # 메인 브랜치 생성
        self._branches[self.MAIN_BRANCH] = BranchInfo(
            name=self.MAIN_BRANCH,
            branch_point_id=root_id,
            head_id=root_id,
            description="Main conversation branch"
        )
        
        return root_id
    
    @property
    def current_branch(self) -> str:
        """현재 브랜치 이름"""
        return self._current_branch
    
    @property
    def head(self) -> SessionNode:
        """현재 헤드 노드"""
        return self._nodes[self._head_id]
    
    @property
    def root(self) -> SessionNode:
        """루트 노드"""
        return self._nodes[self._root_id]
    
    def add_message(
        self,
        role: str,
        content: str,
        metadata: dict[str, Any] | None = None
    ) -> SessionNode:
        """
        메시지 추가
        
        Args:
            role: 역할 ('user', 'assistant', 'tool', 'system')
            content: 메시지 내용
            metadata: 추가 메타데이터
        
        Returns:
            생성된 노드
        """
        node_type = NodeType(role) if role in [t.value for t in NodeType] else NodeType.USER
        
        node_id = f"{role}_{self._generate_id()}"
        node = SessionNode(
            id=node_id,
            type=node_type,
            content=content,
            parent_id=self._head_id,
            metadata=metadata or {},
            branch_name=self._current_branch
        )
        
        # 부모 노드에 자식 추가
        self._nodes[self._head_id].children_ids.append(node_id)
        
        # 노드 저장 및 헤드 이동
        self._nodes[node_id] = node
        self._head_id = node_id
        
        # 브랜치 헤드 업데이트
        self._branches[self._current_branch].head_id = node_id
        
        # 자동 스냅샷
        self._nodes_since_snapshot += 1
        if self._nodes_since_snapshot >= self.config.snapshot_interval:
            self._create_snapshot()
        
        return node
    
    def create_branch(
        self,
        name: str,
        description: str | None = None
    ) -> BranchInfo:
        """
        새 브랜치 생성 및 전환
        
        현재 헤드에서 분기하여 새 브랜치 생성
        
        Args:
            name: 브랜치 이름
            description: 브랜치 설명
        
        Returns:
            생성된 브랜치 정보
        """
        if name in self._branches:
            raise ValueError(f"Branch '{name}' already exists")
        
        # 분기점 노드 생성
        branch_point_id = f"branch_{self._generate_id()}"
        branch_point = SessionNode(
            id=branch_point_id,
            type=NodeType.BRANCH_POINT,
            content=f"Branch point: {name}",
            parent_id=self._head_id,
            metadata={"target_branch": name},
            branch_name=self._current_branch
        )
        
        self._nodes[self._head_id].children_ids.append(branch_point_id)
        self._nodes[branch_point_id] = branch_point
        
        # 브랜치 생성
        branch = BranchInfo(
            name=name,
            branch_point_id=branch_point_id,
            head_id=branch_point_id,
            description=description
        )
        self._branches[name] = branch
        
        # 브랜치로 전환
        self._current_branch = name
        self._head_id = branch_point_id
        
        self._logger.info(f"Created and switched to branch: {name}")
        
        return branch
    
    def switch_branch(self, name: str) -> BranchInfo:
        """
        브랜치 전환
        
        Args:
            name: 전환할 브랜치 이름
        
        Returns:
            전환된 브랜치 정보
        """
        if name not in self._branches:
            raise ValueError(f"Branch '{name}' does not exist")
        
        branch = self._branches[name]
        self._current_branch = name
        self._head_id = branch.head_id
        
        self._logger.info(f"Switched to branch: {name}")
        
        return branch
    
    def list_branches(self) -> list[BranchInfo]:
        """모든 브랜치 목록"""
        return list(self._branches.values())
    
    def get_branch(self, name: str) -> BranchInfo | None:
        """브랜치 정보 조회"""
        return self._branches.get(name)
    
    def rewind(self, target_node_id: str) -> SessionNode:
        """
        특정 노드로 리와인드
        
        현재 브랜치의 헤드를 지정된 노드로 이동
        (노드 삭제 없이 헤드만 이동)
        
        Args:
            target_node_id: 목표 노드 ID
        
        Returns:
            새 헤드 노드
        """
        if target_node_id not in self._nodes:
            raise ValueError(f"Node '{target_node_id}' does not exist")
        
        target_node = self._nodes[target_node_id]
        
        # 현재 브랜치에 속한 노드인지 확인
        if not self._is_ancestor_of_current(target_node_id):
            raise ValueError(f"Node '{target_node_id}' is not in current branch path")
        
        self._head_id = target_node_id
        self._branches[self._current_branch].head_id = target_node_id
        
        self._logger.info(f"Rewound to node: {target_node_id}")
        
        return target_node
    
    def _is_ancestor_of_current(self, node_id: str) -> bool:
        """노드가 현재 경로의 조상인지 확인"""
        current = self._head_id
        while current:
            if current == node_id:
                return True
            current = self._nodes[current].parent_id
        return False
    
    def get_path_to_root(self, node_id: str | None = None) -> list[SessionNode]:
        """
        노드에서 루트까지의 경로
        
        Args:
            node_id: 시작 노드 (기본: 현재 헤드)
        
        Returns:
            노드 리스트 (루트 → 현재)
        """
        path = []
        current = node_id or self._head_id
        
        while current:
            path.append(self._nodes[current])
            current = self._nodes[current].parent_id
        
        path.reverse()
        return path
    
    def get_conversation_history(self, branch: str | None = None) -> list[dict[str, Any]]:
        """
        대화 기록 조회 (OpenAI 메시지 형식)
        
        Args:
            branch: 브랜치 이름 (기본: 현재 브랜치)
        
        Returns:
            메시지 리스트
        """
        target_branch = branch or self._current_branch
        branch_info = self._branches[target_branch]
        path = self.get_path_to_root(branch_info.head_id)
        
        messages = []
        for node in path:
            if node.type in [NodeType.USER, NodeType.ASSISTANT, NodeType.TOOL, NodeType.SYSTEM]:
                messages.append({
                    "role": node.type.value,
                    "content": node.content,
                    "metadata": {
                        "node_id": node.id,
                        "timestamp": node.timestamp.isoformat(),
                        **node.metadata
                    }
                })
            elif node.type == NodeType.SUMMARY:
                messages.append({
                    "role": "system",
                    "content": f"[Branch Summary: {node.branch_name}]\n{node.content}",
                    "metadata": {"node_id": node.id, "type": "branch_summary"}
                })
        
        return messages
    
    async def merge_branch(
        self,
        branch_name: str,
        summarizer: Callable[[list[dict]], str] | None = None
    ) -> str | None:
        """
        브랜치를 메인에 병합
        
        브랜치 작업 내용을 요약하여 메인 브랜치에 주입
        
        Args:
            branch_name: 병합할 브랜치
            summarizer: 요약 함수 (None이면 요약 스킵)
        
        Returns:
            요약 텍스트 또는 None
        """
        if branch_name == self.MAIN_BRANCH:
            raise ValueError("Cannot merge main branch")
        
        if branch_name not in self._branches:
            raise ValueError(f"Branch '{branch_name}' does not exist")
        
        branch = self._branches[branch_name]
        
        # 브랜치 비활성화
        branch.is_active = False
        
        summary = None
        if self.config.auto_summarize_on_merge and summarizer:
            # 브랜치 대화 기록 가져오기
            history = self.get_conversation_history(branch_name)
            
            # 브랜치 시작점 이후의 메시지만
            branch_messages = [
                m for m in history 
                if m.get("metadata", {}).get("node_id") != branch.branch_point_id
            ]
            
            if branch_messages:
                summary = await summarizer(branch_messages)
                branch.summary = summary
                
                # 메인 브랜치에 요약 노드 추가
                self.switch_branch(self.MAIN_BRANCH)
                summary_node = SessionNode(
                    id=f"summary_{self._generate_id()}",
                    type=NodeType.SUMMARY,
                    content=summary,
                    parent_id=self._head_id,
                    branch_name=branch_name,
                    metadata={"merged_from": branch_name}
                )
                self._nodes[self._head_id].children_ids.append(summary_node.id)
                self._nodes[summary_node.id] = summary_node
                self._head_id = summary_node.id
                self._branches[self.MAIN_BRANCH].head_id = summary_node.id
        
        self._logger.info(f"Merged branch: {branch_name}")
        
        return summary
    
    def get_branch_summary(self, branch_name: str) -> str | None:
        """브랜치 요약 조회"""
        branch = self._branches.get(branch_name)
        return branch.summary if branch else None
    
    def _create_snapshot(self) -> SessionSnapshot:
        """스냅샷 생성"""
        snapshot = SessionSnapshot(
            id=f"snap_{self._generate_id()}",
            session_id=self.session_id,
            node_id=self._head_id,
            branch_name=self._current_branch,
            nodes={k: v.to_dict() for k, v in self._nodes.items()},
            branches={k: v.to_dict() for k, v in self._branches.items()}
        )
        
        self._snapshots.append(snapshot)
        self._nodes_since_snapshot = 0
        
        self._logger.debug(f"Snapshot created: {snapshot.id}")
        
        return snapshot
    
    def restore_snapshot(self, snapshot_id: str) -> bool:
        """
        스냅샷에서 복원
        
        Args:
            snapshot_id: 복원할 스냅샷 ID
        
        Returns:
            성공 여부
        """
        snapshot = next((s for s in self._snapshots if s.id == snapshot_id), None)
        if not snapshot:
            return False
        
        # 상태 복원
        self._nodes = {k: SessionNode.from_dict(v) for k, v in snapshot.nodes.items()}
        self._branches = {k: BranchInfo.from_dict(v) for k, v in snapshot.branches.items()}
        self._current_branch = snapshot.branch_name
        self._head_id = snapshot.node_id
        
        self._logger.info(f"Restored from snapshot: {snapshot_id}")
        
        return True
    
    def list_snapshots(self) -> list[dict[str, Any]]:
        """스냅샷 목록"""
        return [
            {
                "id": s.id,
                "node_id": s.node_id,
                "branch": s.branch_name,
                "created_at": s.created_at.isoformat()
            }
            for s in self._snapshots
        ]
    
    def get_tree_stats(self) -> dict[str, Any]:
        """트리 통계"""
        return {
            "session_id": self.session_id,
            "total_nodes": len(self._nodes),
            "total_branches": len(self._branches),
            "active_branches": len([b for b in self._branches.values() if b.is_active]),
            "current_branch": self._current_branch,
            "current_depth": len(self.get_path_to_root()),
            "snapshots": len(self._snapshots)
        }
    
    def to_dict(self) -> dict[str, Any]:
        """트리 전체를 딕셔너리로 직렬화"""
        return {
            "session_id": self.session_id,
            "root_id": self._root_id,
            "head_id": self._head_id,
            "current_branch": self._current_branch,
            "nodes": {k: v.to_dict() for k, v in self._nodes.items()},
            "branches": {k: v.to_dict() for k, v in self._branches.items()},
            "config": {
                "max_depth": self.config.max_depth,
                "auto_summarize_on_merge": self.config.auto_summarize_on_merge,
                "snapshot_interval": self.config.snapshot_interval
            }
        }
    
    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "SessionTree":
        """딕셔너리에서 트리 복원"""
        config = SessionTreeConfig(
            max_depth=data["config"]["max_depth"],
            auto_summarize_on_merge=data["config"]["auto_summarize_on_merge"],
            snapshot_interval=data["config"]["snapshot_interval"]
        )
        
        tree = cls(session_id=data["session_id"], config=config)
        
        # 기존 상태 덮어쓰기
        tree._root_id = data["root_id"]
        tree._head_id = data["head_id"]
        tree._current_branch = data["current_branch"]
        tree._nodes = {k: SessionNode.from_dict(v) for k, v in data["nodes"].items()}
        tree._branches = {k: BranchInfo.from_dict(v) for k, v in data["branches"].items()}
        
        return tree

# ============================================================================
# Session Tree Manager
# ============================================================================

class SessionTreeManager:
    """
    세션 트리 관리자
    
    여러 세션 트리를 관리하고 디스크 영속화 담당
    
    사용 예시:
        >>> manager = SessionTreeManager(session_dir="~/.agent_sessions")
        >>> 
        >>> # 세션 생성 또는 로드
        >>> tree = manager.get_or_create("my-session")
        >>> 
        >>> # 대화 추가
        >>> tree.add_message("user", "Hello")
        >>> 
        >>> # 저장
        >>> manager.save(tree)
        >>> 
        >>> # 세션 목록
        >>> sessions = manager.list_sessions()
    """
    
    def __init__(self, session_dir: str | None = None):
        self.session_dir = Path(session_dir or os.path.expanduser("~/.agent_sessions"))
        self._trees: dict[str, SessionTree] = {}
        self._logger = StructuredLogger("session_tree_manager")
        
        self.session_dir.mkdir(parents=True, exist_ok=True)
    
    def _get_session_path(self, session_id: str) -> Path:
        """세션 파일 경로"""
        return self.session_dir / f"{session_id}.json"
    
    def get_or_create(
        self,
        session_id: str,
        config: SessionTreeConfig | None = None
    ) -> SessionTree:
        """
        세션 조회 또는 생성
        
        메모리에 있으면 반환, 없으면 디스크에서 로드, 없으면 새로 생성
        """
        # 메모리 캐시 확인
        if session_id in self._trees:
            return self._trees[session_id]
        
        # 디스크에서 로드 시도
        session_path = self._get_session_path(session_id)
        if session_path.exists():
            try:
                with open(session_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                tree = SessionTree.from_dict(data)
                self._trees[session_id] = tree
                self._logger.info(f"Loaded session from disk: {session_id}")
                return tree
            except Exception as e:
                self._logger.warning(f"Failed to load session {session_id}: {e}")
        
        # 새로 생성
        tree = SessionTree(session_id, config)
        self._trees[session_id] = tree
        self._logger.info(f"Created new session: {session_id}")
        
        return tree
    
    def save(self, tree: SessionTree) -> bool:
        """세션 저장"""
        try:
            session_path = self._get_session_path(tree.session_id)
            with open(session_path, 'w', encoding='utf-8') as f:
                json.dump(tree.to_dict(), f, ensure_ascii=False, indent=2)
            
            self._logger.info(f"Saved session: {tree.session_id}")
            return True
        except Exception as e:
            self._logger.error(f"Failed to save session {tree.session_id}: {e}")
            return False
    
    def save_all(self) -> None:
        """모든 세션 저장"""
        for tree in self._trees.values():
            self.save(tree)
    
    def delete(self, session_id: str) -> bool:
        """세션 삭제"""
        # 메모리에서 제거
        if session_id in self._trees:
            del self._trees[session_id]
        
        # 디스크에서 제거
        session_path = self._get_session_path(session_id)
        if session_path.exists():
            session_path.unlink()
            self._logger.info(f"Deleted session: {session_id}")
            return True
        
        return False
    
    def list_sessions(self) -> list[dict[str, Any]]:
        """모든 세션 목록"""
        sessions = []
        
        for session_file in self.session_dir.glob("*.json"):
            session_id = session_file.stem
            try:
                with open(session_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                sessions.append({
                    "session_id": session_id,
                    "total_nodes": len(data.get("nodes", {})),
                    "branches": list(data.get("branches", {}).keys()),
                    "current_branch": data.get("current_branch"),
                    "modified": datetime.fromtimestamp(session_file.stat().st_mtime).isoformat()
                })
            except Exception as e:
                logger.warning(f"[세션 로드 실패] {session_id}: {e}")
                sessions.append({
                    "session_id": session_id,
                    "error": f"Failed to load: {e}"
                })
        
        return sessions
    
    def export_conversation(
        self,
        session_id: str,
        branch: str | None = None,
        format: str = "markdown"
    ) -> str:
        """
        대화 내보내기
        
        Args:
            session_id: 세션 ID
            branch: 브랜치 이름 (기본: 현재)
            format: 출력 형식 ('markdown', 'json')
        
        Returns:
            포맷된 대화 내용
        """
        tree = self.get_or_create(session_id)
        history = tree.get_conversation_history(branch)
        
        if format == "json":
            return json.dumps(history, ensure_ascii=False, indent=2)
        
        # Markdown 형식
        lines = [f"# Session: {session_id}", f"Branch: {branch or tree.current_branch}", ""]
        
        for msg in history:
            role = msg["role"].upper()
            content = msg["content"]
            lines.append(f"## {role}")
            lines.append(content)
            lines.append("")
        
        return "\n".join(lines)
