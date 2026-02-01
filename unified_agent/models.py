#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Unified Agent Framework - 데이터 모델 모듈 (Models Module)

================================================================================
📁 파일 위치: unified_agent/models.py
📋 역할: Enum, Pydantic 모델, Dataclass 정의
📅 최종 업데이트: 2026년 2월
================================================================================

🎯 주요 구성 요소:

    📌 Enums:
        - AgentRole: 에이전트 역할 (USER, ASSISTANT, SYSTEM, TOOL)
        - ExecutionStatus: 실행 상태 (PENDING, RUNNING, COMPLETED, FAILED)
        - ApprovalStatus: 승인 상태 (PENDING, APPROVED, REJECTED)
        - WebSocketMessageType: WebSocket 메시지 유형
        - PlanStepStatus: 계획 단계 상태
        - RAICategory: RAI 검증 카테고리

    📌 Pydantic 모델:
        - Message: 대화 메시지
        - AgentState: 에이전트 상태
        - NodeResult: 노드 실행 결과
        - StreamingMessage: 스트리밍 메시지
        - TeamAgent: 팀 에이전트 설정
        - TeamConfiguration: 팀 구성
        - PlanStep: 계획 단계
        - MPlan: 계획 전체
        - RAIValidationResult: RAI 검증 결과

📌 참고:
    - Microsoft Agent Framework: https://github.com/microsoft/agent-framework
    - Pydantic V2: https://docs.pydantic.dev/latest/
"""

from enum import Enum
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Dict, List, Optional, Any, Union
from pydantic import BaseModel, Field

__all__ = [
    # Enums
    "AgentRole",
    "ExecutionStatus",
    "ApprovalStatus",
    "WebSocketMessageType",
    "PlanStepStatus",
    "RAICategory",
    # Pydantic 모델
    "Message",
    "AgentState",
    "NodeResult",
    "StreamingMessage",
    "TeamAgent",
    "TeamConfiguration",
    "PlanStep",
    "MPlan",
    "RAIValidationResult",
]


# ============================================================================
# Enums
# ============================================================================

class AgentRole(str, Enum):
    """에이전트 역할"""
    USER = "user"
    ASSISTANT = "assistant"
    SYSTEM = "system"
    TOOL = "tool"
    SUPERVISOR = "supervisor"
    ROUTER = "router"
    PROXY = "proxy"


class ExecutionStatus(str, Enum):
    """실행 상태"""
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"
    WAITING_APPROVAL = "waiting_approval"


class ApprovalStatus(str, Enum):
    """승인 상태"""
    PENDING = "pending"
    APPROVED = "approved"
    REJECTED = "rejected"
    AUTO_APPROVED = "auto_approved"
    TIMEOUT = "timeout"


class WebSocketMessageType(str, Enum):
    """WebSocket 메시지 유형"""
    AGENT_START = "agent_start"
    AGENT_END = "agent_end"
    STREAM_TOKEN = "stream_token"
    TOOL_CALL = "tool_call"
    TOOL_RESULT = "tool_result"
    ERROR = "error"
    STATUS = "status"
    APPROVAL_REQUEST = "approval_request"
    APPROVAL_RESPONSE = "approval_response"


class PlanStepStatus(str, Enum):
    """계획 단계 상태"""
    NOT_STARTED = "not_started"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    FAILED = "failed"
    SKIPPED = "skipped"
    BLOCKED = "blocked"


class RAICategory(str, Enum):
    """RAI (Responsible AI) 검증 카테고리"""
    HATE = "hate"
    VIOLENCE = "violence"
    SELF_HARM = "self_harm"
    SEXUAL = "sexual"
    JAILBREAK = "jailbreak"
    PROTECTED_MATERIAL = "protected_material"


# ============================================================================
# Pydantic 모델
# ============================================================================

class Message(BaseModel):
    """대화 메시지"""
    role: AgentRole
    content: str
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    agent_name: Optional[str] = None
    metadata: Dict[str, Any] = Field(default_factory=dict)
    
    class Config:
        use_enum_values = True


class AgentState(BaseModel):
    """
    에이전트 상태
    
    워크플로우 실행 중 에이전트 간 전달되는 상태 객체
    """
    session_id: str
    messages: List[Message] = Field(default_factory=list)
    current_node: str = ""
    visited_nodes: List[str] = Field(default_factory=list)
    metadata: Dict[str, Any] = Field(default_factory=dict)
    execution_status: ExecutionStatus = ExecutionStatus.PENDING
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    updated_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    
    def add_message(self, role: Union[AgentRole, str], content: str, agent_name: Optional[str] = None):
        """메시지 추가"""
        if isinstance(role, str):
            role = AgentRole(role)
        self.messages.append(Message(role=role, content=content, agent_name=agent_name))
        self.updated_at = datetime.now(timezone.utc)
    
    def get_conversation_history(self, max_messages: int = 10) -> List[Dict[str, Any]]:
        """대화 기록 조회"""
        return [
            {"role": m.role, "content": m.content}
            for m in self.messages[-max_messages:]
        ]
    
    class Config:
        use_enum_values = True


class NodeResult(BaseModel):
    """노드 실행 결과"""
    node_name: str
    status: ExecutionStatus
    output: Optional[str] = None
    next_node: Optional[str] = None
    metadata: Dict[str, Any] = Field(default_factory=dict)
    duration_ms: float = 0.0
    tokens_used: int = 0
    error: Optional[str] = None
    
    class Config:
        use_enum_values = True


class StreamingMessage(BaseModel):
    """스트리밍 메시지"""
    type: WebSocketMessageType
    content: str = ""
    agent_name: Optional[str] = None
    session_id: Optional[str] = None
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    metadata: Dict[str, Any] = Field(default_factory=dict)
    is_final: bool = False
    
    class Config:
        use_enum_values = True


class TeamAgent(BaseModel):
    """팀 에이전트 설정"""
    name: str
    description: str
    role: AgentRole = AgentRole.ASSISTANT
    system_prompt: Optional[str] = None
    model: Optional[str] = None
    tools: List[str] = Field(default_factory=list)
    metadata: Dict[str, Any] = Field(default_factory=dict)
    
    class Config:
        use_enum_values = True


class TeamConfiguration(BaseModel):
    """팀 구성"""
    name: str
    description: Optional[str] = None
    agents: List[TeamAgent] = Field(default_factory=list)
    orchestration_mode: str = "supervisor"  # supervisor, round_robin, parallel
    max_rounds: int = 10
    metadata: Dict[str, Any] = Field(default_factory=dict)


class PlanStep(BaseModel):
    """계획 단계"""
    index: int
    description: str
    agent_name: str
    status: PlanStepStatus = PlanStepStatus.NOT_STARTED
    depends_on: List[int] = Field(default_factory=list)
    output: Optional[str] = None
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    error: Optional[str] = None
    metadata: Dict[str, Any] = Field(default_factory=dict)
    
    def start(self):
        """단계 시작"""
        self.status = PlanStepStatus.IN_PROGRESS
        self.started_at = datetime.now(timezone.utc)
    
    def complete(self, output: Optional[str] = None):
        """단계 완료"""
        self.status = PlanStepStatus.COMPLETED
        self.completed_at = datetime.now(timezone.utc)
        if output:
            self.output = output
    
    def fail(self, error: str):
        """단계 실패"""
        self.status = PlanStepStatus.FAILED
        self.completed_at = datetime.now(timezone.utc)
        self.error = error
    
    class Config:
        use_enum_values = True


class MPlan(BaseModel):
    """
    구조화된 계획 (Microsoft Agent Framework 패턴)
    
    에이전트 실행 계획을 단계별로 관리
    """
    name: str
    description: Optional[str] = None
    steps: List[PlanStep] = Field(default_factory=list)
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    metadata: Dict[str, Any] = Field(default_factory=dict)
    
    def get_progress(self) -> float:
        """진행률 계산 (0.0 ~ 1.0)"""
        if not self.steps:
            return 0.0
        completed = sum(1 for s in self.steps if s.status == PlanStepStatus.COMPLETED)
        return completed / len(self.steps)
    
    def get_current_step(self) -> Optional[PlanStep]:
        """현재 진행 중인 단계"""
        for step in self.steps:
            if step.status == PlanStepStatus.IN_PROGRESS:
                return step
        return None
    
    def get_next_step(self) -> Optional[PlanStep]:
        """다음 실행 가능한 단계"""
        for step in self.steps:
            if step.status == PlanStepStatus.NOT_STARTED:
                # 의존성 확인
                deps_completed = all(
                    self.steps[dep].status == PlanStepStatus.COMPLETED
                    for dep in step.depends_on
                    if dep < len(self.steps)
                )
                if deps_completed:
                    return step
        return None
    
    def is_completed(self) -> bool:
        """계획 완료 여부"""
        return all(
            s.status in [PlanStepStatus.COMPLETED, PlanStepStatus.SKIPPED]
            for s in self.steps
        )


class RAIValidationResult(BaseModel):
    """RAI (Responsible AI) 검증 결과"""
    is_safe: bool
    categories: Dict[RAICategory, bool] = Field(default_factory=dict)
    scores: Dict[RAICategory, float] = Field(default_factory=dict)
    blocked_content: Optional[str] = None
    details: Optional[str] = None
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    
    class Config:
        use_enum_values = True
