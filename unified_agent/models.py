#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Unified Agent Framework - 데이터 모델 모듈

핵심 데이터 모델, Enum, Pydantic 모델들을 정의합니다.
"""

import time
from datetime import datetime, timezone
from enum import Enum
from typing import Dict, List, Optional, Any

from pydantic import BaseModel, Field

__all__ = [
    # Enum 클래스
    "AgentRole",
    "ExecutionStatus",
    "ApprovalStatus",
    "WebSocketMessageType",
    "PlanStepStatus",
    "RAICategory",

    # 기본 모델
    "Message",
    "AgentState",
    "NodeResult",

    # WebSocket & 스트리밍
    "StreamingMessage",

    # Team & Agent 설정
    "TeamAgent",
    "TeamConfiguration",

    # 계획 시스템
    "PlanStep",
    "MPlan",

    # RAI 검증
    "RAIValidationResult",
]


# ============================================================================
# Enum 클래스
# ============================================================================

class AgentRole(str, Enum):
    """
    Agent 역할 정의

    기존: ASSISTANT, USER, SYSTEM, FUNCTION, ROUTER, ORCHESTRATOR
    추가: SUPERVISOR - 여러 에이전트를 감독하고 조율하는 역할
    """
    ASSISTANT = "assistant"
    USER = "user"
    SYSTEM = "system"
    FUNCTION = "function"
    ROUTER = "router"
    ORCHESTRATOR = "orchestrator"
    SUPERVISOR = "supervisor"


class ExecutionStatus(str, Enum):
    """
    실행 상태 정의

    기존: PENDING, RUNNING, COMPLETED, FAILED, PAUSED, WAITING_APPROVAL
    추가: APPROVED, REJECTED
    """
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    PAUSED = "paused"
    WAITING_APPROVAL = "waiting_approval"
    APPROVED = "approved"
    REJECTED = "rejected"


class ApprovalStatus(str, Enum):
    """
    승인 상태 정의

    - PENDING: 승인 대기 중
    - APPROVED: 사용자가 승인함
    - REJECTED: 사용자가 거부함
    - AUTO_APPROVED: 자동 승인됨 (안전한 작업)
    """
    PENDING = "pending"
    APPROVED = "approved"
    REJECTED = "rejected"
    AUTO_APPROVED = "auto_approved"


class WebSocketMessageType(str, Enum):
    """
    WebSocket 메시지 타입 정의 (Microsoft Pattern)

    실시간 통신을 위한 구조화된 메시지 타입
    """
    # 작업 관련
    START_TASK = "start_task"
    TASK_COMPLETE = "task_complete"
    TASK_PROGRESS = "task_progress"

    # 에이전트 관련
    AGENT_RESPONSE = "agent_response"
    AGENT_SWITCH = "agent_switch"
    AGENT_THINKING = "agent_thinking"

    # 계획 관련
    PLAN_CREATED = "plan_created"
    PLAN_APPROVAL_REQUESTED = "plan_approval_requested"
    PLAN_APPROVED = "plan_approved"
    PLAN_REJECTED = "plan_rejected"
    PLAN_STEP_STARTED = "plan_step_started"
    PLAN_STEP_COMPLETED = "plan_step_completed"

    # 사용자 상호작용
    USER_CLARIFICATION_NEEDED = "user_clarification_needed"
    USER_INPUT_RECEIVED = "user_input_received"

    # 상태
    ERROR = "error"
    HEARTBEAT = "heartbeat"
    CONNECTION_ESTABLISHED = "connection_established"


class PlanStepStatus(str, Enum):
    """계획 단계 상태"""
    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    FAILED = "failed"
    SKIPPED = "skipped"


class RAICategory(str, Enum):
    """RAI 검증 카테고리"""
    HARMFUL_CONTENT = "harmful_content"
    HATE_SPEECH = "hate_speech"
    VIOLENCE = "violence"
    SELF_HARM = "self_harm"
    SEXUAL_CONTENT = "sexual_content"
    JAILBREAK = "jailbreak"
    PII_EXPOSURE = "pii_exposure"
    SAFE = "safe"


# ============================================================================
# 기본 모델
# ============================================================================

class Message(BaseModel):
    """
    메시지 모델

    function_call 필드로 OpenAI Function Calling 지원
    """
    role: AgentRole
    content: str
    metadata: Dict[str, Any] = Field(default_factory=dict)
    timestamp: str = Field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    agent_name: Optional[str] = None
    function_call: Optional[Dict[str, Any]] = None

    class Config:
        use_enum_values = True


class AgentState(BaseModel):
    """
    Agent 상태 - 체크포인팅 및 복원 지원

    - pending_approvals: 승인 대기 중인 요청 목록
    - metrics: 실행 메트릭 (시간, 토큰 등)
    """
    messages: List[Message] = Field(default_factory=list)
    current_node: str = "start"
    visited_nodes: List[str] = Field(default_factory=list)
    metadata: Dict[str, Any] = Field(default_factory=dict)
    session_id: str
    workflow_name: str = "default"
    execution_status: ExecutionStatus = ExecutionStatus.PENDING
    pending_approvals: List[Dict[str, Any]] = Field(default_factory=list)
    metrics: Dict[str, Any] = Field(default_factory=dict)

    def add_message(self, role: AgentRole, content: str, agent_name: Optional[str] = None,
                   function_call: Optional[Dict[str, Any]] = None):
        """메시지 추가"""
        self.messages.append(Message(
            role=role,
            content=content,
            agent_name=agent_name,
            function_call=function_call
        ))

    def get_conversation_history(self, max_messages: int = 10) -> List[Message]:
        """최근 대화 기록"""
        return self.messages[-max_messages:]

    def add_pending_approval(self, approval_request: Dict[str, Any]):
        """승인 대기 요청 추가"""
        self.pending_approvals.append(approval_request)
        self.execution_status = ExecutionStatus.WAITING_APPROVAL


class NodeResult(BaseModel):
    """
    노드 실행 결과

    - requires_approval: 승인이 필요한 작업인지 표시
    - approval_data: 승인 관련 데이터
    """
    node_name: str
    output: str
    next_node: Optional[str] = None
    metadata: Dict[str, Any] = Field(default_factory=dict)
    success: bool = True
    error: Optional[str] = None
    tokens_used: int = 0
    duration_ms: float = 0.0
    requires_approval: bool = False
    approval_data: Optional[Dict[str, Any]] = None


# ============================================================================
# WebSocket & 스트리밍
# ============================================================================

class StreamingMessage(BaseModel):
    """
    WebSocket 스트리밍 메시지 모델

    모든 WebSocket 통신은 이 형식을 따릅니다.
    """
    type: WebSocketMessageType
    content: str = ""
    agent_name: Optional[str] = None
    session_id: Optional[str] = None
    timestamp: str = Field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    metadata: Dict[str, Any] = Field(default_factory=dict)

    # Plan 관련 필드
    plan_id: Optional[str] = None
    step_index: Optional[int] = None
    total_steps: Optional[int] = None

    # 진행률 관련
    progress: Optional[float] = None  # 0.0 ~ 1.0

    def to_json(self) -> str:
        """JSON 문자열로 변환"""
        return self.model_dump_json()

    @classmethod
    def from_json(cls, data: str) -> 'StreamingMessage':
        """JSON 문자열에서 파싱"""
        return cls.model_validate_json(data)


# ============================================================================
# Team & Agent 설정 (Microsoft Pattern)
# ============================================================================

class TeamAgent(BaseModel):
    """
    팀 에이전트 설정 모델 (Microsoft Pattern)

    팀에 속한 개별 에이전트의 설정을 정의합니다.
    """
    name: str
    description: str = ""
    system_prompt: Optional[str] = None

    # 기능 플래그 (Microsoft 패턴)
    use_rag: bool = False           # RAG (Retrieval-Augmented Generation) 사용
    use_mcp: bool = False           # MCP (Model Context Protocol) 사용
    use_reasoning: bool = False     # 추론 강화 모드
    coding_tools: bool = False      # 코딩 도구 사용

    # 도구 설정
    tools: List[str] = Field(default_factory=list)
    mcp_servers: List[str] = Field(default_factory=list)

    # 실행 설정
    max_tokens: int = 1000
    temperature: float = 0.7
    timeout_seconds: int = 60

    # 에이전트 특성
    priority: int = 0               # 실행 우선순위
    is_terminator: bool = False     # 종료 결정 권한
    can_delegate: bool = True       # 다른 에이전트에게 위임 가능

    class Config:
        extra = "allow"


class TeamConfiguration(BaseModel):
    """
    팀 구성 모델 (Microsoft Pattern)

    멀티 에이전트 팀의 전체 구성을 정의합니다.
    """
    name: str
    description: str = ""
    agents: List[TeamAgent] = Field(default_factory=list)

    # 오케스트레이션 설정
    orchestration_mode: str = "supervisor"  # supervisor, sequential, parallel, round_robin
    max_rounds: int = 5
    timeout_seconds: int = 300

    # 계획 설정
    require_plan_approval: bool = False  # 계획 승인 필요 여부
    auto_approve_simple: bool = True     # 간단한 계획 자동 승인

    # RAG 설정 (팀 레벨)
    search_config: Optional[Dict[str, Any]] = None

    # MCP 설정 (팀 레벨)
    mcp_config: Optional[Dict[str, Any]] = None

    # 메타데이터
    created_at: str = Field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    version: str = "1.0"

    def get_agent(self, name: str) -> Optional[TeamAgent]:
        """이름으로 에이전트 조회"""
        for agent in self.agents:
            if agent.name == name:
                return agent
        return None

    def get_terminator_agents(self) -> List[TeamAgent]:
        """종료 결정 권한이 있는 에이전트 목록"""
        return [a for a in self.agents if a.is_terminator]

    def validate_team(self) -> List[str]:
        """팀 구성 검증 - 오류 목록 반환"""
        errors = []
        if not self.agents:
            errors.append("팀에 최소 1개의 에이전트가 필요합니다.")
        if len(set(a.name for a in self.agents)) != len(self.agents):
            errors.append("에이전트 이름은 고유해야 합니다.")
        return errors


# ============================================================================
# 계획 시스템 (Microsoft Pattern)
# ============================================================================

class PlanStep(BaseModel):
    """
    계획 단계 모델 (Microsoft Pattern)

    개별 계획 단계를 정의합니다.
    """
    index: int
    description: str
    agent_name: str
    status: PlanStepStatus = PlanStepStatus.PENDING

    # 실행 결과
    output: Optional[str] = None
    error: Optional[str] = None
    duration_ms: float = 0.0

    # 의존성
    depends_on: List[int] = Field(default_factory=list)  # 선행 단계 인덱스


class MPlan(BaseModel):
    """
    MPlan - 구조화된 실행 계획 (Microsoft Pattern)

    Human-in-the-loop 패턴의 핵심 모델입니다.
    복잡한 작업을 단계별로 분해하고 승인을 관리합니다.
    """
    id: str = Field(default_factory=lambda: f"plan-{int(time.time()*1000)}")
    name: str
    description: str = ""
    steps: List[PlanStep] = Field(default_factory=list)

    # 상태
    status: PlanStepStatus = PlanStepStatus.PENDING
    current_step_index: int = 0

    # 승인 관련
    requires_approval: bool = False
    approval_status: Optional[str] = None  # pending, approved, rejected
    approved_by: Optional[str] = None
    approved_at: Optional[str] = None

    # 복잡도 분석
    complexity: str = "simple"  # simple, moderate, complex
    estimated_duration_seconds: int = 0

    # 메타데이터
    created_at: str = Field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    reasoning: str = ""  # 계획 수립 근거

    def get_next_steps(self) -> List[PlanStep]:
        """실행 가능한 다음 단계들 반환 (의존성 확인)"""
        ready_steps = []
        completed_indices = {s.index for s in self.steps if s.status == PlanStepStatus.COMPLETED}

        for step in self.steps:
            if step.status == PlanStepStatus.PENDING:
                if all(dep in completed_indices for dep in step.depends_on):
                    ready_steps.append(step)
        return ready_steps

    def complete_step(self, index: int, output: str, duration_ms: float = 0.0):
        """단계 완료 처리"""
        for step in self.steps:
            if step.index == index:
                step.status = PlanStepStatus.COMPLETED
                step.output = output
                step.duration_ms = duration_ms
                break
        self._update_status()

    def fail_step(self, index: int, error: str):
        """단계 실패 처리"""
        for step in self.steps:
            if step.index == index:
                step.status = PlanStepStatus.FAILED
                step.error = error
                break
        self.status = PlanStepStatus.FAILED

    def _update_status(self):
        """전체 계획 상태 업데이트"""
        statuses = [s.status for s in self.steps]
        if all(s == PlanStepStatus.COMPLETED for s in statuses):
            self.status = PlanStepStatus.COMPLETED
        elif any(s == PlanStepStatus.FAILED for s in statuses):
            self.status = PlanStepStatus.FAILED
        elif any(s == PlanStepStatus.IN_PROGRESS for s in statuses):
            self.status = PlanStepStatus.IN_PROGRESS

    def request_approval(self):
        """승인 요청"""
        self.requires_approval = True
        self.approval_status = "pending"

    def approve(self, approved_by: str = "user"):
        """계획 승인"""
        self.approval_status = "approved"
        self.approved_by = approved_by
        self.approved_at = datetime.now(timezone.utc).isoformat()

    def reject(self, reason: str = ""):
        """계획 거부"""
        self.approval_status = "rejected"
        self.reasoning = reason

    def get_progress(self) -> float:
        """진행률 반환 (0.0 ~ 1.0)"""
        if not self.steps:
            return 0.0
        completed = sum(1 for s in self.steps if s.status == PlanStepStatus.COMPLETED)
        return completed / len(self.steps)

    def to_summary(self) -> str:
        """계획 요약 문자열"""
        lines = [f"📋 계획: {self.name}"]
        lines.append(f"   설명: {self.description}")
        lines.append(f"   복잡도: {self.complexity}")
        lines.append(f"   단계 수: {len(self.steps)}")
        lines.append(f"   진행률: {self.get_progress()*100:.0f}%")
        lines.append("")
        for step in self.steps:
            status_icon = {
                PlanStepStatus.PENDING: "⏳",
                PlanStepStatus.IN_PROGRESS: "🔄",
                PlanStepStatus.COMPLETED: "✅",
                PlanStepStatus.FAILED: "❌",
                PlanStepStatus.SKIPPED: "⏭️"
            }.get(step.status, "❓")
            lines.append(f"   {status_icon} [{step.index}] {step.description} ({step.agent_name})")
        return "\n".join(lines)


# ============================================================================
# RAI 검증 모델
# ============================================================================

class RAIValidationResult(BaseModel):
    """
    RAI 검증 결과 모델

    AI 출력의 안전성을 검증한 결과를 담습니다.
    """
    is_safe: bool = True
    category: RAICategory = RAICategory.SAFE
    confidence: float = 1.0
    reason: str = ""
    suggestions: List[str] = Field(default_factory=list)
    checked_at: str = Field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
