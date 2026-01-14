#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Unified Agent Framework - Enterprise Edition v2.0

통합 AI 에이전트 프레임워크

특징:
- Azure OpenAI + Semantic Kernel 통합
- Microsoft Multi-Agent-Custom-Automation-Engine 패턴
- Human-in-the-loop 승인 시스템
- MCP (Model Context Protocol) 지원
- Skills 시스템 (Anthropic 패턴)
- 이벤트 기반 아키텍처
- 체크포인트 및 롤백

사용법:
    # 간단한 사용
    from unified_agent import UnifiedAgentFramework, quick_run

    # 방법 1: 빠른 시작
    framework = UnifiedAgentFramework.create()
    response = await framework.quick_chat("안녕하세요!")

    # 방법 2: 워크플로우 실행
    framework.create_simple_workflow("my_workflow")
    state = await framework.run("session-1", "my_workflow", "질문입니다")

    # 방법 3: 팀 기반 멀티에이전트
    team_config = TeamConfiguration(...)
    workflow = framework.create_team_workflow(team_config)

라이선스: MIT
"""

__version__ = "2.0.0"
__author__ = "Enterprise AI Team"

# ============================================================================
# 핵심 Exceptions
# ============================================================================
from .exceptions import (
    FrameworkError,
    ConfigurationError,
    WorkflowError,
    AgentError,
    ApprovalError,
    RAIValidationError,
)

# ============================================================================
# 설정
# ============================================================================
from .config import (
    Settings,
    FrameworkConfig,
    DEFAULT_LLM_MODEL,
    DEFAULT_API_VERSION,
    SUPPORTED_MODELS,
    O_SERIES_MODELS,
    MODELS_WITHOUT_TEMPERATURE,
    supports_temperature,
    create_execution_settings,
)

# ============================================================================
# 모델 (Enums, Pydantic 모델, Dataclasses)
# ============================================================================
from .models import (
    # Enums
    AgentRole,
    ExecutionStatus,
    ApprovalStatus,
    WebSocketMessageType,
    PlanStepStatus,
    RAICategory,
    # Pydantic 모델
    Message,
    AgentState,
    NodeResult,
    StreamingMessage,
    TeamAgent,
    TeamConfiguration,
    PlanStep,
    MPlan,
    RAIValidationResult,
)

# ============================================================================
# 유틸리티
# ============================================================================
from .utils import (
    StructuredLogger,
    retry_with_backoff,
    CircuitBreaker,
    setup_telemetry,
    RAIValidator,
)

# ============================================================================
# 메모리 시스템
# ============================================================================
from .memory import (
    MemoryStore,
    CachedMemoryStore,
    ConversationMessage,
    MemoryHookProvider,
    MemorySessionManager,
    StateManager,
)

# ============================================================================
# 이벤트 시스템
# ============================================================================
from .events import (
    EventType,
    AgentEvent,
    EventBus,
)

# ============================================================================
# Skills 시스템
# ============================================================================
from .skills import (
    SkillResource,
    Skill,
    SkillManager,
)

# ============================================================================
# 도구
# ============================================================================
from .tools import (
    AIFunction,
    ApprovalRequiredAIFunction,
    MockMCPClient,
    MCPTool,
)

# ============================================================================
# 에이전트
# ============================================================================
from .agents import (
    Agent,
    SimpleAgent,
    ApprovalAgent,
    RouterAgent,
    ProxyAgent,
    InvestigationPlan,
    SupervisorAgent,
)

# ============================================================================
# 워크플로우
# ============================================================================
from .workflow import (
    Node,
    Graph,
)

# ============================================================================
# 오케스트레이션
# ============================================================================
from .orchestration import (
    AgentFactory,
    OrchestrationManager,
)

# ============================================================================
# 프레임워크 메인
# ============================================================================
from .framework import (
    UnifiedAgentFramework,
    quick_run,
    create_framework,
)

# ============================================================================
# Public API
# ============================================================================
__all__ = [
    # 버전
    "__version__",
    "__author__",

    # Exceptions
    "FrameworkError",
    "ConfigurationError",
    "WorkflowError",
    "AgentError",
    "ApprovalError",
    "RAIValidationError",

    # Config
    "Settings",
    "FrameworkConfig",
    "DEFAULT_LLM_MODEL",
    "DEFAULT_API_VERSION",
    "SUPPORTED_MODELS",
    "O_SERIES_MODELS",
    "MODELS_WITHOUT_TEMPERATURE",
    "supports_temperature",
    "create_execution_settings",

    # Models - Enums
    "AgentRole",
    "ExecutionStatus",
    "ApprovalStatus",
    "WebSocketMessageType",
    "PlanStepStatus",
    "RAICategory",

    # Models - Pydantic/Dataclass
    "Message",
    "AgentState",
    "NodeResult",
    "StreamingMessage",
    "TeamAgent",
    "TeamConfiguration",
    "PlanStep",
    "MPlan",
    "RAIValidationResult",

    # Utils
    "StructuredLogger",
    "retry_with_backoff",
    "CircuitBreaker",
    "setup_telemetry",
    "RAIValidator",

    # Memory
    "MemoryStore",
    "CachedMemoryStore",
    "ConversationMessage",
    "MemoryHookProvider",
    "MemorySessionManager",
    "StateManager",

    # Events
    "EventType",
    "AgentEvent",
    "EventBus",

    # Skills
    "SkillResource",
    "Skill",
    "SkillManager",

    # Tools
    "AIFunction",
    "ApprovalRequiredAIFunction",
    "MockMCPClient",
    "MCPTool",

    # Agents
    "Agent",
    "SimpleAgent",
    "ApprovalAgent",
    "RouterAgent",
    "ProxyAgent",
    "InvestigationPlan",
    "SupervisorAgent",

    # Workflow
    "Node",
    "Graph",

    # Orchestration
    "AgentFactory",
    "OrchestrationManager",

    # Framework
    "UnifiedAgentFramework",
    "quick_run",
    "create_framework",
]
