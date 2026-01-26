#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Unified Agent Framework - Enterprise Edition v3.1

================================================================================
📋 프로젝트: 통합 AI 에이전트 프레임워크
📅 버전: 3.1.0 (2026년 1월 최신)
📦 Python: 3.11+
================================================================================

🌟 프레임워크 특징:
    ★ Azure OpenAI + Semantic Kernel 통합
    ★ Microsoft Agent Framework MCP 패턴 완전 통합
    ★ GPT-5.2, Claude Opus 4.5, Grok-4 등 2026년 최신 모델 지원
    ★ Human-in-the-loop 승인 시스템
    ★ MCP (Model Context Protocol) 네이티브 지원
    ★ Skills 시스템 (Anthropic 패턴)
    ★ 이벤트 기반 아키텍처 (EventBus)
    ★ 체크포인트 및 롤백
    ★ Adaptive Circuit Breaker (2026년 개선)
    ★ 대용량 컨텍스트 지원 (최대 10M tokens)

📁 모듈 구조:
    unified_agent/
    ├── __init__.py      # 이 파일 - 패키지 진입점
    ├── config.py        # 설정 관리 (Settings, FrameworkConfig)
    ├── models.py        # 데이터 모델 (Enum, Pydantic)
    ├── utils.py         # 유틸리티 (CircuitBreaker, RAIValidator)
    ├── memory.py        # 메모리 시스템 (StateManager, Cache)
    ├── events.py        # 이벤트 시스템 (EventBus)
    ├── skills.py        # 스킬 시스템 (SkillManager)
    ├── tools.py         # 도구 (MCPTool, AIFunction)
    ├── agents.py        # 에이전트 (SimpleAgent, SupervisorAgent)
    ├── workflow.py      # 워크플로우 (Graph, Node)
    ├── orchestration.py # 오케스트레이션 (OrchestrationManager)
    ├── framework.py     # 메인 프레임워크 (UnifiedAgentFramework)
    └── exceptions.py    # 예외 클래스

📌 빠른 시작 가이드:

    예제 1: 간단한 챗봇
    ----------------------------------------
    >>> from unified_agent import UnifiedAgentFramework, Settings
    >>>
    >>> # 2026년 최신 모델 설정
    >>> Settings.DEFAULT_MODEL = "gpt-5.2"
    >>>
    >>> # 프레임워크 생성 및 빠른 챗
    >>> framework = UnifiedAgentFramework.create()
    >>> response = await framework.quick_chat("안녕하세요!")

    예제 2: 워크플로우 실행
    ----------------------------------------
    >>> framework.create_simple_workflow("my_workflow")
    >>> state = await framework.run("session-1", "my_workflow", "질문입니다")

    예제 3: 팀 기반 멀티에이전트
    ----------------------------------------
    >>> from unified_agent import TeamConfiguration, TeamAgent, AgentRole
    >>>
    >>> agent = TeamAgent(
    ...     name="researcher",
    ...     description="Research specialist",
    ...     role=AgentRole.ASSISTANT
    ... )
    >>> team_config = TeamConfiguration(
    ...     name="research_team",
    ...     agents=[agent],
    ...     orchestration_mode="supervisor"
    ... )
    >>> workflow = framework.create_team_workflow(team_config)

    예제 4: MCP 도구 통합 (Microsoft Agent Framework 패턴)
    ----------------------------------------
    >>> from unified_agent import MCPTool
    >>>
    >>> # MCP 도구 생성 (Microsoft Learn 문서 접근)
    >>> mcp_tool = MCPTool(
    ...     name="docs",
    ...     url="https://learn.microsoft.com/api/mcp"
    ... )
    >>>
    >>> # MCP 도구를 사용하는 에이전트 생성
    >>> agent = framework.create_skilled_agent(
    ...     name="assistant",
    ...     tools=[mcp_tool]
    ... )

🔧 환경 설정 (.env 파일):
    AZURE_OPENAI_API_KEY=your-api-key
    AZURE_OPENAI_ENDPOINT=https://your-resource.openai.azure.com/
    AZURE_OPENAI_DEPLOYMENT=gpt-5.2
    AZURE_OPENAI_API_VERSION=2025-12-01-preview

⚠️ 주의사항:
    - Python 3.11 이상 필요
    - 비동기 함수는 asyncio.run() 또는 await로 실행
    - Reasoning 모델(o3, o4-mini 등)은 temperature 미지원
    - MCP 도구 사용 시 Settings.ENABLE_MCP = True 필요

🔗 관련 문서:
    - Azure OpenAI: https://learn.microsoft.com/azure/ai-services/openai/
    - Semantic Kernel: https://github.com/microsoft/semantic-kernel
    - Microsoft Agent Framework: https://github.com/microsoft/agent-framework
    - MCP Protocol: https://modelcontextprotocol.io/

📝 라이선스: MIT
"""

__version__ = "3.1.0"
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
# 설정 (2026년 최신 모델 목록 포함)
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
# 유틸리티 (Adaptive Circuit Breaker 포함)
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
