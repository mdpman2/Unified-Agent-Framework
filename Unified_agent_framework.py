#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Unified Agent Framework - Enterprise Edition v3.0
Microsoft Multi-Agent-Custom-Automation-Engine 패턴 통합 + 완전 모듈화 아키텍처

============================================================================
📌 모듈 정보
============================================================================
버전: 3.0.0
작성자: Enterprise AI Team
라이선스: MIT

🆕 v3.0 주요 변경사항:
- 완전한 모듈화 아키텍처 (6,040줄 → 12개 모듈로 분리)
- 93% 코드 감소 (이 파일은 re-export 래퍼로 변환)
- 79개 테스트 케이스 완전 통과
- 순환 참조 없는 깔끔한 의존성 구조

이 파일은 unified_agent 패키지의 모든 공개 API를 re-export합니다.
실제 구현은 unified_agent/ 패키지의 개별 모듈에 있습니다.

패키지 구조:
    unified_agent/
    ├── __init__.py      # 패키지 진입점 (67개 공개 API export)
    ├── exceptions.py    # 예외 클래스 (FrameworkError, ConfigurationError 등)
    ├── config.py        # 설정 및 상수 (Settings, FrameworkConfig)
    ├── models.py        # 데이터 모델 (Enum, Pydantic, Dataclass)
    ├── utils.py         # 유틸리티 (StructuredLogger, CircuitBreaker, RAIValidator)
    ├── memory.py        # 메모리 시스템 (MemoryStore, CachedMemoryStore)
    ├── events.py        # 이벤트 시스템 (EventBus, EventType)
    ├── skills.py        # Skills 시스템 (Skill, SkillManager)
    ├── tools.py         # 도구 (AIFunction, MCPTool)
    ├── agents.py        # 에이전트 (SimpleAgent, RouterAgent, SupervisorAgent)
    ├── workflow.py      # 워크플로우 (Graph, Node)
    ├── orchestration.py # 오케스트레이션 (AgentFactory, OrchestrationManager)
    └── framework.py     # 메인 프레임워크 (UnifiedAgentFramework)

============================================================================
🚀 빠른 시작 가이드
============================================================================

1. 환경변수 설정 (.env 파일):
   AZURE_OPENAI_API_KEY=your-api-key
   AZURE_OPENAI_ENDPOINT=https://your-endpoint.openai.azure.com/
   AZURE_OPENAI_DEPLOYMENT=your-deployment-name

2. 가장 간단한 사용법:
   ```python
   import asyncio
   from unified_agent import quick_run  # 또는 from Unified_agent_framework import quick_run

   response = asyncio.run(quick_run("파이썬이란 무엇인가요?"))
   print(response)
   ```

3. 프레임워크 직접 사용:
   ```python
   import asyncio
   from unified_agent import UnifiedAgentFramework

   async def main():
       framework = UnifiedAgentFramework.create()
       response = await framework.quick_chat("안녕하세요!")
       print(response)
   asyncio.run(main())
   ```

4. Team 기반 멀티에이전트 (v3.0 NEW!):
   ```python
   from unified_agent import TeamConfiguration, TeamAgent, AgentFactory

   team_config = TeamConfiguration(
       name="research_team",
       agents=[
           TeamAgent(name="researcher", description="연구 담당"),
           TeamAgent(name="writer", description="문서 작성"),
       ]
   )

   factory = AgentFactory(framework)
   team = factory.create_team(team_config)
   ```

5. MPlan 구조화된 계획 시스템 (v3.0 NEW!):
   ```python
   from unified_agent import MPlan, PlanStep

   plan = MPlan(
       name="research_plan",
       steps=[
           PlanStep(index=0, description="데이터 수집", agent_name="researcher"),
           PlanStep(index=1, description="보고서 작성", agent_name="writer", depends_on=[0]),
       ]
   )
   print(f"진행률: {plan.get_progress() * 100}%")
   ```

6. WebSocket 스트리밍 (v3.0 NEW!):
   ```python
   from unified_agent import WebSocketMessageType, StreamingMessage

   msg = StreamingMessage(
       type=WebSocketMessageType.AGENT_RESPONSE,
       content="Hello!",
       agent_name="assistant"
   )
   ```

============================================================================
주요 기능
============================================================================
[핵심 기능]
1. MCP (Model Context Protocol) 서버 통합
2. Human-in-the-loop 승인 시스템
3. 스트리밍 응답 지원
4. 재시도 로직 및 회로 차단기 패턴
5. 비동기 이벤트 시스템 (Pub-Sub)
6. 향상된 메모리 관리 (LRU 캐시)
7. Supervisor Agent 패턴
8. 조건부 라우팅 및 루프 지원
9. 버전 관리 및 롤백
10. 상세 메트릭 및 성능 모니터링
11. Anthropic Skills 시스템

[v3.0 NEW! Microsoft Multi-Agent Engine 통합]
12. WebSocket 메시지 타입 및 실시간 스트리밍
13. Team/Agent Configuration 시스템
14. MPlan 구조화된 계획 시스템 (진행률 추적)
15. ProxyAgent - 사용자 명확화 요청
16. RAI (Responsible AI) 검증 시스템
17. AgentFactory - JSON 기반 에이전트 동적 생성
18. OrchestrationManager - 팀 기반 오케스트레이션

[v3.0 NEW! 모듈화 아키텍처]
19. 12개 독립 모듈로 분리
20. 93% 코드 감소 (유지보수성 향상)
21. 79개 테스트 케이스 통과
22. 순환 참조 없는 의존성 구조

============================================================================
필요 패키지
============================================================================
pip install semantic-kernel python-dotenv opentelemetry-api opentelemetry-sdk pydantic pyyaml
"""

# ============================================================================
# 모듈 메타데이터
# ============================================================================
__version__ = "3.0.0"
__author__ = "Enterprise AI Team"

# ============================================================================
# unified_agent 패키지에서 모든 공개 API re-export
# v3.0: 12개 모듈에서 67개 공개 심볼 export
# ============================================================================
from unified_agent import (
    # ─────────────────────────────────────────────────────────────────────────
    # Exceptions (unified_agent/exceptions.py)
    # 프레임워크 전용 예외 클래스
    # ─────────────────────────────────────────────────────────────────────────
    FrameworkError,
    ConfigurationError,
    WorkflowError,
    AgentError,
    ApprovalError,
    RAIValidationError,

    # ─────────────────────────────────────────────────────────────────────────
    # Configuration (unified_agent/config.py)
    # 중앙 설정 관리 및 상수
    # ─────────────────────────────────────────────────────────────────────────
    Settings,
    FrameworkConfig,
    DEFAULT_LLM_MODEL,
    DEFAULT_API_VERSION,
    SUPPORTED_MODELS,
    O_SERIES_MODELS,
    MODELS_WITHOUT_TEMPERATURE,
    supports_temperature,
    create_execution_settings,

    # ─────────────────────────────────────────────────────────────────────────
    # Models - Enums (unified_agent/models.py)
    # 상태 및 역할 정의 열거형
    # ─────────────────────────────────────────────────────────────────────────
    AgentRole,
    ExecutionStatus,
    ApprovalStatus,
    WebSocketMessageType,
    PlanStepStatus,
    RAICategory,

    # ─────────────────────────────────────────────────────────────────────────
    # Models - Data Classes (unified_agent/models.py)
    # Pydantic/Dataclass 기반 데이터 모델
    # ─────────────────────────────────────────────────────────────────────────
    Message,
    AgentState,
    NodeResult,
    StreamingMessage,
    TeamAgent,
    TeamConfiguration,
    PlanStep,
    MPlan,
    RAIValidationResult,

    # ─────────────────────────────────────────────────────────────────────────
    # Utilities (unified_agent/utils.py)
    # 로깅, 회로차단기, RAI 검증 등 유틸리티
    # ─────────────────────────────────────────────────────────────────────────
    StructuredLogger,
    retry_with_backoff,
    CircuitBreaker,
    setup_telemetry,
    RAIValidator,

    # ─────────────────────────────────────────────────────────────────────────
    # Memory System (unified_agent/memory.py)
    # 대화 기록 및 상태 관리 (LRU 캐시, Hook Provider)
    # ─────────────────────────────────────────────────────────────────────────
    MemoryStore,
    CachedMemoryStore,
    ConversationMessage,
    MemoryHookProvider,
    MemorySessionManager,
    StateManager,

    # ─────────────────────────────────────────────────────────────────────────
    # Event System (unified_agent/events.py)
    # 비동기 Pub-Sub 이벤트 버스
    # ─────────────────────────────────────────────────────────────────────────
    EventType,
    AgentEvent,
    EventBus,

    # ─────────────────────────────────────────────────────────────────────────
    # Skills System (unified_agent/skills.py)
    # Anthropic Skills 패턴 기반 모듈화 전문 지식
    # ─────────────────────────────────────────────────────────────────────────
    SkillResource,
    Skill,
    SkillManager,

    # ─────────────────────────────────────────────────────────────────────────
    # Tools (unified_agent/tools.py)
    # AI Function, MCP Tool, 승인 필요 함수
    # ─────────────────────────────────────────────────────────────────────────
    AIFunction,
    ApprovalRequiredAIFunction,
    MockMCPClient,
    MCPTool,

    # ─────────────────────────────────────────────────────────────────────────
    # Agents (unified_agent/agents.py)
    # 다양한 에이전트 클래스 (Simple, Router, Supervisor, Proxy)
    # ─────────────────────────────────────────────────────────────────────────
    Agent,
    SimpleAgent,
    ApprovalAgent,
    RouterAgent,
    ProxyAgent,
    InvestigationPlan,
    SupervisorAgent,

    # ─────────────────────────────────────────────────────────────────────────
    # Workflow (unified_agent/workflow.py)
    # 상태 기반 그래프 및 조건부 라우팅
    # ─────────────────────────────────────────────────────────────────────────
    Node,
    Graph,

    # ─────────────────────────────────────────────────────────────────────────
    # Orchestration (unified_agent/orchestration.py)
    # v3.0 NEW! 팀 기반 오케스트레이션 및 에이전트 팩토리
    # ─────────────────────────────────────────────────────────────────────────
    AgentFactory,
    OrchestrationManager,

    # ─────────────────────────────────────────────────────────────────────────
    # Framework Main (unified_agent/framework.py)
    # 핵심 프레임워크 클래스 및 헬퍼 함수
    # ─────────────────────────────────────────────────────────────────────────
    UnifiedAgentFramework,
    quick_run,
    create_framework,
)

# ============================================================================
# Public API 정의
# v3.0: 67개 공개 심볼
# ============================================================================
__all__ = [
    # 버전 정보
    "__version__",
    "__author__",

    # 예외 클래스 (unified_agent/exceptions.py)
    "FrameworkError",
    "ConfigurationError",
    "WorkflowError",
    "AgentError",
    "ApprovalError",
    "RAIValidationError",

    # 설정 클래스 (unified_agent/config.py)
    "Settings",
    "FrameworkConfig",
    "DEFAULT_LLM_MODEL",
    "DEFAULT_API_VERSION",
    "SUPPORTED_MODELS",
    "O_SERIES_MODELS",
    "MODELS_WITHOUT_TEMPERATURE",
    "supports_temperature",
    "create_execution_settings",

    # 데이터 모델 - Enums (unified_agent/models.py)
    "AgentRole",
    "ExecutionStatus",
    "ApprovalStatus",
    "WebSocketMessageType",
    "PlanStepStatus",
    "RAICategory",

    # 데이터 모델 - Classes (unified_agent/models.py)
    "Message",
    "AgentState",
    "NodeResult",
    "StreamingMessage",
    "TeamAgent",
    "TeamConfiguration",
    "PlanStep",
    "MPlan",
    "RAIValidationResult",

    # 유틸리티 (unified_agent/utils.py)
    "StructuredLogger",
    "retry_with_backoff",
    "CircuitBreaker",
    "setup_telemetry",
    "RAIValidator",

    # 메모리/상태 관리 (unified_agent/memory.py)
    "MemoryStore",
    "CachedMemoryStore",
    "ConversationMessage",
    "MemoryHookProvider",
    "MemorySessionManager",
    "StateManager",

    # 이벤트 시스템 (unified_agent/events.py)
    "EventType",
    "AgentEvent",
    "EventBus",

    # 스킬 시스템 (unified_agent/skills.py)
    "SkillResource",
    "Skill",
    "SkillManager",

    # 도구 (unified_agent/tools.py)
    "AIFunction",
    "ApprovalRequiredAIFunction",
    "MockMCPClient",
    "MCPTool",

    # 에이전트 클래스 (unified_agent/agents.py)
    "Agent",
    "SimpleAgent",
    "ApprovalAgent",
    "RouterAgent",
    "ProxyAgent",
    "InvestigationPlan",
    "SupervisorAgent",

    # 워크플로우 (unified_agent/workflow.py)
    "Node",
    "Graph",

    # 오케스트레이션 (unified_agent/orchestration.py) - v3.0 NEW!
    "AgentFactory",
    "OrchestrationManager",

    # 핵심 프레임워크 (unified_agent/framework.py)
    "UnifiedAgentFramework",
    "quick_run",
    "create_framework",
]


# ============================================================================
# 하위 호환성을 위한 별칭
# v3.0: TeamService → OrchestrationManager로 통합됨 (Deprecated)
# ============================================================================

# TeamService는 OrchestrationManager로 통합됨
TeamService = OrchestrationManager


# ============================================================================
# 모듈 로드 시 초기화
# v3.0: UTF-8 인코딩 자동 설정 (Windows 환경 지원)
# ============================================================================

def _init_module():
    """
    모듈 초기화

    - UTF-8 인코딩 설정 (Windows 환경)
    - 콘솔 출력 한글 깨짐 방지
    """
    import sys
    # UTF-8 인코딩 설정 (Windows 환경)
    if sys.stdout and hasattr(sys.stdout, 'reconfigure'):
        try:
            sys.stdout.reconfigure(encoding='utf-8')
        except Exception:
            pass
    if sys.stderr and hasattr(sys.stderr, 'reconfigure'):
        try:
            sys.stderr.reconfigure(encoding='utf-8')
        except Exception:
            pass


_init_module()
