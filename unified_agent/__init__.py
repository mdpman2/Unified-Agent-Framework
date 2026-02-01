#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Unified Agent Framework - Enterprise Edition v3.3

================================================================================
📋 프로젝트: 통합 AI 에이전트 프레임워크
📅 버전: 3.3.0 (2026년 1월 최신)
📦 Python: 3.11+
================================================================================

🌟 프레임워크 특징:
    ★ Azure OpenAI + Semantic Kernel 통합
    ★ Microsoft Agent Framework MCP 패턴 완전 통합
    ★ Agent Lightning 패턴 통합 (Tracer, AgentStore, Reward, Adapter, Hooks)
    ★ GPT-5.2, Claude Opus 4.5, Grok-4 등 2026년 최신 모델 지원 (54+)
    ★ Human-in-the-loop 승인 시스템
    ★ MCP (Model Context Protocol) 네이티브 지원
    ★ Skills 시스템 (Anthropic 패턴)
    ★ 이벤트 기반 아키텍처 (EventBus)
    ★ 체크포인트 및 롤백
    ★ Adaptive Circuit Breaker (2026년 개선)
    ★ 대용량 컨텍스트 지원 (최대 10M tokens)
    ★ 영속 메모리 시스템 (Clawdbot 스타일)
    ★ 세션 트리 분기 관리
    ★ 메모리 압축 전략 (Compaction)

📁 모듈 구조 (21개 모듈, 164개 공개 API):
    unified_agent/
    ├── __init__.py          # 이 파일 - 패키지 진입점
    ├── config.py            # 설정 관리 (Settings, FrameworkConfig) - frozenset 최적화
    ├── models.py            # 데이터 모델 (Enum, Pydantic)
    ├── utils.py             # 유틸리티 (CircuitBreaker, RAIValidator)
    ├── memory.py            # 메모리 시스템 (StateManager, Cache)
    ├── persistent_memory.py # 영속 메모리 (PersistentMemory, MemoryLayer)
    ├── compaction.py        # 메모리 압축 (CompactionEngine)
    ├── session_tree.py      # 세션 트리 (SessionTree, BranchInfo)
    ├── events.py            # 이벤트 시스템 (EventBus)
    ├── skills.py            # 스킬 시스템 (SkillManager)
    ├── tools.py             # 도구 (MCPTool, AIFunction)
    ├── agents.py            # 에이전트 (SimpleAgent, SupervisorAgent)
    ├── workflow.py          # 워크플로우 (Graph, Node)
    ├── orchestration.py     # 오케스트레이션 (OrchestrationManager)
    ├── framework.py         # 메인 프레임워크 (UnifiedAgentFramework)
    ├── exceptions.py        # 예외 클래스
    ├── tracer.py            # 분산 추적 (Tracer, SpanContext) - Agent Lightning
    ├── agent_store.py       # 에이전트 저장소 (AgentStore) - bisect 최적화
    ├── reward.py            # 보상 시스템 (RewardEngine) - Agent Lightning
    ├── adapter.py           # 모델 어댑터 (AdapterManager) - Agent Lightning
    └── hooks.py             # 라이프사이클 훅 (HookManager) - bisect 최적화

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

    예제 2: 영속 메모리 사용 (v3.2)
    ----------------------------------------
    >>> from unified_agent import PersistentMemory, MemoryConfig, MemoryLayer
    >>>
    >>> memory = PersistentMemory(MemoryConfig(storage_path="./memory"))
    >>> await memory.initialize()
    >>> await memory.store("핵심 정보", layer=MemoryLayer.CORE)
    >>> results = await memory.search("핵심", top_k=5)

    예제 3: Agent Lightning 추적 (v3.3)
    ----------------------------------------
    >>> from unified_agent import Tracer, TracerConfig, TracerBackend, span
    >>>
    >>> tracer = Tracer(TracerConfig(
    ...     service_name="my-agent",
    ...     backend=TracerBackend.CONSOLE
    ... ))
    >>> tracer.start()
    >>>
    >>> @span(name="process_request")
    >>> def process_request(data):
    ...     return {"result": "success"}

    예제 4: 에이전트 저장소 (v3.3)
    ----------------------------------------
    >>> from unified_agent import AgentStore, AgentStoreConfig, AgentEntry
    >>>
    >>> store = AgentStore(AgentStoreConfig(max_agents=100))
    >>> store.register(AgentEntry(
    ...     agent_id="researcher",
    ...     name="Research Agent",
    ...     capabilities={AgentCapability.REASONING}
    ... ))
    >>> agents = store.find_by_capability(AgentCapability.REASONING)

    예제 5: 보상 시스템 (v3.3)
    ----------------------------------------
    >>> from unified_agent import RewardEngine, RewardConfig, RewardSignal
    >>>
    >>> engine = RewardEngine(RewardConfig(discount_factor=0.99))
    >>> engine.begin_episode("ep-1")
    >>> engine.record(RewardSignal(reward=1.0, step=0))
    >>> summary = engine.end_episode()

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

__version__ = "3.3.0"
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
# 영속 메모리 시스템 (v3.2 NEW! - Clawdbot 스타일)
# ============================================================================
from .persistent_memory import (
    # 메모리 시스템
    PersistentMemory,
    MemoryConfig,
    MemoryLayer,
    # 검색 결과
    MemorySearchResult,
    MemoryChunk,
    # 도구
    MemorySearchTool,
    MemoryGetTool,
    MemoryWriteTool,
    # Bootstrap Files
    BootstrapFileManager,
    BootstrapFileType,
    # 인덱서
    MemoryIndexer,
)

# ============================================================================
# Compaction 시스템 (v3.2 NEW! - 컨텍스트 압축)
# ============================================================================
from .compaction import (
    # 설정
    CompactionConfig,
    PruningConfig,
    MemoryFlushConfig,
    # 핵심 클래스
    ContextCompactor,
    MemoryFlusher,
    CacheTTLPruner,
    # 매니저
    CompactionManager,
    # 모델
    CompactionSummary,
    PruningResult,
    ConversationTurn,
)

# ============================================================================
# 세션 트리 시스템 (v3.2 NEW! - Pi 스타일 브랜칭)
# ============================================================================
from .session_tree import (
    # 설정
    SessionTreeConfig,
    # 노드
    SessionNode,
    NodeType,
    # 트리
    SessionTree,
    BranchInfo,
    # 매니저
    SessionTreeManager,
    # 스냅샷
    SessionSnapshot,
)

# ============================================================================
# Tracer 시스템 (v3.3 NEW! - Agent Lightning 영감)
# ============================================================================
from .tracer import (
    # 스팬
    Span,
    SpanKind,
    SpanStatus,
    SpanContext,
    # 트레이서
    AgentTracer,
    SpanRecordingContext,
    # LLM/Tool 트레이싱
    LLMCallTracer,
    ToolCallTracer,
    # 전역 함수
    get_tracer,
    set_tracer,
    trace_context,
    current_span,
)

# ============================================================================
# Agent Store 시스템 (v3.3 NEW! - LightningStore 영감)
# ============================================================================
from .agent_store import (
    # 롤아웃/어템프트
    Rollout,
    Attempt,
    RolloutStatus,
    AttemptStatus,
    # 리소스
    NamedResource,
    ResourceBundle,
    # 스토어
    AgentStoreBase as AgentStore,
    InMemoryAgentStore,
    SQLiteAgentStore,
    # 전역 함수
    get_store,
    set_store,
)

# ============================================================================
# Reward 시스템 (v3.3 NEW! - 리워드 발행)
# ============================================================================
from .reward import (
    # 레코드
    RewardRecord,
    RewardDimension,
    RewardType,
    SpanCoreFields,
    # 매니저
    RewardManager,
    # 함수
    emit_reward,
    emit_annotation,
    is_reward_span,
    get_reward_value,
    find_reward_spans,
    find_final_reward,
    calculate_cumulative_reward,
    # 데코레이터
    reward,
    reward_async,
)

# ============================================================================
# Adapter 시스템 (v3.3 NEW! - 학습 데이터 변환)
# ============================================================================
from .adapter import (
    # 트리플렛
    Triplet,
    Transition,
    Trajectory,
    # 정책
    RewardMatchPolicy,
    # 어댑터
    Adapter,
    TraceAdapter,
    TracerTraceToTriplet,
    OpenAIMessagesAdapter,
    OpenAIMessage,
    # 트리
    TraceTree,
    # 헬퍼
    build_trajectory,
    export_triplets_to_jsonl,
    export_for_sft,
)

# ============================================================================
# Hook 시스템 (v3.3 NEW! - 라이프사이클 훅)
# ============================================================================
from .hooks import (
    # 우선순위
    HookPriority,
    # 이벤트
    HookEvent,
    # 등록
    HookRegistration,
    # 컨텍스트
    HookContext,
    HookResult,
    # 매니저
    HookManager,
    # 전역 함수
    get_hook_manager,
    set_hook_manager,
    on_trace_start,
    on_trace_end,
    on_span_start,
    on_span_end,
    on_llm_call,
    on_tool_call,
    on_reward,
    emit_hook,
    # 인터셉터
    HookInterceptor,
    BuiltinHooks,
    hooked_context,
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

    # Persistent Memory (v3.2 NEW!)
    "PersistentMemory",
    "MemoryConfig",
    "MemoryLayer",
    "MemorySearchResult",
    "MemoryChunk",
    "MemorySearchTool",
    "MemoryGetTool",
    "MemoryWriteTool",
    "BootstrapFileManager",
    "BootstrapFileType",
    "MemoryIndexer",

    # Compaction (v3.2 NEW!)
    "CompactionConfig",
    "PruningConfig",
    "MemoryFlushConfig",
    "ContextCompactor",
    "MemoryFlusher",
    "CacheTTLPruner",
    "CompactionManager",
    "CompactionSummary",
    "PruningResult",
    "ConversationTurn",

    # Session Tree (v3.2 NEW!)
    "SessionTreeConfig",
    "SessionNode",
    "NodeType",
    "SessionTree",
    "BranchInfo",
    "SessionTreeManager",
    "SessionSnapshot",

    # Tracer (v3.3 NEW!)
    "Span",
    "SpanKind",
    "SpanStatus",
    "SpanContext",
    "AgentTracer",
    "SpanRecordingContext",
    "LLMCallTracer",
    "ToolCallTracer",
    "get_tracer",
    "set_tracer",
    "trace_context",
    "current_span",

    # Agent Store (v3.3 NEW!)
    "Rollout",
    "Attempt",
    "RolloutStatus",
    "AttemptStatus",
    "NamedResource",
    "ResourceBundle",
    "AgentStore",
    "InMemoryAgentStore",
    "SQLiteAgentStore",
    "get_store",
    "set_store",

    # Reward (v3.3 NEW!)
    "RewardRecord",
    "RewardDimension",
    "RewardType",
    "SpanCoreFields",
    "RewardManager",
    "emit_reward",
    "emit_annotation",
    "is_reward_span",
    "get_reward_value",
    "find_reward_spans",
    "find_final_reward",
    "calculate_cumulative_reward",
    "reward",
    "reward_async",

    # Adapter (v3.3 NEW!)
    "Triplet",
    "Transition",
    "Trajectory",
    "RewardMatchPolicy",
    "Adapter",
    "TraceAdapter",
    "TracerTraceToTriplet",
    "OpenAIMessagesAdapter",
    "OpenAIMessage",
    "TraceTree",
    "build_trajectory",
    "export_triplets_to_jsonl",
    "export_for_sft",

    # Hooks (v3.3 NEW!)
    "HookPriority",
    "HookEvent",
    "HookRegistration",
    "HookContext",
    "HookResult",
    "HookManager",
    "get_hook_manager",
    "set_hook_manager",
    "on_trace_start",
    "on_trace_end",
    "on_span_start",
    "on_span_end",
    "on_llm_call",
    "on_tool_call",
    "on_reward",
    "emit_hook",
    "HookInterceptor",
    "BuiltinHooks",
    "hooked_context",

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
