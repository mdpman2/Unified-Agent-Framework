#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Unified Agent Framework - Enterprise Edition v4.1

================================================================================
📋 프로젝트: 통합 AI 에이전트 프레임워크
📅 버전: 4.1.0 (2026년 2월 14일 최신)
📦 Python: 3.11+
👤 테스트: 28개 시나리오 100% 통과
================================================================================

🌟 프레임워크 특징:
    ★ Azure OpenAI + Semantic Kernel 통합
    ★ Microsoft Agent Framework MCP 패턴 완전 통합
    ★ Agent Lightning 패턴 통합 (Tracer, AgentStore, Reward, Adapter, Hooks)
    ★ GPT-5.2, Claude Opus 4.6, Grok-4 등 2026년 최신 모델 지원 — Model-Agnostic
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
    ★ [v3.4] Prompt Caching - 비용 절감
    ★ [v3.4] Durable Agent - 장기 워크플로우
    ★ [v3.4] Concurrent Orchestration - 병렬 실행
    ★ [v3.4] AgentTool Pattern - 에이전트 중첩
    ★ [v3.4] Extended Thinking - Reasoning 추적
    ★ [v3.4] MCP Workbench - 다중 MCP 관리
    ★ [v3.5] Security Guardrails - 프롬프트 보안
    ★ [v3.5] Structured Output - GPT-5.2 구조화된 출력
    ★ [v3.5] Evaluation - PDCA + LLM-as-Judge 평가
    ★ [v4.0 NEW!] Universal Agent Bridge - 16개 프레임워크 통합
    ★ [v4.0 NEW!] Responses API - Stateful 대화 관리
    ★ [v4.0 NEW!] Sora 2/2 Pro - 비디오 생성
    ★ [v4.0 NEW!] GPT Image 1.5 - 이미지 생성
    ★ [v4.0 NEW!] 오픈 웨이트 모델 - gpt-oss-120b/20b
    ★ [v4.0 NEW!] 7개 프레임워크 브릿지 (OpenAI, Google, CrewAI, A2A, MS, AG2, SK)
    ★ [v4.0 NEW!] A2A + MCP 이중 프로토콜
    ★ [v4.1 NEW!] Agent Identity — Microsoft Entra ID 에이전트 인증/RBAC
    ★ [v4.1 NEW!] Browser Automation & CUA — Playwright + Computer Use Agent
    ★ [v4.1 NEW!] Deep Research — o3-deep-research 다단계 자율 연구
    ★ [v4.1 NEW!] Observability — OpenTelemetry 네이티브 분산 추적/메트릭
    ★ [v4.1 NEW!] Middleware Pipeline — 요청/응답 미들웨어 체인
    ★ [v4.1 NEW!] Agent Triggers — 이벤트/스케줄/웹훅 기반 자동 호출

📁 모듈 구조 (49개 모듈, 380개+ 공개 API):
    unified_agent/
    ├── __init__.py          # 이 파일 - 패키지 진입점
    ├── interfaces.py        # 핵심 인터페이스 (IFramework, IOrchestrator)
    ├── config.py            # 설정 관리 (Settings, FrameworkConfig) - frozenset 최적화
    ├── models.py            # 데이터 모델 (Enum, Pydantic)
    ├── utils.py             # 유틸리티 (CircuitBreaker, RAIValidator)
    ├── memory.py            # 메모리 시스템 (StateManager, Cache)
    ├── persistent_memory.py # 영속 메모리 (PersistentMemory, MemoryLayer)
    ├── compaction.py        # 메모리 압축 (CompactionManager)
    ├── session_tree.py      # 세션 트리 (SessionTree, BranchInfo)
    ├── events.py            # 이벤트 시스템 (EventBus)
    ├── skills.py            # 스킬 시스템 (SkillManager)
    ├── tools.py             # 도구 (MCPTool, AIFunction)
    ├── agents.py            # 에이전트 (SimpleAgent, SupervisorAgent)
    ├── workflow.py          # 워크플로우 (Graph, Node)
    ├── orchestration.py     # 오케스트레이션 (OrchestrationManager)
    ├── framework.py         # 메인 프레임워크 (UnifiedAgentFramework)
    ├── exceptions.py        # 예외 클래스
    ├── extensions.py        # [v3.4 NEW!] 확장 허브 (ExtensionsHub)
    ├── tracer.py            # 분산 추적 (AgentTracer, SpanContext) - Agent Lightning
    ├── agent_store.py       # 에이전트 저장소 (AgentStore) - bisect 최적화
    ├── reward.py            # 보상 시스템 (RewardEngine) - Agent Lightning
    ├── adapter.py           # 모델 어댑터 (AdapterManager) - Agent Lightning
    ├── hooks.py             # 라이프사이클 훅 (HookManager) - bisect 최적화
    ├── prompt_cache.py      # [v3.4 NEW!] Prompt Caching
    ├── durable_agent.py     # [v3.4 NEW!] Durable Agent 워크플로우
    ├── concurrent.py        # [v3.4 NEW!] Fan-out/Fan-in 병렬 실행
    ├── agent_tool.py        # [v3.4 NEW!] AgentTool 패턴
    ├── extended_thinking.py # [v3.4 NEW!] Extended Thinking
    ├── mcp_workbench.py     # [v3.4 NEW!] 다중 MCP 서버 관리
    ├── security_guardrails.py # [v3.5] 보안 가드레일 (PromptShield, PIIDetector)
    ├── structured_output.py   # [v3.5] 구조화된 출력 (OutputSchema)
    ├── evaluation.py          # [v3.5] PDCA 평가 (LLMJudge, CheckActIterator)
    ├── responses_api.py       # [v4.0 NEW!] Responses API (Stateful 대화)
    ├── video_generation.py    # [v4.0 NEW!] Sora 2/2 Pro 비디오 생성
    ├── image_generation.py    # [v4.0 NEW!] GPT Image 1.5 이미지 생성
    ├── open_weight.py         # [v4.0 NEW!] 오픈 웨이트 모델 (gpt-oss)
    ├── universal_bridge.py    # [v4.0 NEW!] Universal Agent Bridge (16개 통합)
    ├── openai_agents_bridge.py  # [v4.0 NEW!] OpenAI Agents SDK 브릿지
    ├── google_adk_bridge.py     # [v4.0 NEW!] Google ADK 브릿지
    ├── crewai_bridge.py         # [v4.0 NEW!] CrewAI 브릿지
    ├── a2a_bridge.py            # [v4.0 NEW!] A2A Protocol 브릿지
    ├── ms_agent_bridge.py       # [v4.0 NEW!] MS Agent Framework 브릿지
    ├── ag2_bridge.py            # [v4.0 NEW!] AG2 AgentOS 브릿지
    ├── sk_agent_bridge.py       # [v4.0 NEW!] SK Agent Framework 브릿지
    ├── agent_identity.py        # [v4.1 NEW!] Microsoft Entra Agent Identity/RBAC
    ├── browser_use.py           # [v4.1 NEW!] Playwright + Computer Use Agent
    ├── deep_research.py         # [v4.1 NEW!] o3-deep-research 다단계 연구
    ├── observability.py         # [v4.1 NEW!] OpenTelemetry 분산 추적/메트릭
    ├── middleware.py            # [v4.1 NEW!] 요청/응답 미들웨어 파이프라인
    └── agent_triggers.py        # [v4.1 NEW!] 이벤트 기반 에이전트 자동 호출

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
    >>> memory = PersistentMemory(
    ...     agent_id="my-agent",
    ...     config=MemoryConfig(workspace_dir="./memory")
    ... )
    >>> await memory.initialize()
    >>> await memory.add_long_term_memory("핵심 정보")
    >>> results = await memory.search("핵심", max_results=5)

    예제 3: Agent Lightning 추적 (v3.3)
    ----------------------------------------
    >>> from unified_agent import AgentTracer, SpanKind, SpanStatus
    >>>
    >>> tracer = AgentTracer(name="my-agent")
    >>> await tracer.initialize()
    >>>
    >>> async with tracer.trace_context("task-001", "attempt-1"):
    ...     with tracer.span("llm_call", SpanKind.LLM) as ctx:
    ...         ctx.set_attribute("tokens", 1500)
    ...         # ... LLM 호출 ...

    예제 4: 에이전트 저장소 (v3.3)
    ----------------------------------------
    >>> from unified_agent import AgentStore, Rollout, RolloutStatus
    >>>
    >>> store = AgentStore()
    >>> store.register(Rollout(
    ...     agent_id="researcher",
    ...     name="Research Agent",
    ...     status=RolloutStatus.ACTIVE
    ... ))
    >>> agents = store.list_rollouts()

    예제 5: 보상 시스템 (v3.3)
    ----------------------------------------
    >>> from unified_agent import RewardManager, RewardDimension, RewardType
    >>>
    >>> manager = RewardManager()
    >>> manager.emit_reward(RewardDimension(
    ...     reward=1.0, reward_type=RewardType.INTRINSIC, step=0
    ... ))

    예제 6: Prompt Caching 사용 (v3.4 NEW!)
    ----------------------------------------
    >>> from unified_agent import PromptCache, CacheConfig
    >>>
    >>> cache = PromptCache(CacheConfig(
    ...     max_memory_mb=100,
    ...     ttl_seconds=3600
    ... ))
    >>> # 캐시 히트로 비용 절감
    >>> result, cached = await cache.get_or_call(
    ...     model="gpt-5.2",
    ...     messages=messages,
    ...     call_fn=llm_call_fn
    ... )

    예제 7: Durable Agent 워크플로우 (v3.4 NEW!)
    ----------------------------------------
    >>> from unified_agent import DurableAgent, DurableConfig, workflow, activity
    >>>
    >>> @activity()
    >>> async def send_email(ctx, recipient, content):
    ...     return {"sent": True}
    >>>
    >>> @workflow()
    >>> async def approval_workflow(ctx, data):
    ...     result = await ctx.call_activity(send_email, data["to"], data["msg"])
    ...     return result

    예제 8: Concurrent Orchestration (v3.4 NEW!)
    ----------------------------------------
    >>> from unified_agent import ConcurrentOrchestrator, FanOutConfig
    >>>
    >>> orchestrator = ConcurrentOrchestrator([agent1, agent2, agent3])
    >>> results = await orchestrator.fan_out(
    ...     task="시장 분석",
    ...     aggregation="majority"
    ... )

    예제 9: AgentTool 패턴 (v3.4 NEW!)
    ----------------------------------------
    >>> from unified_agent import AgentTool, AgentToolRegistry
    >>>
    >>> registry = AgentToolRegistry()
    >>> registry.register(AgentTool.from_agent(
    ...     research_agent,
    ...     name="research_expert",
    ...     description="심층 연구 수행"
    ... ))

    예제 10: Extended Thinking (v3.4 NEW!)
    ----------------------------------------
    >>> from unified_agent import ThinkingTracker, ThinkingConfig, ThinkingMode
    >>>
    >>> tracker = ThinkingTracker(ThinkingConfig(mode=ThinkingMode.FULL))
    >>> with tracker.track_thinking("task-1") as thinking:
    ...     thinking.add_step(ThinkingStepType.OBSERVATION, "입력 분석...")
    ...     thinking.add_step(ThinkingStepType.REASONING, "추론 수행...")

    예제 11: MCP Workbench (v3.4 NEW!)
    ----------------------------------------
    >>> from unified_agent import McpWorkbench, McpServerConfig
    >>>
    >>> workbench = McpWorkbench()
    >>> workbench.register_server(McpServerConfig(
    ...     name="filesystem",
    ...     uri="stdio://mcp-server-filesystem",
    ...     capabilities=["read_file", "write_file"]
    ... ))
    >>> await workbench.connect_all()
    >>> result = await workbench.call_tool("read_file", path="/etc/hosts")

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

from __future__ import annotations

import importlib
import sys
from typing import Final

__version__: Final = "4.1.0"
__author__: Final = "Enterprise AI Team"

# ============================================================================
# Lazy Import Infrastructure (PEP 562)
# ============================================================================
# 모든 모듈을 즉시 로드하지 않고, 실제 접근 시에만 로드합니다.
# - import unified_agent           → 즉시 완료 (~50ms)
# - unified_agent.AgentTracer      → tracer 모듈만 로드
# - from unified_agent import X    → X가 속한 모듈만 로드
# ============================================================================

_MODULE_EXPORTS: dict[str, list[str]] = {
    ".exceptions": [
        "FrameworkError", "ConfigurationError", "WorkflowError",
        "AgentError", "ApprovalError", "RAIValidationError",
    ],
    ".config": [
        "Settings", "FrameworkConfig", "DEFAULT_LLM_MODEL",
        "DEFAULT_API_VERSION", "SUPPORTED_MODELS", "O_SERIES_MODELS",
        "MODELS_WITHOUT_TEMPERATURE", "supports_temperature",
        "create_execution_settings",
    ],
    ".models": [
        "AgentRole", "ExecutionStatus", "ApprovalStatus",
        "WebSocketMessageType", "PlanStepStatus", "RAICategory",
        "Message", "AgentState", "NodeResult", "StreamingMessage",
        "TeamAgent", "TeamConfiguration", "PlanStep", "MPlan",
        "RAIValidationResult",
    ],
    ".utils": [
        "StructuredLogger", "retry_with_backoff", "CircuitBreaker",
        "setup_telemetry", "RAIValidator",
    ],
    ".memory": [
        "MemoryStore", "CachedMemoryStore", "ConversationMessage",
        "MemoryHookProvider", "MemorySessionManager", "StateManager",
    ],
    ".persistent_memory": [
        "PersistentMemory", "MemoryConfig", "MemoryLayer",
        "MemorySearchResult", "MemoryChunk", "MemorySearchTool",
        "MemoryGetTool", "MemoryWriteTool", "BootstrapFileManager",
        "BootstrapFileType", "MemoryIndexer",
    ],
    ".compaction": [
        "CompactionConfig", "PruningConfig", "MemoryFlushConfig",
        "ContextCompactor", "MemoryFlusher", "CacheTTLPruner",
        "CompactionManager", "CompactionSummary", "PruningResult",
        "ConversationTurn",
    ],
    ".session_tree": [
        "SessionTreeConfig", "SessionNode", "NodeType", "SessionTree",
        "BranchInfo", "SessionTreeManager", "SessionSnapshot",
    ],
    ".tracer": [
        "Span", "SpanKind", "SpanStatus", "SpanContext", "AgentTracer",
        "SpanRecordingContext", "LLMCallTracer", "ToolCallTracer",
        "get_tracer", "set_tracer", "trace_context", "current_span",
    ],
    ".agent_store": [
        "Rollout", "Attempt", "RolloutStatus", "AttemptStatus",
        "NamedResource", "ResourceBundle", "InMemoryAgentStore",
        "SQLiteAgentStore", "get_store", "set_store",
    ],
    ".reward": [
        "RewardRecord", "RewardDimension", "RewardType", "SpanCoreFields",
        "RewardManager", "emit_reward", "emit_annotation", "is_reward_span",
        "get_reward_value", "find_reward_spans", "find_final_reward",
        "calculate_cumulative_reward", "reward", "reward_async",
    ],
    ".adapter": [
        "Triplet", "Transition", "Trajectory", "RewardMatchPolicy",
        "Adapter", "TraceAdapter", "TracerTraceToTriplet",
        "OpenAIMessagesAdapter", "OpenAIMessage", "TraceTree",
        "build_trajectory", "export_triplets_to_jsonl", "export_for_sft",
    ],
    ".hooks": [
        "HookPriority", "HookEvent", "HookRegistration", "HookContext",
        "HookResult", "HookManager", "get_hook_manager", "set_hook_manager",
        "on_trace_start", "on_trace_end", "on_span_start", "on_span_end",
        "on_llm_call", "on_tool_call", "on_reward", "emit_hook",
        "HookInterceptor", "BuiltinHooks", "hooked_context",
    ],
    ".events": ["EventType", "AgentEvent", "EventBus"],
    ".skills": ["SkillResource", "Skill", "SkillManager"],
    ".tools": [
        "AIFunction", "ApprovalRequiredAIFunction", "MockMCPClient",
        "MCPTool",
    ],
    ".agents": [
        "Agent", "SimpleAgent", "ApprovalAgent", "RouterAgent",
        "ProxyAgent", "InvestigationPlan", "SupervisorAgent",
    ],
    ".workflow": ["Node", "Graph"],
    ".orchestration": ["AgentFactory", "OrchestrationManager"],
    ".interfaces": [
        "IFramework", "IOrchestrator", "IMemoryProvider",
        "ICacheProvider", "IThinkingProvider",
    ],
    ".framework": ["UnifiedAgentFramework", "quick_run", "create_framework"],
    ".extensions": ["Extensions", "ExtensionsConfig"],
    ".prompt_cache": [
        "CacheConfig", "CacheEntry", "CacheStats", "CacheBackend",
        "MemoryCacheBackend", "DiskCacheBackend", "TwoLevelCacheBackend",
        "PromptCache",
    ],
    ".durable_agent": [
        "DurableConfig", "WorkflowState", "WorkflowStatus", "CheckpointData",
        "ActivityResult", "DurableContext", "DurableAgent",
        "DurableOrchestrator", "WorkflowStore", "FileWorkflowStore",
        "activity", "workflow",
    ],
    ".concurrent": [
        "FanOutConfig", "AggregationStrategy", "ParallelResult",
        "AggregatedResult", "ConcurrentOrchestrator", "ResultAggregator",
        "MapReducePattern", "ScatterGatherPattern", "ConditionalFanOut",
    ],
    ".agent_tool": [
        "AgentToolConfig", "DelegationPolicy", "DelegationResult",
        "AgentTool", "AgentToolRegistry", "DelegationManager",
        "AgentChain", "ChainStep",
    ],
    ".extended_thinking": [
        "ThinkingConfig", "ThinkingMode", "ThinkingStepType", "ThinkingStep",
        "ThinkingChain", "ThinkingContext", "ThinkingTracker",
        "ThinkingAnalyzer", "ThinkingMetrics", "ThinkingStore",
    ],
    ".mcp_workbench": [
        "McpServerConfig", "McpWorkbenchConfig", "ConnectionState",
        "LoadBalanceStrategy", "McpServerConnection", "McpServerInfo",
        "McpWorkbench", "McpToolRegistry", "McpRouter", "CapabilityRouter",
        "RoundRobinRouter", "HealthChecker", "HealthStatus",
    ],
    ".security_guardrails": [
        "ThreatLevel", "AttackType", "PIIType", "ValidationStage",
        "SecurityConfig", "ShieldResult", "JailbreakResult", "PIIResult",
        "GroundednessResult", "ValidationResult", "AuditLogEntry",
        "PromptShield", "JailbreakDetector", "OutputValidator",
        "GroundednessChecker", "PIIDetector", "SecurityOrchestrator",
        "SecurityAuditLogger",
    ],
    ".structured_output": [
        "OutputSchema", "StructuredOutputConfig", "StructuredOutputParser",
        "StructuredOutputValidator", "StructuredOutputClient",
        "structured_output", "pydantic_to_schema",
    ],
    ".evaluation": [
        "PDCAPhase", "EvaluationDimension", "QualityLevel", "GapSeverity",
        "EvaluationConfig", "JudgeConfig", "IterationConfig",
        "EvaluationResult", "JudgeVerdict", "GapAnalysisResult",
        "IterationResult", "BenchmarkResult", "QualityReport",
        "PDCAEvaluator", "LLMJudge", "CheckActIterator", "GapAnalyzer",
        "AgentBenchmark", "QualityMetrics", "Evaluator", "Optimizer",
    ],
    ".responses_api": [
        "ResponsesClient", "ConversationState", "BackgroundMode",
        "ResponseConfig", "ResponseObject", "ResponseStatus", "ToolType",
    ],
    ".video_generation": [
        "VideoGenerator", "Sora2Client", "VideoConfig", "VideoResult",
        "VideoModel", "VideoStatus",
    ],
    ".image_generation": [
        "ImageModel", "ImageGenerator", "GPTImage1_5Client",
        "ImageConfig", "ImageResult",
    ],
    ".open_weight": [
        "OpenWeightAdapter", "OSSModelConfig", "OpenWeightRegistry",
        "OSSLicense", "OSSModelInfo",
    ],
    ".universal_bridge": ["UniversalAgentBridge", "BridgeProtocol"],
    ".openai_agents_bridge": [
        "OpenAIAgentsBridge", "AgentHandoff", "SessionBackend",
    ],
    ".google_adk_bridge": ["GoogleADKBridge"],
    ".crewai_bridge": ["CrewAIBridge"],
    ".a2a_bridge": ["A2ABridge", "AgentCard", "TaskMode"],
    ".ms_agent_bridge": ["MicrosoftAgentBridge"],
    ".ag2_bridge": ["AG2Bridge"],
    ".sk_agent_bridge": ["SemanticKernelAgentBridge"],
    ".agent_identity": [
        "AgentIdentity", "AgentCredential", "AgentRBACManager",
        "AgentIdentityProvider", "AgentDelegation", "IdentityRegistry",
        "ScopedPermission", "PermissionScope", "IdentityStatus",
        "AuthMethod", "IdentityAuditEntry",
    ],
    ".browser_use": [
        "BrowserAutomation", "ComputerUseAgent", "BrowserSession",
        "SafetyChecker", "ActionRecorder", "BrowserConfig", "BrowserAction",
        "ActionResult", "CUAConfig", "CUAResult", "ScreenCapture",
        "ActionType", "BrowserStatus", "CUAEnvironment",
    ],
    ".deep_research": [
        "DeepResearchAgent", "SourceCollector", "SynthesisEngine",
        "CitationManager", "ResearchConfig", "ResearchPlan", "ResearchStep",
        "SourceDocument", "Citation", "ResearchResult",
        "ResearchCheckpoint", "ResearchPhase", "SourceType",
        "ResearchStatus", "SearchProvider",
    ],
    ".observability": [
        "ObservabilityPipeline", "AgentTelemetry", "MetricsCollector",
        "TraceExporter", "AlertManager", "AgentDashboard",
        "ObservabilityConfig", "TelemetrySpan", "MetricRecord",
        "AlertRule", "AlertEvent", "DashboardData", "MetricType",
        "ExportTarget", "TelemetryLevel",
    ],
    ".middleware": [
        "MiddlewareManager", "MiddlewareChain", "BaseMiddleware",
        "RequestMiddleware", "ResponseMiddleware", "LoggingMiddleware",
        "AuthMiddleware", "RateLimitMiddleware", "RetryMiddleware",
        "ContentFilterMiddleware", "CacheMiddleware", "MiddlewareConfig",
        "MiddlewareContext", "MiddlewareResult", "MiddlewareMetrics",
        "MiddlewarePhase", "MiddlewarePriority", "MiddlewareStatus",
    ],
    ".agent_triggers": [
        "TriggerManager", "EventTrigger", "ScheduleTrigger",
        "WebhookTrigger", "QueueTrigger", "FileChangeTrigger",
        "AgentCompletionTrigger", "BaseTrigger", "TriggerConfig",
        "TriggerEvent", "TriggerCondition", "TriggerResult",
        "TriggerMetrics", "TriggerType", "TriggerStatus", "TriggerPriority",
    ],
}

# ============================================================================
# Aliased Imports — 원본 이름과 공개 이름이 다른 심볼
# ============================================================================
_ALIASES: dict[str, tuple[str, str]] = {
    "AgentStore": (".agent_store", "AgentStoreBase"),
    "AgentIdentityRole": (".agent_identity", "AgentRole"),
    "StructuredValidationError": (".structured_output", "ValidationError"),
}

# ============================================================================
# Reverse Lookup Table (자동 생성)
# ============================================================================
_SYMBOL_TO_MODULE: dict[str, str] = {}
for _mod, _syms in _MODULE_EXPORTS.items():
    for _sym in _syms:
        _SYMBOL_TO_MODULE[_sym] = _mod
del _mod, _syms, _sym  # cleanup namespace


def __getattr__(name: str):
    """PEP 562 lazy module attribute access.

    첫 접근 시 해당 서브모듈만 import하고, 같은 모듈의 모든 심볼을
    globals()에 캐싱하여 이후 접근은 dict lookup만으로 처리합니다.
    """
    _this = sys.modules[__name__]

    # 1) Aliased imports (AgentStore, AgentIdentityRole, StructuredValidationError)
    if name in _ALIASES:
        mod_path, original_name = _ALIASES[name]
        module = importlib.import_module(mod_path, __package__)
        value = getattr(module, original_name)
        setattr(_this, name, value)
        return value

    # 2) Regular lazy imports — load entire submodule on first access
    if name in _SYMBOL_TO_MODULE:
        mod_path = _SYMBOL_TO_MODULE[name]
        module = importlib.import_module(mod_path, __package__)
        # Cache ALL symbols from this module at once (batch)
        for sym in _MODULE_EXPORTS[mod_path]:
            setattr(_this, sym, getattr(module, sym))
        return getattr(_this, name)

    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__() -> list[str]:
    """IDE autocomplete 지원 — 모든 공개 API 심볼을 노출합니다."""
    return list(__all__)


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

    # Interfaces (v3.4 NEW! - 순환 의존 해소)
    "IFramework",
    "IOrchestrator",
    "IMemoryProvider",
    "ICacheProvider",
    "IThinkingProvider",

    # Framework
    "UnifiedAgentFramework",
    "quick_run",
    "create_framework",

    # Extensions Hub (v3.4 NEW!)
    "Extensions",
    "ExtensionsConfig",

    # Prompt Cache (v3.4 NEW!)
    "CacheConfig",
    "CacheEntry",
    "CacheStats",
    "CacheBackend",
    "MemoryCacheBackend",
    "DiskCacheBackend",
    "TwoLevelCacheBackend",
    "PromptCache",

    # Durable Agent (v3.4 NEW!)
    "DurableConfig",
    "WorkflowState",
    "WorkflowStatus",
    "CheckpointData",
    "ActivityResult",
    "DurableContext",
    "DurableAgent",
    "DurableOrchestrator",
    "WorkflowStore",
    "FileWorkflowStore",
    "activity",
    "workflow",

    # Concurrent Orchestration (v3.4 NEW!)
    "FanOutConfig",
    "AggregationStrategy",
    "ParallelResult",
    "AggregatedResult",
    "ConcurrentOrchestrator",
    "ResultAggregator",
    "MapReducePattern",
    "ScatterGatherPattern",
    "ConditionalFanOut",

    # AgentTool (v3.4 NEW!)
    "AgentToolConfig",
    "DelegationPolicy",
    "DelegationResult",
    "AgentTool",
    "AgentToolRegistry",
    "DelegationManager",
    "AgentChain",
    "ChainStep",

    # Extended Thinking (v3.4 NEW!)
    "ThinkingConfig",
    "ThinkingMode",
    "ThinkingStepType",
    "ThinkingStep",
    "ThinkingChain",
    "ThinkingContext",
    "ThinkingTracker",
    "ThinkingAnalyzer",
    "ThinkingMetrics",
    "ThinkingStore",

    # MCP Workbench (v3.4 NEW!)
    "McpServerConfig",
    "McpWorkbenchConfig",
    "ConnectionState",
    "LoadBalanceStrategy",
    "McpServerConnection",
    "McpServerInfo",
    "McpWorkbench",
    "McpToolRegistry",
    "McpRouter",
    "CapabilityRouter",
    "RoundRobinRouter",
    "HealthChecker",
    "HealthStatus",

    # Security Guardrails (v3.5 NEW!)
    "ThreatLevel",
    "AttackType",
    "PIIType",
    "ValidationStage",
    "SecurityConfig",
    "ShieldResult",
    "JailbreakResult",
    "PIIResult",
    "GroundednessResult",
    "ValidationResult",
    "AuditLogEntry",
    "PromptShield",
    "JailbreakDetector",
    "OutputValidator",
    "GroundednessChecker",
    "PIIDetector",
    "SecurityOrchestrator",
    "SecurityAuditLogger",

    # Structured Output (v3.5 NEW!)
    "OutputSchema",
    "StructuredOutputConfig",
    "StructuredOutputParser",
    "StructuredOutputValidator",
    "StructuredValidationError",
    "StructuredOutputClient",
    "structured_output",
    "pydantic_to_schema",

    # Evaluation (v3.5 NEW!)
    "PDCAPhase",
    "EvaluationDimension",
    "QualityLevel",
    "GapSeverity",
    "EvaluationConfig",
    "JudgeConfig",
    "IterationConfig",
    "EvaluationResult",
    "JudgeVerdict",
    "GapAnalysisResult",
    "IterationResult",
    "BenchmarkResult",
    "QualityReport",
    "PDCAEvaluator",
    "LLMJudge",
    "CheckActIterator",
    "GapAnalyzer",
    "AgentBenchmark",
    "QualityMetrics",
    "Evaluator",
    "Optimizer",
    # ── v4.0 NEW: Responses API ──
    "ResponsesClient",
    "ConversationState",
    "BackgroundMode",
    "ResponseConfig",
    "ResponseObject",
    "ResponseStatus",
    "ToolType",
    # ── v4.0 NEW: Video Generation ──
    "VideoGenerator",
    "Sora2Client",
    "VideoConfig",
    "VideoResult",
    "VideoModel",
    "VideoStatus",
    # ── v4.0 NEW: Image Generation ──
    "ImageModel",
    "ImageGenerator",
    "GPTImage1_5Client",
    "ImageConfig",
    "ImageResult",
    # ── v4.0 NEW: Open Weight Models ──
    "OpenWeightAdapter",
    "OSSModelConfig",
    "OpenWeightRegistry",
    "OSSLicense",
    "OSSModelInfo",
    # ── v4.0 NEW: Universal Agent Bridge ──
    "UniversalAgentBridge",
    "BridgeProtocol",
    # ── v4.0 NEW: Framework Bridges ──
    "OpenAIAgentsBridge",
    "AgentHandoff",
    "SessionBackend",
    "GoogleADKBridge",
    "CrewAIBridge",
    "A2ABridge",
    "AgentCard",
    "TaskMode",
    "MicrosoftAgentBridge",
    "AG2Bridge",
    "SemanticKernelAgentBridge",
    # ── v4.1 NEW: Agent Identity (Microsoft Entra ID) ──
    "AgentIdentity",
    "AgentCredential",
    "AgentRBACManager",
    "AgentIdentityProvider",
    "AgentDelegation",
    "IdentityRegistry",
    "ScopedPermission",
    "PermissionScope",
    "IdentityStatus",
    "AuthMethod",
    "AgentIdentityRole",
    "IdentityAuditEntry",
    # ── v4.1 NEW: Browser Automation & Computer Use ──
    "BrowserAutomation",
    "ComputerUseAgent",
    "BrowserSession",
    "SafetyChecker",
    "ActionRecorder",
    "BrowserConfig",
    "BrowserAction",
    "ActionResult",
    "CUAConfig",
    "CUAResult",
    "ScreenCapture",
    "ActionType",
    "BrowserStatus",
    "CUAEnvironment",
    # ── v4.1 NEW: Deep Research ──
    "DeepResearchAgent",
    "SourceCollector",
    "SynthesisEngine",
    "CitationManager",
    "ResearchConfig",
    "ResearchPlan",
    "ResearchStep",
    "SourceDocument",
    "Citation",
    "ResearchResult",
    "ResearchCheckpoint",
    "ResearchPhase",
    "SourceType",
    "ResearchStatus",
    "SearchProvider",
    # ── v4.1 NEW: Observability (OpenTelemetry) ──
    "ObservabilityPipeline",
    "AgentTelemetry",
    "MetricsCollector",
    "TraceExporter",
    "AlertManager",
    "AgentDashboard",
    "ObservabilityConfig",
    "TelemetrySpan",
    "MetricRecord",
    "AlertRule",
    "AlertEvent",
    "DashboardData",
    "MetricType",
    "ExportTarget",
    "TelemetryLevel",
    # ── v4.1 NEW: Middleware Pipeline ──
    "MiddlewareManager",
    "MiddlewareChain",
    "BaseMiddleware",
    "RequestMiddleware",
    "ResponseMiddleware",
    "LoggingMiddleware",
    "AuthMiddleware",
    "RateLimitMiddleware",
    "RetryMiddleware",
    "ContentFilterMiddleware",
    "CacheMiddleware",
    "MiddlewareConfig",
    "MiddlewareContext",
    "MiddlewareResult",
    "MiddlewareMetrics",
    "MiddlewarePhase",
    "MiddlewarePriority",
    "MiddlewareStatus",
    # ── v4.1 NEW: Agent Triggers (Event-Driven) ──
    "TriggerManager",
    "EventTrigger",
    "ScheduleTrigger",
    "WebhookTrigger",
    "QueueTrigger",
    "FileChangeTrigger",
    "AgentCompletionTrigger",
    "BaseTrigger",
    "TriggerConfig",
    "TriggerEvent",
    "TriggerCondition",
    "TriggerResult",
    "TriggerMetrics",
    "TriggerType",
    "TriggerStatus",
    "TriggerPriority",
]
