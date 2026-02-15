#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Unified Agent Framework - Enterprise Edition v4.1
16개 AI Agent 프레임워크 통합 · 7가지 핵심 기술 혁신 · Model-Agnostic 설계

============================================================================
📌 모듈 정보
============================================================================
버전: 4.1.0
작성자: Enterprise AI Team
라이선스: MIT
최종 업데이트: 2026년 2월

💡 설계 철학:
    "복잡한 것을 단순하게, 단순한 것을 강력하게"
    - 창의적 기술: 단순 래핑이 아닌, 프레임워크 고유의 혁신적 설계 패턴
    - 실용적 유용성: 실제 프로덕션에서 바로 사용 가능한 구조와 안전장치
    - 좋은 아이디어: 16개 프레임워크의 최고 아이디어를 통합하고 발전
    - 쉬운 사용법: 3줄이면 시작, 점진적으로 확장 가능 (Progressive Disclosure)
    - Model-Agnostic: 특정 모델에 종속되지 않음 — 한 줄로 모델 전환

🧠 7가지 핵심 기술 혁신:
    1. Universal Agent Bridge — 16개 프레임워크를 하나의 인터페이스로 (전환 비용 0)
    2. Session Tree — Git 스타일 대화 분기/병합/리와인드
    3. Adaptive Circuit Breaker — 실시간 메트릭 기반 동적 타임아웃
    4. Security Guardrails Pipeline — PromptShield + JailbreakDetector + PIIDetector
    5. PDCA Auto Quality Loop — LLMJudge → GapAnalyzer → CheckActIterator 자동 개선
    6. Responses API Stateful — 대화 상태 서버사이드 관리, Background Mode
    7. A2A + MCP Dual Protocol — Agent-to-Agent 프로토콜 + MCP 도구 통합

🆕 v4.0 주요 변경사항 (2026년 2월, v4.1에서 확장):
- Universal Agent Bridge: 16개 프레임워크 통합 (OpenAI Agents SDK, Google ADK, CrewAI 등)
- 7개 프레임워크 브릿지 모듈 추가 (openai_agents_bridge, google_adk_bridge 등)
- A2A Protocol v0.3.0 기반 에이전트 간 협업
- Responses API 기반 Stateful 대화 관리
- Sora 2/2 Pro 비디오 생성 통합
- GPT Image 1.5 이미지 생성 통합
- 오픈 웨이트 모델 지원 (gpt-oss-120b/20b)
- Model-Agnostic 설계 완성 — 모든 핵심 기술이 모델 독립적으로 작동

🆕 v4.1 주요 변경사항 (2026년 2월):
- Agent Identity: Microsoft Entra ID 에이전트 전용 ID/RBAC
- Browser Automation & CUA: Playwright + OpenAI Computer Use Agent 통합
- Deep Research: o3-deep-research 다단계 자율 연구 에이전트
- Observability: OpenTelemetry 네이티브 분산 추적/메트릭/로깅
- Middleware Pipeline: 요청/응답 미들웨어 체인 (인증, 캐시, 필터링)
- Agent Triggers: 이벤트/스케줄/웹훅 기반 에이전트 자동 호출

📋 v3.5 주요 변경사항 (2026년 2월):
- Security Guardrails Pipeline (PromptShield, JailbreakDetector, PIIDetector)
- Structured Output (OutputSchema, StructuredOutputParser)
- PDCA Evaluation (PDCAEvaluator, LLMJudge, CheckActIterator, GapAnalyzer)
- 성능 최적화: frozenset, bisect.insort, 패턴 캐싱, LRU 캐시, 연결 풀링

📋 v3.4 주요 변경사항 (2026년 1월):
- Prompt Cache (PromptCache, CacheConfig)
- Durable Agent (DurableOrchestrator, DurableConfig)
- Concurrent Orchestration (ConcurrentOrchestrator, FanOutConfig)
- Agent-as-Tool 패턴 (AgentToolRegistry, DelegationManager)
- Extended Thinking (ThinkingTracker, ThinkingConfig)
- MCP Workbench (McpWorkbench, McpServerConfig)
- Extensions Hub (ExtensionsHub)

📋 v3.3 주요 변경사항 (2026년 1월):
- Agent Lightning 통합 (분산 추적, 보상 시스템, 모델 어댑터)
- AgentTracer: OpenTelemetry 기반 분산 추적 (SpanKind, SpanStatus)
- AgentStore: 우선순위 기반 에이전트 저장소 (bisect 최적화)
- RewardEngine: 실시간 보상 신호 기반 에이전트 자가 개선
- AdapterManager: 다중 LLM 제공자 통합 어댑터
- HookManager: 라이프사이클 훅 포인트 (PreProcess, PostProcess, OnError)

📋 v3.2 주요 변경사항 (2026년 2월):
- 영속 메모리 시스템 (Clawdbot 스타일 2계층 메모리)
- Compaction 시스템 (컨텍스트 압축, Cache-TTL Pruning)
- 세션 트리 시스템 (Pi 스타일 대화 브랜칭/리와인드)

📋 v3.1 주요 변경사항 (2026년 1월):
- Microsoft Agent Framework MCP 패턴 완전 통합
- Adaptive Circuit Breaker (동적 타임아웃)
- RAI 강화 검증 (Azure Content Safety 통합)
- 병렬 도구 호출 (최대 5개 동시)

이 파일은 unified_agent 패키지의 모든 공개 API를 re-export합니다.
실제 구현은 unified_agent/ 패키지의 개별 모듈에 있습니다.

패키지 구조 (49개 모듈):
    unified_agent/
    ├── __init__.py              # 패키지 진입점 (380개 공개 API export)
    ├── interfaces.py            # 핵심 인터페이스 (IFramework, IOrchestrator, IMemoryProvider)
    ├── exceptions.py            # 예외 클래스 (FrameworkError, ConfigurationError 등)
    ├── config.py                # 설정 및 상수 (Settings, FrameworkConfig) - frozenset 최적화
    ├── models.py                # 데이터 모델 (Enum, Pydantic, Dataclass)
    ├── utils.py                 # 유틸리티 (StructuredLogger, CircuitBreaker, RAIValidator)
    ├── memory.py                # 메모리 시스템 (MemoryStore, CachedMemoryStore)
    ├── persistent_memory.py     # v3.2 영속 메모리 (PersistentMemory, MemoryLayer)
    ├── compaction.py            # v3.2 메모리 압축 (CompactionEngine, CompactionStrategy)
    ├── session_tree.py          # v3.2 세션 트리 (SessionTree, BranchInfo)
    ├── events.py                # 이벤트 시스템 (EventBus, EventType)
    ├── skills.py                # Skills 시스템 (Skill, SkillManager)
    ├── tools.py                 # 도구 (AIFunction, MCPTool)
    ├── agents.py                # 에이전트 (SimpleAgent, RouterAgent, SupervisorAgent)
    ├── workflow.py              # 워크플로우 (Graph, Node)
    ├── orchestration.py         # 오케스트레이션 (AgentFactory, OrchestrationManager)
    ├── framework.py             # 메인 프레임워크 (UnifiedAgentFramework)
    ├── tracer.py                # v3.3 분산 추적 (AgentTracer, SpanContext)
    ├── agent_store.py           # v3.3 에이전트 저장소 (AgentStore, AgentEntry)
    ├── reward.py                # v3.3 보상 시스템 (RewardEngine, RewardSignal)
    ├── adapter.py               # v3.3 모델 어댑터 (AdapterManager, ModelAdapter)
    ├── hooks.py                 # v3.3 라이프사이클 훅 (HookManager, HookPoint)
    ├── prompt_cache.py          # v3.4 프롬프트 캐싱 (PromptCache, CacheConfig)
    ├── durable_agent.py         # v3.4 내구성 에이전트 (DurableOrchestrator, DurableConfig)
    ├── concurrent.py            # v3.4 병렬 오케스트레이션 (ConcurrentOrchestrator, FanOutConfig)
    ├── agent_tool.py            # v3.4 에이전트 도구 패턴 (AgentToolRegistry, DelegationManager)
    ├── extended_thinking.py     # v3.4 확장 사고 (ThinkingTracker, ThinkingConfig)
    ├── mcp_workbench.py         # v3.4 MCP 워크벤치 (McpWorkbench, McpServerConfig)
    ├── extensions.py            # v3.4 확장 허브 (ExtensionsHub)
    ├── security_guardrails.py   # v3.5 보안 가드레일 (PromptShield, JailbreakDetector, PIIDetector)
    ├── structured_output.py     # v3.5 구조화된 출력 (OutputSchema, StructuredOutputParser)
    ├── evaluation.py            # v3.5 PDCA 평가 (PDCAEvaluator, LLMJudge, CheckActIterator)
    ├── responses_api.py         # v4.0 Responses API (ResponsesClient, ConversationState)
    ├── video_generation.py      # v4.0 비디오 생성 (VideoGenerator, Sora2Client)
    ├── image_generation.py      # v4.0 이미지 생성 (ImageGenerator, GPTImage1_5Client)
    ├── open_weight.py           # v4.0 오픈 웨이트 모델 (OpenWeightAdapter, OSSModelConfig)
    ├── universal_bridge.py      # v4.0 통합 브릿지 (UniversalAgentBridge, 16개 프레임워크)
    ├── openai_agents_bridge.py  # v4.0 OpenAI Agents SDK 브릿지 (Handoff, Session, HITL)
    ├── google_adk_bridge.py     # v4.0 Google ADK 브릿지 (Workflow Agent, A2A 통합)
    ├── crewai_bridge.py         # v4.0 CrewAI 브릿지 (Crews + Flows 아키텍처)
    ├── a2a_bridge.py            # v4.0 A2A Protocol 브릿지 (Agent Card, JSON-RPC 2.0)
    ├── ms_agent_bridge.py       # v4.0 Microsoft Agent Framework 브릿지 (Graph Workflow)
    ├── ag2_bridge.py            # v4.0 AG2 AgentOS 브릿지 (Universal Interop)
    ├── sk_agent_bridge.py       # v4.0 SK Agent Framework 브릿지 (Orchestration 패턴)
    ├── agent_identity.py        # v4.1 Agent Identity (Microsoft Entra ID RBAC)
    ├── browser_use.py           # v4.1 Browser Automation + CUA (Playwright, Computer Use)
    ├── deep_research.py         # v4.1 Deep Research (o3-deep-research)
    ├── observability.py         # v4.1 Observability (OpenTelemetry 네이티브)
    ├── middleware.py            # v4.1 Middleware Pipeline (요청/응답 체인)
    └── agent_triggers.py        # v4.1 Agent Triggers (이벤트 기반 자동 호출)

============================================================================
🚀 빠른 시작 가이드
============================================================================

1. 3줄로 시작하기 (Model-Agnostic):
   ```python
   from unified_agent import UnifiedAgentFramework, Settings

   Settings.DEFAULT_MODEL = "gpt-5.2"           # 모델 하나만 설정 (어떤 모델이든 OK)
   framework = UnifiedAgentFramework.create()   # 끝! 바로 사용 가능
   result = await framework.run("보고서를 작성해주세요")  # 모든 기능 자동 활성화
   ```

   > Model-Agnostic: GPT, Claude, Grok, Llama 등 어떤 모델이든 한 줄로 전환 가능

2. Universal Agent Bridge (v4.0 NEW! - 16개 프레임워크 통합):
   ```python
   from unified_agent import UniversalAgentBridge, OpenAIAgentsBridge, GoogleADKBridge

   bridge = UniversalAgentBridge()
   bridge.register("openai", OpenAIAgentsBridge())
   bridge.register("google", GoogleADKBridge())

   # 동일한 인터페이스로 프레임워크 전환 — 전환 비용 0
   result = await bridge.run("openai", task="코드 리뷰")
   result = await bridge.run("google", task="데이터 분석")  # 코드 변경 없이 전환
   ```

3. Session Tree — Git 스타일 대화 분기 (v3.2):
   ```python
   from unified_agent import SessionTree

   tree = SessionTree(session_id="conversation_1")
   branch = tree.create_branch("alternative_approach")
   tree.merge_branch(branch.branch_id, target_branch_id="main")
   ```

4. Security Guardrails Pipeline (v3.5):
   ```python
   from unified_agent import SecurityOrchestrator, SecurityConfig

   security = SecurityOrchestrator(SecurityConfig(
       enable_prompt_shield=True,
       enable_jailbreak_detection=True,
       enable_pii_detection=True,
   ))
   result = await security.validate(user_input)
   ```

5. PDCA 자동 품질 루프 (v3.5):
   ```python
   from unified_agent import PDCAEvaluator, LLMJudge

   evaluator = PDCAEvaluator(judge=LLMJudge())
   improved = await evaluator.evaluate_and_improve(agent_output)
   ```

6. A2A 프로토콜 에이전트 간 협업 (v4.0):
   ```python
   from unified_agent import A2ABridge, AgentCard

   a2a = A2ABridge()
   card = AgentCard(name="researcher", capabilities=["search", "summarize"])
   await a2a.publish_card(card)
   result = await a2a.delegate("summarize", input_data)
   ```

7. Responses API Stateful 대화 (v4.0):
   ```python
   from unified_agent import ResponsesClient, ConversationState

   client = ResponsesClient()
   state = ConversationState()
   response = await client.send("프로젝트 상태 알려줘", state=state)
   # 서버가 상태를 관리 — 클라이언트는 state ID만 전달
   ```

8. 영속 메모리 시스템 (v3.2 - Clawdbot 스타일):
   ```python
   from unified_agent import PersistentMemory

   memory = PersistentMemory(agent_id="main")
   await memory.initialize()
   await memory.add_long_term_memory("TypeScript 선호", section="User Preferences")
   results = await memory.search("API 설계")  # 하이브리드 검색 (Vector 70% + BM25 30%)
   ```

============================================================================
주요 기능 (v4.1 — 7가지 핵심 기술 혁신 + 49개 모듈)
============================================================================
[핵심 기술 혁신 — v4.1]
1. Universal Agent Bridge (16개 프레임워크 통합, 전환 비용 0)
2. Session Tree (Git 스타일 대화 분기/병합/리와인드)
3. Adaptive Circuit Breaker (실시간 메트릭 기반 동적 타임아웃)
4. Security Guardrails Pipeline (PromptShield + JailbreakDetector + PIIDetector)
5. PDCA Auto Quality Loop (LLMJudge → GapAnalyzer → CheckActIterator)
6. Responses API Stateful (대화 상태 서버사이드 관리, Background Mode)
7. A2A + MCP Dual Protocol (Agent-to-Agent + MCP 도구 통합)

[프레임워크 브릿지 — v4.0 NEW!]
8. OpenAI Agents SDK Bridge (Handoff, Session, HITL)
9. Google ADK Bridge (Workflow Agent, A2A 통합)
10. CrewAI Bridge (Crews + Flows 아키텍처)
11. A2A Protocol Bridge (Agent Card, JSON-RPC 2.0)
12. Microsoft Agent Framework Bridge (Graph Workflow)
13. AG2 AgentOS Bridge (Universal Interop)
14. SK Agent Framework Bridge (Orchestration 패턴)

[멀티모달 생성 — v4.0 NEW!]
15. Responses API (ResponsesClient, ConversationState, Background Mode)
16. Sora 2/2 Pro 비디오 생성 (VideoGenerator, Sora2Client)
17. GPT Image 1.5 이미지 생성 (ImageGenerator, GPTImage1_5Client)
18. 오픈 웨이트 모델 지원 (gpt-oss-120b/20b, OpenWeightAdapter)

[보안 / 평가 / 구조화 — v3.5]
19. Security Guardrails Pipeline (멀티레이어 방어)
20. Structured Output (OutputSchema, StructuredOutputParser)
21. PDCA Evaluation (PDCAEvaluator, LLMJudge, CheckActIterator)

[고급 오케스트레이션 — v3.4]
22. Prompt Cache (프롬프트 캐싱, 비용 최적화)
23. Durable Agent (내구성 에이전트, 장기 실행 태스크)
24. Concurrent Orchestration (Fan-Out/Fan-In 병렬 처리)
25. Agent-as-Tool (AgentToolRegistry, DelegationManager)
26. Extended Thinking (Claude/GPT 확장 사고 추적)
27. MCP Workbench (MCP 서버 통합 관리)
28. Extensions Hub (플러그인 확장 시스템)

[Agent Lightning — v3.3]
29. AgentTracer: OpenTelemetry 기반 분산 추적
30. AgentStore: 우선순위 기반 에이전트 저장소 (bisect 최적화)
31. RewardEngine: 실시간 보상 신호 기반 자가 개선
32. AdapterManager: 다중 LLM 제공자 통합 어댑터
33. HookManager: 라이프사이클 훅 포인트

[핵심 인프라 — v3.0~v3.2]
34. MCP (Model Context Protocol) 서버 통합
35. Human-in-the-loop 승인 시스템
36. 스트리밍 응답 지원 (기본 활성화)
37. 영속 메모리 시스템 (2계층, 하이브리드 검색)
38. Compaction 시스템 (컨텍스트 압축, Cache-TTL Pruning)
39. 비동기 이벤트 시스템 (Pub-Sub)
40. Supervisor Agent 패턴
41. MPlan 구조화된 계획 시스템
42. Team 기반 오케스트레이션

[최신 기술 통합 — v4.1 NEW!]
43. Agent Identity (Microsoft Entra ID 에이전트 전용 ID/RBAC)
44. Browser Automation & CUA (Playwright + Computer Use Agent)
45. Deep Research (o3-deep-research 다단계 자율 연구)
46. Observability (OpenTelemetry 네이티브 분산 추적/메트릭)
47. Middleware Pipeline (요청/응답 미들웨어 체인)
48. Agent Triggers (이벤트/스케줄/웹훅 기반 자동 호출)

============================================================================
필요 패키지
============================================================================
pip install semantic-kernel python-dotenv opentelemetry-api opentelemetry-sdk pydantic pyyaml
# MCP 통합 (선택)
pip install agent-framework-azure-ai --pre
# OpenAI Agents SDK (선택)
pip install openai-agents
# A2A Protocol (선택)
pip install a2a-sdk
# Google ADK (선택)
pip install google-adk
"""

# ============================================================================
# 모듈 메타데이터
# ============================================================================
__version__ = "4.1.0"
__author__ = "Enterprise AI Team"

# ============================================================================
# unified_agent 패키지 lazy re-export (PEP 562)
# v4.1: 49개 모듈에서 380+ 공개 심볼 — 즉시 로드 없이 위임
# ============================================================================
import sys as _sys
import unified_agent as _ua

def __getattr__(name: str):
    """unified_agent 패키지의 모든 공개 API를 lazy하게 re-export합니다."""
    # 하위 호환성 별칭
    if name == "TeamService":
        value = getattr(_ua, "OrchestrationManager")
        setattr(_sys.modules[__name__], name, value)
        return value
    # __all__에 정의된 심볼만 위임
    if name in __all__:
        value = getattr(_ua, name)
        setattr(_sys.modules[__name__], name, value)  # cache
        return value
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

def __dir__():
    return list(__all__)

# ============================================================================
# Public API 정의
# v4.1: 49개 모듈에서 380+ 공개 심볼
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

    # 설정 클래스 (unified_agent/config.py) - Model-Agnostic 설계
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

    # 영속 메모리 시스템 (unified_agent/persistent_memory.py) - v3.2
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

    # Compaction 시스템 (unified_agent/compaction.py) - v3.2
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

    # 세션 트리 시스템 (unified_agent/session_tree.py) - v3.2
    "SessionTreeConfig",
    "SessionNode",
    "NodeType",
    "SessionTree",
    "BranchInfo",
    "SessionTreeManager",
    "SessionSnapshot",

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

    # 오케스트레이션 (unified_agent/orchestration.py)
    "AgentFactory",
    "OrchestrationManager",

    # 핵심 프레임워크 (unified_agent/framework.py)
    "UnifiedAgentFramework",
    "quick_run",
    "create_framework",

    # ─── v3.3 Agent Lightning ────────────────────────────────────────────

    # 분산 추적 (unified_agent/tracer.py)
    "AgentTracer",
    "SpanKind",
    "SpanStatus",
    "SpanContext",

    # 에이전트 저장소 (unified_agent/agent_store.py)
    "AgentStore",
    "Rollout",
    "Attempt",
    "RolloutStatus",

    # 보상 시스템 (unified_agent/reward.py)
    "RewardManager",
    "RewardRecord",
    "RewardDimension",

    # 모델 어댑터 (unified_agent/adapter.py)
    "Adapter",
    "TraceAdapter",

    # 라이프사이클 훅 (unified_agent/hooks.py)
    "HookManager",
    "HookEvent",
    "HookPriority",

    # ─── v3.4 Advanced Orchestration ─────────────────────────────────────

    # 프롬프트 캐시 (unified_agent/prompt_cache.py)
    "PromptCache",
    "CacheConfig",
    "CacheEntry",

    # 내구성 에이전트 (unified_agent/durable_agent.py)
    "DurableAgent",
    "DurableConfig",
    "DurableOrchestrator",
    "WorkflowStore",

    # 병렬 오케스트레이션 (unified_agent/concurrent.py)
    "ConcurrentOrchestrator",
    "FanOutConfig",
    "AggregationStrategy",
    "ParallelResult",

    # Agent-as-Tool (unified_agent/agent_tool.py)
    "AgentTool",
    "AgentToolRegistry",
    "DelegationManager",

    # 확장 사고 (unified_agent/extended_thinking.py)
    "ThinkingTracker",
    "ThinkingConfig",
    "ThinkingStep",

    # MCP 워크벤치 (unified_agent/mcp_workbench.py)
    "McpWorkbench",
    "McpServerConfig",
    "McpToolRegistry",

    # 확장 (unified_agent/extensions.py)
    "Extensions",
    "ExtensionsConfig",

    # ─── v3.5 Security & Evaluation ──────────────────────────────────────

    # 보안 가드레일 (unified_agent/security_guardrails.py)
    "SecurityOrchestrator",
    "SecurityConfig",
    "PromptShield",
    "JailbreakDetector",
    "PIIDetector",
    "ShieldResult",

    # 구조화된 출력 (unified_agent/structured_output.py)
    "StructuredOutputClient",
    "OutputSchema",
    "StructuredOutputParser",

    # PDCA 평가 (unified_agent/evaluation.py)
    "PDCAEvaluator",
    "LLMJudge",
    "CheckActIterator",
    "GapAnalyzer",
    "QualityMetrics",

    # ─── v4.0 Universal Bridge & Multimodal ──────────────────────────────

    # Responses API (unified_agent/responses_api.py)
    "ResponsesClient",
    "ConversationState",
    "BackgroundMode",
    "ResponseConfig",
    "ResponseObject",
    "ResponseStatus",
    "ToolType",

    # 비디오 생성 (unified_agent/video_generation.py)
    "VideoGenerator",
    "Sora2Client",
    "VideoConfig",
    "VideoResult",
    "VideoModel",
    "VideoStatus",

    # 이미지 생성 (unified_agent/image_generation.py)
    "ImageGenerator",
    "GPTImage1_5Client",
    "ImageConfig",
    "ImageResult",
    "ImageModel",

    # 오픈 웨이트 모델 (unified_agent/open_weight.py)
    "OpenWeightAdapter",
    "OSSModelConfig",
    "OpenWeightRegistry",
    "OSSLicense",
    "OSSModelInfo",

    # Universal Agent Bridge (unified_agent/universal_bridge.py)
    "UniversalAgentBridge",
    "BridgeProtocol",

    # 프레임워크 브릿지 모듈
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

    # ─── v4.1 Agent Identity (Microsoft Entra ID) ────────────────────────
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

    # ─── v4.1 Browser Automation & CUA ───────────────────────────────────
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

    # ─── v4.1 Deep Research ──────────────────────────────────────────────
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

    # ─── v4.1 Observability (OpenTelemetry) ──────────────────────────────
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

    # ─── v4.1 Middleware Pipeline ────────────────────────────────────────
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

    # ─── v4.1 Agent Triggers (Event-Driven) ──────────────────────────────
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


# ============================================================================
# 하위 호환성을 위한 별칭
# v3.0: TeamService → OrchestrationManager로 통합됨 (Deprecated)
# v4.0: 레거시 별칭 유지
# ============================================================================

# TeamService는 OrchestrationManager로 통합됨 (lazy — __getattr__에서 처리)


# ============================================================================
# 모듈 로드 시 초기화
# v4.0: UTF-8 인코딩 자동 설정 (Windows 환경 지원)
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
