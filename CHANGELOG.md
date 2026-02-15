# 📝 Changelog

All notable changes to Unified Agent Framework will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [5.0.0] - 2026-02-15

### 🔄 Breaking Changes — Runner-Centric Redesign

v4.1의 49개 모듈을 **9개 모듈**로, 16개 프레임워크 브릿지를 **Top 3 엔진**으로 전면 재설계.

#### 축소 (82% 모듈 감소)
- 49개 모듈 → 9개 모듈 (`unified_agent_v5/` 패키지)
- 380+ 공개 API → ~20개 공개 API
- 16개 프레임워크 브릿지 → 3개 엔진 (Direct, LangChain, CrewAI)
- 필수 의존성: 8개 (`semantic-kernel` 포함) → 2개 (`openai`, `python-dotenv`)

#### 신규 모듈 (`unified_agent_v5/`)
- `runner.py`: Runner 중심 설계 — `run_agent("질문")` 한 줄 진입점
- `types.py`: OpenAI ChatCompletion 표준 통합 I/O (Message, AgentResult)
- `config.py`: 최소 설정 (Settings, AgentConfig)
- `memory.py`: List[Message] + 슬라이딩 윈도우 + JSON 직렬화
- `tools.py`: MCP 표준 Tool + `@mcp_tool` 데코레이터
- `callback.py`: OTEL 표준 어댑터 (CallbackHandler, OTelCallbackHandler)
- `engines/direct.py`: OpenAI/Azure API 직접 호출 엔진
- `engines/langchain_engine.py`: LangChain 체인/RAG 엔진
- `engines/crewai_engine.py`: CrewAI 멀티 에이전트 엔진
- `plugins/`: v4 비핵심 기능 마이그레이션 대상

#### 설계 원칙 변경
1. **Top 3 + Direct**: 16개 → 3개 엔진 (실무 사용 빈도 기준 선정)
2. **OTEL 표준 어댑터**: 자체 Tracer/Dashboard 제거 → CallbackHandler 패턴
3. **핵심 3기능 집중**: Unified I/O, Memory, Tool Use
4. **Runner 중심**: "만드는 것"은 엔진이, "실행하는 것"은 Runner가

#### v4.1 아카이브
- `unified_agent/` 패키지는 `_legacy/` 디렉토리로 아카이브
- `unified_agent_v5/`는 독립적으로 동작
- v4.1 데모, 테스트, README도 `_legacy/`에 포함

---

## [4.1.0] - 2026-02-14

### 🆕 Added

#### Agent Identity (agent_identity.py)
- `AgentIdentity`: Microsoft Entra ID 에이전트 전용 ID 관리
- `AgentCredential`: 에이전트 자격 증명
- `AgentRBACManager`: RBAC 기반 권한 관리 (최소 권한 원칙)
- `AgentIdentityProvider`: ID 프로비저닝 및 라이프사이클 관리
- `AgentDelegation`: 에이전트 간 위임 인증
- `IdentityRegistry`, `ScopedPermission`, `PermissionScope`

#### Browser Automation & CUA (browser_use.py)
- `BrowserAutomation`: Playwright 기반 헤드리스 브라우저 자동화
- `ComputerUseAgent`: OpenAI Computer Use Agent (CUA) 통합
- `BrowserSession`, `SafetyChecker`, `ActionRecorder`
- `BrowserConfig`, `CUAConfig`, `CUAEnvironment`

#### Deep Research (deep_research.py)
- `DeepResearchAgent`: 다단계 자율 연구 에이전트 (o3-deep-research)
- `SourceCollector`: 다중 소스 문서 수집 (Web, Academic, API)
- `SynthesisEngine`: 연구 결과 종합 엔진
- `CitationManager`: 인용 관리 및 검증
- `ResearchConfig`, `ResearchPlan`, `ResearchStep`

#### Observability (observability.py)
- `ObservabilityPipeline`: OpenTelemetry 네이티브 분산 추적/메트릭/로깅
- `MetricsCollector`: 에이전트 메트릭 수집
- `TraceExporter`: 분산 추적 익스포터 (Azure Monitor, Jaeger 등)
- `AlertManager`, `AgentDashboard`
- `ObservabilityConfig`, `TelemetrySpan`, `MetricRecord`

#### Middleware Pipeline (middleware.py)
- `MiddlewareManager`: 요청/응답 미들웨어 파이프라인
- `MiddlewareChain`: 체인 패턴 미들웨어 실행
- `AuthMiddleware`, `RateLimitMiddleware`, `RetryMiddleware`
- `ContentFilterMiddleware`, `CacheMiddleware`, `LoggingMiddleware`
- `MiddlewareConfig`, `MiddlewareContext`, `MiddlewareResult`

#### Agent Triggers (agent_triggers.py)
- `TriggerManager`: 이벤트 기반 에이전트 자동 호출
- `EventTrigger`, `ScheduleTrigger`, `WebhookTrigger`
- `QueueTrigger`, `FileChangeTrigger`, `AgentCompletionTrigger`
- `TriggerConfig`, `TriggerEvent`, `TriggerCondition`

#### framework.py v4.1 팩토리 메서드 추가
- `create_agent_identity_provider()`: Agent Identity 프로바이더 생성
- `create_browser_automation()`: 브라우저 자동화 인스턴스 생성
- `create_deep_research_agent()`: Deep Research 에이전트 생성
- `create_observability_pipeline()`: Observability 파이프라인 생성
- `create_middleware_manager()`: 미들웨어 매니저 생성
- `create_trigger_manager()`: 트리거 매니저 생성

### 🔧 Changed
- 49개 모듈, 380+ 공개 API로 확장 (v4.0: 43개 → v4.1: 49개)
- 모든 v4.1 모듈 자체 완결형 (순환 참조 없음)
- README.md v4.1 전면 개편 (6가지 최신 기술 통합)
- 테스트 시나리오 22개 → 28개로 확장

### ✅ Tests
- `test_v41_all_scenarios.py`: 28개 시나리오, 49개 모듈, 100% 통과

---

## [4.0.0] - 2026-02-08

### 🆕 Added

#### Responses API 통합 (responses_api.py)
- `ResponsesClient`: OpenAI Responses API 기반 Stateful 대화 클라이언트
- `ConversationState`: 대화 상태 관리 (세션, 턴 히스토리)
- `BackgroundMode`: 비동기 백그라운드 실행 지원
- `ResponseConfig`, `ResponseObject`, `ResponseStatus`, `ToolType`

#### Sora 2 비디오 생성 (video_generation.py)
- `VideoGenerator`: Sora 2/2 Pro 비디오 생성 파이프라인
- `Sora2Client`: Sora 2 API 직접 호출 클라이언트
- `VideoConfig`, `VideoResult`, `VideoModel`, `VideoStatus`

#### GPT Image 1.5 이미지 생성 (image_generation.py)
- `ImageGenerator`: GPT-image-1.5 이미지 생성기
- `GPTImage1_5Client`: 이미지 생성 API 클라이언트
- `ImageConfig`, `ImageResult`, `ImageModel`

#### 오픈 웨이트 모델 (open_weight.py)
- `OpenWeightAdapter`: gpt-oss-120b/20b 등 오픈 소스 모델 어댑터
- `OpenWeightRegistry`: 모델 레지스트리 (Llama 4, Phi-4, Mistral 등)
- `OSSModelConfig`, `OSSModelInfo`, `OSSLicense`

#### Universal Agent Bridge (universal_bridge.py)
- `UniversalAgentBridge`: 7개 프레임워크 통합 실행 레이어
- `BridgeProtocol`: 브릿지 프로토콜 인터페이스

#### 7개 프레임워크 브릿지 모듈
- `SemanticKernelAgentBridge` (sk_agent_bridge.py) — SK Orchestration 패턴
- `OpenAIAgentsBridge` (openai_agents_bridge.py) — Handoff, Session, Human-in-the-Loop
- `GoogleADKBridge` (google_adk_bridge.py) — Workflow Agent, A2A 프로토콜
- `CrewAIBridge` (crewai_bridge.py) — Crews + Flows 아키텍처
- `AG2Bridge` (ag2_bridge.py) — Universal Interop, AutoGen 진화
- `MicrosoftAgentBridge` (ms_agent_bridge.py) — Graph Workflow, Declarative Agents
- `A2ABridge` (a2a_bridge.py) — A2A Protocol v0.3.0 (AgentCard, JSON-RPC 2.0)

#### framework.py v4.0 팩토리 메서드
- `create_responses_client()`: Responses API 클라이언트 생성
- `create_video_generator()`: 비디오 생성기 팩토리
- `create_image_generator()`: 이미지 생성기 팩토리
- `create_open_weight_adapter()`: 오픈 웨이트 어댑터 팩토리
- `create_universal_bridge()`: Universal Bridge 팩토리
- `get_bridge(protocol)`: 프로토콜별 브릿지 인스턴스 반환

### 🔧 Changed
- 43개 모듈, 380+ 공개 API로 확장 (v3.5: 31개 → v4.0: 43개)
- 모든 v4.0 모듈 자체 완결형 (순환 참조 없음)
- 모든 bridge `run(*, task=...)` 시그니처 통일
- Config dataclass에 `frozen=True, slots=True` 적용
- 비-dataclass 클래스에 `__repr__` 추가
- 미사용 import 전면 제거

### 📚 Documentation
- README.md v4.0 전면 개편 (7가지 핵심 기술 혁신)
- 22개 시나리오 테스트 문서화

### ✅ Tests
- `test_v40_all_scenarios.py`: 22개 시나리오, 43개 모듈, 100% 통과

---

## [3.5.0] - 2026-02-01

### 🆕 Added

#### 보안 가드레일 (security_guardrails.py)
- `PromptShield`: 프롬프트 인젝션 방어
- `JailbreakDetector`: 탈옥 시도 탐지
- `PIIDetector`: 개인정보(PII) 탐지 및 마스킹

#### 구조화된 출력 (structured_output.py)
- `OutputSchema`: JSON Schema 기반 출력 스키마
- `StructuredParser`: 구조화된 파싱
- `OutputValidator`: 출력 유효성 검증

#### PDCA 평가 (evaluation.py)
- `PDCAEvaluator`: Plan-Do-Check-Act 평가 프레임워크
- `LLMJudge`: LLM 기반 품질 판정
- `GapAnalyzer`: 기대-실제 갭 분석
- `QualityMetrics`: 품질 메트릭 통합

### ✅ Tests
- 22개 시나리오 (v3.5 모듈 포함) 전체 통과

---

## [3.4.0] - 2026-01-20

### 🆕 Added

#### Extensions Hub (extensions.py)
- `Extensions`: 확장 모듈 통합 허브
- `ExtensionsConfig`: 확장 설정

#### 프롬프트 캐싱 (prompt_cache.py)
- `PromptCache`: 프롬프트 캐시 (LRU + TTL)
- `CacheConfig`: 캐시 설정

#### 확장 사고 (extended_thinking.py)
- `ThinkingTracker`: 사고 과정 추적
- `ThinkingConfig`, `ThinkingStep`

#### MCP 워크벤치 (mcp_workbench.py)
- `McpWorkbench`: MCP 서버 관리 워크벤치
- `McpServerConfig`: 서버 설정

#### 병렬 오케스트레이션 (concurrent.py)
- `ConcurrentOrchestrator`: Fan-Out/Fan-In 병렬 실행
- `FanOutConfig`: 병렬 설정

#### AgentTool 패턴 (agent_tool.py)
- `AgentToolRegistry`: 에이전트-도구 레지스트리
- `DelegationManager`: 위임 관리

#### 내구성 에이전트 (durable_agent.py)
- `DurableOrchestrator`: 장기 실행 워크플로우
- `DurableConfig`: 내구성 설정
- `@workflow` 데코레이터

#### 인터페이스 (interfaces.py)
- `IFramework`, `IOrchestrator`, `IMemoryProvider` 인터페이스 정의

---

## [3.3.0] - 2026-01-15

### 🆕 Added

#### Agent Lightning (tracer.py, hooks.py, reward.py)
- `AgentTracer`: 분산 추적 (OpenTelemetry 호환)
- `SpanKind`: INTERNAL, LLM, TOOL, RETRIEVER 등
- `HookManager`, `HookEvent`: 라이프사이클 훅
- `RewardManager`, `emit_reward()`: 보상 시스템

#### 모델 어댑터 (adapter.py)
- `ModelAdapter`: 비-OpenAI 모델 프로바이더 통합

#### 에이전트 저장소 (agent_store.py)
- `AgentStore`: 에이전트 설정 YAML 기반 영속화
- `AgentSnapshot`: 에이전트 스냅샷

---

## [3.2.0] - 2026-01-10

### 🆕 Added

#### 영속 메모리 (persistent_memory.py)
- `PersistentMemory`: 장기 기억 시스템
- `MemoryConfig`, `MemoryLayer`

#### 컴팩션 (compaction.py)
- `CompactionManager`: 컨텍스트 윈도우 관리
- `ContextCompactor`: 대화 요약 압축
- `CompactionConfig`

#### 세션 트리 (session_tree.py)
- `SessionTree`: 세션 브랜칭 (Git 스타일)
- `BranchInfo`: 브랜치 메타데이터

---

## [3.1.0] - 2026-01-26

### 🆕 Added

#### 최신 AI 모델 지원 (40+ 모델)
- **GPT-5.2 시리즈**: gpt-5.2, gpt-5.2-chat, gpt-5.2-codex (400K context)
- **GPT-5.1 Codex 시리즈**: gpt-5.1-codex, gpt-5.1-codex-mini, gpt-5.1-codex-max
- **Claude 4.5 시리즈**: claude-opus-4-5, claude-sonnet-4-5, claude-haiku-4-5 (Microsoft Foundry)
- **Grok-4 시리즈**: grok-4, grok-4-fast-reasoning (2M context), grok-4-fast-non-reasoning
- **o4-mini**: o3-mini 후속 Reasoning 모델
- **DeepSeek**: deepseek-v3.2, deepseek-v3.2-speciale, deepseek-r1-0528
- **Llama 4**: llama-4-maverick-17b, llama-4-scout-17b (10M context!)
- **Phi-4**: phi-4, phi-4-reasoning, phi-4-multimodal-instruct
- **Mistral**: mistral-large-3, mistral-medium-2505, mistral-small-2503

#### 유틸리티 함수
- `is_multimodal_model()`: 멀티모달(이미지/오디오 입력) 지원 모델 확인
- `is_large_context_model()`: 대용량 컨텍스트(100K+) 지원 모델 확인
- `get_model_context_window()`: 모델별 컨텍스트 윈도우 크기 반환

#### MCP (Model Context Protocol) 설정
- `ENABLE_MCP`: MCP 활성화 플래그
- `MCP_AUTO_CONNECT`: 자동 연결 설정
- `MCP_RECONNECT_ATTEMPTS`: 재연결 시도 횟수
- `MCP_REQUEST_TIMEOUT`: 요청 타임아웃
- `MCP_APPROVAL_MODE`: 승인 모드 (always/never/selective)

#### Multi-Agent 오케스트레이션 설정
- `ORCHESTRATION_MODE`: 오케스트레이션 모드 (supervisor/sequential/parallel/adaptive)
- `MAX_CONCURRENT_AGENTS`: 최대 동시 에이전트 수
- `ENABLE_HANDOFF`: 에이전트 간 Handoff 활성화
- `ENABLE_REFLECTION`: 반성(Reflection) 패턴 활성화

#### RAI (Responsible AI) 설정
- `ENABLE_RAI_VALIDATION`: RAI 검증 활성화
- `RAI_STRICT_MODE`: RAI 엄격 모드
- `RAI_CONTENT_SAFETY_LEVEL`: 콘텐츠 안전 레벨 (low/medium/high)
- `ENABLE_PII_DETECTION`: PII 감지 활성화

#### 상세 한글 주석
- 모든 12개 모듈에 상세한 한글 주석 추가
- 각 클래스/함수별 역할, 사용 예시, 주의사항 포함
- ASCII 다이어그램을 통한 상태 전환 설명 (CircuitBreaker)
- 참고 링크 및 관련 문서 연결

### 🔧 Changed

#### Adaptive Circuit Breaker 개선
- `success_threshold` 파라미터 추가 (HALF_OPEN → CLOSED 전환에 연속 성공 필요)
- `adaptive_timeout` 옵션 추가 (평균 응답 시간 기반 동적 타임아웃)
- 메트릭 수집 기능 추가 (`get_metrics()`)
- `reset()` 메서드 추가 (수동 리셋)

#### Settings 클래스 확장
- `DEFAULT_API_VERSION`: 2025-12-01-preview로 업데이트
- `DEFAULT_MAX_TOKENS`: 1000 → 4096으로 증가
- `DEFAULT_CONTEXT_WINDOW`: 200,000 토큰 기본값
- `MAX_SUPERVISOR_ROUNDS`: 5 → 10으로 증가
- `MAX_CACHE_SIZE`: 100 → 500으로 증가
- `MAX_MEMORY_TURNS`: 20 → 50으로 증가
- `SESSION_TTL_HOURS`: 24 → 72시간으로 증가
- `ENABLE_STREAMING`: False → True (기본 활성화)
- `ENABLE_PARALLEL_TOOLS`: 병렬 도구 호출 활성화
- `MAX_PARALLEL_TOOL_CALLS`: 최대 5개 동시 호출

### 📚 Documentation
- README.md 전면 업데이트 (2026년 1월 최신 모델 정보)
- 모든 docstring 한글 상세화
- GitHub 오픈소스 파일 추가 (LICENSE, CONTRIBUTING, etc.)

### ✅ Tests
- 79개 테스트 전체 통과 유지

---

## [3.0.0] - 2025-12-01

### 🆕 Added

#### 완전한 모듈화 아키텍처
- 12개 독립 모듈로 분리
- 67개 공개 API export
- 순환 참조 없는 깔끔한 구조

#### Microsoft Multi-Agent Engine 통합
- WebSocket 스트리밍 (WebSocketMessageType)
- MPlan 계획 시스템 (PlanStep, PlanStepStatus)
- ProxyAgent (사용자 명확화)
- RAIValidator (Responsible AI 검증)
- AgentFactory & OrchestrationManager

#### Skills 시스템
- Skill, SkillManager, SkillResource 클래스
- Progressive Disclosure 패턴
- 자동 스킬 활성화

#### Memory Hook Provider
- MemoryHookProvider 클래스
- ConversationMessage 모델
- MemorySessionManager

### 🔧 Changed
- 메인 파일 93.5% 코드 감소 (6,040줄 → 325줄)
- Settings 클래스로 중앙 설정 관리
- FrameworkConfig 데이터클래스 도입

### ✅ Tests
- 79개 단위 테스트 추가
- 100% 모듈 커버리지

---

## [2.0.0] - 2025-06-01

### 🆕 Added
- SupervisorAgent 추가
- CircuitBreaker 패턴 도입
- OpenTelemetry 통합
- GPT-5 시리즈 지원

### 🔧 Changed
- Pydantic v2 마이그레이션
- Python 3.10+ 요구사항

---

## [1.0.0] - 2025-01-01

### 🆕 Added
- 초기 릴리스
- SimpleAgent, RouterAgent
- Graph 워크플로우
- Azure OpenAI 통합
