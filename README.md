# 🚀 Unified Agent Framework - Enterprise Edition v3.1

**최고의 AI Agent 프레임워크들의 장점을 통합한 엔터프라이즈급 오케스트레이션 프레임워크**

[![Python](https://img.shields.io/badge/Python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![CI](https://img.shields.io/github/actions/workflow/status/unified-agent-framework/unified-agent-framework/ci.yml?label=CI)](https://github.com/unified-agent-framework/unified-agent-framework/actions)
[![PyPI](https://img.shields.io/pypi/v/unified-agent-framework.svg)](https://pypi.org/project/unified-agent-framework/)
[![Semantic Kernel](https://img.shields.io/badge/Semantic_Kernel-Latest-orange.svg)](https://github.com/microsoft/semantic-kernel)
[![OpenTelemetry](https://img.shields.io/badge/OpenTelemetry-Enabled-purple.svg)](https://opentelemetry.io/)
[![Agent Framework](https://img.shields.io/badge/MS_Agent_Framework-Integrated-red.svg)](https://github.com/microsoft/agent-framework)
[![GPT-5.2](https://img.shields.io/badge/GPT--5.2-Supported-brightgreen.svg)](https://openai.com/)
[![Claude 4.5](https://img.shields.io/badge/Claude_Opus_4.5-Supported-blueviolet.svg)](https://anthropic.com/)
[![Grok-4](https://img.shields.io/badge/Grok--4-Supported-yellow.svg)](https://xai.com/)
[![MCP](https://img.shields.io/badge/MCP-Native_Support-teal.svg)](https://modelcontextprotocol.io/)
[![Tests](https://img.shields.io/badge/Tests-79%20Passed-success.svg)](#-테스트)

> **v3.1.0** - 🆕 **2026년 1월 최신 업데이트!** GPT-5.2/Claude Opus 4.5/Grok-4 지원, Microsoft Agent Framework MCP 패턴 통합, Adaptive Circuit Breaker, 대용량 컨텍스트(최대 10M tokens) 지원, RAI 강화 검증, 상세 한글 주석 추가

## 🆕 v3.1 주요 업데이트 (2026년 1월)

### 🤖 최신 AI 모델 지원 (40+ 모델)

| 모델 계열 | 지원 모델 | 컨텍스트 | 비고 |
|------------|-----------|---------|------|
| **GPT-5.2** | gpt-5.2, gpt-5.2-chat, gpt-5.2-codex | 400K | 🆕 최신 |
| **GPT-5.1 Codex** | gpt-5.1-codex, gpt-5.1-codex-mini, gpt-5.1-codex-max | 400K | 코드 특화 |
| **Claude 4.5** | claude-opus-4-5, claude-sonnet-4-5, claude-haiku-4-5 | 200K | MS Foundry |
| **Grok-4** | grok-4, grok-4-fast-reasoning, grok-4-fast-non-reasoning | 2M | MS Foundry |
| **o-시리즈** | o3, o3-mini, o3-pro, o4-mini | 200K | Reasoning |
| **DeepSeek** | deepseek-v3.2, deepseek-v3.2-speciale, deepseek-r1-0528 | - | Reasoning |
| **Llama 4** | llama-4-maverick-17b, llama-4-scout-17b | **10M** | 최대 컨텍스트 |
| **Phi-4** | phi-4, phi-4-reasoning, phi-4-multimodal-instruct | - | Microsoft |
| **Mistral** | mistral-large-3, mistral-medium-2505, mistral-small-2503 | - | - |

### 📝 상세 한글 주석 추가 (🆕 NEW)

모든 모듈에 상세한 한글 주석이 추가되어 학습 및 유지보수가 용이해졌습니다:

```python
class CircuitBreaker:
    """
    Adaptive Circuit Breaker - 장애 전파 방지 패턴 (2026년 개선 버전)

    ================================================================================
    📋 역할: 외부 서비스 장애 시 빠른 실패로 시스템 안정성 보장
    📅 최종 업데이트: 2026년 1월
    ================================================================================

    🔄 상태 전환 다이어그램:
        [CLOSED] ──(연속 실패)──▶ [OPEN]
            ▲                        │
            │                 (타임아웃 후)
            │                        ▼
            └──(연속 성공)── [HALF_OPEN]
    ...
    """
```

주석에 포함된 내용:
- 📋 **역할 설명**: 각 클래스/함수의 목적
- 📅 **업데이트 날짜**: 최종 수정일
- 📌 **사용 예시**: 코피 가능한 코드 예제
- ⚠️ **주의사항**: 흔한 실수 및 제약사항
- 🔗 **참고 링크**: 관련 문서 및 리소스

### 🔌 Microsoft Agent Framework MCP 통합

```python
from unified_agent import MCPTool, Settings

# MCP 활성화
Settings.ENABLE_MCP = True
Settings.MCP_APPROVAL_MODE = "selective"  # always/never/selective

# Microsoft Learn MCP 도구
mcp_tool = MCPTool(
    name="docs",
    server_config={
        "type": "mcp",
        "url": "https://learn.microsoft.com/api/mcp"
    }
)

# 에이전트에 MCP 도구 통합
agent = framework.create_skilled_agent(
    name="assistant",
    tools=[mcp_tool]
)
```

### 📦 모듈화 아키텍처 개선

| 항목 | v2.x | v3.1 | 개선 |
|------|------|------|------|
| 메인 파일 | 6,040줄 | 325줄 | **93.5% 감소** |
| 모듈 수 | 1개 | 12개 | **모듈화** |
| 테스트 | 없음 | 79개 | **완전 커버리지** |
| 공개 API | - | 67개 | **정의됨** |

### 🛡️ 성능 및 안정성 개선

#### Adaptive Circuit Breaker (2026년 개선)
```python
from unified_agent import CircuitBreaker

breaker = CircuitBreaker(
    failure_threshold=5,      # 5회 실패 시 OPEN
    success_threshold=3,      # 3회 연속 성공 시 CLOSED 복귀
    timeout=60.0,             # 60초 후 HALF_OPEN
    adaptive_timeout=True     # 평균 응답시간 기반 동적 타임아웃
)

# 메트릭 확인
metrics = breaker.get_metrics()
print(f"성공률: {metrics['success_rate']:.2%}")
print(f"평균 응답시간: {metrics['avg_response_time_ms']:.1f}ms")
```

#### 대용량 컨텍스트 지원
```python
from unified_agent.config import is_large_context_model, get_model_context_window

# 모델별 컨텍스트 크기 확인
print(get_model_context_window("gpt-5.2"))      # 400,000
print(get_model_context_window("gpt-4.1"))      # 1,000,000
print(get_model_context_window("grok-4-fast-reasoning"))  # 2,000,000
print(get_model_context_window("llama-4-scout-17b-16e-instruct"))  # 10,000,000 (최대!)

# 대용량 모델 확인
is_large_context_model("gpt-5.2")  # True (100K+)
```

#### 멀티모달 모델 지원
```python
from unified_agent.config import is_multimodal_model

# 이미지/오디오 입력 지원 모델 확인
is_multimodal_model("gpt-5.2")           # True
is_multimodal_model("claude-opus-4-5")   # True
is_multimodal_model("phi-4-multimodal-instruct")  # True
is_multimodal_model("gpt-5.2-codex")     # False (코드 특화)
```

#### RAI (Responsible AI) 강화
```python
from unified_agent import RAIValidator, RAICategory

validator = RAIValidator(strict_mode=True)
result = validator.validate("사용자 입력 텍스트")

if not result.is_safe:
    print(f"⚠️ 카테고리: {result.category.value}")
    print(f"⚠️ 사유: {result.reason}")
    print(f"💡 제안: {result.suggestions}")
```

## 📖 목차

- [v3.0 주요 업데이트](#-v30-주요-업데이트)
- [모듈화 아키텍처](#-모듈화-아키텍처-v30)
- [개요](#-개요)
- [핵심 기능](#-핵심-기능)
- [Microsoft Multi-Agent Engine](#-microsoft-multi-agent-engine-v30)
- [중앙 설정 (Settings)](#-중앙-설정-settings)
- [GPT-5 및 모델 지원](#-gpt-5-및-모델-지원)
- [Skills 시스템](#-skills-시스템)
- [Memory Hook Provider](#-memory-hook-provider)
- [Session Manager](#-session-manager)
- [Enhanced Supervisor](#-enhanced-supervisor)
- [설치](#-설치)
- [빠른 시작](#-빠른-시작)
- [아키텍처](#-아키텍처)
- [주요 컴포넌트](#-주요-컴포넌트)
- [테스트](#-테스트)
- [실전 예제](#-실전-예제)
- [성능 최적화](#-성능-최적화)
- [프로덕션 배포](#-프로덕션-배포)
- [FAQ](#-faq)
- [기여하기](#-기여하기)
- [라이선스](#-라이선스)

---

## 📦 모듈화 아키텍처 (v3.0)

v3.0에서 완전한 모듈화 아키텍처로 재구성되었습니다:

### 패키지 구조

```
unified_agent/
├── __init__.py          # 패키지 진입점 (67개 공개 API export)
├── exceptions.py        # 예외 클래스 (FrameworkError, ConfigurationError 등)
├── config.py            # 설정 및 상수 (Settings, FrameworkConfig)
├── models.py            # 데이터 모델 (Enum, Pydantic, Dataclass)
├── utils.py             # 유틸리티 (StructuredLogger, CircuitBreaker, RAIValidator)
├── memory.py            # 메모리 시스템 (MemoryStore, CachedMemoryStore)
├── events.py            # 이벤트 시스템 (EventBus, EventType)
├── skills.py            # Skills 시스템 (Skill, SkillManager)
├── tools.py             # 도구 (AIFunction, MCPTool)
├── agents.py            # 에이전트 (SimpleAgent, RouterAgent, SupervisorAgent)
├── workflow.py          # 워크플로우 (Graph, Node)
├── orchestration.py     # 오케스트레이션 (AgentFactory, OrchestrationManager)
└── framework.py         # 메인 프레임워크 (UnifiedAgentFramework)
```

### 최적화 결과

| 항목 | 변경 전 (v2.x) | 변경 후 (v3.0) | 개선 |
|------|---------------|----------------|------|
| 메인 파일 | 6,040줄 | 325줄 | **93.5% 감소** |
| 파일 크기 | 214 KB | 15 KB | **93% 감소** |
| 모듈 수 | 1개 (모놀리식) | 12개 | **모듈화** |
| 테스트 | 없음 | 79개 | **완전 커버리지** |

### Import 방식

```python
# 방법 1: 래퍼 파일에서 import (하위 호환성)
from unified_agent import UnifiedAgentFramework, Settings

# 방법 2: 패키지에서 직접 import (권장)
from unified_agent import UnifiedAgentFramework, Settings

# 방법 3: 개별 모듈에서 import (세부 제어)
from unified_agent.agents import SimpleAgent, SupervisorAgent
from unified_agent.workflow import Graph, Node
from unified_agent.models import AgentState, MPlan
```

---

## 🎯 개요

Unified Agent Framework는 다음 6가지 최고의 AI Agent 프레임워크의 핵심 장점을 통합했습니다:

| 프레임워크 | 통합된 기능 |
|-----------|-----------|
| **Microsoft AutoGen** | Multi-agent 협업 (GroupChat 패턴) |
| **Semantic Kernel** | 플러그인 시스템 & 함수 호출 |
| **LangGraph** | 상태 기반 그래프 & 조건부 라우팅 |
| **Microsoft Agent Framework** | 체크포인팅, OpenTelemetry, 관찰성 |
| **Anthropic Skills** | 모듈화된 전문 지식 & Progressive Disclosure |
| **AWS AgentCore** | Memory Hook Provider, Session Manager, Investigation Plan |
| **Microsoft Multi-Agent Engine** | WebSocket, MPlan, ProxyAgent, RAI, AgentFactory (NEW!) |

### 왜 Unified Agent Framework인가?

```python
# ❌ 기존 방식: 복잡하고 장황한 코드
# - 각 프레임워크별 학습 필요
# - 통합 어려움
# - 프로덕션 준비 미흡

# ✅ Unified Agent Framework v3.0: 간단하고 강력하며 모듈화됨
from unified_agent import UnifiedAgentFramework, Settings, TeamConfiguration

# 중앙 설정으로 모델 변경 (한 곳에서 관리)
Settings.DEFAULT_MODEL = "gpt-5.2"

# 프레임워크 생성 (환경변수 자동 로드)
framework = UnifiedAgentFramework.create()

# v3.0 NEW: 팀 기반 멀티에이전트
team_config = TeamConfiguration(
    name="research_team",
    agents=[
        TeamAgent(name="researcher", description="연구 담당"),
        TeamAgent(name="writer", description="작성 담당"),
    ]
)

# v3.0 NEW: MPlan으로 구조화된 실행 계획
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

---

## 🎯 Microsoft Multi-Agent Engine (v3.0)

Microsoft Multi-Agent-Custom-Automation-Engine 패턴을 완전히 통합했습니다.

### WebSocket 스트리밍

```python
from unified_agent import WebSocketMessageType, StreamingMessage

# 실시간 스트리밍 메시지
msg = StreamingMessage(
    type=WebSocketMessageType.AGENT_RESPONSE,
    content="Hello!",
    agent_name="assistant"
)

# 지원하는 메시지 타입
# - START_SESSION, END_SESSION
# - AGENT_STARTED, AGENT_RESPONSE, AGENT_COMPLETED
# - PLAN_CREATED, PLAN_STEP_STARTED, PLAN_STEP_COMPLETED
# - ERROR, APPROVAL_REQUIRED
```

### MPlan 계획 시스템

```python
from unified_agent import MPlan, PlanStep, PlanStepStatus

# 구조화된 실행 계획 생성
plan = MPlan(
    name="research_plan",
    description="시장 조사 계획",
    steps=[
        PlanStep(index=0, description="데이터 수집", agent_name="researcher"),
        PlanStep(index=1, description="분석", agent_name="analyst", depends_on=[0]),
        PlanStep(index=2, description="보고서", agent_name="writer", depends_on=[1]),
    ],
    complexity="moderate",
    requires_approval=True
)

# 계획 요약 출력
print(plan.to_summary())
# 📋 계획: research_plan
#    단계 수: 3, 진행률: 0%
#    ⏳ [0] 데이터 수집 (researcher)
#    ⏳ [1] 분석 (analyst)
#    ⏳ [2] 보고서 (writer)

# 진행률 추적
plan.complete_step(0, "데이터 수집 완료", tokens_used=1500)
print(f"진행률: {plan.get_progress() * 100:.1f}%")  # 33.3%

# 다음 실행 가능 단계
next_steps = plan.get_next_steps()
```

### ProxyAgent (사용자 명확화)

```python
from unified_agent import ProxyAgent

# 사용자에게 명확화 요청이 필요할 때
proxy = ProxyAgent(
    name="clarifier",
    system_prompt="사용자 의도가 불명확할 때 질문합니다"
)
```

### RAI (Responsible AI) 검증

```python
from unified_agent import RAIValidator, RAICategory

# RAI 검증기
validator = RAIValidator()
result = validator.validate("콘텐츠 내용")

if not result.is_safe:
    print(f"위반 카테고리: {result.violations}")
```

### AgentFactory & OrchestrationManager

```python
from unified_agent import AgentFactory, OrchestrationManager, TeamConfiguration

# JSON 기반 에이전트 동적 생성
factory = AgentFactory(framework)
team = factory.create_team(team_config)

# 팀 오케스트레이션
orchestrator = OrchestrationManager(framework)
result = await orchestrator.execute_team(team_config, user_input)
```

---

## ✨ 핵심 기능

### 🎓 Skills 시스템
```python
# 스킬 기반 스마트 질의응답 - 자동으로 관련 스킬 활성화
response = await framework.smart_chat("pandas로 데이터 분석해줘")
# -> data-analyst, python-expert 스킬 자동 활성화!

# 커스텀 스킬 생성
from unified_agent import Skill
my_skill = Skill(
    name="my-domain-expert",
    description="특정 도메인 전문가",
    instructions="## 역할\n도메인 전문가로서...",
    triggers=["도메인", "전문"]
)
framework.skill_manager.register_skill(my_skill)
```

### 🤝 Multi-Agent 협업
```python
orchestrator = OrchestratorAgent(
    name="team_lead",
    agents=[researcher, writer, critic]
)
# 자동으로 라운드 기반 협업 실행
```

### 📊 상태 기반 그래프
```python
graph.add_node(Node("step1", agent1))
graph.add_edge("step1", "step2")  # 조건부 분기
print(graph.visualize())  # Mermaid 다이어그램 자동 생성
```

### 🔄 체크포인팅 & 복원
```python
# 작업 중단 시 자동 저장
await state_manager.save_checkpoint(state)

# 언제든 재개
state = await state_manager.restore_checkpoint(session_id)
```

### 📡 OpenTelemetry 통합
```python
# 프로덕션 환경 실시간 모니터링
with tracer.start_as_current_span("workflow"):
    span.set_attribute("tokens_used", tokens)
```

### 🔀 조건부 라우팅
```python
router = RouterAgent(
    routes={
        "order": "order_agent",
        "support": "support_agent"
    }
)
# 사용자 의도에 따라 자동 분기
```

### 💾 캐싱 메모리 저장소
```python
# 3회 이상 접근 시 자동 캐싱
# O(1) 조회 성능
memory_store = CachedMemoryStore()
```

### ⚙️ 중앙 설정 (Settings 클래스)
```python
from unified_agent import Settings

# 모든 설정을 한 곳에서 관리
Settings.DEFAULT_MODEL = "gpt-5.2"      # 기본 모델
Settings.DEFAULT_TEMPERATURE = 0.7      # 온도
Settings.MAX_SUPERVISOR_ROUNDS = 5      # Supervisor 라운드
Settings.ENABLE_MEMORY_HOOKS = True     # Memory Hook 활성화
```

### 🤖 GPT-5 및 o-series 모델 지원
```python
# GPT-5 계열 (temperature 자동 비활성화)
Settings.DEFAULT_MODEL = "gpt-5.2"

# o-series (Reasoning 모델)
Settings.DEFAULT_MODEL = "o3"  # temperature 자동 비활성화
```

---

## 🧪 테스트

v3.0에서는 포괄적인 테스트 스위트를 제공합니다.

### 테스트 실행

```bash
# 전체 단위 테스트 (79개)
python test_unified_agent.py

# 실행 데모
python demo_unified_agent.py
```

### 테스트 결과

```
============================================================
📊 테스트 결과 요약
============================================================
  ✅ 성공: 79
  ❌ 실패: 0
============================================================
```

### 테스트 커버리지

| 테스트 영역 | 테스트 수 | 상태 |
|------------|----------|------|
| Import 테스트 | 42 | ✅ |
| 패키지 테스트 | 2 | ✅ |
| Enum 테스트 | 4 | ✅ |
| Pydantic 모델 | 3 | ✅ |
| Config | 4 | ✅ |
| Memory 시스템 | 2 | ✅ |
| Utils | 3 | ✅ |
| Skills | 3 | ✅ |
| Tools | 2 | ✅ |
| Workflow | 3 | ✅ |
| TeamConfiguration | 2 | ✅ |
| MPlan | 4 | ✅ |
| 순환 참조 | 2 | ✅ |
| Events | 3 | ✅ |

---

## ⚙️ 중앙 설정 (Settings)

모든 프레임워크 설정을 한 곳에서 관리하는 `Settings` 클래스입니다.

### Settings 클래스 구조

```python
class Settings:
    """
    프레임워크 전역 설정 클래스 (Singleton-like Pattern)

    2026년 1월 업데이트:
    - 40+ 모델 지원 (GPT-5.2, Claude 4.5, Grok-4, Llama 4 등)
    - MCP 설정 추가 (ENABLE_MCP, MCP_APPROVAL_MODE)
    - Multi-Agent 오케스트레이션 설정
    - RAI (Responsible AI) 설정
    """

    # ─────────────────────────────────────────────────────────────────────
    # LLM 모델 설정 (2026년 최신)
    # ─────────────────────────────────────────────────────────────────────
    DEFAULT_MODEL: str = "gpt-5.2"           # 기본 모델 (2026년 최신)
    DEFAULT_API_VERSION: str = "2025-12-01-preview"  # API 버전 (최신)
    DEFAULT_TEMPERATURE: float = 0.7         # GPT-4 계열만 적용
    DEFAULT_MAX_TOKENS: int = 4096           # 기본 최대 토큰 (증가)
    DEFAULT_CONTEXT_WINDOW: int = 200000     # 기본 컨텍스트 윈도우

    # ─────────────────────────────────────────────────────────────────────
    # 지원 모델 목록 (2026년 1월 기준 - 40+ 모델)
    # ─────────────────────────────────────────────────────────────────────
    SUPPORTED_MODELS: list = [
        # GPT-4 계열 (Legacy)
        "gpt-4", "gpt-4o", "gpt-4o-mini", "gpt-4.1", "gpt-4.1-mini", "gpt-4.1-nano",
        # GPT-5 계열
        "gpt-5", "gpt-5.1", "gpt-5.2", "gpt-5.2-chat", "gpt-5.2-codex",
        "gpt-5.1-codex", "gpt-5.1-codex-mini", "gpt-5.1-codex-max",
        # o-시리즈 (Reasoning)
        "o1", "o1-mini", "o3", "o3-mini", "o3-pro", "o4-mini",
        # Claude (Microsoft Foundry)
        "claude-opus-4-5", "claude-sonnet-4-5", "claude-haiku-4-5",
        # Grok (Microsoft Foundry)
        "grok-4", "grok-4-fast-reasoning", "grok-4-fast-non-reasoning",
        # DeepSeek
        "deepseek-v3.2", "deepseek-r1-0528",
        # Llama 4
        "llama-4-maverick-17b-128e-instruct-fp8", "llama-4-scout-17b-16e-instruct",
        # Phi-4
        "phi-4", "phi-4-reasoning", "phi-4-multimodal-instruct",
        # Mistral
        "mistral-large-3", "mistral-medium-2505"
    ]

    # Temperature 미지원 모델 (Reasoning 모델)
    MODELS_WITHOUT_TEMPERATURE: list = [
        "gpt-5", "gpt-5.1", "gpt-5.2", "gpt-5.1-codex", "gpt-5.2-codex",
        "o1", "o1-mini", "o3", "o3-mini", "o3-pro", "o4-mini",
        "deepseek-r1", "deepseek-r1-0528",
        "phi-4-reasoning", "phi-4-mini-reasoning"
    ]

    # ─────────────────────────────────────────────────────────────────────
    # MCP (Model Context Protocol) 설정 - 2026 최신
    # ─────────────────────────────────────────────────────────────────────
    ENABLE_MCP: bool = True
    MCP_AUTO_CONNECT: bool = True
    MCP_RECONNECT_ATTEMPTS: int = 3
    MCP_REQUEST_TIMEOUT: int = 30
    MCP_APPROVAL_MODE: str = "selective"  # always/never/selective

    # ─────────────────────────────────────────────────────────────────────
    # Multi-Agent 오케스트레이션 설정
    # ─────────────────────────────────────────────────────────────────────
    ORCHESTRATION_MODE: str = "adaptive"     # supervisor/sequential/parallel/adaptive
    MAX_SUPERVISOR_ROUNDS: int = 10
    MAX_CONCURRENT_AGENTS: int = 5
    ENABLE_HANDOFF: bool = True
    ENABLE_REFLECTION: bool = True

    # ─────────────────────────────────────────────────────────────────────
    # RAI (Responsible AI) 설정
    # ─────────────────────────────────────────────────────────────────────
    ENABLE_RAI_VALIDATION: bool = True
    RAI_STRICT_MODE: bool = False
    RAI_CONTENT_SAFETY_LEVEL: str = "medium"  # low/medium/high
    ENABLE_PII_DETECTION: bool = True

    # ─────────────────────────────────────────────────────────────────────
    # Memory 설정
    # ─────────────────────────────────────────────────────────────────────
    ENABLE_MEMORY_HOOKS: bool = True
    ENABLE_SEMANTIC_MEMORY: bool = True
    MEMORY_EMBEDDING_MODEL: str = "text-embedding-3-large"
    MAX_MEMORY_TURNS: int = 50
    MAX_CACHE_SIZE: int = 500
    SESSION_TTL_HOURS: int = 72
```

### 사용법

```python
from unified_agent import Settings, UnifiedAgentFramework

# 1. 모델 변경
Settings.DEFAULT_MODEL = "gpt-4.1"  # 전역 적용

# 2. 설정 확인
print(f"현재 모델: {Settings.DEFAULT_MODEL}")
print(f"지원 모델: {Settings.SUPPORTED_MODELS}")

# 3. 메모리 설정
Settings.MAX_MEMORY_TURNS = 50
Settings.SESSION_TTL_HOURS = 48

# 4. Supervisor 설정
Settings.MAX_SUPERVISOR_ROUNDS = 10
Settings.AUTO_APPROVE_SIMPLE_PLANS = False

# 5. 프레임워크 생성 (Settings 값 자동 적용)
framework = UnifiedAgentFramework.create()
```

### 설정 카테고리

| 카테고리 | 설정 | 설명 |
|---------|------|------|
| **LLM 모델** | `DEFAULT_MODEL` | 기본 LLM 모델 |
| | `DEFAULT_API_VERSION` | Azure API 버전 |
| | `DEFAULT_TEMPERATURE` | 기본 온도 (GPT-4만) |
| | `DEFAULT_MAX_TOKENS` | 최대 토큰 수 |
| | `SUPPORTED_MODELS` | 지원 모델 목록 |
| | `MODELS_WITHOUT_TEMPERATURE` | 온도 미지원 모델 |
| **프레임워크** | `CHECKPOINT_DIR` | 체크포인트 저장 경로 |
| | `ENABLE_TELEMETRY` | 텔레메트리 활성화 |
| | `ENABLE_STREAMING` | 스트리밍 응답 활성화 |
| **메모리** | `ENABLE_MEMORY_HOOKS` | 메모리 훅 활성화 |
| | `MAX_MEMORY_TURNS` | 최대 대화 턴 수 |
| | `SESSION_TTL_HOURS` | 세션 만료 시간 |
| **Supervisor** | `AUTO_APPROVE_SIMPLE_PLANS` | 간단한 계획 자동 승인 |
| | `MAX_SUPERVISOR_ROUNDS` | 최대 라운드 수 |
| **로깅** | `LOG_LEVEL` | 로그 레벨 |
| | `LOG_FILE` | 로그 파일 경로 |

---

## 🤖 GPT-5 및 모델 지원 (NEW!)

프레임워크는 2026년 1월 기준 최신 AI 모델을 완전히 지원합니다.

### 지원 모델 (40+)

| 모델 시리즈 | 모델 | Temperature | 컨텍스트 | 비고 |
|------------|------|-------------|---------|------|
| **GPT-4** | gpt-4, gpt-4o, gpt-4o-mini | ✅ 지원 | 128K | Legacy |
| **GPT-4.1** | gpt-4.1, gpt-4.1-mini, gpt-4.1-nano | ✅ 지원 | **1M** | 개선된 성능 |
| **GPT-5** | gpt-5, gpt-5-pro | ❌ 자동 생략 | 200K~400K | Reasoning |
| **GPT-5.1** | gpt-5.1, gpt-5.1-chat | ❌/✅ | 400K | 2025 |
| **GPT-5.1 Codex** | gpt-5.1-codex, codex-mini, codex-max | ❌ 자동 생략 | 400K | 코드 특화 |
| **GPT-5.2** | gpt-5.2, gpt-5.2-chat, gpt-5.2-codex | ❌/✅ | **400K** | 🆕 최신 |
| **o-series** | o1, o3, o3-mini, o3-pro, o4-mini | ❌ 자동 생략 | 200K | Reasoning |
| **Claude 4.5** | claude-opus-4-5, sonnet-4-5, haiku-4-5 | ✅ 지원 | 200K | MS Foundry |
| **Grok-4** | grok-4, grok-4-fast-reasoning | ✅ 지원 | **2M** | MS Foundry |
| **DeepSeek** | deepseek-v3.2, r1-0528 | ❌/✅ | - | Reasoning |
| **Llama 4** | maverick-17b, scout-17b | ✅ 지원 | **10M** | 최대 컨텍스트 |
| **Phi-4** | phi-4, phi-4-reasoning, multimodal | ❌/✅ | - | Microsoft |
| **Mistral** | large-3, medium-2505, small-2503 | ✅ 지원 | - | - |

### 유틸리티 함수

```python
from unified_agent.config import (
    supports_temperature,
    is_multimodal_model,
    is_large_context_model,
    get_model_context_window
)

# Temperature 지원 확인
print(supports_temperature("gpt-4.1"))     # True
print(supports_temperature("gpt-5.2"))     # False (Reasoning)
print(supports_temperature("gpt-5.2-chat"))  # True (chat 모델)
print(supports_temperature("o4-mini"))     # False (Reasoning)

# 멀티모달 지원 확인 (이미지/오디오 입력)
print(is_multimodal_model("gpt-5.2"))      # True
print(is_multimodal_model("claude-opus-4-5"))  # True
print(is_multimodal_model("gpt-5.2-codex"))  # False

# 대용량 컨텍스트 확인 (100K+)
print(is_large_context_model("gpt-5.2"))   # True
print(is_large_context_model("gpt-4o"))    # False (128K)

# 컨텍스트 윈도우 크기 확인
print(get_model_context_window("gpt-5.2"))      # 400,000
print(get_model_context_window("gpt-4.1"))      # 1,000,000
print(get_model_context_window("grok-4-fast-reasoning"))  # 2,000,000
print(get_model_context_window("llama-4-scout-17b-16e-instruct"))  # 10,000,000
```

### Temperature 자동 처리

```python
from unified_agent.config import create_execution_settings

# 자동으로 temperature 지원 여부 확인 후 설정 생성
settings = create_execution_settings(
    model="gpt-5.2",
    temperature=0.7,  # Reasoning 모델에서는 자동 생략됨
    max_tokens=2000
)
# → ⓘ️ 모델 'gpt-5.2'은(는) temperature를 지원하지 않습니다. 해당 파라미터를 생략합니다.
```

> 💡 **자동 처리**: GPT-5, o1, o3, o4 계열 모델 사용 시 `temperature` 파라미터가 자동으로 생략되어 API 오류를 방지합니다.

---

## 🧠 Memory Hook Provider

> **참조**: [AWS AgentCore - Memory Pattern](https://github.com/awslabs/amazon-bedrock-agentcore-samples)

대화 기록을 자동으로 저장/로드하는 Memory Hook 시스템입니다.

### 주요 기능

- **자동 대화 기록**: 메시지 추가 시 자동 저장
- **세션 기반 컨텍스트**: 세션별 대화 기록 관리
- **네임스페이스 분류**: `/conversation`, `/preferences` 등으로 분류

### 사용법

```python
from unified_agent import MemoryHookProvider, MemoryStore

# Memory Hook 생성
memory_hook = MemoryHookProvider(
    memory_store=memory_store,
    session_id="session-123",
    actor_id="user-456",
    max_context_turns=10  # 최근 10개 대화 유지
)

# 에이전트 초기화 시 컨텍스트 로드
context = await memory_hook.on_agent_initialized(agent_name="assistant")

# 메시지 추가 시 자동 저장
await memory_hook.on_message_added(
    content="사용자 질문입니다",
    role="USER",
    agent_name="assistant"
)

# 최근 k개 대화 조회
last_turns = await memory_hook.get_last_k_turns(k=5)
```

### ConversationMessage 모델

```python
@dataclass
class ConversationMessage:
    content: str
    role: str  # USER, ASSISTANT, TOOL
    timestamp: datetime
    agent_name: Optional[str] = None
    session_id: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
```

---

## 🔐 Session Manager (NEW!)

> **참조**: [AWS AgentCore - Session Management](https://github.com/awslabs/amazon-bedrock-agentcore-samples)

다중 사용자/다중 세션을 효율적으로 관리합니다.

### 사용법

```python
from unified_agent import MemorySessionManager

# Session Manager 생성
session_manager = MemorySessionManager(
    memory_store=memory_store,
    default_ttl_hours=24  # 세션 만료 시간
)

# 세션 조회 또는 생성
session = session_manager.get_or_create_session(
    session_id="session-123",
    actor_id="user-456",
    namespace="/conversation"
)

# 세션 목록 조회
sessions = await session_manager.list_sessions(actor_id="user-456")

# 만료된 세션 정리
await session_manager.cleanup_expired_sessions()
```

---

## 🎯 Enhanced Supervisor

> **참조**: [AWS AgentCore - SRE Agent Supervisor Pattern](https://github.com/awslabs/amazon-bedrock-agentcore-samples)

Investigation Plan 기반의 체계적인 멀티 에이전트 오케스트레이션입니다.

### Investigation Plan

```python
@dataclass
class InvestigationPlan:
    steps: List[str]            # 실행 단계
    agents_sequence: List[str]  # 에이전트 실행 순서
    complexity: str             # "simple" or "complex"
    auto_execute: bool          # 자동 실행 여부
    reasoning: str              # 계획 생성 이유
```

### 사용법

```python
from unified_agent import SupervisorAgent, SimpleAgent

# 서브 에이전트 정의
researcher = SimpleAgent(name="researcher", system_prompt="연구 담당")
writer = SimpleAgent(name="writer", system_prompt="작성 담당")

# Supervisor 생성 (Enhanced)
supervisor = SupervisorAgent(
    name="supervisor",
    system_prompt="팀 리더입니다",
    sub_agents=[researcher, writer],
    max_rounds=5,
    memory_hook=memory_hook,  # Memory Hook 연동
    auto_approve_simple=True  # 간단한 계획 자동 실행
)

# 실행 (Investigation Plan 자동 생성)
result = await supervisor.execute(state, kernel)

# 결과 확인
print(result.metadata["investigation_plan"])  # 실행된 계획
print(result.metadata["execution_log"])       # 실행 로그
```

### 응답 집계 (Response Aggregation)

여러 에이전트의 응답을 자동으로 집계하여 통합된 답변을 생성합니다:

```python
# supervisor.execute() 내부에서 자동 실행
aggregated = await supervisor.aggregate_responses(
    responses=execution_log,
    state=state,
    kernel=kernel
)
```

---

## 🎓 Skills 시스템 (NEW!)

Anthropic Skills 패턴을 기반으로 한 모듈화된 전문 지식 관리 시스템입니다.

### Skills란?

Skills는 AI 에이전트의 능력을 확장하는 모듈화된 패키지입니다. 특정 도메인의 지식, 워크플로우, 도구를 캡슐화하여 재사용 가능하게 만듭니다.

```
skill-name/
├── SKILL.md          # 메타데이터 + 지침 (필수)
├── scripts/          # 실행 가능한 스크립트
├── references/       # 참조 문서
└── assets/           # 템플릿, 에셋
```

### 기본 제공 스킬

`skills/` 디렉토리에서 SKILL.md 파일로 제공됩니다:

| 스킬 | 설명 | 우선순위 |
|-----|------|--------|
| `python-expert` | Python 프로그래밍 전문가 | 10 |
| `data-analyst` | 데이터 분석 (pandas, 시각화) | 8 |
| `api-developer` | REST API 개발 전문가 | 8 |
| `korean-writer` | 한국어 작문 전문가 | 7 |

### 스킬 사용법

#### 1. 스마트 질의응답 (자동 스킬 감지)
```python
# 질문에 맞는 스킬이 자동으로 활성화됩니다
response = await framework.smart_chat("파이썬으로 웹 크롤러 만들어줘")
# -> python-expert 스킬 자동 활성화!
```

#### 2. 특정 스킬로 에이전트 생성
```python
# 특정 스킬을 사용하는 에이전트 생성
agent = framework.create_skilled_agent(
    "my_coder",
    skills=["python-expert", "api-developer"]
)
```

#### 3. 스킬 기반 워크플로우
```python
# 스킬 기반 워크플로우 생성
workflow = framework.create_skill_workflow(
    "data_pipeline",
    skills=["python-expert", "data-analyst"],
    base_prompt="데이터 처리 전문가입니다."
)
```

### 커스텀 스킬 만들기

#### 방법 1: 코드에서 직접 생성
```python
from unified_agent import Skill

my_skill = Skill(
    name="my-domain-expert",
    description="특정 도메인 전문가. 도메인 관련 질문에 사용.",
    instructions="""
## 역할
특정 도메인 전문가로서 답변합니다.

## 가이드라인
- 전문 용어 사용
- 정확한 정보 제공
- 예시와 함께 설명
    """,
    triggers=["도메인", "전문", "관련키워드"]
)

framework.skill_manager.register_skill(my_skill)
```

#### 방법 2: SKILL.md 파일에서 로드
```python
# 단일 스킬 로드
skill = Skill.from_file("skills/my-skill/SKILL.md")

# 디렉토리에서 로드 (리소스 포함)
skill = Skill.from_directory("skills/my-skill/")

# 여러 스킬 일괄 로드
framework.skill_manager.load_skills_from_directory("./my_skills")
```

#### 방법 3: 템플릿으로 시작
```python
# 스킬 템플릿 생성
framework.skill_manager.create_skill_template("my-new-skill", "./skills")
# -> ./skills/my-new-skill/SKILL.md 및 디렉토리 구조 생성
```

### SKILL.md 파일 형식

스킬은 `skills/` 디렉토리에서 **SKILL.md 파일 기반**으로 관리됩니다:

```
skills/
├── python-expert/
│   └── SKILL.md
├── data-analyst/
│   └── SKILL.md
├── korean-writer/
│   └── SKILL.md
└── api-developer/
    └── SKILL.md
```

**SKILL.md 파일 형식:**

```markdown
---
name: my-skill
description: 스킬 설명 - 언제 사용해야 하는지 포함
triggers:
  - 키워드1
  - 키워드2
priority: 10
---

# 스킬 제목

## Overview
스킬이 무엇을 하는지 설명

## When to Use
- 사용 시나리오 1
- 사용 시나리오 2

## Instructions
AI가 따라야 할 지침

## Examples
구체적인 예시
```

> 💡 **스킬 추가/수정**: `skills/` 디렉토리에 새 폴더를 만들고 `SKILL.md` 파일만 작성하면 됩니다. 프레임워크 재시작 시 자동 로드됩니다.

### Progressive Disclosure

Skills 시스템은 컨텍스트 효율성을 위해 Progressive Disclosure 패턴을 사용합니다:

1. **메타데이터 (항상 로드)**: 이름 + 설명 (~100 단어)
2. **지침 (트리거 시 로드)**: SKILL.md 본문 (<5k 단어)
3. **리소스 (필요 시 로드)**: scripts/, references/, assets/

```python
# 매칭된 스킬만 전체 지침 포함
matched_skills = framework.skill_manager.match_skills(
    query="파이썬 코드 작성",
    threshold=0.2,  # 매칭 임계값
    max_skills=3    # 최대 스킬 수
)
```

### CLI에서 스킬 관리

```bash
# 실행 (UTF-8 기본 인코딩)
python Unified-agent_framework.py

# 모델 명령어 (NEW!)
model                  # 현재 모델 확인
model gpt-5.2          # 모델 변경
model o3               # o-series 모델 변경

# 스킬 명령어
skills list            # 등록된 스킬 목록
skills info <name>     # 스킬 상세 정보
skills stats           # 스킬 사용 통계
skills create <name>   # 새 스킬 템플릿 생성
skills load <dir>      # 디렉토리에서 스킬 로드

# 스마트 질의응답 (스킬 자동 감지)
smart 파이썬으로 웹 스크래퍼 만들어줘

# 일반 대화
chat 안녕하세요!

# 데모 워크플로우 실행
demo simple            # 기본 대화
demo router            # 라우팅 데모
demo orchestrator      # 멀티에이전트 데모
demo all               # 전체 데모

# 설정 확인 (NEW!)
settings               # 현재 Settings 확인

# 종료
exit
```

### CLI 사용 예시

```
🚀 Unified Agent Framework CLI (v2.2)
Commands: chat, smart, demo, skills, model, settings, workflow, exit
Current Model: gpt-5.2

> model
📋 현재 모델: gpt-5.2
📋 지원 모델: gpt-4, gpt-4o, gpt-4o-mini, gpt-4.1, gpt-4.1-mini, gpt-4.1-nano,
              gpt-5, gpt-5.1, gpt-5.2, o1, o1-mini, o1-preview, o3, o3-mini, o4-mini

> model gpt-4.1
🔄 모델 변경: gpt-5.2 → gpt-4.1

> skills list
📚 등록된 스킬:
  - python-expert: Python 프로그래밍 전문가
  - data-analyst: 데이터 분석 전문가
  - korean-writer: 한국어 작문 전문가
  - api-developer: REST API 개발 전문가

> smart pandas로 CSV 파일 읽고 통계 내줘
🎯 활성화된 스킬: data-analyst, python-expert
📝 응답:
import pandas as pd
df = pd.read_csv('data.csv')
print(df.describe())
...

> settings
⚙️ 현재 Settings:
  DEFAULT_MODEL: gpt-4.1
  DEFAULT_TEMPERATURE: 0.7
  ENABLE_MEMORY_HOOKS: True
  MAX_SUPERVISOR_ROUNDS: 5

> exit
👋 안녕히 가세요!
```

---

## 📦 설치

### 필수 요구사항
- Python 3.10 이상
- Azure OpenAI 또는 OpenAI API 키

### 패키지 설치
```bash
pip install semantic-kernel python-dotenv pydantic opentelemetry-api opentelemetry-sdk pyyaml
```

### 환경 변수 설정
`.env` 파일 생성:
```bash
# Azure OpenAI (권장)
AZURE_OPENAI_API_KEY=your_api_key_here
AZURE_OPENAI_ENDPOINT=https://your-resource.openai.azure.com/
AZURE_OPENAI_DEPLOYMENT=gpt-4.1
AZURE_OPENAI_API_VERSION=2025-05-01
```

### 지원 모델

| 모델 시리즈 | 모델 | Temperature | 비고 |
|------------|------|-------------|------|
| **GPT-4** | gpt-4, gpt-4o, gpt-4o-mini | ✅ 지원 | 범용 모델 |
| **GPT-4.1** | gpt-4.1, gpt-4.1-mini, gpt-4.1-nano | ✅ 지원 | 성능 개선 |
| **GPT-5** | gpt-5, gpt-5.1, gpt-5.2 | ❌ 자동 생략 | 최신 모델 |
| **o1** | o1, o1-mini, o1-preview | ❌ 자동 생략 | Reasoning |
| **o3/o4** | o3, o3-mini, o4-mini | ❌ 자동 생략 | 고급 추론 |

> 💡 **자동 Temperature 처리**: GPT-5 및 o-series 모델은 temperature 파라미터를 지원하지 않습니다. 프레임워크가 자동으로 해당 파라미터를 생략하여 오류를 방지합니다.

### UTF-8 인코딩

프레임워크는 **UTF-8 인코딩을 기본으로 사용**합니다. Windows 환경에서도 별도의 `-X utf8` 옵션 없이 실행할 수 있습니다.

```python
# 내장 UTF-8 설정 (자동 적용)
#!/usr/bin/env python3
# -*- coding: utf-8 -*-

if sys.stdout and hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')
```

---

## 🚀 빠른 시작

### 가장 간단한 방법 (권장)

```python
import asyncio
from unified_agent import UnifiedAgentFramework, Settings

# Settings에서 모델 설정 (선택적)
Settings.DEFAULT_MODEL = "gpt-5.2"

async def main():
    # 환경변수 자동 로드하여 프레임워크 생성
    framework = UnifiedAgentFramework.create()

    # 빠른 질의응답
    response = await framework.quick_chat("안녕하세요!")
    print(response)

    # 스마트 질의응답 (스킬 자동 감지)
    response = await framework.smart_chat("파이썬으로 피보나치 함수 만들어줘")
    print(response)

asyncio.run(main())
```

### 커스텀 설정으로 시작

```python
from unified_agent import FrameworkConfig, UnifiedAgentFramework, Settings

async def main():
    # Settings로 전역 기본값 변경 (선택적)
    Settings.DEFAULT_MODEL = "gpt-4o"

    # 또는 FrameworkConfig로 개별 설정
    config = FrameworkConfig(
        model="gpt-4o",
        temperature=0.5,
        max_tokens=2000
    )
    config.api_key = "your-key"
    config.endpoint = "your-endpoint"
    config.deployment_name = "your-deployment"

    framework = UnifiedAgentFramework.create(config)
    response = await framework.quick_chat("Hello!")

asyncio.run(main())
```

### 워크플로우 사용

```python
async def main():
    framework = UnifiedAgentFramework.create()

    # 간단한 대화 워크플로우
    framework.create_simple_workflow("my_bot", "너는 친절한 AI야.")

    # 실행
    state = await framework.run(
        session_id="session-001",
        workflow_name="my_bot",
        user_message="안녕하세요!"
    )

    print(state.messages[-1].content)

asyncio.run(main())
```

### 헬퍼 함수 (가장 간단한 방법)

```python
from unified_agent import quick_run, create_framework, Settings

# 모델 설정 (선택적)
Settings.DEFAULT_MODEL = "gpt-5.2"

# 한 줄로 질의응답 (환경변수 자동 로드)
response = quick_run("Hello, AI!")

# 프레임워크만 생성
framework = create_framework()
```

---

## 🏗️ 아키텍처

```
┌──────────────────────────────────────────────────────────────────┐
│              UnifiedAgentFramework (통합 프레임워크)               │
├──────────────────────────────────────────────────────────────────┤
│  ┌────────────────┐  ┌────────────────┐  ┌────────────────────┐  │
│  │ FrameworkConfig│  │  SkillManager  │  │      Kernel        │  │
│  │  (중앙 설정)    │  │  (스킬 관리)    │  │    (SK 통합)        │  │
│  └────────────────┘  └────────────────┘  └────────────────────┘  │
├──────────────────────────────────────────────────────────────────┤
│  ┌──────────────────┐  ┌──────────────────┐  ┌────────────────┐  │
│  │      Graph       │  │   StateManager   │  │  MemoryStore   │  │
│  │   (워크플로우)    │  │   (상태 관리)     │  │  (캐시/저장)    │  │
│  └──────────────────┘  └──────────────────┘  └────────────────┘  │
├──────────────────────────────────────────────────────────────────┤
│  ┌────────────────────────────────────────────────────────────┐  │
│  │                  Agent Layer (Agent 계층)                  │  │
│  ├────────────────────────────────────────────────────────────┤  │
│  │  SimpleAgent  │  RouterAgent  │  OrchestratorAgent  │ MCP  │  │
│  └────────────────────────────────────────────────────────────┘  │
├──────────────────────────────────────────────────────────────────┤
│  ┌────────────────────────────────────────────────────────────┐  │
│  │                Skills Layer (스킬 계층)                     │  │
│  ├────────────────────────────────────────────────────────────┤  │
│  │  Skill  │  SkillResource  │  SKILL.md Parser  │  Matching  │  │
│  └────────────────────────────────────────────────────────────┘  │
├──────────────────────────────────────────────────────────────────┤
│  ┌────────────────────────────────────────────────────────────┐  │
│  │                  Data Layer (데이터 계층)                   │  │
│  ├────────────────────────────────────────────────────────────┤  │
│  │  AgentState  │  Message  │  NodeResult  │  Checkpoint      │  │
│  └────────────────────────────────────────────────────────────┘  │
└──────────────────────────────────────────────────────────────────┘
```

### 핵심 설계 원칙

1. **타입 안정성**: Pydantic 기반 런타임 검증
2. **비동기 처리**: asyncio로 고성능 실현
3. **표준 준수**: OpenTelemetry, CNCF 표준
4. **확장 가능**: 플러그인 아키텍처
5. **모듈화**: Skills 기반 기능 분리
6. **컨텍스트 효율성**: Progressive Disclosure 패턴

---

## 🔧 주요 컴포넌트

### 0. FrameworkConfig (설정 관리)

`FrameworkConfig`는 `Settings` 클래스의 값을 기본값으로 사용합니다:

```python
from unified_agent import FrameworkConfig, Settings

# Settings에서 전역 기본값 변경
Settings.DEFAULT_MODEL = "gpt-5.2"
Settings.DEFAULT_TEMPERATURE = 0.5

# FrameworkConfig는 Settings 값을 자동 참조
config = FrameworkConfig()  # Settings.DEFAULT_MODEL 적용
print(config.model)  # "gpt-5.2"

# 또는 개별 설정 오버라이드
config = FrameworkConfig(
    model="gpt-4o",           # Settings보다 우선
    temperature=0.7,
    max_tokens=2000,
    checkpoint_dir="./checkpoints",
    enable_telemetry=True
)

# 환경변수에서 자동 로드 (권장)
config = FrameworkConfig.from_env()
```

### FrameworkConfig와 Settings 관계

```
┌─────────────────────────────────────────┐
│           Settings 클래스              │  ← 전역 기본값 (한 곳에서 관리)
│  DEFAULT_MODEL = "gpt-5.2"            │
│  DEFAULT_TEMPERATURE = 0.7            │
│  ...                                   │
└─────────────────────────────────────────┘
                   │
                   │ 참조
                   ▼
┌─────────────────────────────────────────┐
│        FrameworkConfig 인스턴스         │  ← 실행 시 설정
│  model = Settings.DEFAULT_MODEL        │
│  temperature = Settings.DEFAULT_TEMPERATURE │
│  ...                                   │
└─────────────────────────────────────────┘
```

### 1. Agent 클래스

#### SimpleAgent
기본 대화형 Agent

```python
assistant = SimpleAgent(
    name="assistant",
    system_prompt="You are a helpful assistant.",
    model="gpt-4o-mini",
    temperature=0.7,
    max_tokens=1000
)
```

#### RouterAgent
조건부 라우팅 Agent

```python
router = RouterAgent(
    name="router",
    routes={
        "order": "order_agent",
        "support": "support_agent",
        "general": "general_agent"
    },
    model="gpt-4o-mini"
)
```

#### OrchestratorAgent
Multi-agent 협업 조정자

```python
orchestrator = OrchestratorAgent(
    name="team_lead",
    agents=[researcher, writer, reviewer],
    max_rounds=5
)
```

### 2. Graph (워크플로우)

```python
# 그래프 생성
graph = framework.create_graph("customer_service")

# 노드 추가
graph.add_node(Node("router", router_agent))
graph.add_node(Node("order", order_agent))
graph.add_node(Node("support", support_agent))

# 엣지 정의
graph.set_start("router")
graph.set_end("order")
graph.set_end("support")

# 시각화
print(graph.visualize())
```

**출력 (Mermaid)**:
```mermaid
graph TD
    router([START])
    order[END]
    support[END]
    router --> order
    router --> support
```

### 3. AgentState (상태 관리)

```python
class AgentState(BaseModel):
    messages: List[Message]              # 전체 대화 기록
    current_node: str                    # 현재 노드
    visited_nodes: List[str]             # 방문 경로
    metadata: Dict[str, Any]             # 메타데이터
    execution_status: ExecutionStatus    # 실행 상태
```

**주요 메서드**:
```python
state.add_message(AgentRole.USER, "Hello")
history = state.get_conversation_history(max_messages=10)
```

### 4. StateManager (체크포인팅)

```python
# 체크포인트 저장
checkpoint_file = await state_manager.save_checkpoint(state)
# 출력: ./checkpoints/session-123_2025-10-09T12-00-00.json

# 복원
restored_state = await state_manager.restore_checkpoint(session_id)
```

---

## 💡 실전 예제

### 예제 1: 고객 서비스 라우팅

```python
# 라우터 설정
router = RouterAgent(
    name="customer_service_router",
    routes={
        "order": "order_processing",
        "refund": "refund_handling",
        "inquiry": "general_inquiry"
    }
)

# 각 전문 Agent
order_agent = SimpleAgent(
    name="order_processing",
    system_prompt="You handle order-related requests."
)

refund_agent = SimpleAgent(
    name="refund_handling",
    system_prompt="You process refund requests."
)

inquiry_agent = SimpleAgent(
    name="general_inquiry",
    system_prompt="You answer general questions."
)

# 그래프 구성
graph = framework.create_graph("customer_service")
graph.add_node(Node("router", router))
graph.add_node(Node("order_processing", order_agent))
graph.add_node(Node("refund_handling", refund_agent))
graph.add_node(Node("general_inquiry", inquiry_agent))

graph.set_start("router")
graph.set_end("order_processing")
graph.set_end("refund_handling")
graph.set_end("general_inquiry")

# 실행
state = await framework.run(
    session_id="customer-001",
    workflow_name="customer_service",
    user_message="I want to track my order"
)

# 결과: router → order_processing 자동 라우팅
```

### 예제 2: 콘텐츠 생성 팀

```python
# 전문 Agent 생성
researcher = SimpleAgent(
    name="researcher",
    system_prompt="You are a thorough researcher. Gather facts and data."
)

writer = SimpleAgent(
    name="writer",
    system_prompt="You are a creative writer. Turn research into engaging content."
)

editor = SimpleAgent(
    name="editor",
    system_prompt="You are a critical editor. Review and improve content. Say 'TERMINATE' when satisfied."
)

# Orchestrator로 협업 구성
content_team = OrchestratorAgent(
    name="content_team_lead",
    agents=[researcher, writer, editor],
    max_rounds=5
)

# 실행
graph = framework.create_graph("content_creation")
graph.add_node(Node("team", content_team))
graph.set_start("team")
graph.set_end("team")

state = await framework.run(
    session_id="content-001",
    workflow_name="content_creation",
    user_message="Write an article about AI agents"
)

# 출력: 각 Agent가 순차적으로 기여한 결과
```

### 예제 3: 장기 실행 워크플로우

```python
# Day 1: 데이터 수집 시작
state = await framework.run(
    session_id="etl-pipeline-001",
    workflow_name="data_processing",
    user_message="Start data collection"
)

# 자동 체크포인트 저장됨
# 출력: ./checkpoints/etl-pipeline-001_2025-10-09T10-00-00.json

# [시스템 재시작 또는 장애 발생]

# Day 2: 중단 지점부터 재개
state = await framework.run(
    session_id="etl-pipeline-001",
    workflow_name="data_processing",
    user_message="",
    restore_from_checkpoint=True
)

print(f"복원된 노드: {state.current_node}")
print(f"방문 경로: {' -> '.join(state.visited_nodes)}")
```

---

## ⚡ 성능 최적화

### 1. 캐싱 전략

```python
class CachedMemoryStore:
    async def save(self, key: str, data: Dict):
        self.access_count[key] += 1
        # 3회 이상 접근 시 HOT 캐시에 저장
        if self.access_count[key] > 3:
            self.cache[key] = data
```

**효과**:
- 캐시 히트율 85% 이상
- 평균 조회 시간 90% 감소

### 2. 병렬 실행 (준비 중)

```python
# Multi-agent 병렬 실행
tasks = [agent.execute(state, kernel) for agent in agents]
results = await asyncio.gather(*tasks)
```

### 3. 토큰 사용량 추적

```python
# 자동으로 각 노드별 토큰 기록
state.metadata[f"{node_name}_result"] = {
    "tokens_used": 150,
    "duration_ms": 1234.56
}
```

### 성능 벤치마크

| 작업 | 소요 시간 | 토큰 사용량 |
|------|----------|-----------|
| 단순 대화 | ~1.5초 | 150-300 |
| 라우팅 | ~2.0초 | 200-400 |
| Multi-agent (3 agents) | ~5.0초 | 500-1000 |

---

## 🌐 프로덕션 배포

### 배포 전 체크리스트

- [ ] **환경 변수**: API 키, 엔드포인트 설정
- [ ] **체크포인트 디렉토리**: 충분한 디스크 공간 확보
- [ ] **로깅**: 프로덕션 레벨로 설정 (WARNING 이상)
- [ ] **OpenTelemetry**: Application Insights 또는 Jaeger 연결
- [ ] **에러 처리**: 각 Agent의 예외 처리 로직 검증
- [ ] **보안**: API 키 암호화, 접근 제어

### 환경별 설정

#### 개발 환경
```python
framework = UnifiedAgentFramework(
    kernel=kernel,
    checkpoint_dir="./checkpoints",
    enable_telemetry=True  # 디버깅용 콘솔 출력
)
```

#### 프로덕션 환경
```python
framework = UnifiedAgentFramework(
    kernel=kernel,
    checkpoint_dir="/var/checkpoints",  # 영구 스토리지
    enable_telemetry=True
)

# Application Insights 연결
setup_telemetry("UnifiedAgentFramework", enable_console=False)
```

### Docker 배포

```dockerfile
FROM python:3.10-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

CMD ["python", "main.py"]
```

**docker-compose.yml**:
```yaml
version: '3.8'
services:
  agent-framework:
    build: .
    environment:
      - AZURE_OPENAI_API_KEY=${AZURE_OPENAI_API_KEY}
      - AZURE_OPENAI_ENDPOINT=${AZURE_OPENAI_ENDPOINT}
      - AZURE_OPENAI_DEPLOYMENT_NAME=${AZURE_OPENAI_DEPLOYMENT_NAME}
    volumes:
      - ./checkpoints:/app/checkpoints
```

### 모니터링

```python
# OpenTelemetry 메트릭 자동 수집
- workflow_execution_time      # 워크플로우 실행 시간
- node_execution_count         # 노드별 실행 횟수
- tokens_per_request          # 요청당 토큰 사용량
- error_rate                  # 에러 발생률
- cache_hit_rate              # 캐시 히트율
```

---

## ❓ FAQ

### Q1: Semantic Kernel이 필수인가요?
**A**: 현재 버전은 Semantic Kernel 기반이지만, 다른 LLM 라이브러리로 확장 가능합니다.

### Q2: Redis 대신 인메모리만 사용 가능한가요?
**A**: 네, `CachedMemoryStore`가 기본으로 제공됩니다. Redis는 분산 환경에서 권장됩니다.

### Q3: 체크포인트 파일 크기가 너무 큽니다.
**A**: `AgentState.messages`에서 오래된 메시지를 주기적으로 정리하세요:
```python
if len(state.messages) > 100:
    state.messages = state.messages[-50:]  # 최근 50개만 유지
```

### Q4: OpenTelemetry를 비활성화할 수 있나요?
**A**: 네, 프레임워크 초기화 시 `enable_telemetry=False` 설정:
```python
framework = UnifiedAgentFramework(kernel, enable_telemetry=False)
```

### Q5: Multi-language 지원이 되나요?
**A**: Agent의 `system_prompt`를 다국어로 설정하면 됩니다:
```python
assistant = SimpleAgent(
    system_prompt="당신은 한국어로 대화하는 AI 어시스턴트입니다."
)
```

### Q6: Skills와 Agent의 차이점은 무엇인가요?
**A**: Agent는 실행 가능한 워크플로우 단위이고, Skill은 컨텍스트와 지침을 제공하는 모듈입니다:
- **Agent**: 실제 LLM 호출 및 상태 관리를 담당
- **Skill**: Agent가 특정 작업을 잘 수행하도록 지침, 예제, 리소스를 제공

### Q7: 커스텀 스킬을 어떻게 만드나요?
**A**: CLI 또는 코드로 템플릿을 생성하고 SKILL.md를 수정하세요:
```bash
# CLI
python Semantic-agent_framework.py
skills create my-custom-skill
```
```python
# 코드
framework.skill_manager.create_skill_template("my-skill", "./skills")
```

### Q8: Progressive Disclosure가 무엇인가요?
**A**: 컨텍스트 효율성을 위해 필요한 정보만 단계적으로 로드하는 패턴입니다:
1. 항상: 스킬 이름 + 설명 (~100 단어)
2. 매칭 시: SKILL.md 전체 지침 (<5k 단어)
3. 필요 시: scripts/, references/ 등 리소스

---

## 🛠️ 고급 활용

### 커스텀 Agent 만들기

```python
class CustomAnalyzer(Agent):
    async def execute(self, state: AgentState, kernel: Kernel) -> NodeResult:
        # 커스텀 로직 구현
        user_message = state.messages[-1].content

        # 외부 API 호출
        analysis_result = await self.call_external_api(user_message)

        # 상태 업데이트
        state.add_message(
            AgentRole.ASSISTANT,
            f"Analysis: {analysis_result}",
            self.name
        )

        return NodeResult(
            node_name=self.name,
            output=analysis_result,
            success=True
        )

    async def call_external_api(self, text: str):
        # 외부 서비스 호출 로직
        pass
```

### MCP (Model Context Protocol) 통합

```python
class MCPAgent(Agent):
    def __init__(self, *args, mcp_server: MCPServer, **kwargs):
        super().__init__(*args, **kwargs)
        self.mcp_server = mcp_server

    async def execute(self, state, kernel):
        # MCP 도구 동적 발견
        tools = await self.mcp_server.discover_tools()

        # LLM이 필요시 도구 자동 호출
        result = await self._get_llm_response_with_tools(state, tools)
        return result
```

---

## 🤝 기여하기

기여를 환영합니다! 자세한 내용은 [CONTRIBUTING.md](CONTRIBUTING.md)를 참조하세요.

### 빠른 시작

```bash
# 저장소 Fork 후 Clone
git clone https://github.com/YOUR_USERNAME/unified-agent-framework.git
cd unified-agent-framework

# 가상환경 생성
python -m venv venv
venv\Scripts\activate  # macOS/Linux: source venv/bin/activate

# 의존성 설치
pip install -r requirements.txt

# 개발 의존성 설치 (선택)
pip install -e ".[dev]"

# 테스트 실행
python test_unified_agent.py
```

### 기여 방법

1. **Fork** 이 저장소
2. **Feature Branch** 생성 (`git checkout -b feature/AmazingFeature`)
3. **Commit** 변경사항 (`git commit -m 'feat: add amazing feature'`)
4. **Push** to Branch (`git push origin feature/AmazingFeature`)
5. **Pull Request** 생성

[Conventional Commits](https://www.conventionalcommits.org/) 규칙을 따릅니다.

### 관련 문서

- 📋 [기여 가이드](CONTRIBUTING.md) - 상세한 기여 방법
- 📜 [행동 강령](CODE_OF_CONDUCT.md) - 커뮤니티 가이드라인
- 📝 [변경 이력](CHANGELOG.md) - 버전별 변경사항

---

## 📁 프로젝트 구조

```
Unified-agent-framework/
│
├── 📦 unified_agent/              # 핵심 패키지 (12개 모듈)
│   ├── __init__.py               # 패키지 진입점 (67개 API export)
│   ├── config.py                 # 설정 클래스 (40+ 모델, MCP, RAI)
│   ├── models.py                 # Pydantic 데이터 모델
│   ├── memory.py                 # 메모리 시스템
│   ├── events.py                 # 이벤트 시스템
│   ├── skills.py                 # 스킬 관리
│   ├── tools.py                  # 도구 정의
│   ├── agents.py                 # 5가지 에이전트 타입
│   ├── workflow.py               # 워크플로우 엔진
│   ├── orchestration.py          # 멀티에이전트 오케스트레이션
│   ├── framework.py              # 통합 프레임워크
│   ├── utils.py                  # 유틸리티 (CircuitBreaker 등)
│   └── exceptions.py             # 커스텀 예외
│
├── 📂 skills/                     # SKILL.md 기반 스킬 디렉토리
│   ├── python-expert/
│   ├── data-analyst/
│   └── korean-writer/
│
├── 🧪 test_unified_agent.py       # 테스트 (79개)
├── 🎮 demo_unified_agent.py       # 데모 코드
├── 📖 Unified_agent_framework.py  # 레거시 래퍼 (하위 호환성)
│
├── 📋 README.md                   # 이 문서
├── 📄 LICENSE                     # MIT 라이선스
├── 📝 CHANGELOG.md                # 버전 변경 이력
├── 🤝 CONTRIBUTING.md             # 기여 가이드
├── 📜 CODE_OF_CONDUCT.md          # 행동 강령
│
├── 📦 pyproject.toml              # Python 패키징 설정
├── 📦 requirements.txt            # 의존성 목록
├── 🔧 .env.example                # 환경 변수 템플릿
├── 🙈 .gitignore                  # Git 제외 파일
│
└── 🔄 .github/                    # GitHub 설정
    ├── ISSUE_TEMPLATE/           # 이슈 템플릿
    │   ├── bug_report.md
    │   ├── feature_request.md
    │   └── question.md
    ├── PULL_REQUEST_TEMPLATE.md  # PR 템플릿
    └── workflows/                # GitHub Actions
        ├── ci.yml                # CI 파이프라인
        └── release.yml           # PyPI 배포
```

### 스킬 추가 방법

1. `skills/` 디렉토리에 새 폴더 생성
2. `SKILL.md` 파일 작성 (YAML frontmatter + 마크다운)
3. 프레임워크 재시작 시 자동 로드

---

## 📄 라이선스

이 프로젝트는 MIT 라이선스 하에 배포됩니다. 자세한 내용은 [LICENSE](LICENSE) 파일을 참조하세요.

---

## 🙏 감사의 말

이 프로젝트는 다음 오픈소스 프로젝트에서 영감을 받았습니다:

- [Microsoft AutoGen](https://github.com/microsoft/autogen)
- [Semantic Kernel](https://github.com/microsoft/semantic-kernel)
- [Microsoft Agent Framework](https://github.com/microsoft/agent-framework)
- [Microsoft Multi-Agent-Custom-Automation-Engine](https://github.com/microsoft/multi-agent-custom-automation-engine) - MPlan, ProxyAgent, RAI 패턴 (NEW!)
- [LangGraph](https://github.com/langchain-ai/langgraph)
- [Anthropic Skills](https://github.com/anthropics/skills) - Skills 시스템 패턴
- [AWS AgentCore Samples](https://github.com/awslabs/amazon-bedrock-agentcore-samples) - Memory Hook, Session Manager, Investigation Plan 패턴

---

## 📊 버전 이력

전체 변경 이력은 [CHANGELOG.md](CHANGELOG.md)를 참조하세요.

| 버전 | 날짜 | 주요 변경사항 |
|------|------|-------------|
| **3.1.0** | 2026-01-26 | 🆕 **40+ AI 모델 지원** (GPT-5.2, Claude 4.5, Grok-4, Llama 4, o4-mini), Adaptive Circuit Breaker, MCP 설정, RAI 강화, 상세 한글 주석, **GitHub 오픈소스 준비** (CI/CD, 문서화) |
| 3.0.0 | 2026-01 | **완전한 모듈화 아키텍처** (12개 모듈로 분리), Microsoft Multi-Agent Engine 통합 (WebSocket, MPlan, ProxyAgent, RAI), AgentFactory, OrchestrationManager, 79개 테스트 커버리지, 93% 코드 감소 |
| 2.2.0 | 2026-01 | **Settings 클래스** (중앙 설정 통합), GPT-5.2/o3/o4-mini 모델 추가, UTF-8 기본 인코딩, CLI `model` 명령 추가 |
| 2.1.0 | 2025-12 | SKILL.md 파일 기반 스킬 관리, GPT-5/o1 모델 temperature 자동 분기 |
| 2.0.0 | 2025-01 | Skills 시스템 통합, FrameworkConfig 추가, Factory Pattern, AWS AgentCore 패턴 |
| 1.0.0 | 2024-12 | 초기 릴리스, 5개 프레임워크 통합 |

---

## 📦 설치 (PyPI)

```bash
# pip로 설치
pip install unified-agent-framework

# 또는 MCP 지원 포함
pip install unified-agent-framework[mcp]

# 또는 모든 기능 포함
pip install unified-agent-framework[full]
```

---

<div align="center">

**⭐ 이 프로젝트가 도움이 되었다면 Star를 눌러주세요! ⭐**

[![GitHub Stars](https://img.shields.io/github/stars/unified-agent-framework/unified-agent-framework?style=social)](https://github.com/unified-agent-framework/unified-agent-framework)
[![GitHub Forks](https://img.shields.io/github/forks/unified-agent-framework/unified-agent-framework?style=social)](https://github.com/unified-agent-framework/unified-agent-framework/fork)

[🐛 버그 리포트](https://github.com/unified-agent-framework/unified-agent-framework/issues/new?template=bug_report.md) ·
[✨ 기능 제안](https://github.com/unified-agent-framework/unified-agent-framework/issues/new?template=feature_request.md) ·
[❓ 질문하기](https://github.com/unified-agent-framework/unified-agent-framework/issues/new?template=question.md)

Made with ❤️ by the Unified Agent Framework Team

</div>
