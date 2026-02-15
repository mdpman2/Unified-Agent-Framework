# Unified Agent Framework v5 — Runner-Centric Design

**"가장 잘 동작하는 것 하나를 쉽고 안정적으로"**

[![Python](https://img.shields.io/badge/Python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Engines](https://img.shields.io/badge/Engines-Direct%20|%20LangChain%20|%20CrewAI-orange.svg)](#엔진)
[![OTEL](https://img.shields.io/badge/Observability-OpenTelemetry-teal.svg)](#모니터링-otel-어댑터)

> **v5.0.0** (2026-02-15) — v4.1의 49개 모듈·16개 프레임워크 브릿지를 **9개 모듈·Top 3 엔진**으로 전면 재설계.
> `run_agent("질문")` 한 줄로 끝냅니다.

---

## 목차

- [왜 v5인가? — 축소 이유](#왜-v5인가--축소-이유)
- [v4.1 → v5 모듈 축소 상세 매핑](#v41--v5-모듈-축소-상세-매핑)
- [설계 원칙 4가지](#설계-원칙-4가지)
- [빠른 시작](#빠른-시작)
- [설치](#설치)
- [핵심 사용법](#핵심-사용법)
- [아키텍처 (v5 모듈 구조)](#아키텍처-v5-모듈-구조)
- [엔진](#엔진)
- [도구 (MCP 표준)](#도구-mcp-표준)
- [메모리](#메모리)
- [모니터링 (OTEL 어댑터)](#모니터링-otel-어댑터)
- [플러그인 (v4 기능 마이그레이션)](#플러그인-v4-기능-마이그레이션)
- [기존 v4.1 코드와의 관계](#기존-v41-코드와의-관계)

---

## 왜 v5인가? — 축소 이유

v4.1은 49개 모듈, 380+ 공개 API, 16개 프레임워크 브릿지를 가진 **"메타 프레임워크"** 였습니다.
실무에서는 다음과 같은 문제가 발생했습니다:

| 문제 | 설명 |
|------|------|
| **유지보수 불가** | 49개 모듈 간 의존성 파악이 어렵고, 한 모듈 변경 시 연쇄 수정이 필요 |
| **사용자 진입 장벽** | 380+ API 중 "어떤 것을 써야 하는지" 파악하는 데만 시간 소비 |
| **16개 브릿지 환상** | 모든 프레임워크를 지원한다는 것 ≠ 모든 프레임워크가 잘 동작한다는 것. 대부분 사용자는 1~2개만 사용 |
| **자체 모니터링 과부하** | Tracer, Dashboard, MetricsCollector, AlertManager 등 자체 관찰성 스택이 프레임워크를 무겁게 만듦 |
| **메모리 복잡도** | PersistentMemory, Compaction, SessionTree, MemoryLayer 등 6개 메모리 시스템은 실무에서 `List[Message]`면 충분 |
| **도구 파편화** | 프레임워크마다 다른 도구 정의 방식(AIFunction, MCPTool, StructuredTool)이 혼재 |

**핵심 인사이트:** 사용자는 "모든 것"을 원하지 않습니다. **"가장 잘 동작하는 것 하나"를 쉽고 안정적으로 쓰고 싶어 합니다.**

---

## v4.1 → v5 모듈 축소 상세 매핑

### 숫자로 보는 변화

| 지표 | v4.1 | v5 | 축소율 |
|------|------|----|--------|
| 파이썬 파일 수 | 49개 | **9개** | 82% ↓ |
| 공개 API 수 | 380+ | **~20개** | 95% ↓ |
| 프레임워크 브릿지 | 16개 | **3개 엔진** | 81% ↓ |
| 필수 의존성 | 8개 (`semantic-kernel` 포함) | **2개** (`openai`, `python-dotenv`) | 75% ↓ |

### 핵심 모듈 매핑: v4.1 → v5 (9개 모듈로 통합)

| v4.1 모듈 | v5 매핑 | 축소/변경 이유 |
|-----------|---------|---------------|
| `framework.py` (875줄) | **`runner.py`** (297줄) | Runner 중심으로 재설계. Kernel 의존성 제거, `run_agent()` 한 줄 진입점 |
| `config.py` (668줄) | **`config.py`** (160줄) | 54개 모델 목록·Settings 클래스를 실무 필수 항목만으로 축소 |
| `models.py` + `interfaces.py` | **`types.py`** (137줄) | AgentRole/AgentState/PlanStep 등 30+ 모델 → Message/Role/AgentResult 6개로 단순화. OpenAI ChatCompletion 표준 통일 |
| `memory.py` (586줄) + `persistent_memory.py` + `compaction.py` + `session_tree.py` | **`memory.py`** (147줄) | 6개 메모리 시스템 → 단순 `List[Message]` + 슬라이딩 윈도우. 영속화는 `to_json()`으로 외부 위임 |
| `tools.py` (316줄) + `mcp_workbench.py` | **`tools.py`** (220줄) | AIFunction/MCPTool/ApprovalRequiredAIFunction → MCP 표준 `Tool` 하나로 일원화, `@mcp_tool` 데코레이터 |
| `tracer.py` (851줄) + `observability.py` (721줄) + `hooks.py` + `reward.py` + `agent_store.py` | **`callback.py`** (330줄) | 자체 Tracer/Dashboard/Alert/Reward를 전부 제거. OTEL 표준 `CallbackHandler` 어댑터만 남김 |
| `universal_bridge.py` + 7개 브릿지 모듈 | **`engines/`** (4파일) | 16개 브릿지 → Top 3 엔진(Direct, LangChain, CrewAI)으로 축소. `EngineProtocol` 인터페이스 |
| `events.py` + `workflow.py` + `orchestration.py` + `agents.py` (931줄) | **`runner.py`의 Runner 클래스** | 5종 에이전트(Simple/Router/Supervisor/Proxy/Approval) + EventBus + Graph를 Runner 실행 루프로 단순화 |

### 제거된 v4.1 모듈 상세 (40개)

| v4.1 모듈 | 제거 이유 | 대안 |
|-----------|---------|------|
| `universal_bridge.py` | "16개 통합"이 유지보수 불가. 실무에서 1~2개만 사용 | `engines/` 3개 엔진으로 대체 |
| `openai_agents_bridge.py` | 브릿지 패턴 제거 → 엔진 패턴으로 전환 | `engines/direct.py`가 OpenAI 네이티브 지원 |
| `google_adk_bridge.py` | 실무 사용 빈도 낮음 | 필요 시 플러그인 |
| `crewai_bridge.py` | v5 엔진으로 승격 | `engines/crewai_engine.py` |
| `a2a_bridge.py` | 프로토콜 아직 초기 단계 (Linux Foundation) | 필요 시 플러그인 |
| `ms_agent_bridge.py` | Preview 단계, API 불안정 | 필요 시 플러그인 |
| `ag2_bridge.py` | AutoGen 생태계 변경 잦음 (v0.2→v0.4→AgentOS) | CrewAI가 더 안정적 |
| `sk_agent_bridge.py` | Semantic Kernel은 자체로 충분히 강력 → 래핑 불필요 | 직접 사용 권장 |
| `persistent_memory.py` | `Memory.to_json()` + Redis/CosmosDB 직접 연동으로 대체 | 플러그인 |
| `compaction.py` | 메모리 압축 → 슬라이딩 윈도우로 단순화 | 불필요 |
| `session_tree.py` | Git 스타일 분기 — 실무에서 거의 미사용 | 플러그인 |
| `tracer.py` (851줄) | 자체 Span/Trace 구현 → OTEL 표준 위임 | `callback.py`의 `OTelCallbackHandler` |
| `observability.py` (721줄) | 자체 Dashboard/Metrics/Alert → 전문 도구에 위임 | OTEL → Azure Monitor / Datadog |
| `agent_store.py` | 에이전트 레지스트리 → 미사용 | 불필요 |
| `reward.py` | 보상 시스템 → 실무 미활용 | 불필요 |
| `adapter.py` | Agent Lightning 어댑터 → `EngineProtocol`로 대체 | 불필요 |
| `hooks.py` | 라이프사이클 훅 → `CallbackHandler`로 대체 | 통합 |
| `events.py` | EventBus pub/sub → 콜백 패턴으로 단순화 | `callback.py` |
| `agents.py` (931줄) | 5종 에이전트 클래스 → Runner + 엔진 조합으로 대체 | `runner.py` |
| `workflow.py` | Graph/Node 워크플로우 → LangGraph에 위임 | `engines/langchain_engine.py` |
| `orchestration.py` | OrchestrationManager → CrewAI/LangGraph에 위임 | `engines/` |
| `skills.py` | SKILL.md 기반 스킬 → 시스템 프롬프트로 단순화 | `AgentConfig.system_prompt` |
| `extensions.py` | ExtensionsHub → 모듈 자체가 최소화되어 불필요 | 불필요 |
| `prompt_cache.py` | OpenAI Prompt Caching은 API 레벨 자동 적용 | 불필요 (API 내장) |
| `durable_agent.py` | 장기 워크플로우 → 외부 도구(Temporal, Durable Functions) 권장 | 플러그인 |
| `concurrent.py` | Fan-out/Fan-in → `asyncio.gather()` 직접 사용 | 언어 내장 |
| `agent_tool.py` | AgentTool 중첩 패턴 → 거의 미사용 | 불필요 |
| `extended_thinking.py` | Reasoning 추적 → OTEL span 속성으로 대체 | `callback.py` |
| `security_guardrails.py` | 프롬프트 보안 → Azure Content Safety API 직접 호출 권장 | 플러그인 |
| `structured_output.py` | GPT JSON Schema → `openai` 패키지 `response_format` 직접 사용 | 불필요 (API 내장) |
| `evaluation.py` | PDCA 평가 → LangSmith/Braintrust 등 전문 도구 위임 | 외부 도구 |
| `responses_api.py` | Stateful 대화 → `Memory` 클래스로 충분 | `memory.py` |
| `video_generation.py` | Sora 비디오 생성 → AI 에이전트 프레임워크 범위 밖 | 별도 프로젝트 |
| `image_generation.py` | GPT Image → AI 에이전트 프레임워크 범위 밖 | 별도 프로젝트 |
| `open_weight.py` | 오픈 웨이트 모델 → Direct엔진이 모든 OpenAI 호환 API 지원 | `engines/direct.py` |
| `agent_identity.py` | Entra ID → Azure SDK 직접 사용 권장 | 플러그인 |
| `browser_use.py` | Playwright CUA → 별도 도구(Browser Use 패키지) 권장 | 플러그인 |
| `deep_research.py` | 다단계 연구 → 별도 에이전트로 구현 | 플러그인 |
| `middleware.py` | 미들웨어 파이프라인 → 콜백 패턴으로 대체 | `callback.py` |
| `agent_triggers.py` | 이벤트 트리거 → Azure Logic Apps/Functions 직접 사용 | 외부 도구 |
| `exceptions.py` | 커스텀 예외 → 표준 Python 예외 사용 | 불필요 |
| `utils.py` | CircuitBreaker/RAIValidator → 외부 라이브러리 권장 | 불필요 |

### 엔진 축소 상세: 16개 → 3개

| 순위 | v5 엔진 | 선정 이유 | v4.1 대응 브릿지 |
|------|---------|---------|-----------------|
| 1 | **`direct`** (Direct API) | 프레임워크 없이 가장 가볍게, `openai` 패키지만으로 동작 | `openai_agents_bridge` + `open_weight` |
| 2 | **`langchain`** (LangChain/LangGraph) | 범용 체인/RAG 구성에서 사실상 표준. 생태계가 가장 넓음 | `sk_agent_bridge` + `workflow` |
| 3 | **`crewai`** (CrewAI) | 멀티 에이전트 협업에서 가장 직관적. AutoGen보다 API가 단순 | `crewai_bridge` + `ag2_bridge` + `agents(SupervisorAgent)` |

**제외된 13개 브릿지의 대안:**

| 제외 브릿지 | 이유 | 대안 |
|------------|------|------|
| Google ADK | 생태계 아직 소규모 | LangChain에서 Google 모델 사용 가능 |
| A2A Protocol | 아직 초기 단계 (Linux Foundation) | 성숙 시 플러그인으로 추가 |
| MS Agent Framework | Preview 단계, API 변경 잦음 | Azure SDK 직접 사용 |
| AG2/AutoGen | API 변경이 너무 잦음 (v0.2→v0.4→AgentOS) | CrewAI가 더 안정적 |
| Semantic Kernel | SK는 자체로 충분히 강력 → 프레임워크 래핑 불필요 | 직접 사용 권장 |
| 나머지 8개 | 실무 사용 빈도 극히 낮음 | 필요 시 플러그인 |

---

## 설계 원칙 4가지

### 1. Top 3 + Direct — 엔진 축소

> "모든 것을 지원"하는 메타 프레임워크보다, 실무에서 가장 많이 쓰이는 조합으로 범위를 좁힌다.

- **Direct** — 프레임워크 없이 `openai` 패키지만으로 가볍게
- **LangChain** — 범용 체인/RAG 구성
- **CrewAI** — 멀티 에이전트 협업
- 나머지는 `plugins/`로 분리하거나 삭제

### 2. OTEL 표준 어댑터 — 모니터링 경량화

> 자체 로깅 시스템을 최소화하고, `CallbackHandler` 하나만 잘 만들면 어디든 연결.

- `callback.py`에 `CallbackHandler` ABC 인터페이스 정의
- 내장: `OTelCallbackHandler`, `LoggingCallbackHandler`
- 사용자가 Datadog/Arize/WandB 커스텀 핸들러 작성 가능
- **프레임워크 자체는 데이터를 저장하지 않음** (Export only)

### 3. 핵심 3기능 집중 — Unified I/O, Memory, Tool Use

| 기능 | v4.1 | v5 | 개선 |
|------|------|----|------|
| **Unified I/O** | 다수 모델 + 인터페이스 혼재 | `Message`/`AgentResult` | OpenAI ChatCompletion 표준으로 통일 |
| **Memory** | 6개 메모리 시스템 | `Memory` (List[Message]) | 슬라이딩 윈도우 + JSON 직렬화만으로 충분 |
| **Tool Use** | AIFunction/MCPTool/StructuredTool | `Tool` + `@mcp_tool` | MCP 표준으로 일원화, OpenAI 스키마 자동 생성 |

### 4. Runner 중심 설계 — 실행에 집중

> "만드는 것"은 LangChain/CrewAI에 맡기고, "만들어진 에이전트를 쉽게 실행하고 결과를 추적하는 Runner"에 집중.

```python
# v4.1 — 복잡한 설정 필요
config = FrameworkConfig.from_env()
config.validate()
framework = UnifiedAgentFramework.create(config)
graph = framework.create_simple_workflow("chat")
state = await framework.run("session-1", "chat", "질문")
# state에서 응답 추출...

# v5 — 한 줄
result = await run_agent("질문")
print(result.content)
```

---

## 빠른 시작

```python
from unified_agent_v5 import run_agent

# 가장 간단한 사용
result = await run_agent("파이썬으로 피보나치 함수 작성해줘")
print(result.content)
```

---

## 설치

```bash
# 필수 (Direct API만 사용)
pip install openai python-dotenv

# LangChain 엔진 추가
pip install langchain langchain-openai langchain-core

# CrewAI 엔진 추가
pip install crewai

# OTEL 모니터링 추가
pip install opentelemetry-api opentelemetry-sdk opentelemetry-exporter-otlp
```

환경변수:

```bash
# Azure OpenAI (권장)
AZURE_OPENAI_ENDPOINT=https://your-resource.openai.azure.com/
AZURE_OPENAI_API_KEY=your-key
AZURE_OPENAI_DEPLOYMENT=your-deployment

# 또는 OpenAI 직접
OPENAI_API_KEY=sk-...

# 기본 엔진/모델
AGENT_ENGINE=direct
AGENT_MODEL=gpt-5.2
```

---

## 핵심 사용법

### 1. 가장 간단한 호출

```python
from unified_agent_v5 import run_agent

result = await run_agent("안녕하세요!")
print(result.content)      # "안녕하세요! 무엇을 도와드릴까요?"
print(result.usage)         # {'input_tokens': 12, 'output_tokens': 15, ...}
print(result.duration_ms)   # 340.5
```

### 2. 대화 이어가기 (Memory)

```python
from unified_agent_v5 import run_agent, Memory

memory = Memory(system_prompt="You are a Python expert.")
r1 = await run_agent("내 이름은 철수야", memory=memory)
r2 = await run_agent("내 이름이 뭐였지?", memory=memory)
print(r2.content)  # "철수입니다!"
```

### 3. 도구 사용

```python
from unified_agent_v5 import run_agent, mcp_tool

@mcp_tool(description="날씨 조회")
async def get_weather(city: str) -> str:
    return f"{city}: 맑음, 22°C"

result = await run_agent("서울 날씨 알려줘", tools=[get_weather])
```

### 4. 엔진 선택

```python
# Direct API (기본, 가장 가벼움)
result = await run_agent("코드 리뷰해줘", engine="direct")

# LangChain (체인/RAG 구성)
result = await run_agent("RAG 검색 후 요약", engine="langchain")

# CrewAI (멀티 에이전트 협업)
result = await run_agent(
    "시장 분석 보고서 작성",
    engine="crewai",
    crew_agents=[
        {"role": "Researcher", "goal": "데이터 수집 및 분석"},
        {"role": "Writer", "goal": "보고서 작성"},
    ]
)
```

### 5. 모니터링 콜백

```python
from unified_agent_v5 import run_agent, OTelCallbackHandler, LoggingCallbackHandler

# OTEL로 트레이스 Export
otel = OTelCallbackHandler(endpoint="http://localhost:4318")

# 로깅 (디버깅용)
log = LoggingCallbackHandler()

result = await run_agent("질문", callbacks=[otel, log])
```

### 6. Runner 클래스 (고급 사용)

```python
from unified_agent_v5 import Runner, AgentConfig, Memory

runner = Runner(config=AgentConfig(
    model="gpt-5.2",
    engine="direct",
    system_prompt="You are a Python expert.",
    max_tool_rounds=5,
))

memory = Memory()
r1 = await runner.run("데코레이터 설명해줘", memory=memory)
r2 = await runner.run("예시 코드도 보여줘", memory=memory)
```

---

## 엔진

| 엔진 | 용도 | 필수 패키지 | 선정 이유 |
|------|------|-------------|---------|
| `direct` | 범용 | `openai` | 가장 가볍고 빠름. 의존성 최소. LiteLLM 대안 |
| `langchain` | 체인/RAG | `langchain`, `langchain-openai` | 범용 체인 구성에서 사실상 표준. 생태계 최대 |
| `crewai` | 멀티 에이전트 | `crewai` | 역할 기반 협업이 직관적. AutoGen보다 API 안정 |

모든 엔진은 `EngineProtocol`을 구현 → **전환 비용 0**:

```python
# 동일한 run_agent() 인터페이스로 엔진만 바꾸면 끝
result = await run_agent("질문", engine="direct")     # Direct
result = await run_agent("질문", engine="langchain")   # LangChain
result = await run_agent("질문", engine="crewai")      # CrewAI
```

---

## 도구 (MCP 표준)

프레임워크별로 다른 도구 정의 방식을 **MCP(Model Context Protocol) 표준으로 일원화**합니다.

```python
from unified_agent_v5 import Tool, mcp_tool, ToolRegistry

# 방법 1: Tool 직접 생성
tool = Tool(
    name="web_search",
    description="웹 검색",
    parameters={"query": {"type": "string", "description": "검색어"}},
    fn=my_search_function,
)

# 방법 2: @mcp_tool 데코레이터 (권장)
@mcp_tool(description="코드 검색")
async def search_code(query: str, language: str = "python") -> str:
    return f"Found code for: {query}"

# 레지스트리로 도구 관리
registry = ToolRegistry()
registry.register(tool)
registry.register(search_code)

# OpenAI Function Calling 스키마 자동 생성
schemas = registry.get_openai_schemas()
```

---

## 메모리

```python
from unified_agent_v5 import Memory

memory = Memory(system_prompt="You are helpful.", max_messages=50)
memory.add_user("안녕")
memory.add_assistant("안녕하세요!")

# OpenAI API 호환 형식 반환
messages = memory.get_messages()

# 직렬화 → 외부 스토어(Redis, CosmosDB) 연동
json_str = memory.to_json()
restored = Memory.from_json(json_str)
```

**왜 `List[Message]`로 충분한가?**
- v4의 PersistentMemory/Compaction/SessionTree는 실무에서 거의 미사용
- 영속화가 필요하면 `to_json()` → Redis/CosmosDB에 직접 저장
- 토큰 제한은 슬라이딩 윈도우(`max_messages`)가 처리

---

## 모니터링 (OTEL 어댑터)

**자체 모니터링 시스템을 구축하지 않습니다.** `CallbackHandler` 하나만 잘 만들어두면, 사용자가 원하는 툴에 연결만 하면 됩니다.

```python
from unified_agent_v5 import CallbackHandler, OTelCallbackHandler

# 내장 핸들러
otel = OTelCallbackHandler(endpoint="http://otel-collector:4318")
# → Azure Monitor, Datadog, Jaeger, Zipkin 등으로 Export

# 커스텀 핸들러 (예: Arize, WandB)
class ArizeHandler(CallbackHandler):
    async def on_llm_end(self, content, usage=None, **kwargs):
        arize_client.log(prediction=content, tokens=usage)
```

지원 연동 대상:
- **Azure Monitor / Application Insights** (via OTEL)
- **Datadog** (via OTEL 또는 커스텀 핸들러)
- **LangSmith** (커스텀 핸들러)
- **Arize** (커스텀 핸들러)
- **WandB** (커스텀 핸들러)
- **Jaeger / Zipkin / Grafana Tempo** (via OTEL)

---

## 아키텍처 (v5 모듈 구조)

```
unified_agent_v5/                    # 9개 모듈 (v4.1의 49개 → 82% 축소)
├── __init__.py                      # 패키지 진입점 — 공개 API 20개
│                                    #   (v4.1: 380+ → 95% 축소)
│
├── runner.py                        # [핵심] Runner 중심 설계
│                                    #   - run_agent() 최상위 함수
│                                    #   - Runner 클래스 (실행 + 추적)
│                                    #   ← v4.1: framework.py + orchestration.py
│                                    #           + workflow.py + agents.py 통합
│
├── types.py                         # 통합 I/O 타입
│                                    #   - Message, Role (OpenAI ChatCompletion 표준)
│                                    #   - AgentResult, ToolCall, ToolResult
│                                    #   ← v4.1: models.py + interfaces.py 통합
│
├── config.py                        # 설정 (최소한)
│                                    #   - Settings (전역), AgentConfig (인스턴스)
│                                    #   ← v4.1: config.py 668줄 → 160줄
│
├── memory.py                        # 메모리 = List[Message]
│                                    #   - 슬라이딩 윈도우 + JSON 직렬화
│                                    #   ← v4.1: memory.py + persistent_memory.py
│                                    #           + compaction.py + session_tree.py 통합
│
├── tools.py                         # MCP 표준 도구
│                                    #   - Tool, @mcp_tool, ToolRegistry
│                                    #   ← v4.1: tools.py + mcp_workbench.py 통합
│
├── callback.py                      # OTEL 표준 어댑터
│                                    #   - CallbackHandler (인터페이스)
│                                    #   - OTelCallbackHandler (OTEL Export)
│                                    #   - LoggingCallbackHandler (디버깅)
│                                    #   ← v4.1: tracer.py(851줄) + observability.py(721줄)
│                                    #           + hooks.py + reward.py + events.py 통합
│
├── engines/                         # Top 3 엔진 (v4.1의 16개 브릿지 → 3개)
│   ├── __init__.py                  #   엔진 패키지 진입점
│   ├── base.py                      #   EngineProtocol + 레지스트리
│   ├── direct.py                    #   Direct API — openai 패키지 직접 호출
│   ├── langchain_engine.py          #   LangChain — 체인/RAG 구성
│   └── crewai_engine.py             #   CrewAI — 멀티 에이전트 협업
│
└── plugins/                         # v4 비핵심 기능 → 선택적 플러그인
    └── __init__.py                  #   필요 시 lazy import
```

---

## 플러그인 (v4 기능 마이그레이션)

기존 v4.1의 비핵심 기능은 `plugins/`에서 선택적으로 사용할 수 있습니다:

| v4.1 기능 | 플러그인 상태 | 대안 |
|-----------|-------------|------|
| Security Guardrails | 이관 가능 | Azure Content Safety API 직접 호출 |
| Structured Output | 이관 가능 | `openai` 패키지 `response_format` 사용 |
| Persistent Memory | 이관 가능 | `Memory.to_json()` + Redis/CosmosDB |
| Session Tree | 이관 가능 | 대부분 실무에서 미사용 |
| Browser Automation & CUA | 이관 가능 | `browser-use` 패키지 |
| Deep Research | 이관 가능 | 별도 에이전트 구현 |
| Agent Identity (Entra ID) | 이관 가능 | `azure-identity` SDK 직접 사용 |

---

## 기존 v4.1 코드와의 관계

기존 `unified_agent/` 패키지는 **그대로 유지**됩니다. v5(`unified_agent_v5/`)는 독립적으로 동작하며, 점진적 마이그레이션이 가능합니다.

```
Unified-agent-framework/
├── unified_agent/        # v4.1 (49개 모듈) — 기존 코드 보존
├── unified_agent_v5/     # v5 (9개 모듈) — 새로운 Runner 중심 설계
├── demo_unified_agent.py # v4.1 데모
└── demo_v5.py            # v5 데모 (6/6 시나리오 통과)
```

---

## 라이선스

[MIT License](LICENSE)
