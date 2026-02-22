# Unified Agent Framework v6

**Microsoft Agent Framework 1.0.0-rc1 호환 — Agent 클래스 기반 AI 에이전트 프레임워크**

[![Python](https://img.shields.io/badge/Python-3.11+-blue.svg)](https://www.python.org)
[![Version](https://img.shields.io/badge/Version-6.0.0-green.svg)](./CHANGELOG.md)
[![License](https://img.shields.io/badge/License-MIT-yellow.svg)](./LICENSE)
[![Agent Framework](https://img.shields.io/badge/Agent_Framework-1.0.0--rc1-orange.svg)](https://github.com/microsoft/agent-framework)

---

## 개요 (Overview)

Unified Agent Framework v6는 [Microsoft Agent Framework 1.0.0-rc1](https://github.com/microsoft/agent-framework)의
공식 API 패턴을 따르는 Python 기반 AI 에이전트 프레임워크입니다.

v5(Runner 중심)에서 **Agent 클래스 기반 설계**로 전면 재설계되었으며,
ChatClient 주입 · 도구 자동 스키마 · 세션 관리 · 멀티 에이전트 · 미들웨어 · 분산 추적을
하나의 통합된 API로 제공합니다.

### 핵심 특징

| 기능 | 설명 |
|------|------|
| **Agent 클래스** | `Agent.run()` / `Agent.run(stream=True)` 단일 진입점 |
| **ChatClient 주입** | OpenAI · Azure OpenAI 등 LLM 프로바이더 자유 교체 |
| **@tool 데코레이터** | 함수 → FunctionTool 자동 변환, OpenAI 스키마 자동 생성 |
| **AgentSession** | `ContextProvider` 기반 멀티턴 대화 히스토리 자동 관리 |
| **멀티 에이전트** | `agent.as_tool()`로 에이전트를 다른 에이전트의 도구로 위임 |
| **스트리밍** | `async for update in agent.run(..., stream=True)` 실시간 응답 |
| **미들웨어** | 로깅 · 재시도 등 파이프라인 전/후 처리 |
| **OpenTelemetry** | 분산 추적 + Azure Monitor 연동 |

---

## 아키텍처 (Architecture)

```
┌──────────────────────────────────────────────────────────┐
│                    사용자 코드 (User Code)                  │
│   agent = Agent(client, instructions, tools, providers)  │
│   response = await agent.run("질문")                      │
└────────────────────────┬─────────────────────────────────┘
                         │
         ┌───────────────▼───────────────┐
         │         Agent.run()           │
         │  ┌─────────────────────────┐  │
         │  │ MiddlewarePipeline      │  │  ← LoggingMiddleware, RetryMiddleware
         │  └────────────┬────────────┘  │
         │               │               │
         │  ┌────────────▼────────────┐  │
         │  │ ContextProvider 파이프라인│  │  ← before_run → after_run
         │  │ (History, Time, Custom) │  │
         │  └────────────┬────────────┘  │
         │               │               │
         │  ┌────────────▼────────────┐  │
         │  │  ChatClient.get_response│  │  ← OpenAIChatClient (Azure / OpenAI)
         │  │  ┌───────────────────┐  │  │
         │  │  │ Tool 자동 실행 루프 │  │  │  ← FunctionTool.invoke() × N
         │  │  │ (최대 10 라운드)   │  │  │
         │  │  └───────────────────┘  │  │
         │  └────────────┬────────────┘  │
         │               │               │
         │  ┌────────────▼────────────┐  │
         │  │    AgentResponse        │  │  ← Message(role, [Content])
         │  │    (text, usage, raw)   │  │
         │  └─────────────────────────┘  │
         └───────────────────────────────┘
```

---

## 빠른 시작 (Quick Start)

### 설치

```bash
pip install openai python-dotenv
```

### 최소 예제

```python
import asyncio
from unified_agent_v6 import Agent, OpenAIChatClient

client = OpenAIChatClient(model_id="gpt-5.2")
agent = Agent(client=client, instructions="당신은 친절한 AI 어시스턴트입니다.")

response = asyncio.run(agent.run("안녕하세요!"))
print(response.text)
```

---

## 사용법 (Usage)

### 1. 도구 사용 (@tool)

`@tool` 데코레이터로 일반 함수를 AI 에이전트의 도구로 자동 변환합니다.
함수 시그니처와 docstring에서 OpenAI function calling 스키마가 자동 생성됩니다.

```python
from unified_agent_v6 import Agent, OpenAIChatClient, tool

@tool
def get_weather(city: str) -> str:
    """도시의 날씨를 반환합니다."""
    return f"{city}: 맑음 22°C"

@tool
def calculate(expression: str) -> str:
    """수식을 계산합니다."""
    return str(eval(expression))

client = OpenAIChatClient(model_id="gpt-5.2")
agent = Agent(
    client=client,
    instructions="당신은 날씨와 계산을 도와주는 AI입니다.",
    tools=[get_weather, calculate],
)

response = await agent.run("서울 날씨와 15*23 계산해줘")
print(response.text)
```

> **참고**: 동기 함수도 `asyncio.to_thread()`를 통해 비동기로 자동 실행됩니다.
> Python 3.10+ 파이프 문법(`str | None`)도 스키마 변환을 지원합니다.

### 2. 멀티턴 대화 (AgentSession)

`InMemoryHistoryProvider`를 사용하면 대화 히스토리가 슬라이딩 윈도우 방식으로 자동 관리됩니다.

```python
from unified_agent_v6 import Agent, OpenAIChatClient, InMemoryHistoryProvider

agent = Agent(
    client=OpenAIChatClient(model_id="gpt-5.2"),
    instructions="사용자의 이전 발화를 기억하세요.",
    context_providers=[InMemoryHistoryProvider(max_messages=50)],
)

session = agent.create_session()
await agent.run("제 이름은 김철수입니다.", session=session)
response = await agent.run("제 이름이 뭐죠?", session=session)
print(response.text)  # → "김철수"
```

> **참고**: 세션만 전달하고 `context_providers`를 지정하지 않으면,
> `InMemoryHistoryProvider`가 자동 주입됩니다 (Agent 상태를 변경하지 않는 로컬 복사본).

### 3. 커스텀 ContextProvider

`BaseContextProvider`를 상속하여 `before_run` / `after_run` 훅으로
동적 지시사항 · 메시지 · 도구를 실행 시점에 주입할 수 있습니다.

```python
from unified_agent_v6 import BaseContextProvider

class TimeAwareProvider(BaseContextProvider):
    """현재 시간 정보를 자동으로 주입하는 프로바이더."""
    DEFAULT_SOURCE_ID = "time_aware"

    async def before_run(self, *, agent, session, context, state):
        from datetime import datetime
        now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        context.extend_instructions(
            self.source_id,
            f"현재 시각은 {now}입니다.",
        )
```

### 4. 멀티 에이전트 (Agent as Tool)

`agent.as_tool()`로 전문가 에이전트를 다른 에이전트의 도구로 등록하여
오케스트레이터 패턴을 구현합니다.

```python
weather_agent = Agent(
    client=client,
    instructions="날씨 전문가입니다.",
    tools=[get_weather],
    name="weather_expert",
    description="날씨 관련 질문을 처리",
)

calc_agent = Agent(
    client=client,
    instructions="계산 전문가입니다.",
    tools=[calculate],
    name="calc_expert",
    description="수학 계산을 처리",
)

orchestrator = Agent(
    client=client,
    instructions="적절한 전문가에게 위임하세요.",
    tools=[weather_agent.as_tool(), calc_agent.as_tool()],
)

response = await orchestrator.run("서울 날씨와 123 * 456 알려줘")
```

### 5. 스트리밍

```python
async for update in agent.run("Python의 장점 3가지", stream=True):
    print(update.text, end="", flush=True)
```

### 6. 미들웨어

```python
from unified_agent_v6 import Agent, LoggingMiddleware, RetryMiddleware

agent = Agent(
    client=client,
    instructions="...",
    middleware=[
        LoggingMiddleware(log_level=logging.INFO),
        RetryMiddleware(max_retries=3, delay_seconds=1.0),
    ],
)
```

### 7. v5 호환 모드 (run_agent)

기존 v5 코드와의 하위 호환을 위해 `run_agent()` 래퍼 함수를 제공합니다.

```python
from unified_agent_v6 import run_agent

result = await run_agent("안녕하세요!", model="gpt-5.2")
print(result.text)
```

---

## 환경변수 설정

`.env` 파일 또는 시스템 환경변수로 설정합니다.

```bash
# ── Azure OpenAI (권장) ──────────────────────────────────
AZURE_OPENAI_API_KEY=your-key
AZURE_OPENAI_ENDPOINT=https://your-resource.openai.azure.com/
AZURE_OPENAI_DEPLOYMENT=gpt-5.2
AZURE_OPENAI_API_VERSION=2025-01-01-preview

# ── OpenAI 직접 사용 ─────────────────────────────────────
OPENAI_API_KEY=sk-...
OPENAI_CHAT_MODEL_ID=gpt-5.2

# ── 에이전트 동작 (선택) ─────────────────────────────────
AGENT_TEMPERATURE=0.7
AGENT_MAX_TOKENS=4096
AGENT_MAX_TOOL_ROUNDS=10
AGENT_STREAM=false
AGENT_LOG_LEVEL=INFO
```

---

## 데모 시나리오 (demo_v6.py)

7개의 시나리오로 프레임워크 전체 기능을 검증합니다.

```bash
# 전체 실행
python demo_v6.py

# 특정 시나리오만 실행
python demo_v6.py 1 3 5
```

| # | 시나리오 | 검증 기능 |
|---|---------|----------|
| 1 | 기본 Agent | `Agent.run()`, `AgentResponse`, `UsageDetails` |
| 2 | 도구 사용 | `@tool`, `FunctionTool`, 자동 도구 호출 루프 |
| 3 | 멀티턴 대화 | `AgentSession`, `InMemoryHistoryProvider`, 슬라이딩 윈도우 |
| 4 | 커스텀 Provider | `BaseContextProvider.before_run()`, 동적 지시사항 주입 |
| 5 | 멀티 에이전트 | `agent.as_tool()`, 오케스트레이터 패턴 |
| 6 | 스트리밍 | `agent.run(stream=True)`, `AgentResponseUpdate` |
| 7 | v5 호환 | `run_agent()` 래퍼, `AgentResult` 별칭 |

---

## 프로젝트 구조

```
Unified-agent-framework/
├── unified_agent_v6/           # 메인 패키지
│   ├── __init__.py             # 패키지 엔트리 — 36개 public exports
│   ├── types.py                # Content, Message, AgentResponse, UsageDetails
│   ├── agents.py               # Agent, AgentSession, ContextProvider, ChatClient
│   ├── tools.py                # @tool 데코레이터, FunctionTool, normalize_tools
│   ├── middleware.py           # AgentMiddleware, LoggingMiddleware, RetryMiddleware
│   ├── config.py               # AgentConfig TypedDict, load_config()
│   └── observability.py       # configure_tracing(), get_tracer() (OpenTelemetry)
├── demo_v6.py                  # 7개 시나리오 데모
├── requirements.txt            # 의존성 정의
├── pyproject.toml              # 빌드 / 린터 / 테스트 설정
├── CHANGELOG.md                # 버전별 변경 이력
├── README.md                   # 이 문서
└── _legacy/                    # v5 아카이브 (unified_agent_v5/, demo_v5.py 등)
```

---

## API 레퍼런스

### Core Classes

| 클래스 | 모듈 | 설명 |
|--------|------|------|
| `Agent` | agents.py | AI 에이전트 — `run()` / `run(stream=True)` / `as_tool()` |
| `AgentSession` | agents.py | 대화 세션 — 세션 ID + state 딕셔너리 |
| `OpenAIChatClient` | agents.py | OpenAI / Azure OpenAI ChatClient (lazy 초기화) |
| `BaseChatClient` | agents.py | 커스텀 ChatClient 베이스 클래스 |

### Types

| 타입 | 모듈 | 설명 |
|------|------|------|
| `Content` | types.py | 통합 콘텐츠 컨테이너 (text, error, function_call 등) |
| `Message` | types.py | Content 기반 메시지 (role + contents[]) |
| `AgentResponse` | types.py | `Agent.run()` 반환 — messages, usage_details |
| `AgentResponseUpdate` | types.py | 스트리밍 청크 — contents, role |
| `UsageDetails` | types.py | 토큰 사용량 TypedDict (input/output/total) |
| `ChatOptions` | types.py | 채팅 요청 옵션 TypedDict |

### Tools

| 항목 | 모듈 | 설명 |
|------|------|------|
| `@tool` | tools.py | 함수 → FunctionTool 변환 데코레이터 |
| `FunctionTool` | tools.py | 도구 클래스 — 스키마 자동 생성, `invoke()` |
| `normalize_tools` | tools.py | 다양한 형식을 `list[FunctionTool]`로 정규화 |

### Context Providers

| 프로바이더 | 모듈 | 설명 |
|-----------|------|------|
| `BaseContextProvider` | agents.py | 커스텀 프로바이더 베이스 (`before_run` / `after_run`) |
| `InMemoryHistoryProvider` | agents.py | 인메모리 대화 히스토리 + 슬라이딩 윈도우 |
| `SessionContext` | agents.py | 단일 실행 컨텍스트 — 메시지/지시사항/도구 동적 추가 |

### Middleware

| 미들웨어 | 모듈 | 설명 |
|----------|------|------|
| `AgentMiddleware` | middleware.py | `Agent.run()` 전/후 처리 베이스 |
| `ChatMiddleware` | middleware.py | LLM API 호출 전/후 처리 |
| `FunctionMiddleware` | middleware.py | 도구 함수 호출 전/후 처리 |
| `LoggingMiddleware` | middleware.py | 자동 로깅 (실행 시간, 입출력) |
| `RetryMiddleware` | middleware.py | 실패 시 지수 백오프 재시도 |
| `MiddlewarePipeline` | middleware.py | 미들웨어 체인 실행 관리 |

### Config & Observability

| 항목 | 모듈 | 설명 |
|------|------|------|
| `AgentConfig` | config.py | 환경변수 기반 설정 TypedDict |
| `load_config()` | config.py | `.env` 로드 + 환경변수 파싱 |
| `configure_tracing()` | observability.py | OpenTelemetry 트레이싱 설정 |
| `get_tracer()` | observability.py | 현재 트레이서 인스턴스 반환 |

---

## 성능 최적화 사항

v6는 다음과 같은 성능 최적화가 적용되어 있습니다:

| 최적화 | 대상 | 효과 |
|--------|------|------|
| `__slots__` | Content, Message, AgentResponse, AgentResponseUpdate | 메모리 사용량 감소, 속성 접근 속도 향상 |
| `_SERIALIZE_FIELDS` 튜플 | Content | 직렬화 시 동적 필드 탐색 제거 |
| `{*usage1, *usage2}` 셋 언패킹 | `add_usage_details()` | 키 합산 시 set 생성 최적화 |
| 모듈 레벨 임포트 | agents.py (`json`, `os`, `re`) | 함수 호출마다 임포트하는 오버헤드 제거 |
| 모듈 레벨 `_env()` | agents.py | 클래스 내부 → 모듈 수준으로 승격 |
| `asyncio.to_thread()` | tools.py | 더 이상 사용되지 않는 `get_event_loop()` 대체 |
| `types.UnionType` 지원 | tools.py | Python 3.10+ 파이프 문법(`X | Y`) 스키마 변환 |
| 클로저 캡처 수정 | middleware.py `_bind_middleware()` | 루프 내 async 클로저 변수 캡처 버그 방지 |
| 로컬 복사본 사용 | agents.py `context_providers` | `auto-inject` 시 Agent 인스턴스 상태 오염 방지 |

---

## v5 → v6 마이그레이션 가이드

| v5 (Runner 기반) | v6 (Agent 클래스 기반) | 비고 |
|---|---|---|
| `run_agent("질문", model="gpt-5.2")` | `Agent(client=OpenAIChatClient(model_id="gpt-5.2"))` | 클래스 기반 진입점 |
| `run_agent("질문")` | `await agent.run("질문")` | async/await |
| `Memory()` | `InMemoryHistoryProvider()` | ContextProvider 패턴 |
| `@mcp_tool` | `@tool` | 간소화된 데코레이터 |
| `AgentResult` | `AgentResponse` (별칭 유지) | Content 기반 |
| `StreamChunk` | `AgentResponseUpdate` (별칭 유지) | Content 기반 |
| 3개 엔진 (Direct/LangChain/CrewAI) | `ChatClient` 주입 | 프로바이더 교체 |
| 없음 | `AgentMiddleware` | 미들웨어 파이프라인 (신규) |
| 없음 | `agent.as_tool()` | 네이티브 멀티 에이전트 (신규) |

> **하위 호환**: `run_agent()` 함수와 `AgentResult` / `StreamChunk` 별칭은 v6에서도 그대로 사용 가능합니다.

---

## 공식 레퍼런스

- [Microsoft Agent Framework 1.0.0-rc1](https://github.com/microsoft/agent-framework)
- [Azure OpenAI Service](https://learn.microsoft.com/azure/ai-services/openai/)
- [OpenAI Function Calling](https://platform.openai.com/docs/guides/function-calling)
- [OpenTelemetry Python](https://opentelemetry.io/docs/languages/python/)

---

## 라이선스 (License)

MIT License — see [LICENSE](./LICENSE) for details.
