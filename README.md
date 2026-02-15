# Unified Agent Framework

**"가장 잘 동작하는 것 하나를 쉽고 안정적으로"**

[![Python](https://img.shields.io/badge/Python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![v5](https://img.shields.io/badge/Latest-v5.0.0-blue.svg)](unified_agent_v5/README.md)
[![Engines](https://img.shields.io/badge/Engines-Direct%20|%20LangChain%20|%20CrewAI-orange.svg)](unified_agent_v5/README.md#엔진)
[![OTEL](https://img.shields.io/badge/Observability-OpenTelemetry-teal.svg)](unified_agent_v5/README.md#모니터링-otel-어댑터)
[![Tests](https://img.shields.io/badge/Tests-6%2F6%20Passed-success.svg)](demo_v5.py)

> **v5.0.0** (2026-02-15) — v4.1의 49개 모듈을 **9개 모듈**로, 16개 브릿지를 **Top 3 엔진**으로 전면 재설계.
> `run_agent("질문")` 한 줄로 끝냅니다.

```python
from unified_agent_v5 import run_agent

result = await run_agent("파이썬으로 피보나치 함수 작성해줘")
print(result.content)
```

---

## 목차

- [빠른 시작](#빠른-시작)
- [설치](#설치)
- [v4.1 → v5 변경 요약](#v41--v5-변경-요약)
- [프로젝트 구조](#프로젝트-구조)
- [v5 상세 문서](#v5-상세-문서)
- [v4.1 레거시](#v41-레거시)
- [라이선스](#라이선스)

---

## 빠른 시작

### 1. 가장 간단한 호출

```python
from unified_agent_v5 import run_agent

result = await run_agent("안녕하세요!")
print(result.content)      # "안녕하세요! 무엇을 도와드릴까요?"
print(result.usage)         # {'input_tokens': 12, 'output_tokens': 15, ...}
print(result.duration_ms)   # 340.5
```

### 2. 대화 이어가기

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
from unified_agent_v5 import run_agent, OTelCallbackHandler

otel = OTelCallbackHandler(endpoint="http://localhost:4318")
result = await run_agent("질문", callbacks=[otel])
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

환경변수 (`.env`):

```bash
# Azure OpenAI (권장)
AZURE_OPENAI_ENDPOINT=https://your-resource.openai.azure.com/
AZURE_OPENAI_API_KEY=your-key
AZURE_OPENAI_DEPLOYMENT=your-deployment

# 또는 OpenAI 직접
OPENAI_API_KEY=sk-...

# 기본 모델/엔진
AGENT_MODEL=gpt-5.2
AGENT_ENGINE=direct
```

---

## v4.1 → v5 변경 요약

| 항목 | v4.1 | v5 | 축소율 |
|------|------|----|--------|
| 모듈 수 | 49개, 380+ API | **9개, ~20 API** | 82% ↓ |
| 엔진 | 16개 프레임워크 브릿지 | **3개 (Direct, LangChain, CrewAI)** | 81% ↓ |
| 모니터링 | 자체 Tracer/Dashboard/DB | **OTEL 어댑터 (Export only)** | — |
| 메모리 | 6개 메모리 시스템 | **List[Message] + JSON** | — |
| 도구 | 프레임워크별 다른 방식 | **MCP 표준 + OpenAI 스키마** | — |
| 진입점 | `UnifiedAgentFramework.create()` | **`run_agent("질문")`** | — |
| 필수 의존성 | 8개 (`semantic-kernel` 포함) | **2개 (`openai`, `python-dotenv`)** | 75% ↓ |

> 상세 축소 이유 및 모듈별 매핑: [v5 README — 모듈 축소 상세 매핑](unified_agent_v5/README.md#v41--v5-모듈-축소-상세-매핑)

---

## 프로젝트 구조

```
Unified-agent-framework/
│
├── unified_agent_v5/              # ★ v5 — Runner 중심 설계 (9개 모듈)
│   ├── __init__.py                #   공개 API (~20개)
│   ├── runner.py                  #   run_agent() 핵심 진입점
│   ├── types.py                   #   Message, AgentResult (OpenAI 표준)
│   ├── config.py                  #   Settings, AgentConfig
│   ├── memory.py                  #   List[Message] 메모리
│   ├── tools.py                   #   MCP 표준 Tool + @mcp_tool
│   ├── callback.py                #   OTEL 콜백 어댑터
│   ├── engines/                   #   Top 3 엔진
│   │   ├── base.py                #     EngineProtocol + 레지스트리
│   │   ├── direct.py              #     OpenAI/Azure 직접 호출
│   │   ├── langchain_engine.py    #     LangChain 체인/RAG
│   │   └── crewai_engine.py       #     CrewAI 멀티 에이전트
│   ├── plugins/                   #   v4 기능 마이그레이션 (선택)
│   └── README.md                  #   v5 상세 문서
│
├── _legacy/                       # v4.1 아카이브 (49개 모듈, 참고용)
├── skills/                        # v4.1 스킬 정의 (SKILL.md)
│
├── demo_v5.py                     # v5 데모 (6/6 통과)
│
├── README.md                      # ← 지금 보는 파일
├── CHANGELOG.md                   # 변경 이력
├── pyproject.toml                 # 패키지 설정
├── requirements.txt               # 의존성
├── LICENSE                        # MIT
└── .github/                       # CI/CD
```

---

## v5 상세 문서

v5의 설계 원칙, 모듈 축소 이유, API 레퍼런스 등 상세 내용:

→ **[unified_agent_v5/README.md](unified_agent_v5/README.md)**

주요 섹션:
- [왜 v5인가? — 축소 이유](unified_agent_v5/README.md#왜-v5인가--축소-이유)
- [모듈 축소 상세 매핑 (49개 → 9개)](unified_agent_v5/README.md#v41--v5-모듈-축소-상세-매핑)
- [설계 원칙 4가지](unified_agent_v5/README.md#설계-원칙-4가지)
- [엔진 (Direct, LangChain, CrewAI)](unified_agent_v5/README.md#엔진)
- [도구 (MCP 표준)](unified_agent_v5/README.md#도구-mcp-표준)
- [메모리](unified_agent_v5/README.md#메모리)
- [모니터링 (OTEL 어댑터)](unified_agent_v5/README.md#모니터링-otel-어댑터)
- [플러그인](unified_agent_v5/README.md#플러그인-v4-기능-마이그레이션)

---

## v4.1 레거시

v4.1 코드는 `_legacy/` 디렉토리에 아카이브되어 있습니다. 참고용으로만 보존됩니다.

---

## 라이선스

[MIT License](LICENSE)
