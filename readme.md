# 🚀 Unified Agent Framework - Enterprise Edition (v2.0)

**Microsoft Agent Framework 패턴(Semantic Kernel, AutoGen) 통합 및 엔터프라이즈급 안정성/성능이 강화된 차세대 AI 에이전트 오케스트레이션 프레임워크**

[![Python](https://img.shields.io/badge/Python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Semantic Kernel](https://img.shields.io/badge/Semantic_Kernel-Latest-orange.svg)](https://github.com/microsoft/semantic-kernel)
[![OpenTelemetry](https://img.shields.io/badge/OpenTelemetry-Enabled-purple.svg)](https://opentelemetry.io/)
[![Enterprise](https://img.shields.io/badge/Edition-Enterprise_v2.0-blueviolet.svg)]()

## 📖 목차

- [개요](#-개요)
- [v2.0 주요 업데이트](#-v20-주요-업데이트)
- [핵심 기능](#-핵심-기능)
- [설치](#-설치)
- [빠른 시작](#-빠른-시작)
- [아키텍처](#️-아키텍처)
- [주요 컴포넌트](#-주요-컴포넌트)
- [실전 예제 (Demos)](#-실전-예제-demos)
- [성능 및 안정성](#-성능-및-안정성)
- [운영 및 모니터링](#-운영-및-모니터링)
- [문제 해결 (Troubleshooting)](#-문제-해결-troubleshooting)

---

## 🎯 개요

**Unified Agent Framework - Enterprise Edition (v2.0)**은 단순한 LLM 래퍼를 넘어, 복잡한 비즈니스 프로세스를 자동화할 수 있는 **자율 에이전트 오케스트레이션 플랫폼**입니다.

Microsoft의 **Semantic Kernel**을 기반으로 **AutoGen**의 멀티 에이전트 협업 패턴을 통합하고, 실제 운영 환경에 필수적인 **MCP(Model Context Protocol)**, **회로 차단기**, **구조화된 로깅**, **고성능 캐시** 등을 내장했습니다.

### 버전 비교

| 구분 | 기존 버전 (v1.0) | Optimized Edition (v1.5) | Enterprise Edition (v2.0) |
|------|------------------|--------------------------|---------------------------|
| **협업 방식** | 단일/순차 실행 | 조건부 라우팅 | **Dynamic Supervisor (LLM 기반 동적 협업)** |
| **외부 연동** | 하드코딩된 함수 | 플러그인 구조 | **MCP (Model Context Protocol) 표준 연동** |
| **안정성** | 기본 예외 처리 | 회로 차단기 (Circuit Breaker) | **Backoff 재시도 + Smart Approval + 회로 차단기** |
| **로깅** | 텍스트 로그 | 기본 메트릭 | **JSON 구조화 로깅 (ELK 연동) + OpenTelemetry** |
| **성능** | 기본 메모리 | TTL LRU 캐시 | **최적화된 비동기 파이프라인 + 캐시** |

---

## 🚀 v2.0 주요 업데이트

### 1. Dynamic Supervisor Agent (AutoGen 패턴)
기존의 정해진 순서대로 실행하는 방식이 아닌, **LLM Supervisor**가 대화의 맥락과 목표를 분석하여 **다음에 실행할 에이전트를 동적으로 결정**합니다. 이를 통해 훨씬 유연하고 복잡한 문제 해결이 가능합니다.

### 2. MCP (Model Context Protocol) 통합
Anthropic이 제안한 **MCP 표준**을 지원하여, 로컬 또는 원격에 있는 다양한 도구(데이터베이스, API, 파일 시스템 등)를 표준화된 방식으로 에이전트에 연결할 수 있습니다. (현재 Mock Client 내장)

### 3. Smart Approval (지능형 승인)
모든 작업에 대해 사람의 승인을 기다리는 비효율을 개선했습니다. **읽기 전용(Read-only) 작업은 자동으로 승인**하고, 데이터 변경이나 비용이 발생하는 민감한 작업만 선별적으로 승인을 요청합니다.

### 4. Enterprise-Grade Reliability
- **Structured Logging**: 모든 로그를 JSON 포맷으로 출력하여 기계적 분석을 용이하게 했습니다.
- **Retry with Backoff**: 일시적인 API 오류 발생 시 지수 백오프(Exponential Backoff) 전략으로 스마트하게 재시도합니다.

---

## 📦 설치

### 필수 요구사항
- Python 3.10 이상
- Azure OpenAI 또는 OpenAI API 키

### 패키지 설치
```bash
pip install semantic-kernel python-dotenv pydantic opentelemetry-api opentelemetry-sdk
```

### 환경 변수 설정 (.env)
```bash
AZURE_OPENAI_API_KEY=your_api_key
AZURE_OPENAI_ENDPOINT=https://your-resource.openai.azure.com/
AZURE_OPENAI_DEPLOYMENT_NAME=gpt-4
```

---

## 🚀 빠른 시작

### 1분 만에 Supervisor Agent 실행하기

```python
import asyncio
from Semantic_agent_framework import (
    UnifiedAgentFramework, SimpleAgent, SupervisorAgent,
    Node, AgentRole, DEFAULT_LLM_MODEL
)
from semantic_kernel import Kernel

async def main():
    # 1. 프레임워크 초기화
    kernel = Kernel()
    # ... (Kernel 서비스 설정 생략, 전체 코드는 데모 참조) ...

    framework = UnifiedAgentFramework(kernel=kernel)

    # 2. 하위 에이전트 생성
    researcher = SimpleAgent(name="researcher", system_prompt="Research data.", model=DEFAULT_LLM_MODEL)
    writer = SimpleAgent(name="writer", system_prompt="Write content.", model=DEFAULT_LLM_MODEL)

    # 3. Supervisor 생성 (동적 라우팅)
    supervisor = SupervisorAgent(
        name="supervisor",
        sub_agents=[researcher, writer],
        max_rounds=5,
        model=DEFAULT_LLM_MODEL
    )

    # 4. 그래프 구성 및 실행
    graph = framework.create_graph("collaboration")
    graph.add_node(Node("supervisor", supervisor))
    graph.set_start("supervisor")
    graph.set_end("supervisor")

    # 5. 실행
    state = await framework.run_workflow(
        "collaboration",
        session_id="test-001",
        initial_input="AI 트렌드에 대해 조사하고 블로그 글을 써줘."
    )
    print(state.messages[-1].content)

if __name__ == "__main__":
    asyncio.run(main())
```

---

## 🏗️ 아키텍처

```
┌─────────────────────────────────────────────────────────────┐
│           Unified Agent Framework - Enterprise v2.0         │
├─────────────────────────────────────────────────────────────┤
│  ┌────────────────────────────────────────────────────────┐ │
│  │               Orchestration Layer                      │ │
│  ├────────────────────────────────────────────────────────┤ │
│  │  Dynamic Supervisor (LLM) │  Workflow Graph Engine     │ │
│  │  Smart Approval System    │  Event Bus (Pub/Sub)       │ │
│  └────────────────────────────────────────────────────────┘ │
├─────────────────────────────────────────────────────────────┤
│  ┌────────────────────────────────────────────────────────┐ │
│  │                  Agent Layer                           │ │
│  ├────────────────────────────────────────────────────────┤ │
│  │ SimpleAgent │ RouterAgent │ SupervisorAgent │ MCPTool  │ │
│  └────────────────────────────────────────────────────────┘ │
├─────────────────────────────────────────────────────────────┤
│  ┌────────────────────────────────────────────────────────┐ │
│  │               Infrastructure Layer                     │ │
│  ├────────────────────────────────────────────────────────┤ │
│  │ Structured Logger │ Retry/Backoff │ Circuit Breaker    │ │
│  │ TTL LRU Cache     │ State Manager │ OpenTelemetry      │ │
│  └────────────────────────────────────────────────────────┘ │
└─────────────────────────────────────────────────────────────┘
```

---

## 🔧 주요 컴포넌트

### 1. SupervisorAgent
Microsoft AutoGen의 패턴을 구현했습니다. 하위 에이전트들의 목록을 가지고 있으며, 현재 대화 상태를 기반으로 **"누가 다음에 말해야 하는가?"**를 LLM에게 물어보고 실행 흐름을 제어합니다.

### 2. MCPTool (MockMCPClient)
외부 MCP 서버와 통신하여 도구를 동적으로 로드하고 실행합니다. 현재는 데모용 `MockMCPClient`가 내장되어 있어 별도의 서버 없이도 `calculator`, `web_search` 등의 도구 사용 시나리오를 테스트할 수 있습니다.

### 3. StructuredLogger
모든 로그를 다음과 같은 JSON 포맷으로 남깁니다.
```json
{
  "timestamp": "2025-11-27T10:00:00Z",
  "level": "INFO",
  "message": "Supervisor Decision",
  "agent": "supervisor",
  "decision": "researcher"
}
```

### 4. CircuitBreaker & Retry
외부 API 호출 실패 시 즉시 에러를 내지 않고, 지수 백오프(Exponential Backoff)로 재시도합니다. 실패가 지속되면 회로 차단기(Circuit Breaker)가 작동하여 장애 확산을 방지합니다.

---

## 💡 실전 예제 (Demos)

프레임워크에는 4가지 핵심 데모가 포함되어 있습니다 (`main()` 함수 참조):

1.  **`demo_simple_chat`**: 가장 기본적인 단일 에이전트 대화.
2.  **`demo_routing_workflow`**: 사용자의 의도(Intent)를 분류하여 적절한 전문가 에이전트에게 라우팅.
3.  **`demo_supervisor_workflow`** (v2.0 New): Supervisor가 Researcher와 Writer를 조율하여 복합 작업 수행.
4.  **`demo_conditional_workflow`**: 질문의 복잡도에 따라 처리 경로를 분기하는 조건부 로직.

---

## ⚡ 성능 및 안정성

### TTL 기반 LRU 캐시
메모리 사용량을 일정 수준으로 유지하면서 자주 사용되는 데이터(인텐트 분류 결과 등)를 캐싱하여 응답 속도를 획기적으로 개선했습니다.

### 메모리 자동 정리
오래된 대화 기록이나 사용되지 않는 세션 데이터를 백그라운드에서 자동으로 정리하여 메모리 누수를 방지합니다.

### 회로 차단기 (Circuit Breaker)
- **Closed**: 정상 상태
- **Open**: 장애 발생 (일정 횟수 실패 시). 즉시 에러 반환하여 시스템 보호.
- **Half-Open**: 복구 시도. 일부 요청만 허용하여 정상화 여부 확인.

---

## ❓ 문제 해결 (Troubleshooting)

### ModuleNotFoundError: 'semantic_kernel'
패키지가 설치되지 않았습니다.
```bash
pip install semantic_kernel
```

### 401 Access Denied
`.env` 파일의 API Key와 Endpoint가 정확한지 확인하세요. 특히 Key에 따옴표(`"`)가 포함되어 있지 않은지 확인하세요.

### Supervisor가 엉뚱한 에이전트를 선택함
System Prompt를 튜닝하거나, 모델을 더 똑똑한 모델(gpt-4 등)로 변경해보세요. `DEFAULT_LLM_MODEL` 상수를 변경하면 됩니다.

---

<div align="center">
  <strong>Enterprise-Grade AI Agent Orchestration</strong><br>
  Made with ❤️ by the Unified Agent Framework Team
</div>