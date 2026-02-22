#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Unified Agent Framework v6 — Demo (7개 시나리오)

================================================================================
Microsoft Agent Framework 1.0.0-rc1 API 패턴으로 재설계된 데모.

7개 시나리오로 프레임워크의 전체 기능을 검증합니다:
    1. 기본 Agent       — Agent.run(), AgentResponse, UsageDetails
    2. 도구 사용         — @tool, FunctionTool, 자동 도구 호출 루프
    3. 멀티턴 대화       — AgentSession, InMemoryHistoryProvider
    4. 커스텀 Provider   — BaseContextProvider.before_run(), 동적 지시사항
    5. 멀티 에이전트      — agent.as_tool(), 오케스트레이터 패턴
    6. 스트리밍          — agent.run(stream=True), AgentResponseUpdate
    7. v5 호환           — run_agent() 래퍼, AgentResult 별칭

사전 준비:
    pip install openai python-dotenv
    # .env 파일에 OPENAI_API_KEY 또는 AZURE_OPENAI_* 설정

실행:
    python demo_v6.py           # 전체 시나리오
    python demo_v6.py 1 3 5     # 특정 시나리오만
================================================================================
"""

import asyncio
import os
import sys

# 패키지 경로 추가
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# .env 파일 로드 (상위 디렉토리 포함)
from dotenv import load_dotenv
load_dotenv(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".env"), override=False)
load_dotenv(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".env"), override=False)

from unified_agent_v6 import (
    Agent,
    AgentSession,
    BaseContextProvider,
    InMemoryHistoryProvider,
    OpenAIChatClient,
    Content,
    Message,
    AgentResponse,
    tool,
    load_config,
)


# ─── 도구 정의 ───────────────────────────────────────────────

@tool
def get_weather(city: str) -> str:
    """도시의 현재 날씨를 반환합니다.

    Args:
        city: 날씨를 확인할 도시 이름
    """
    weather_data = {
        "서울": "맑음 22°C, 습도 45%",
        "부산": "흐림 18°C, 습도 70%",
        "제주": "비 16°C, 습도 85%",
        "New York": "Sunny 24°C, Humidity 50%",
        "London": "Cloudy 15°C, Humidity 65%",
    }
    return weather_data.get(city, f"{city}: 날씨 정보를 찾을 수 없습니다.")


@tool
def calculate(expression: str) -> str:
    """수학 수식을 계산합니다.

    Args:
        expression: 계산할 수식 (예: "2 + 3 * 4")
    """
    try:
        # 안전한 eval (기본 수학만 허용)
        allowed = {
            "__builtins__": {},
            "abs": abs, "round": round, "min": min, "max": max,
            "pow": pow, "sum": sum, "len": len,
        }
        result = eval(expression, allowed, {})
        return f"{expression} = {result}"
    except Exception as e:
        return f"계산 오류: {e}"


@tool(name="search_knowledge", description="내부 지식 베이스를 검색합니다")
def search_knowledge(query: str, max_results: int = 3) -> str:
    """내부 지식 베이스 검색 시뮬레이션."""
    return f"검색 결과 ({query}): Azure OpenAI Service는 GPT-4o, o1 등의 모델을 제공합니다."


# ─── 시나리오 1: 기본 에이전트 ────────────────────────────────

async def scenario_1_basic_agent():
    """시나리오 1: Agent 기본 사용법"""
    print("\n" + "=" * 60)
    print("시나리오 1: 기본 Agent 사용법")
    print("=" * 60)

    client = OpenAIChatClient()
    agent = Agent(
        client=client,
        instructions="당신은 친절한 한국어 AI 어시스턴트입니다. 간결하게 답변하세요.",
        name="기본 어시스턴트",
    )

    response = await agent.run("안녕하세요! 오늘 기분이 어떠세요?")
    print(f"\n🤖 응답: {response.text}")
    print(f"📊 토큰 사용량: {response.usage_details}")


# ─── 시나리오 2: 도구 사용 ────────────────────────────────────

async def scenario_2_tools():
    """시나리오 2: @tool 데코레이터로 도구 사용"""
    print("\n" + "=" * 60)
    print("시나리오 2: 도구(Tool) 사용")
    print("=" * 60)

    client = OpenAIChatClient()
    agent = Agent(
        client=client,
        instructions="당신은 날씨와 계산을 도와주는 AI 어시스턴트입니다.",
        tools=[get_weather, calculate],
        name="도구 어시스턴트",
    )

    # 날씨 질문 → get_weather 도구 호출
    response = await agent.run("서울과 부산의 날씨를 알려줘")
    print(f"\n🌤️ 날씨 응답: {response.text}")

    # 계산 질문 → calculate 도구 호출
    response = await agent.run("(15 * 23) + (47 * 8)을 계산해줘")
    print(f"\n🧮 계산 응답: {response.text}")


# ─── 시나리오 3: 멀티턴 대화 (세션) ──────────────────────────

async def scenario_3_session():
    """시나리오 3: AgentSession을 사용한 멀티턴 대화"""
    print("\n" + "=" * 60)
    print("시나리오 3: 멀티턴 대화 (AgentSession)")
    print("=" * 60)

    client = OpenAIChatClient()
    agent = Agent(
        client=client,
        instructions="당신은 친절한 AI 어시스턴트입니다. 사용자의 이전 발화를 기억하세요.",
        context_providers=[InMemoryHistoryProvider(max_messages=50)],
        name="대화 어시스턴트",
    )

    session = agent.create_session()

    # 대화 1
    response = await agent.run("제 이름은 김철수이고, 서울에 살고 있어요.", session=session)
    print(f"\n턴 1 🤖: {response.text}")

    # 대화 2 — 이전 대화 기억 확인
    response = await agent.run("제 이름이 뭐였죠?", session=session)
    print(f"턴 2 🤖: {response.text}")

    # 대화 3 — 추가 맥락 기억
    response = await agent.run("저는 어디에 살고 있었나요?", session=session)
    print(f"턴 3 🤖: {response.text}")


# ─── 시나리오 4: 커스텀 ContextProvider ──────────────────────

async def scenario_4_custom_provider():
    """시나리오 4: 커스텀 ContextProvider로 동적 지시사항 주입"""
    print("\n" + "=" * 60)
    print("시나리오 4: 커스텀 ContextProvider")
    print("=" * 60)

    class TimeAwareProvider(BaseContextProvider):
        """현재 시간 정보를 자동으로 주입하는 프로바이더."""

        DEFAULT_SOURCE_ID = "time_aware"

        async def before_run(self, *, agent, session, context, state):
            from datetime import datetime
            now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            context.extend_instructions(
                self.source_id,
                f"현재 시각은 {now}입니다. 시간 관련 질문에 이 정보를 사용하세요.",
            )

    client = OpenAIChatClient()
    agent = Agent(
        client=client,
        instructions="당신은 시간 정보를 포함하여 답변하는 AI 어시스턴트입니다.",
        context_providers=[
            TimeAwareProvider(),
            InMemoryHistoryProvider(),
        ],
        name="시간 인식 어시스턴트",
    )

    session = agent.create_session()
    response = await agent.run("지금 몇 시인가요?", session=session)
    print(f"\n🕐 응답: {response.text}")


# ─── 시나리오 5: Agent를 도구로 사용 (멀티 에이전트) ──────────

async def scenario_5_agent_as_tool():
    """시나리오 5: 에이전트를 다른 에이전트의 도구로 사용"""
    print("\n" + "=" * 60)
    print("시나리오 5: 에이전트를 도구로 사용 (멀티 에이전트)")
    print("=" * 60)

    client = OpenAIChatClient()

    # 전문가 에이전트 1: 날씨 전문가
    weather_agent = Agent(
        client=client,
        instructions="당신은 날씨 전문가입니다. 날씨 관련 질문에만 답변하세요.",
        tools=[get_weather],
        name="weather_expert",
        description="날씨 관련 질문을 처리하는 전문가 에이전트",
    )

    # 전문가 에이전트 2: 계산 전문가
    calc_agent = Agent(
        client=client,
        instructions="당신은 수학 계산 전문가입니다. 계산 관련 질문에만 답변하세요.",
        tools=[calculate],
        name="calc_expert",
        description="수학 계산을 처리하는 전문가 에이전트",
    )

    # 오케스트레이터 에이전트
    orchestrator = Agent(
        client=client,
        instructions=(
            "당신은 오케스트레이터입니다. 사용자의 질문을 분석하여 "
            "적절한 전문가 에이전트에게 위임하세요."
        ),
        tools=[
            weather_agent.as_tool(),
            calc_agent.as_tool(),
        ],
        name="오케스트레이터",
    )

    response = await orchestrator.run("서울 날씨와 123 * 456을 동시에 알려줘")
    print(f"\n🎯 오케스트레이터 응답: {response.text}")


# ─── 시나리오 6: 스트리밍 ────────────────────────────────────

async def scenario_6_streaming():
    """시나리오 6: 스트리밍 응답"""
    print("\n" + "=" * 60)
    print("시나리오 6: 스트리밍 응답")
    print("=" * 60)

    client = OpenAIChatClient()
    agent = Agent(
        client=client,
        instructions="당신은 간결하게 답변하는 AI 어시스턴트입니다.",
    )

    print("\n🔄 스트리밍: ", end="")
    async for update in agent.run("Python의 장점 3가지를 짧게 설명해줘", stream=True):
        print(update.text, end="", flush=True)
    print()  # 줄바꿈


# ─── 시나리오 7: v5 호환 run_agent() ─────────────────────────

async def scenario_7_legacy_compat():
    """시나리오 7: v5 호환 run_agent() 함수"""
    print("\n" + "=" * 60)
    print("시나리오 7: v5 호환 run_agent() 함수")
    print("=" * 60)

    from unified_agent_v6 import run_agent

    response = await run_agent(
        "Python에서 리스트 컴프리헨션의 예시를 보여줘",
        instructions="당신은 Python 프로그래밍 교사입니다. 간결하게 답변하세요.",
    )
    print(f"\n📝 응답: {response.text}")


# ─── 메인 실행 ───────────────────────────────────────────────

async def main():
    """모든 시나리오 실행."""
    print("🚀 Unified Agent Framework v6.0.0 — Demo")
    print(f"   (Microsoft Agent Framework 1.0.0-rc1 호환)")
    print("=" * 60)

    # 설정 로드
    config = load_config()
    print(f"📋 모델: {config.get('openai_model', 'gpt-5.2')}")

    scenarios = [
        ("1", "기본 Agent", scenario_1_basic_agent),
        ("2", "도구 사용", scenario_2_tools),
        ("3", "멀티턴 대화", scenario_3_session),
        ("4", "커스텀 Provider", scenario_4_custom_provider),
        ("5", "멀티 에이전트", scenario_5_agent_as_tool),
        ("6", "스트리밍", scenario_6_streaming),
        ("7", "v5 호환 모드", scenario_7_legacy_compat),
    ]

    # 인수로 특정 시나리오 선택 가능
    selected = sys.argv[1:] if len(sys.argv) > 1 else None

    for num, name, func in scenarios:
        if selected and num not in selected:
            continue
        try:
            await func()
        except Exception as e:
            print(f"\n❌ 시나리오 {num} ({name}) 실패: {e}")
            import traceback
            traceback.print_exc()


if __name__ == "__main__":
    asyncio.run(main())
