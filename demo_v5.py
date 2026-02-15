#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Unified Agent Framework v5 — 실행 데모

v5의 Runner 중심 설계를 보여주는 데모입니다.
API 키 없이도 구조 검증이 가능합니다.
"""

import asyncio
import sys
from pathlib import Path

# Windows cp949 환경에서 이모지 출력 시 UnicodeEncodeError 방지
if sys.stdout and hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if sys.stderr and hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

sys.path.insert(0, str(Path(__file__).parent))


async def demo_1_basic_types():
    """데모 1: 핵심 타입 검증"""
    print("\n" + "=" * 60)
    print("📦 데모 1: Core Types (Message, Memory, Tool)")
    print("=" * 60)

    from unified_agent_v5 import Message, Role, Memory, Tool, AgentConfig

    # Message
    msg = Message.user("안녕하세요!")
    print(f"  ✅ Message: {msg.to_dict()}")

    msg2 = Message.assistant("반갑습니다!")
    print(f"  ✅ Message: {msg2.to_dict()}")

    # Memory
    memory = Memory(system_prompt="You are a Python expert.")
    memory.add_user("파이썬이란?")
    memory.add_assistant("파이썬은 프로그래밍 언어입니다.")
    memory.add_user("더 자세히 알려줘")

    print(f"  ✅ Memory: {memory}")
    print(f"     메시지 수: {len(memory)}")
    print(f"     히스토리:")
    for m in memory.get_messages():
        print(f"       [{m['role']}] {m['content'][:50]}...")

    # Memory 직렬화
    json_str = memory.to_json()
    restored = Memory.from_json(json_str)
    print(f"  ✅ Memory 직렬화/복원: {restored}")

    # Tool
    async def search(query: str) -> str:
        return f"검색 결과: {query}"

    tool = Tool(
        name="web_search",
        description="웹 검색",
        parameters={"query": {"type": "string", "description": "검색어"}},
        fn=search,
    )
    schema = tool.to_openai_schema()
    print(f"  ✅ Tool 스키마: {schema['function']['name']}")

    result = await tool.execute(query="Python tutorial")
    print(f"  ✅ Tool 실행: {result}")

    # Config
    config = AgentConfig(model="gpt-5.2", engine="direct")
    print(f"  ✅ Config: model={config.model}, engine={config.engine}")

    print("\n  🎉 모든 Core Types 검증 완료!")
    return True


async def demo_2_tool_decorator():
    """데모 2: @mcp_tool 데코레이터"""
    print("\n" + "=" * 60)
    print("🔧 데모 2: @mcp_tool 데코레이터")
    print("=" * 60)

    from unified_agent_v5 import mcp_tool, ToolRegistry

    @mcp_tool(description="날씨 조회")
    async def get_weather(city: str) -> str:
        """도시의 현재 날씨를 조회합니다.
        city: 조회할 도시 이름
        """
        return f"{city}: 맑음, 22°C"

    @mcp_tool(description="환율 조회")
    async def get_exchange_rate(from_currency: str, to_currency: str) -> str:
        """환율을 조회합니다."""
        return f"1 {from_currency} = 1,350 {to_currency}"

    print(f"  ✅ get_weather 스키마: {get_weather.to_openai_schema()['function']['name']}")
    print(f"  ✅ get_exchange_rate 스키마: {get_exchange_rate.to_openai_schema()['function']['name']}")

    # 레지스트리
    registry = ToolRegistry()
    registry.register(get_weather)
    registry.register(get_exchange_rate)
    print(f"  ✅ Registry: {registry}")
    print(f"     스키마: {len(registry.get_openai_schemas())}개")

    # 실행
    result = await get_weather.execute(city="서울")
    print(f"  ✅ 실행 결과: {result}")

    print("\n  🎉 데코레이터 검증 완료!")
    return True


async def demo_3_callbacks():
    """데모 3: 콜백 시스템"""
    print("\n" + "=" * 60)
    print("📊 데모 3: Callback / Observability")
    print("=" * 60)

    from unified_agent_v5 import (
        LoggingCallbackHandler,
        OTelCallbackHandler,
        CompositeCallbackHandler,
        AgentResult,
    )
    from unified_agent_v5.types import ToolCall, ToolResult

    # Logging 콜백
    log_cb = LoggingCallbackHandler()
    await log_cb.on_agent_start("테스트 질문")

    result = AgentResult(
        content="테스트 응답",
        model="gpt-5.2",
        engine="direct",
        usage={"input_tokens": 10, "output_tokens": 20, "total_tokens": 30},
        duration_ms=150.0,
    )
    await log_cb.on_agent_end(result)
    print("  ✅ LoggingCallbackHandler 동작 확인")

    # OTEL 콜백 (설치 안 되어 있어도 안전)
    otel_cb = OTelCallbackHandler(service_name="test-agent")
    await otel_cb.on_agent_start("OTEL 테스트")
    print("  ✅ OTelCallbackHandler 초기화 (graceful degradation)")

    # Composite
    composite = CompositeCallbackHandler([log_cb, otel_cb])
    await composite.on_agent_start("복합 테스트")
    await composite.on_llm_start("gpt-5.2", [{"role": "user", "content": "test"}])
    await composite.on_tool_start(ToolCall(id="tc-1", name="search", arguments={"q": "test"}))
    await composite.on_tool_end(ToolResult(tool_call_id="tc-1", name="search", content="result"))
    await composite.on_llm_end("응답", {"total_tokens": 50})
    await composite.on_agent_end(result)
    print("  ✅ CompositeCallbackHandler 동작 확인")

    print("\n  🎉 콜백 시스템 검증 완료!")
    return True


async def demo_4_engine_registry():
    """데모 4: 엔진 레지스트리"""
    print("\n" + "=" * 60)
    print("⚙️ 데모 4: Engine Registry")
    print("=" * 60)

    from unified_agent_v5.engines import get_engine

    # Direct 엔진 (항상 사용 가능)
    engine = get_engine("direct")
    print(f"  ✅ Direct 엔진: {type(engine).__name__}")

    # LangChain 엔진 (설치 여부에 따라)
    try:
        lc_engine = get_engine("langchain")
        print(f"  ✅ LangChain 엔진: {type(lc_engine).__name__}")
    except (ValueError, ImportError) as e:
        print(f"  ⏭️ LangChain 미설치: {e}")

    # CrewAI 엔진 (설치 여부에 따라)
    try:
        crew_engine = get_engine("crewai")
        print(f"  ✅ CrewAI 엔진: {type(crew_engine).__name__}")
    except (ValueError, ImportError) as e:
        print(f"  ⏭️ CrewAI 미설치: {e}")

    # 존재하지 않는 엔진
    try:
        get_engine("nonexistent")
    except ValueError as e:
        print(f"  ✅ 유효성 검증: {e}")

    print("\n  🎉 엔진 레지스트리 검증 완료!")
    return True


async def demo_5_runner():
    """데모 5: Runner 구조 검증 (API 호출 없이)"""
    print("\n" + "=" * 60)
    print("🚀 데모 5: Runner 설계 검증")
    print("=" * 60)

    from unified_agent_v5 import Runner, AgentConfig, Memory

    # Runner 생성
    runner = Runner(config=AgentConfig(
        model="gpt-5.2",
        engine="direct",
        system_prompt="You are a Python expert.",
    ))
    print(f"  ✅ Runner 생성: engine={runner.config.engine}, model={runner.config.model}")

    # Memory를 통한 대화 관리
    memory = Memory(system_prompt="You are a helpful assistant.")
    print(f"  ✅ Memory 생성: {memory}")

    # run_agent 함수 import 확인
    from unified_agent_v5 import run_agent
    print(f"  ✅ run_agent 함수 사용 가능")

    print("""
    💡 실제 사용 예시 (API 키가 있을 때):

        # 가장 간단한 사용
        result = await run_agent("파이썬 피보나치 함수 작성해줘")
        print(result.content)

        # 대화 이어가기
        memory = Memory(system_prompt="You are helpful.")
        r1 = await run_agent("내 이름은 철수야", memory=memory)
        r2 = await run_agent("내 이름이 뭐였지?", memory=memory)

        # 도구 사용
        @mcp_tool(description="날씨 조회")
        async def get_weather(city: str) -> str:
            return f"{city}: 맑음"
        result = await run_agent("서울 날씨", tools=[get_weather])

        # 멀티 에이전트 (CrewAI)
        result = await run_agent(
            "시장 분석 보고서",
            engine="crewai",
            crew_agents=[
                {"role": "Researcher", "goal": "데이터 수집"},
                {"role": "Writer", "goal": "보고서 작성"},
            ]
        )
    """)

    print("  🎉 Runner 설계 검증 완료!")
    return True


async def demo_6_comparison():
    """데모 6: v4.1 vs v5 비교"""
    print("\n" + "=" * 60)
    print("📊 데모 6: v4.1 → v5 개선 비교")
    print("=" * 60)

    print("""
    ┌─────────────────┬──────────────────────────────┬──────────────────────────────┐
    │ 항목            │ v4.1 (기존)                   │ v5 (개선)                     │
    ├─────────────────┼──────────────────────────────┼──────────────────────────────┤
    │ 모듈 수         │ 49개 모듈, 380+ API           │ 9개 모듈, 20개 API            │
    │ 엔진            │ 16개 프레임워크 브릿지         │ Top 3 + Direct               │
    │ 모니터링        │ 자체 Tracer/Dashboard/DB      │ OTEL 어댑터 (Export only)     │
    │ 메모리          │ 6개 메모리 시스템              │ List[Message] + JSON 직렬화  │
    │ 도구            │ 프레임워크별 다른 방식          │ MCP 표준 + OpenAI 스키마     │
    │ 진입점          │ UnifiedAgentFramework.create() │ run_agent("질문")            │
    │ 의존성          │ semantic-kernel 필수           │ openai만 필수, 나머지 선택    │
    │ 사용 난이도     │ 높음 (설정/이해 필요)          │ 낮음 (한 줄로 시작)           │
    └─────────────────┴──────────────────────────────┴──────────────────────────────┘

    🎯 핵심 변경:
    1. "16개 지원" → Top 3 + Direct (LangChain, CrewAI, Direct API)
    2. 자체 모니터링 → OTEL 표준 어댑터 (callback_handler 패턴)
    3. 복잡한 메모리 → 단순 List[Message]
    4. 프레임워크별 도구 → MCP 표준 일원화
    5. "만드는 도구" → "실행하는 Runner"
    """)

    print("  🎉 비교 완료!")
    return True


async def main():
    """전체 데모 실행"""
    print("=" * 60)
    print("🚀 Unified Agent Framework v5 — 데모")
    print("   Runner-Centric Design")
    print("=" * 60)

    demos = [
        ("Core Types", demo_1_basic_types),
        ("@mcp_tool Decorator", demo_2_tool_decorator),
        ("Callback System", demo_3_callbacks),
        ("Engine Registry", demo_4_engine_registry),
        ("Runner Design", demo_5_runner),
        ("v4.1 vs v5 Comparison", demo_6_comparison),
    ]

    results = []
    for name, demo_fn in demos:
        try:
            success = await demo_fn()
            results.append((name, success))
        except Exception as e:
            print(f"\n  ❌ {name} 실패: {e}")
            import traceback
            traceback.print_exc()
            results.append((name, False))

    # 결과 요약
    print("\n" + "=" * 60)
    print("📋 데모 결과 요약")
    print("=" * 60)
    passed = sum(1 for _, s in results if s)
    total = len(results)
    for name, success in results:
        status = "✅" if success else "❌"
        print(f"  {status} {name}")
    print(f"\n  결과: {passed}/{total} 통과")
    print("=" * 60)


if __name__ == "__main__":
    asyncio.run(main())
