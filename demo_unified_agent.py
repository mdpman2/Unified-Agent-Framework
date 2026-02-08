#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Unified Agent Framework - 실행 데모

실제 프레임워크 기능 테스트
"""

import asyncio
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))


async def demo_framework_creation():
    """프레임워크 생성 테스트"""
    print("\n" + "="*60)
    print("🚀 프레임워크 생성 테스트")
    print("="*60)

    try:
        from unified_agent import (
            UnifiedAgentFramework,
            FrameworkConfig,
            AgentState,
            SimpleAgent,
            Graph,
            Node,
            EventBus,
            EventType,
        )

        # 1. 설정 객체 생성 (환경변수 없이 테스트용)
        print("\n📋 FrameworkConfig 생성...")
        config = FrameworkConfig(
            deployment_name="test-deployment",
            api_key="test-key",
            endpoint="https://test.openai.azure.com/",
            model="gpt-4o",
            temperature=0.7,
            max_tokens=1000
        )
        print(f"  ✅ 설정 생성 완료")
        print(f"     - 모델: {config.model}")
        print(f"     - Temperature: {config.temperature}")
        print(f"     - Max Tokens: {config.max_tokens}")

        # 2. 에이전트 생성
        print("\n📋 SimpleAgent 생성...")
        agent = SimpleAgent(
            name="test_agent",
            system_prompt="You are a helpful assistant.",
            model=config.model,
            temperature=config.temperature
        )
        print(f"  ✅ 에이전트 생성: {agent.name}")

        # 3. 그래프 생성
        print("\n📋 Workflow Graph 생성...")
        graph = Graph(name="test_workflow")
        node = Node(name="assistant", agent=agent)
        graph.add_node(node)
        graph.set_start("assistant")
        graph.set_end("assistant")
        print(f"  ✅ 그래프 생성: {graph.name}")
        print(f"     - 노드 수: {len(graph.nodes)}")

        # 4. 그래프 시각화
        print("\n📋 Workflow 시각화...")
        viz = graph.visualize()
        print(viz)

        # 5. 이벤트 버스 테스트
        print("\n📋 EventBus 테스트...")
        event_bus = EventBus()

        events_received = []
        async def event_handler(event):
            events_received.append(event)
            print(f"  📢 이벤트 수신: {event.event_type.value}")

        event_bus.subscribe(EventType.AGENT_STARTED, event_handler)
        event_bus.subscribe(EventType.AGENT_COMPLETED, event_handler)

        from unified_agent.events import AgentEvent

        await event_bus.publish(AgentEvent(
            event_type=EventType.AGENT_STARTED,
            agent_name="test_agent",
            data={"test": True}
        ))

        await asyncio.sleep(0.1)

        print(f"  ✅ 이벤트 수신 완료: {len(events_received)}개")

        # 6. AgentState 테스트
        print("\n📋 AgentState 테스트...")
        from unified_agent.models import AgentRole

        state = AgentState(
            session_id="demo-session",
            workflow_name="test_workflow"
        )
        state.add_message(AgentRole.USER, "Hello, assistant!")
        state.add_message(AgentRole.ASSISTANT, "Hello! How can I help you?", "test_agent")

        print(f"  ✅ 상태 생성: {state.session_id}")
        print(f"     - 메시지 수: {len(state.messages)}")

        history = state.get_conversation_history(max_messages=5)
        for msg in history:
            role = msg["role"]
            content = msg["content"]
            print(f"     - [{role}] {content[:50]}...")

        print("\n" + "="*60)
        print("✅ 프레임워크 생성 테스트 완료!")
        print("="*60)

        return True

    except Exception as e:
        print(f"\n❌ 오류 발생: {e}")
        import traceback
        traceback.print_exc()
        return False


async def demo_team_workflow():
    """팀 워크플로우 테스트"""
    print("\n" + "="*60)
    print("👥 팀 워크플로우 테스트")
    print("="*60)

    try:
        from unified_agent.models import TeamConfiguration, TeamAgent, AgentRole
        from unified_agent.orchestration import AgentFactory

        # 팀 설정 생성
        print("\n📋 TeamConfiguration 생성...")
        team_config = TeamConfiguration(
            name="research_team",
            description="연구 및 분석 팀",
            agents=[
                TeamAgent(
                    name="researcher",
                    description="데이터 수집 및 연구 담당",
                    role=AgentRole.ASSISTANT,
                    system_prompt="You are a research specialist."
                ),
                TeamAgent(
                    name="analyst",
                    description="데이터 분석 담당",
                    role=AgentRole.ASSISTANT,
                    system_prompt="You are a data analyst."
                ),
                TeamAgent(
                    name="writer",
                    description="보고서 작성 담당",
                    role=AgentRole.ASSISTANT,
                    system_prompt="You are a technical writer."
                ),
            ],
            orchestration_mode="supervisor",
            max_rounds=3
        )

        print(f"  ✅ 팀 설정 생성: {team_config.name}")
        print(f"     - 에이전트 수: {len(team_config.agents)}")
        for agent in team_config.agents:
            print(f"       • {agent.name}: {agent.description}")

        print("\n" + "="*60)
        print("✅ 팀 워크플로우 테스트 완료!")
        print("="*60)

        return True

    except Exception as e:
        print(f"\n❌ 오류 발생: {e}")
        import traceback
        traceback.print_exc()
        return False


async def demo_mplan():
    """MPlan 테스트"""
    print("\n" + "="*60)
    print("📋 MPlan 실행 계획 테스트")
    print("="*60)

    try:
        from unified_agent.models import MPlan, PlanStep, PlanStepStatus

        # 계획 생성
        print("\n📋 MPlan 생성...")
        plan = MPlan(
            name="research_plan",
            description="시장 조사 및 분석 계획",
            steps=[
                PlanStep(
                    index=0,
                    description="시장 데이터 수집",
                    agent_name="researcher"
                ),
                PlanStep(
                    index=1,
                    description="데이터 분석 및 인사이트 도출",
                    agent_name="analyst",
                    depends_on=[0]
                ),
                PlanStep(
                    index=2,
                    description="최종 보고서 작성",
                    agent_name="writer",
                    depends_on=[1]
                ),
            ],
        )

        print(f"  ✅ 계획 생성: {plan.name}")
        print(f"     - 단계 수: {len(plan.steps)}")

        # 진행률 확인
        print(f"\n📊 진행률: {plan.get_progress() * 100:.1f}%")

        # 다음 단계 확인
        current = plan.get_current_step()
        if current:
            print(f"\n📋 현재 단계:")
            print(f"   - Step {current.index}: {current.description} ({current.agent_name})")
        else:
            print(f"\n📋 모든 단계 대기 중")

        # 단계 완료 시뮬레이션
        print("\n📋 단계 완료 시뮬레이션...")
        plan.steps[0].start()
        plan.steps[0].complete("시장 데이터 수집 완료")
        print(f"  ✅ Step 0 완료")
        print(f"  📊 진행률: {plan.get_progress() * 100:.1f}%")

        plan.steps[1].start()
        plan.steps[1].complete("분석 완료")
        print(f"  ✅ Step 1 완료")
        print(f"  📊 진행률: {plan.get_progress() * 100:.1f}%")

        plan.steps[2].start()
        plan.steps[2].complete("보고서 작성 완료")
        print(f"  ✅ Step 2 완료")
        print(f"  📊 진행률: {plan.get_progress() * 100:.1f}%")

        print("\n" + "="*60)
        print("✅ MPlan 테스트 완료!")
        print("="*60)

        return True

    except Exception as e:
        print(f"\n❌ 오류 발생: {e}")
        import traceback
        traceback.print_exc()
        return False


async def demo_skills():
    """Skills 시스템 테스트"""
    print("\n" + "="*60)
    print("🎯 Skills 시스템 테스트")
    print("="*60)

    try:
        from unified_agent.skills import Skill, SkillManager, SkillResource

        # 스킬 매니저 생성
        print("\n📋 SkillManager 생성...")
        manager = SkillManager()

        # 스킬 등록
        print("\n📋 스킬 등록...")

        python_skill = Skill(
            name="python-expert",
            description="Python 프로그래밍 전문가",
            instructions="""You are a Python programming expert.
- Write clean, PEP 8 compliant code
- Use type hints for better code clarity
- Include docstrings for functions and classes
- Handle exceptions properly
""",
            triggers=["python", "파이썬", "코드", "프로그래밍"],
            priority=10
        )
        manager.register_skill(python_skill)
        print(f"  ✅ 스킬 등록: {python_skill.name}")

        azure_skill = Skill(
            name="azure-expert",
            description="Azure 클라우드 전문가",
            instructions="""You are an Azure cloud expert.
- Follow Azure best practices
- Recommend appropriate Azure services
- Consider security and cost optimization
""",
            triggers=["azure", "클라우드", "cloud", "애저"],
            priority=8
        )
        manager.register_skill(azure_skill)
        print(f"  ✅ 스킬 등록: {azure_skill.name}")

        # 스킬 조회
        print("\n📋 등록된 스킬 목록:")
        for skill in manager.list_skills():
            print(f"   • {skill.name}: {skill.description}")

        # 스킬 매칭 테스트
        print("\n📋 스킬 매칭 테스트...")
        test_queries = [
            "Python으로 웹 크롤러 만들어줘",
            "Azure에 애플리케이션 배포하고 싶어",
            "날씨 어때?"
        ]

        for query in test_queries:
            matched = manager.match_skills(query, threshold=0.2, max_skills=2)
            skill_names = [s.name for s in matched] if matched else ["(매칭 없음)"]
            print(f"   '{query[:30]}...' → {', '.join(skill_names)}")

        # 시스템 프롬프트 생성
        print("\n📋 시스템 프롬프트 생성...")
        prompt = manager.build_system_prompt(
            [python_skill, azure_skill],
            base_prompt="You are a helpful assistant."
        )
        print(f"   프롬프트 길이: {len(prompt)} 문자")

        print("\n" + "="*60)
        print("✅ Skills 시스템 테스트 완료!")
        print("="*60)

        return True

    except Exception as e:
        print(f"\n❌ 오류 발생: {e}")
        import traceback
        traceback.print_exc()
        return False


# ============================================================================
# v4.0 신규 기능 데모
# ============================================================================

async def demo_responses_api():
    """v4.0 Responses API 데모"""
    print("\n" + "="*60)
    print("📡 v4.0 Responses API 데모")
    print("="*60)

    try:
        from unified_agent import (
            ResponsesClient, ConversationState, BackgroundMode,
            ResponseConfig, ResponseObject, ResponseStatus, ToolType
        )

        # ResponseConfig 생성
        config = ResponseConfig(model="gpt-5.2", max_tokens=4096)
        print(f"   ResponseConfig: model={config.model}, max_tokens={config.max_tokens}")

        # ResponsesClient 생성
        client = ResponsesClient()
        print(f"   ResponsesClient: 생성 완료")

        # ConversationState 생성
        state = ConversationState()
        print(f"   ConversationState: session={state.session_id}, turns={state.turn_count}")

        # BackgroundMode 생성
        bg = BackgroundMode()
        print(f"   BackgroundMode: 생성 완료")

        # ResponseObject / ResponseStatus 확인
        statuses = [s.value for s in ResponseStatus]
        print(f"   ResponseStatus: {statuses}")

        # 실제 API 호출 (Mock)
        response = await client.create("테스트 프롬프트")
        print(f"   client.create(): output={response.output[:50]}...")

        # 응답을 ConversationState에 추가
        state.add_response(response)
        print(f"   ConversationState: turns={state.turn_count} (응답 추가 후)")

        print("\n   ✅ Responses API 데모 완료!")
        return True

    except Exception as e:
        print(f"\n   ❌ 오류: {e}")
        import traceback
        traceback.print_exc()
        return False


async def demo_multimodal_generation():
    """v4.0 비디오/이미지 생성 데모"""
    print("\n" + "="*60)
    print("🎥 v4.0 비디오/이미지 생성 데모")
    print("="*60)

    try:
        from unified_agent import (
            VideoGenerator, VideoConfig, VideoModel, VideoStatus, VideoResult,
            ImageGenerator, ImageConfig, ImageModel, ImageResult
        )

        # 비디오 생성
        vconfig = VideoConfig(model=VideoModel.SORA_2_PRO, duration=10, resolution="4k")
        vgen = VideoGenerator()
        vresult = await vgen.generate("A sunrise over mountains", config=vconfig)
        print(f"   VideoGenerator: status={vresult.status.value}, model={vconfig.model.value}")

        # 이미지 생성
        iconfig = ImageConfig(model=ImageModel.GPT_IMAGE_1_5, quality="hd")
        igen = ImageGenerator()
        iresult = await igen.generate("A futuristic cityscape", config=iconfig)
        print(f"   ImageGenerator: urls={len(iresult.image_urls)}, model={iconfig.model.value}")

        print("\n   ✅ 멀티모달 생성 데모 완료!")
        return True

    except Exception as e:
        print(f"\n   ❌ 오류: {e}")
        import traceback
        traceback.print_exc()
        return False


async def demo_open_weight():
    """v4.0 오픈 웨이트 모델 데모"""
    print("\n" + "="*60)
    print("🏋️ v4.0 오픈 웨이트 모델 데모")
    print("="*60)

    try:
        from unified_agent import (
            OpenWeightAdapter, OSSModelConfig, OpenWeightRegistry,
            OSSLicense, OSSModelInfo
        )

        # 레지스트리 조회
        registry = OpenWeightRegistry()
        models = registry.list_models()
        print(f"   등록 모델: {len(models)}개")
        for m in models:
            print(f"     - {m.name}: {m.parameters}, {m.license.value}")

        # 어댑터 생성 및 실행
        config = OSSModelConfig(max_tokens=4096)
        adapter = OpenWeightAdapter()
        result = await adapter.generate(model="gpt-oss-120b", prompt="Hello, open-weight model!", config=config)
        print(f"   adapter.generate(): model={result['model']}")

        print("\n   ✅ 오픈 웨이트 모델 데모 완료!")
        return True

    except Exception as e:
        print(f"\n   ❌ 오류: {e}")
        import traceback
        traceback.print_exc()
        return False


async def demo_universal_bridge():
    """v4.0 Universal Agent Bridge 데모"""
    print("\n" + "="*60)
    print("🌉 v4.0 Universal Agent Bridge 데모")
    print("="*60)

    try:
        from unified_agent import (
            UniversalAgentBridge, BridgeProtocol,
            OpenAIAgentsBridge, GoogleADKBridge, CrewAIBridge,
            A2ABridge, MicrosoftAgentBridge, AG2Bridge,
            SemanticKernelAgentBridge, AgentCard, TaskMode
        )

        # 브릿지 생성 및 프레임워크 등록
        bridge = UniversalAgentBridge()
        frameworks = {
            "openai_agents": OpenAIAgentsBridge(),
            "google_adk": GoogleADKBridge(),
            "crewai": CrewAIBridge(),
            "a2a": A2ABridge(),
            "microsoft": MicrosoftAgentBridge(),
            "ag2": AG2Bridge(),
            "semantic_kernel": SemanticKernelAgentBridge(),
        }

        for name, fw in frameworks.items():
            bridge.register(name, fw)
            print(f"   ✓ {name}: {type(fw).__name__} 등록")

        print(f"   등록된 프레임워크: {len(bridge.registered_frameworks)}개")

        # 실제 실행 테스트
        for fw in ["openai_agents", "google_adk", "crewai"]:
            result = await bridge.run(framework=fw, task="데이터 분석 수행")
            print(f"   bridge.run('{fw}'): ✓ 성공")

        print("\n   ✅ Universal Bridge 데모 완료!")
        return True

    except Exception as e:
        print(f"\n   ❌ 오류: {e}")
        import traceback
        traceback.print_exc()
        return False


async def demo_framework_v40_factories():
    """v4.0 framework.py 팩토리 메서드 데모"""
    print("\n" + "="*60)
    print("🏭 v4.0 Framework 팩토리 메서드 데모")
    print("="*60)

    try:
        from unified_agent.framework import UnifiedAgentFramework

        # 팩토리 메서드 존재 확인
        factory_methods = [
            "create_responses_client",
            "create_video_generator",
            "create_image_generator",
            "create_open_weight_adapter",
            "create_universal_bridge",
            "get_bridge",
        ]

        for method in factory_methods:
            assert hasattr(UnifiedAgentFramework, method), f"Missing: {method}"
            print(f"   ✓ UnifiedAgentFramework.{method}() 확인")

        print("\n   ✅ Framework 팩토리 메서드 데모 완료!")
        return True

    except Exception as e:
        print(f"\n   ❌ 오류: {e}")
        import traceback
        traceback.print_exc()
        return False


async def main():
    """메인 실행 함수"""
    print("\n" + "="*60)
    print("🧪 Unified Agent Framework v4.0 - 실행 데모")
    print("="*60)

    results = []

    # Core 데모 (v3.x)
    results.append(("프레임워크 생성", await demo_framework_creation()))
    results.append(("팀 워크플로우", await demo_team_workflow()))
    results.append(("MPlan", await demo_mplan()))
    results.append(("Skills 시스템", await demo_skills()))

    # v4.0 신규 데모
    results.append(("v4.0 Responses API", await demo_responses_api()))
    results.append(("v4.0 멀티모달 생성", await demo_multimodal_generation()))
    results.append(("v4.0 오픈 웨이트", await demo_open_weight()))
    results.append(("v4.0 Universal Bridge", await demo_universal_bridge()))
    results.append(("v4.0 팩토리 메서드", await demo_framework_v40_factories()))

    # 결과 요약
    print("\n" + "="*60)
    print("📊 데모 실행 결과")
    print("="*60)

    all_passed = True
    for name, result in results:
        status = "✅ 성공" if result else "❌ 실패"
        print(f"   {name}: {status}")
        if not result:
            all_passed = False

    print("="*60)

    if all_passed:
        print("\n🎉 모든 데모가 성공적으로 완료되었습니다!")
    else:
        print("\n⚠️ 일부 데모가 실패했습니다.")

    return all_passed


if __name__ == "__main__":
    success = asyncio.run(main())
    sys.exit(0 if success else 1)
