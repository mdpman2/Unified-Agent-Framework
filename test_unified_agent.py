#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Unified Agent Framework - 테스트 스위트

모듈화된 프레임워크의 전체 테스트
"""

import asyncio
import sys
import os
from pathlib import Path

# 모듈 경로 추가
sys.path.insert(0, str(Path(__file__).parent))

# 테스트 결과 저장
test_results = {
    "passed": 0,
    "failed": 0,
    "errors": []
}


def test_passed(name: str):
    test_results["passed"] += 1
    print(f"  ✅ {name}")


def test_failed(name: str, error: str):
    test_results["failed"] += 1
    test_results["errors"].append(f"{name}: {error}")
    print(f"  ❌ {name}: {error}")


def print_section(title: str):
    print(f"\n{'='*60}")
    print(f"📋 {title}")
    print('='*60)


# ============================================================================
# 1. Import 테스트
# ============================================================================
def test_imports():
    print_section("Import 테스트")

    modules = [
        ("exceptions", ["FrameworkError", "ConfigurationError", "WorkflowError", "AgentError"]),
        ("config", ["Settings", "FrameworkConfig", "DEFAULT_LLM_MODEL", "supports_temperature"]),
        ("models", ["AgentRole", "ExecutionStatus", "Message", "AgentState", "NodeResult", "TeamConfiguration"]),
        ("utils", ["StructuredLogger", "CircuitBreaker", "RAIValidator", "setup_telemetry"]),
        ("memory", ["MemoryStore", "CachedMemoryStore", "StateManager"]),
        ("events", ["EventType", "EventBus", "AgentEvent"]),
        ("skills", ["Skill", "SkillManager", "SkillResource"]),
        ("tools", ["AIFunction", "ApprovalRequiredAIFunction", "MCPTool"]),
        ("agents", ["Agent", "SimpleAgent", "RouterAgent", "SupervisorAgent", "ProxyAgent"]),
        ("workflow", ["Node", "Graph"]),
        ("orchestration", ["OrchestrationManager", "AgentFactory"]),
        ("framework", ["UnifiedAgentFramework", "quick_run", "create_framework"]),
    ]

    for module_name, classes in modules:
        try:
            module = __import__(f"unified_agent.{module_name}", fromlist=classes)
            for cls_name in classes:
                if hasattr(module, cls_name):
                    test_passed(f"{module_name}.{cls_name}")
                else:
                    test_failed(f"{module_name}.{cls_name}", "클래스 없음")
        except Exception as e:
            test_failed(f"unified_agent.{module_name}", str(e))


# ============================================================================
# 2. 패키지 import 테스트
# ============================================================================
def test_package_import():
    print_section("패키지 Import 테스트")

    try:
        from unified_agent import (
            __version__,
            UnifiedAgentFramework,
            FrameworkConfig,
            AgentRole,
            AgentState,
            Message,
            SimpleAgent,
            Graph,
            Node,
            EventBus,
        )
        test_passed("패키지 전체 import")
        test_passed(f"버전: {__version__}")
    except Exception as e:
        test_failed("패키지 import", str(e))


# ============================================================================
# 3. Enum 테스트
# ============================================================================
def test_enums():
    print_section("Enum 테스트")

    try:
        from unified_agent.models import AgentRole, ExecutionStatus, ApprovalStatus, PlanStepStatus

        assert AgentRole.USER.value == "user"
        assert AgentRole.ASSISTANT.value == "assistant"
        assert AgentRole.SYSTEM.value == "system"
        test_passed("AgentRole enum")

        assert ExecutionStatus.PENDING.value == "pending"
        assert ExecutionStatus.COMPLETED.value == "completed"
        test_passed("ExecutionStatus enum")

        assert ApprovalStatus.PENDING.value == "pending"
        assert ApprovalStatus.APPROVED.value == "approved"
        test_passed("ApprovalStatus enum")

        assert PlanStepStatus.IN_PROGRESS.value == "in_progress"
        test_passed("PlanStepStatus enum")

    except Exception as e:
        test_failed("Enum 테스트", str(e))


# ============================================================================
# 4. Pydantic 모델 테스트
# ============================================================================
def test_pydantic_models():
    print_section("Pydantic 모델 테스트")

    try:
        from unified_agent.models import Message, AgentState, NodeResult, AgentRole

        msg = Message(role=AgentRole.USER, content="Hello")
        assert msg.role == AgentRole.USER
        assert msg.content == "Hello"
        test_passed("Message 모델")

        state = AgentState(session_id="test-session", workflow_name="test")
        assert state.session_id == "test-session"
        test_passed("AgentState 모델")

        result = NodeResult(node_name="test", output="output", success=True)
        assert result.success == True
        test_passed("NodeResult 모델")

    except Exception as e:
        test_failed("Pydantic 모델", str(e))


# ============================================================================
# 5. Config 테스트
# ============================================================================
def test_config():
    print_section("Config 테스트")

    try:
        from unified_agent.config import (
            FrameworkConfig, Settings,
            supports_temperature, SUPPORTED_MODELS, O_SERIES_MODELS
        )

        assert supports_temperature("gpt-4o") == True
        assert supports_temperature("o1") == False
        test_passed("supports_temperature 함수")

        assert "gpt-4o" in SUPPORTED_MODELS
        assert "o1" in SUPPORTED_MODELS
        test_passed("SUPPORTED_MODELS 상수")

        assert "o1" in O_SERIES_MODELS
        assert "o3" in O_SERIES_MODELS
        test_passed("O_SERIES_MODELS 상수")

        config = FrameworkConfig(
            deployment_name="test",
            api_key="test-key",
            endpoint="https://test.openai.azure.com/"
        )
        assert config.deployment_name == "test"
        test_passed("FrameworkConfig 생성")

    except Exception as e:
        test_failed("Config", str(e))


# ============================================================================
# 6. Memory 시스템 테스트
# ============================================================================
def test_memory():
    print_section("Memory 시스템 테스트")

    try:
        from unified_agent.memory import CachedMemoryStore, StateManager

        store = CachedMemoryStore(max_cache_size=100)
        test_passed("CachedMemoryStore 생성")

        manager = StateManager(store, "./test_checkpoints")
        test_passed("StateManager 생성")

    except Exception as e:
        test_failed("Memory 시스템", str(e))


# ============================================================================
# 7. Event 시스템 테스트
# ============================================================================
async def test_events_async():
    print_section("Event 시스템 테스트")

    try:
        from unified_agent.events import EventBus, EventType, AgentEvent

        bus = EventBus()
        test_passed("EventBus 생성")

        received_events = []

        async def handler(event):
            received_events.append(event)

        bus.subscribe(EventType.AGENT_STARTED, handler)
        test_passed("이벤트 구독")

        event = AgentEvent(
            event_type=EventType.AGENT_STARTED,
            agent_name="test_agent",
            data={"test": True}
        )
        await bus.publish(event)
        await asyncio.sleep(0.1)

        assert len(received_events) > 0
        test_passed("이벤트 발행 및 수신")

    except Exception as e:
        test_failed("Event 시스템", str(e))


# ============================================================================
# 8. Utils 테스트
# ============================================================================
def test_utils():
    print_section("Utils 테스트")

    try:
        from unified_agent.utils import StructuredLogger, CircuitBreaker, RAIValidator

        logger = StructuredLogger("test")
        logger.info("테스트 메시지", key="value")
        test_passed("StructuredLogger")

        breaker = CircuitBreaker(failure_threshold=3, timeout=30.0)
        assert breaker.failure_count == 0
        test_passed("CircuitBreaker 생성")

        validator = RAIValidator()
        result = validator.validate("This is a safe text")
        assert result.is_safe == True
        test_passed("RAIValidator")

    except Exception as e:
        test_failed("Utils", str(e))


# ============================================================================
# 9. Skills 테스트
# ============================================================================
def test_skills():
    print_section("Skills 테스트")

    try:
        from unified_agent.skills import Skill, SkillManager, SkillResource

        resource = SkillResource(
            resource_type="reference",
            name="test.md",
            path="./test.md",
            content="Test content"
        )
        assert resource.name == "test.md"
        test_passed("SkillResource")

        skill = Skill(
            name="test-skill",
            description="Test skill",
            instructions="Do something",
            triggers=["test", "demo"]
        )
        assert skill.name == "test-skill"
        test_passed("Skill 생성")

        manager = SkillManager()
        manager.register_skill(skill)
        retrieved = manager.get_skill("test-skill")
        assert retrieved is not None
        test_passed("SkillManager")

    except Exception as e:
        test_failed("Skills", str(e))


# ============================================================================
# 10. Tools 테스트
# ============================================================================
def test_tools():
    print_section("Tools 테스트")

    try:
        from unified_agent.tools import MockMCPClient, MCPTool

        client = MockMCPClient(config={"test": True})
        test_passed("MockMCPClient 생성")

        tool = MCPTool(
            name="test_tool",
            server_config={"type": "mock"}
        )
        assert tool.name == "test_tool"
        test_passed("MCPTool 생성")

    except Exception as e:
        test_failed("Tools", str(e))


# ============================================================================
# 11. Workflow 테스트
# ============================================================================
def test_workflow():
    print_section("Workflow 테스트")

    try:
        from unified_agent.workflow import Graph

        graph = Graph(name="test_workflow")
        assert graph.name == "test_workflow"
        test_passed("Graph 생성")

        stats = graph.get_statistics()
        assert "total_nodes" in stats
        test_passed("Graph.get_statistics()")

        viz = graph.visualize()
        assert "mermaid" in viz
        test_passed("Graph.visualize()")

    except Exception as e:
        test_failed("Workflow", str(e))


# ============================================================================
# 12. TeamConfiguration 테스트
# ============================================================================
def test_team_config():
    print_section("TeamConfiguration 테스트")

    try:
        from unified_agent.models import TeamConfiguration, TeamAgent, AgentRole

        agent = TeamAgent(
            name="researcher",
            description="Research specialist",
            role=AgentRole.ASSISTANT
        )
        assert agent.name == "researcher"
        test_passed("TeamAgent 생성")

        config = TeamConfiguration(
            name="research_team",
            description="Research team",
            agents=[agent],
            orchestration_mode="supervisor"
        )
        assert config.name == "research_team"
        assert len(config.agents) == 1
        test_passed("TeamConfiguration 생성")

    except Exception as e:
        test_failed("TeamConfiguration", str(e))


# ============================================================================
# 13. MPlan 테스트
# ============================================================================
def test_mplan():
    print_section("MPlan 테스트")

    try:
        from unified_agent.models import MPlan, PlanStep, PlanStepStatus

        step = PlanStep(
            index=0,
            description="Step 1",
            agent_name="agent1"
        )
        assert step.index == 0
        test_passed("PlanStep 생성")

        plan = MPlan(
            name="test_plan",
            description="Test plan",
            steps=[step]
        )
        assert plan.name == "test_plan"
        test_passed("MPlan 생성")

        summary = plan.to_summary()
        assert "test_plan" in summary
        test_passed("MPlan.to_summary()")

        progress = plan.get_progress()
        assert progress >= 0.0
        test_passed("MPlan.get_progress()")

    except Exception as e:
        test_failed("MPlan", str(e))


# ============================================================================
# 14. 순환 참조 테스트
# ============================================================================
def test_circular_imports():
    print_section("순환 참조 테스트")

    try:
        import unified_agent.exceptions
        import unified_agent.config
        import unified_agent.models
        import unified_agent.utils
        import unified_agent.memory
        import unified_agent.events
        import unified_agent.skills
        import unified_agent.tools
        import unified_agent.agents
        import unified_agent.workflow
        import unified_agent.orchestration
        import unified_agent.framework

        test_passed("순환 참조 없음")

        import unified_agent.framework
        import unified_agent.orchestration
        import unified_agent.workflow
        import unified_agent.agents

        test_passed("역순 import 성공")

    except Exception as e:
        test_failed("순환 참조", str(e))


# ============================================================================
# 메인 실행
# ============================================================================
async def main():
    print("\n" + "="*60)
    print("🧪 Unified Agent Framework - 테스트 스위트")
    print("="*60)

    # 동기 테스트
    test_imports()
    test_package_import()
    test_enums()
    test_pydantic_models()
    test_config()
    test_memory()
    test_utils()
    test_skills()
    test_tools()
    test_workflow()
    test_team_config()
    test_mplan()
    test_circular_imports()

    # 비동기 테스트
    await test_events_async()

    # 결과 출력
    print("\n" + "="*60)
    print("📊 테스트 결과 요약")
    print("="*60)
    print(f"  ✅ 성공: {test_results['passed']}")
    print(f"  ❌ 실패: {test_results['failed']}")

    if test_results['errors']:
        print("\n❌ 실패한 테스트:")
        for error in test_results['errors']:
            print(f"  - {error}")

    print("="*60)

    return test_results['failed'] == 0


if __name__ == "__main__":
    success = asyncio.run(main())
    sys.exit(0 if success else 1)
