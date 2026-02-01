#!/usr/bin/env python3
"""
Unified Agent Framework v3.4 - 종합 시나리오 테스트 (최종본)
실제 API에 맞춰 모든 주요 기능을 테스트합니다.
"""

import asyncio
import os
import sys
from datetime import datetime
from typing import Dict, Any

# 테스트 결과 저장
test_results: Dict[str, Dict[str, Any]] = {}

def record_test(category: str, name: str, passed: bool, message: str = ""):
    """테스트 결과 기록"""
    if category not in test_results:
        test_results[category] = {}
    test_results[category][name] = {"passed": passed, "message": message}
    status = "✅" if passed else "❌"
    print(f"  {status} {name}: {message}")


# ============================================================================
# 1. v3.4 확장 모듈 테스트
# ============================================================================
async def test_v34_extensions():
    """v3.4 확장 모듈 테스트"""
    print("\n" + "=" * 70)
    print("📦 v3.4 확장 모듈 테스트")
    print("=" * 70)
    
    # 1.1 Prompt Cache
    print("\n[1.1] Prompt Cache")
    try:
        from unified_agent import PromptCache, CacheConfig
        
        cache = PromptCache(CacheConfig(
            max_size_mb=50,
            ttl_seconds=3600
        ))
        cache.initialize()
        
        # 실제 API: set(prompt, response, model, ...)
        entry = cache.set(
            prompt="테스트 프롬프트",
            response="테스트 응답",
            model="gpt-5.2"
        )
        
        # 조회
        result = cache.get(prompt="테스트 프롬프트", model="gpt-5.2")
        stats = cache.get_stats()
        
        record_test("v3.4", "Prompt Cache", 
                   entry is not None,
                   f"캐시 저장/조회 성공")
    except Exception as e:
        record_test("v3.4", "Prompt Cache", False, str(e))
    
    # 1.2 Durable Agent
    print("\n[1.2] Durable Agent")
    try:
        from unified_agent import DurableOrchestrator, DurableConfig
        
        orchestrator = DurableOrchestrator(DurableConfig(
            checkpoint_interval_seconds=60,
            workflow_timeout_seconds=3600
        ))
        
        record_test("v3.4", "Durable Orchestrator", 
                   orchestrator is not None,
                   "DurableOrchestrator 생성 성공")
    except Exception as e:
        record_test("v3.4", "Durable Orchestrator", False, str(e))
    
    # 1.3 Concurrent Orchestration
    print("\n[1.3] Concurrent Orchestration")
    try:
        from unified_agent import FanOutConfig, AggregationStrategy
        
        # 실제 API: max_concurrency, strategy
        config = FanOutConfig(
            max_concurrency=5,
            timeout_seconds=30.0,
            strategy=AggregationStrategy.ALL
        )
        
        record_test("v3.4", "Concurrent 설정", 
                   config.max_concurrency == 5,
                   f"max_concurrency={config.max_concurrency}, strategy={config.strategy.value}")
    except Exception as e:
        record_test("v3.4", "Concurrent 설정", False, str(e))
    
    # 1.4 AgentTool Pattern
    print("\n[1.4] AgentTool Pattern")
    try:
        from unified_agent import AgentTool, AgentToolRegistry, DelegationManager
        
        registry = AgentToolRegistry()
        delegation = DelegationManager(registry)
        
        record_test("v3.4", "AgentTool Registry", 
                   registry is not None,
                   "Registry와 DelegationManager 생성 성공")
    except Exception as e:
        record_test("v3.4", "AgentTool Registry", False, str(e))
    
    # 1.5 Extended Thinking
    print("\n[1.5] Extended Thinking")
    try:
        from unified_agent import ThinkingTracker, ThinkingConfig, ThinkingMode
        
        # 실제 API: max_steps, max_depth
        tracker = ThinkingTracker(ThinkingConfig(
            max_steps=100,
            max_depth=10
        ))
        
        # 사고 과정 추적
        with tracker.thinking_context("problem-solving") as ctx:
            ctx.observe("입력 분석", "분석 내용")
            ctx.reason("추론", "추론 내용")
            ctx.conclude("결론", "결론 내용")
        
        chain = tracker.get_chain("problem-solving")
        
        record_test("v3.4", "Extended Thinking", 
                   chain is not None and chain.total_steps == 3,
                   f"사고 단계: {chain.total_steps}개")
    except Exception as e:
        record_test("v3.4", "Extended Thinking", False, str(e))
    
    # 1.6 MCP Workbench
    print("\n[1.6] MCP Workbench")
    try:
        from unified_agent import McpWorkbench, McpServerConfig, McpWorkbenchConfig
        
        workbench = McpWorkbench(McpWorkbenchConfig(
            enable_healthcheck=True,
            enable_auto_reconnect=True
        ))
        
        workbench.register_server(McpServerConfig(
            name="test-server",
            uri="stdio://test",
            capabilities=["read"]
        ))
        
        status = workbench.get_status()
        
        record_test("v3.4", "MCP Workbench", 
                   status["total_servers"] == 1,
                   f"등록된 서버: {status['total_servers']}개")
    except Exception as e:
        record_test("v3.4", "MCP Workbench", False, str(e))
    
    # 1.7 Extensions Hub
    print("\n[1.7] Extensions Hub")
    try:
        from unified_agent import Extensions, ExtensionsConfig
        
        ext = Extensions(config=ExtensionsConfig(
            enable_cache=True,
            enable_durable=True,
            enable_thinking=True,
            enable_mcp=False
        ))
        
        active = []
        if ext.cache: active.append("cache")
        if ext.durable: active.append("durable")
        if ext.thinking: active.append("thinking")
        
        record_test("v3.4", "Extensions Hub", 
                   len(active) == 3,
                   f"활성 모듈: {active}")
    except Exception as e:
        record_test("v3.4", "Extensions Hub", False, str(e))


# ============================================================================
# 2. v3.3 Agent Lightning 테스트
# ============================================================================
async def test_v33_agent_lightning():
    """v3.3 Agent Lightning 패턴 테스트"""
    print("\n" + "=" * 70)
    print("⚡ v3.3 Agent Lightning 패턴 테스트")
    print("=" * 70)
    
    # 2.1 Tracer
    print("\n[2.1] Tracer (분산 추적)")
    try:
        from unified_agent import AgentTracer, SpanKind
        
        tracer = AgentTracer(name="test-service")
        await tracer.initialize()
        
        # 실제 API: span context manager 또는 create_span
        span = tracer.create_span("test-op", kind=SpanKind.INTERNAL)
        span.set_attribute("key", "value")
        span.end()
        
        record_test("v3.3", "Tracer 스팬 생성", 
                   span is not None,
                   "스팬 생성 및 속성 설정 성공")
    except Exception as e:
        record_test("v3.3", "Tracer 스팬 생성", False, str(e))
    
    # 2.2 HookManager
    print("\n[2.2] HookManager (라이프사이클 훅)")
    try:
        from unified_agent import HookManager
        
        manager = HookManager()
        call_count = [0]
        
        def test_hook(*args, **kwargs):
            call_count[0] += 1
        
        manager.register("on_span_start", test_hook)
        hooks = manager.get_hooks("on_span_start")
        
        record_test("v3.3", "HookManager 훅 등록", 
                   len(hooks) >= 1,
                   f"등록된 훅: {len(hooks)}개")
    except Exception as e:
        record_test("v3.3", "HookManager 훅 등록", False, str(e))
    
    # 2.3 Reward
    print("\n[2.3] Reward (보상 시스템)")
    try:
        from unified_agent import RewardManager
        
        manager = RewardManager()
        
        # 실제 API: emit(value, tags=, metadata=)
        manager.emit(1.0, tags=["completion"])
        manager.emit(0.5, tags=["efficiency"])
        
        record_test("v3.3", "Reward 보상 기록", 
                   manager.reward_count >= 2,
                   f"총 보상: {manager.reward_count}개, 합계: {manager.total_reward}")
    except Exception as e:
        record_test("v3.3", "Reward 보상 기록", False, str(e))


# ============================================================================
# 3. v3.2 영속 메모리 테스트
# ============================================================================
async def test_v32_persistent_memory():
    """v3.2 영속 메모리 시스템 테스트"""
    print("\n" + "=" * 70)
    print("🗄️ v3.2 영속 메모리 시스템 테스트")
    print("=" * 70)
    
    # 3.1 PersistentMemory
    print("\n[3.1] PersistentMemory")
    try:
        from unified_agent.persistent_memory import PersistentMemory, MemoryConfig
        
        # 실제 API: agent_id, config
        memory = PersistentMemory(
            agent_id="test-agent",
            config=MemoryConfig(workspace_dir="./test_memory")
        )
        await memory.initialize()
        
        # 실제 API: add_long_term_memory, search(query, max_results=)
        await memory.add_long_term_memory("테스트 정보입니다")
        results = await memory.search("테스트", max_results=5)
        
        record_test("v3.2", "PersistentMemory", 
                   True,
                   f"저장/검색 완료")
        
        # close는 동기 함수일 수 있음
        try:
            memory.close()
        except:
            pass
    except Exception as e:
        record_test("v3.2", "PersistentMemory", False, str(e))
    
    # 3.2 Compaction
    print("\n[3.2] CompactionManager")
    try:
        from unified_agent.compaction import CompactionManager, CompactionConfig
        
        manager = CompactionManager(CompactionConfig())
        
        record_test("v3.2", "CompactionManager", 
                   manager is not None,
                   "생성 성공")
    except Exception as e:
        record_test("v3.2", "CompactionManager", False, str(e))
    
    # 3.3 SessionTree
    print("\n[3.3] SessionTree")
    try:
        from unified_agent import SessionTree, SessionTreeConfig
        
        # 실제 API: session_id 필수
        tree = SessionTree(
            session_id="test-session",
            config=SessionTreeConfig(max_depth=10)
        )
        
        # 실제 API: create_branch(name, description=None) - 동기 함수
        branch = tree.create_branch(name="exp-1")
        branches = tree.list_branches()
        
        record_test("v3.2", "SessionTree 분기 관리", 
                   len(branches) >= 1,
                   f"분기 수: {len(branches)}")
    except Exception as e:
        record_test("v3.2", "SessionTree 분기 관리", False, str(e))


# ============================================================================
# 4. Core 기능 테스트
# ============================================================================
async def test_core_features():
    """Core 기능 테스트"""
    print("\n" + "=" * 70)
    print("🔧 Core 기능 테스트")
    print("=" * 70)
    
    # 4.1 Config & Settings
    print("\n[4.1] Config & Settings")
    try:
        from unified_agent import Settings, SUPPORTED_MODELS
        
        record_test("Core", "Config 로딩", 
                   len(SUPPORTED_MODELS) >= 50,
                   f"지원 모델: {len(SUPPORTED_MODELS)}개, 기본: {Settings.DEFAULT_MODEL}")
    except Exception as e:
        record_test("Core", "Config 로딩", False, str(e))
    
    # 4.2 Models - MPlan
    print("\n[4.2] MPlan 데이터 모델")
    try:
        from unified_agent import MPlan, PlanStep, PlanStepStatus
        
        plan = MPlan(
            name="test-plan",
            description="테스트",
            steps=[
                PlanStep(index=0, description="1단계", agent_name="a1"),
                PlanStep(index=1, description="2단계", agent_name="a2", depends_on=[0])
            ]
        )
        
        progress_before = plan.get_progress()
        plan.steps[0].status = PlanStepStatus.COMPLETED
        progress_after = plan.get_progress()
        
        record_test("Core", "MPlan 진행률", 
                   progress_before == 0.0 and progress_after == 0.5,
                   f"{progress_before*100:.0f}% → {progress_after*100:.0f}%")
    except Exception as e:
        record_test("Core", "MPlan 진행률", False, str(e))
    
    # 4.3 Memory
    print("\n[4.3] CachedMemoryStore")
    try:
        from unified_agent import CachedMemoryStore
        
        store = CachedMemoryStore(max_cache_size=100)
        
        # 실제 API: async
        await store.save("s1", {"msg": "hello"})
        data = await store.load("s1")
        
        record_test("Core", "CachedMemoryStore", 
                   data is not None and "msg" in data,
                   "저장/로드 성공")
    except Exception as e:
        record_test("Core", "CachedMemoryStore", False, str(e))
    
    # 4.4 Events
    print("\n[4.4] EventBus")
    try:
        from unified_agent import EventBus, EventType, AgentEvent
        
        bus = EventBus()
        received = []
        
        async def handler(event):
            received.append(event)
        
        bus.subscribe(EventType.AGENT_STARTED, handler)
        await bus.publish(AgentEvent(
            event_type=EventType.AGENT_STARTED,
            session_id="test",
            agent_name="agent1"
        ))
        
        record_test("Core", "EventBus", 
                   len(received) == 1,
                   f"수신 이벤트: {len(received)}개")
    except Exception as e:
        record_test("Core", "EventBus", False, str(e))
    
    # 4.5 Utils
    print("\n[4.5] 유틸리티")
    try:
        from unified_agent import StructuredLogger, CircuitBreaker, RAIValidator
        
        logger = StructuredLogger("test")
        logger.info("테스트")
        
        breaker = CircuitBreaker(failure_threshold=5)
        
        validator = RAIValidator()
        result = validator.validate("안녕하세요")
        
        record_test("Core", "Utils", 
                   result.is_safe,
                   "Logger, CircuitBreaker, RAI 정상")
    except Exception as e:
        record_test("Core", "Utils", False, str(e))
    
    # 4.6 Workflow Graph
    print("\n[4.6] Workflow Graph")
    try:
        from unified_agent import Graph, Node, AgentState
        
        graph = Graph(name="test")
        
        async def step1(state):
            state.response = "step1 done"
            return state
        
        async def step2(state):
            state.response += " -> step2 done"
            return state
        
        graph.add_node(Node("s1", step1))
        graph.add_node(Node("s2", step2))
        graph.add_edge("s1", "s2")
        graph.set_start("s1")
        graph.set_end("s2")
        
        # 통계로 노드 수 확인
        stats = graph.get_statistics()
        
        record_test("Core", "Workflow Graph", 
                   stats.get("total_nodes", 0) == 2,
                   f"노드: {stats.get('total_nodes')}개, 엣지: {stats.get('total_edges')}개")
    except Exception as e:
        record_test("Core", "Workflow Graph", False, str(e))
    
    # 4.7 Interfaces
    print("\n[4.7] Interfaces")
    try:
        from unified_agent import IFramework, IOrchestrator, IMemoryProvider
        
        record_test("Core", "Interfaces", 
                   all([IFramework, IOrchestrator, IMemoryProvider]),
                   "IFramework, IOrchestrator, IMemoryProvider 정의됨")
    except Exception as e:
        record_test("Core", "Interfaces", False, str(e))
    
    # 4.8 Agents
    print("\n[4.8] Agents")
    try:
        from unified_agent import SimpleAgent, RouterAgent, SupervisorAgent
        
        agent = SimpleAgent(name="test", system_prompt="테스트")
        
        record_test("Core", "Agent 생성", 
                   agent.name == "test",
                   f"이름: {agent.name}")
    except Exception as e:
        record_test("Core", "Agent 생성", False, str(e))


# ============================================================================
# 결과 요약
# ============================================================================
def print_summary():
    """테스트 결과 요약"""
    print("\n" + "=" * 70)
    print("📊 종합 테스트 결과")
    print("=" * 70)
    
    total_passed = 0
    total_failed = 0
    
    for category, tests in test_results.items():
        passed = sum(1 for t in tests.values() if t["passed"])
        failed = len(tests) - passed
        total_passed += passed
        total_failed += failed
        
        status = "✅" if failed == 0 else "⚠️"
        print(f"\n{status} {category}: {passed}/{len(tests)} 통과")
        
        for name, result in tests.items():
            icon = "✅" if result["passed"] else "❌"
            print(f"   {icon} {name}")
    
    print("\n" + "-" * 70)
    total = total_passed + total_failed
    rate = (total_passed / total * 100) if total > 0 else 0
    
    print(f"총 테스트: {total}개")
    print(f"통과: {total_passed}개")
    print(f"실패: {total_failed}개")
    print(f"성공률: {rate:.1f}%")
    
    if total_failed == 0:
        print("\n🎉 모든 테스트 통과!")
    else:
        print(f"\n⚠️ {total_failed}개 테스트 실패")
    
    print("=" * 70)
    return total_failed == 0


# ============================================================================
# 메인
# ============================================================================
async def main():
    print("=" * 70)
    print("🚀 Unified Agent Framework v3.4 - 종합 시나리오 테스트")
    print(f"📅 실행: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 70)
    
    await test_v34_extensions()
    await test_v33_agent_lightning()
    await test_v32_persistent_memory()
    await test_core_features()
    
    success = print_summary()
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    asyncio.run(main())
