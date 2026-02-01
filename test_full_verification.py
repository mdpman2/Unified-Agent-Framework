#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Unified Agent Framework v3.3 - 전체 기능 검증 테스트

모든 모듈과 기능을 종합적으로 검증합니다:
- Core: Settings, Config, Models
- v3.1: Memory, Events, Skills, Tools, Agents, Workflow
- v3.2: PersistentMemory, Compaction, SessionTree
- v3.3: Tracer, AgentStore, Reward, Adapter, Hooks
"""

import asyncio
import sys
import os
import tempfile

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


def print_section(title: str):
    print(f"\n{'='*70}")
    print(f"  {title}")
    print('='*70)


def print_check(name: str, success: bool, detail: str = ""):
    status = "[OK]" if success else "[FAIL]"
    print(f"  {status} {name}")
    if detail and not success:
        print(f"       -> {detail}")


def test_core_imports():
    """Core 모듈 import 테스트"""
    print_section("1. Core Modules Import Test")
    results = []
    
    try:
        from unified_agent import __version__, __author__
        print_check(f"Version: {__version__}", True)
        results.append(True)
    except Exception as e:
        print_check("Version import", False, str(e))
        results.append(False)
    
    try:
        from unified_agent import (
            Settings, FrameworkConfig,
            DEFAULT_LLM_MODEL, SUPPORTED_MODELS
        )
        print_check(f"Settings (default model: {DEFAULT_LLM_MODEL})", True)
        print_check(f"Supported models: {len(SUPPORTED_MODELS)} models", True)
        results.append(True)
    except Exception as e:
        print_check("Config import", False, str(e))
        results.append(False)
    
    try:
        from unified_agent import (
            AgentRole, ExecutionStatus, ApprovalStatus,
            Message, AgentState, TeamAgent, TeamConfiguration
        )
        print_check("Models (Enums, Pydantic)", True)
        results.append(True)
    except Exception as e:
        print_check("Models import", False, str(e))
        results.append(False)
    
    try:
        from unified_agent import (
            FrameworkError, ConfigurationError, WorkflowError,
            AgentError, ApprovalError, RAIValidationError
        )
        print_check("Exceptions", True)
        results.append(True)
    except Exception as e:
        print_check("Exceptions import", False, str(e))
        results.append(False)
    
    return all(results)


def test_v31_modules():
    """v3.1 모듈 import 테스트"""
    print_section("2. v3.1 Modules Import Test")
    results = []
    
    modules = [
        ("Utils", ["StructuredLogger", "CircuitBreaker", "RAIValidator"]),
        ("Memory", ["MemoryStore", "CachedMemoryStore", "StateManager"]),
        ("Events", ["EventType", "AgentEvent", "EventBus"]),
        ("Skills", ["Skill", "SkillManager", "SkillResource"]),
        ("Tools", ["AIFunction", "MCPTool", "ApprovalRequiredAIFunction"]),
        ("Agents", ["Agent", "SimpleAgent", "SupervisorAgent", "RouterAgent"]),
        ("Workflow", ["Node", "Graph"]),
        ("Orchestration", ["AgentFactory", "OrchestrationManager"]),
        ("Framework", ["UnifiedAgentFramework", "quick_run", "create_framework"]),
    ]
    
    for name, imports in modules:
        try:
            exec(f"from unified_agent import {', '.join(imports)}")
            print_check(f"{name}: {', '.join(imports[:2])}...", True)
            results.append(True)
        except Exception as e:
            print_check(f"{name}", False, str(e))
            results.append(False)
    
    return all(results)


def test_v32_modules():
    """v3.2 모듈 import 테스트"""
    print_section("3. v3.2 Modules Import Test (NEW)")
    results = []
    
    # PersistentMemory
    try:
        from unified_agent import (
            PersistentMemory, MemoryConfig, MemoryLayer,
            MemorySearchTool, MemoryGetTool, MemoryWriteTool,
            BootstrapFileManager, MemoryIndexer
        )
        print_check("PersistentMemory system", True)
        results.append(True)
    except Exception as e:
        print_check("PersistentMemory", False, str(e))
        results.append(False)
    
    # Compaction
    try:
        from unified_agent import (
            CompactionConfig, ContextCompactor, CompactionManager,
            MemoryFlusher, CacheTTLPruner, CompactionSummary
        )
        print_check("Compaction system", True)
        results.append(True)
    except Exception as e:
        print_check("Compaction", False, str(e))
        results.append(False)
    
    # SessionTree
    try:
        from unified_agent import (
            SessionTree, SessionNode, NodeType,
            SessionTreeManager, SessionTreeConfig, BranchInfo
        )
        print_check("SessionTree system", True)
        results.append(True)
    except Exception as e:
        print_check("SessionTree", False, str(e))
        results.append(False)
    
    return all(results)


def test_v33_modules():
    """v3.3 모듈 import 테스트"""
    print_section("4. v3.3 Modules Import Test (NEW - Agent Lightning)")
    results = []
    
    # Tracer
    try:
        from unified_agent import (
            AgentTracer, Span, SpanKind, SpanStatus,
            LLMCallTracer, ToolCallTracer,
            get_tracer, set_tracer
        )
        print_check("Tracer system (OpenTelemetry-style)", True)
        results.append(True)
    except Exception as e:
        print_check("Tracer", False, str(e))
        results.append(False)
    
    # AgentStore
    try:
        from unified_agent import (
            Rollout, Attempt, RolloutStatus, AttemptStatus,
            InMemoryAgentStore, SQLiteAgentStore,
            get_store, set_store
        )
        print_check("AgentStore system (LightningStore)", True)
        results.append(True)
    except Exception as e:
        print_check("AgentStore", False, str(e))
        results.append(False)
    
    # Reward
    try:
        from unified_agent import (
            emit_reward, emit_annotation, RewardManager,
            is_reward_span, get_reward_value, find_reward_spans,
            reward, reward_async
        )
        print_check("Reward system (emit_reward)", True)
        results.append(True)
    except Exception as e:
        print_check("Reward", False, str(e))
        results.append(False)
    
    # Adapter
    try:
        from unified_agent import (
            Triplet, Trajectory, TracerTraceToTriplet,
            TraceTree, RewardMatchPolicy,
            build_trajectory, export_triplets_to_jsonl
        )
        print_check("Adapter system (Span->Triplet)", True)
        results.append(True)
    except Exception as e:
        print_check("Adapter", False, str(e))
        results.append(False)
    
    # Hooks
    try:
        from unified_agent import (
            HookManager, HookEvent, HookPriority,
            HookContext, HookResult,
            get_hook_manager, on_trace_start, on_llm_call
        )
        print_check("Hooks system (Lifecycle)", True)
        results.append(True)
    except Exception as e:
        print_check("Hooks", False, str(e))
        results.append(False)
    
    return all(results)


async def test_functional_tracer():
    """Tracer 기능 테스트"""
    print_section("5. Functional Test: Tracer")
    results = []
    
    from unified_agent import AgentTracer, SpanKind, set_tracer
    
    collected = []
    tracer = AgentTracer(name="func-test")
    tracer.add_callback_on_span_end(lambda s: collected.append(s))
    set_tracer(tracer)
    
    # 스팬 생성 테스트
    with tracer.span("test-operation", SpanKind.INTERNAL) as span:
        span.set_attribute("test", True)
    
    success = len(collected) == 1 and collected[0].name == "test-operation"
    print_check("Span creation and collection", success)
    results.append(success)
    
    return all(results)


async def test_functional_store():
    """Store 기능 테스트"""
    print_section("6. Functional Test: AgentStore")
    results = []
    
    from unified_agent import Rollout, InMemoryAgentStore, RolloutStatus
    
    store = InMemoryAgentStore()
    await store.initialize()
    
    # 롤아웃 생성 및 큐잉
    rollout = Rollout(task="Test task", priority=5)
    await store.enqueue_rollout(rollout)
    
    # 롤아웃 가져오기
    dequeued = await store.dequeue_rollout()
    
    success = (
        dequeued is not None and
        dequeued.task == "Test task" and
        dequeued.status == RolloutStatus.IN_PROGRESS
    )
    print_check("Rollout enqueue/dequeue", success)
    results.append(success)
    
    return all(results)


async def test_functional_reward():
    """Reward 기능 테스트"""
    print_section("7. Functional Test: Reward")
    results = []
    
    from unified_agent import (
        AgentTracer, SpanKind, set_tracer,
        emit_reward, find_reward_spans, get_reward_value
    )
    
    collected = []
    tracer = AgentTracer(name="reward-test")
    tracer.add_callback_on_span_end(lambda s: collected.append(s))
    set_tracer(tracer)
    
    # 리워드 발행
    with tracer.span("task", SpanKind.AGENT):
        emit_reward(0.95)
    
    reward_spans = find_reward_spans(collected)
    success = len(reward_spans) == 1 and get_reward_value(reward_spans[0]) == 0.95
    print_check("Reward emission and retrieval", success)
    results.append(success)
    
    return all(results)


async def test_functional_hooks():
    """Hooks 기능 테스트"""
    print_section("8. Functional Test: Hooks")
    results = []
    
    from unified_agent import HookManager, HookEvent
    
    hooks = HookManager()
    called = [False]
    
    @hooks.on_trace_start()
    async def on_start(ctx):
        called[0] = True
    
    await hooks.emit(HookEvent.TRACE_START)
    
    success = called[0]
    print_check("Hook registration and execution", success)
    results.append(success)
    
    return all(results)


async def test_functional_adapter():
    """Adapter 기능 테스트"""
    print_section("9. Functional Test: Adapter")
    results = []
    
    import time
    from unified_agent import Span, SpanKind, TracerTraceToTriplet
    
    base_time = time.time()
    
    # 테스트 스팬 생성
    span = Span(
        name="llm_call",
        kind=SpanKind.LLM,
        span_id="test-1",
        trace_id="trace-1",
        start_time=base_time,
        sequence_id=1,
    )
    span.attributes["llm.prompt"] = "Hello"
    span.attributes["llm.response.content"] = "Hi"
    
    adapter = TracerTraceToTriplet()
    triplets = adapter.adapt([span])
    
    success = len(triplets) == 1 and triplets[0].prompt.get("text") == "Hello"
    print_check("Span to Triplet conversion", success)
    results.append(success)
    
    return all(results)


async def test_integration():
    """통합 테스트"""
    print_section("10. Integration Test: Full Pipeline")
    results = []
    
    from unified_agent import (
        AgentTracer, SpanKind, set_tracer,
        InMemoryAgentStore, Rollout, set_store,
        emit_reward, find_reward_spans,
        TracerTraceToTriplet,
        HookManager, HookEvent
    )
    
    # 1. 컴포넌트 초기화
    collected_spans = []
    tracer = AgentTracer(name="integration")
    tracer.add_callback_on_span_end(lambda s: collected_spans.append(s))
    set_tracer(tracer)
    
    store = InMemoryAgentStore()
    await store.initialize()
    set_store(store)
    
    hooks = HookManager()
    events = []
    
    @hooks.on_trace_start()
    async def log_start(ctx):
        events.append("start")
    
    @hooks.on_trace_end()
    async def log_end(ctx):
        events.append("end")
    
    # 2. 롤아웃 생성
    rollout = Rollout(task="Integration test", priority=10)
    await store.enqueue_rollout(rollout)
    rollout = await store.dequeue_rollout()
    
    # 3. 에이전트 실행 시뮬레이션
    await hooks.emit(HookEvent.TRACE_START)
    
    with tracer.span("agent_run", SpanKind.WORKFLOW):
        with tracer.span("llm_call", SpanKind.LLM) as llm:
            llm.set_attribute("llm.model", "gpt-5")
            llm.set_attribute("llm.prompt", "Test")
            llm.set_attribute("llm.response.content", "OK")
        
        emit_reward(1.0)
    
    await hooks.emit(HookEvent.TRACE_END)
    
    # 4. 트리플렛 변환
    adapter = TracerTraceToTriplet()
    triplets = adapter.adapt(collected_spans)
    
    # 5. 검증
    checks = [
        ("Spans collected", len(collected_spans) >= 3),
        ("Hooks triggered", events == ["start", "end"]),
        ("Rollout processed", rollout is not None),
        ("Triplets generated", len(triplets) >= 1),
        ("Reward recorded", len(find_reward_spans(collected_spans)) == 1),
    ]
    
    for name, check in checks:
        print_check(name, check)
        results.append(check)
    
    return all(results)


def count_exports():
    """Export 개수 확인"""
    print_section("11. Export Count Verification")
    
    from unified_agent import __all__
    
    categories = {
        "Exceptions": 0,
        "Config": 0,
        "Models": 0,
        "Utils": 0,
        "Memory": 0,
        "PersistentMemory": 0,
        "Compaction": 0,
        "SessionTree": 0,
        "Tracer": 0,
        "AgentStore": 0,
        "Reward": 0,
        "Adapter": 0,
        "Hooks": 0,
        "Events": 0,
        "Skills": 0,
        "Tools": 0,
        "Agents": 0,
        "Workflow": 0,
        "Framework": 0,
    }
    
    # 간단한 카운트
    total = len(__all__)
    print_check(f"Total exports: {total}", total > 100)
    
    # v3.3 exports 확인
    v33_exports = [
        "Span", "SpanKind", "AgentTracer",
        "Rollout", "Attempt", "InMemoryAgentStore",
        "emit_reward", "RewardManager",
        "Triplet", "TracerTraceToTriplet",
        "HookManager", "HookEvent"
    ]
    
    missing = [e for e in v33_exports if e not in __all__]
    print_check(f"v3.3 exports present", len(missing) == 0, f"Missing: {missing}")
    
    return len(missing) == 0


async def main():
    print("\n" + "=" * 70)
    print("  UNIFIED AGENT FRAMEWORK v3.3 - FULL VERIFICATION")
    print("=" * 70)
    
    test_results = {}
    
    # Import 테스트
    test_results["core_imports"] = test_core_imports()
    test_results["v31_modules"] = test_v31_modules()
    test_results["v32_modules"] = test_v32_modules()
    test_results["v33_modules"] = test_v33_modules()
    
    # 기능 테스트
    test_results["func_tracer"] = await test_functional_tracer()
    test_results["func_store"] = await test_functional_store()
    test_results["func_reward"] = await test_functional_reward()
    test_results["func_hooks"] = await test_functional_hooks()
    test_results["func_adapter"] = await test_functional_adapter()
    
    # 통합 테스트
    test_results["integration"] = await test_integration()
    
    # Export 검증
    test_results["exports"] = count_exports()
    
    # 최종 결과
    print_section("FINAL RESULTS")
    
    passed = 0
    failed = 0
    
    for name, result in test_results.items():
        status = "[PASS]" if result else "[FAIL]"
        print(f"  {status} {name}")
        if result:
            passed += 1
        else:
            failed += 1
    
    print("\n" + "-" * 70)
    print(f"  Total: {passed}/{passed + failed} tests passed")
    
    if failed == 0:
        print("\n  [SUCCESS] ALL VERIFICATION TESTS PASSED!")
        print("  Unified Agent Framework v3.3 is fully functional.")
    else:
        print(f"\n  [WARNING] {failed} test(s) failed. Check logs above.")
    
    print("=" * 70)
    
    return failed == 0


if __name__ == "__main__":
    success = asyncio.run(main())
    sys.exit(0 if success else 1)
