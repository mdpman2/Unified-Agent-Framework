#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Unified Agent Framework - 프레임워크 메인 모듈

UnifiedAgentFramework 클래스 및 데모/유틸리티 함수들
"""

from __future__ import annotations

import asyncio
import json
import time
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, TYPE_CHECKING

from opentelemetry import trace

from semantic_kernel import Kernel
from semantic_kernel.connectors.ai.open_ai import AzureChatCompletion

from .config import (
    FrameworkConfig, DEFAULT_LLM_MODEL, SUPPORTED_MODELS,
    supports_temperature, create_execution_settings
)
from .exceptions import ConfigurationError
from .models import (
    AgentRole, AgentState, ExecutionStatus, ApprovalStatus,
    TeamConfiguration, TeamAgent, PlanStep, MPlan, PlanStepStatus,
    WebSocketMessageType, StreamingMessage, RAICategory
)
from .memory import MemoryStore, CachedMemoryStore, StateManager
from .events import EventType, EventBus, AgentEvent
from .skills import SkillManager
from .tools import MCPTool
from .agents import SimpleAgent, RouterAgent, SupervisorAgent, ProxyAgent
from .workflow import Node, Graph
from .orchestration import OrchestrationManager, AgentFactory
from .utils import StructuredLogger, RAIValidator, setup_telemetry

# v3.3 Agent Lightning 통합
from .tracer import AgentTracer, SpanKind, get_tracer, set_tracer
from .hooks import HookManager, HookEvent, get_hook_manager, set_hook_manager
from .reward import emit_reward, RewardManager

# v3.4 Extensions 통합
from .extensions import Extensions, ExtensionsConfig

# v4.0 핵심 모듈 (지연 임포트 아닌 직접 임포트 — TYPE_CHECKING 불필요)
from .responses_api import ResponsesClient, ConversationState
from .video_generation import VideoGenerator, VideoConfig
from .image_generation import ImageGenerator, ImageConfig
from .open_weight import OpenWeightAdapter, OSSModelConfig
from .universal_bridge import UniversalAgentBridge, BridgeProtocol

# v4.0 프레임워크 브릿지 (지연 임포트 — get_bridge()에서 로드)

__all__ = [
    "UnifiedAgentFramework",
    "quick_run",
    "create_framework",
]

# 기본 스킬 디렉토리
BUILTIN_SKILLS_DIR = Path(__file__).parent.parent / "skills"

# ============================================================================
# UnifiedAgentFramework - 통합 에이전트 프레임워크
# ============================================================================

class UnifiedAgentFramework:
    """
    통합 Agent 프레임워크 - Enterprise Edition

    간편한 사용법:
        # 1. 가장 간단한 방법 (환경변수에서 자동 로드)
        framework = UnifiedAgentFramework.create()

        # 2. 설정 객체 사용
        config = FrameworkConfig.from_env()
        framework = UnifiedAgentFramework.create(config)

        # 3. 빠른 질의응답
        response = await framework.quick_chat("안녕하세요!")

        # 4. 워크플로우 실행
        state = await framework.run("session-1", "simple_chat", "질문입니다")

        # 5. Skills 기반 에이전트
        agent = framework.create_skilled_agent("coder", skills=["python-expert"])

    주요 기능:
    - MCP 도구 관리
    - 이벤트 시스템 (Pub-Sub)
    - 전역 메트릭 수집
    - 체크포인트 및 롤백
    - Human-in-the-loop 승인
    - Skills 시스템 (Anthropic 패턴)
    """

    def __init__(
        self,
        kernel: Kernel,
        config: FrameworkConfig | None = None,
        memory_store: MemoryStore | None = None,
        checkpoint_dir: str = "./checkpoints",
        enable_telemetry: bool = True,
        enable_events: bool = True,
        skill_dirs: list[str] | None = None,
        load_builtin_skills: bool = True,
        extensions_config: ExtensionsConfig | None = None,
    ):
        """
        프레임워크 초기화

        Args:
            kernel: Semantic Kernel 인스턴스
            config: 프레임워크 설정
            memory_store: 메모리 저장소
            checkpoint_dir: 체크포인트 디렉토리
            enable_telemetry: 텔레메트리 활성화 여부
            enable_events: 이벤트 시스템 활성화 여부
            skill_dirs: 스킬 디렉토리 목록
            load_builtin_skills: 기본 스킬 로드 여부
            extensions_config: v3.4 확장 모듈 설정
        """
        self.kernel = kernel
        self.config = config or FrameworkConfig()
        self.memory_store = memory_store or CachedMemoryStore(max_cache_size=self.config.max_cache_size)
        self.state_manager = StateManager(self.memory_store, checkpoint_dir)
        self.graphs: dict[str, Graph] = {}
        self.mcp_tools: dict[str, MCPTool] = {}
        self.event_bus = EventBus() if enable_events else None

        # Skills 시스템 초기화
        self.skill_manager = SkillManager(skill_dirs)
        if load_builtin_skills:
            self._load_builtin_skills()

        # v3.3 Agent Lightning 통합
        self.agent_tracer = AgentTracer(name="unified-agent-framework")
        set_tracer(self.agent_tracer)
        
        # v3.3 Hook Manager 통합
        self.hook_manager = HookManager()
        set_hook_manager(self.hook_manager)
        
        # v3.3 Reward Manager 통합
        self.reward_manager = RewardManager()
        
        # v3.4 Extensions 통합 (Prompt Cache, Durable, Concurrent, AgentTool, Thinking, MCP)
        self.extensions = Extensions(
            framework=self,
            config=extensions_config or ExtensionsConfig()
        )

        if enable_telemetry:
            self.tracer = trace.get_tracer(__name__)
        else:
            self.tracer = None

        self.global_metrics = {
            "total_workflows": 0,
            "total_executions": 0,
            "total_failures": 0,
            "start_time": datetime.now(timezone.utc).isoformat()
        }

    def _load_builtin_skills(self):
        """기본 제공 스킬 로드 (SKILL.md 파일 기반)"""
        if BUILTIN_SKILLS_DIR.exists():
            loaded = self.skill_manager.load_skills_from_directory(str(BUILTIN_SKILLS_DIR))
            logging.info(f"📚 SKILL.md 기반 스킬 {loaded}개 로드 완료 (from {BUILTIN_SKILLS_DIR})")
        else:
            logging.warning(f"⚠️ 기본 스킬 디렉토리가 없습니다: {BUILTIN_SKILLS_DIR}")
            logging.info("💡 'skills' 디렉토리를 생성하고 SKILL.md 파일을 추가하세요.")

    def _create_kernel(self) -> Kernel:
        """Kernel 인스턴스 생성 (내부 메서드)"""
        kernel = Kernel()
        chat_service = AzureChatCompletion(
            deployment_name=self.config.deployment_name,
            api_key=self.config.api_key,
            endpoint=self.config.endpoint,
            service_id=self.config.deployment_name,
            api_version=self.config.api_version
        )
        kernel.add_service(chat_service)
        return kernel

    @classmethod
    def create(
        cls,
        config: FrameworkConfig | None = None,
        skill_dirs: list[str] | None = None,
        load_builtin_skills: bool = True,
        extensions_config: ExtensionsConfig | None = None,
    ) -> 'UnifiedAgentFramework':
        """
        프레임워크 간편 생성 (권장)

        사용법:
            # 환경변수에서 자동 로드
            framework = UnifiedAgentFramework.create()

            # 커스텀 설정 + 스킬 디렉토리
            framework = UnifiedAgentFramework.create(
                skill_dirs=["./my_skills", "./team_skills"]
            )
            
            # v3.4 확장 모듈 설정
            framework = UnifiedAgentFramework.create(
                extensions_config=ExtensionsConfig(
                    enable_cache=True,
                    enable_mcp=True
                )
            )
        """
        if config is None:
            config = FrameworkConfig.from_env()

        config.validate()

        # Kernel 초기화
        kernel = Kernel()
        chat_service = AzureChatCompletion(
            deployment_name=config.deployment_name,
            api_key=config.api_key,
            endpoint=config.endpoint,
            service_id=config.deployment_name,
            api_version=config.api_version
        )
        kernel.add_service(chat_service)

        return cls(
            kernel=kernel,
            config=config,
            checkpoint_dir=config.checkpoint_dir,
            enable_telemetry=config.enable_telemetry,
            enable_events=config.enable_events,
            skill_dirs=skill_dirs,
            load_builtin_skills=load_builtin_skills,
            extensions_config=extensions_config,
        )

    async def quick_chat(self, message: str, system_prompt: str = "You are a helpful assistant.") -> str:
        """
        빠른 질의응답 (워크플로우 없이)

        사용법:
            response = await framework.quick_chat("파이썬이란 무엇인가요?")
            print(response)
        """
        # v3.3: 자동 추적 시작
        with self.agent_tracer.span("quick_chat", SpanKind.WORKFLOW) as span:
            span.set_attribute("user.message", message[:100])  # 처음 100자만
            
            # v3.3: Hook 이벤트 발행
            await self.hook_manager.emit(HookEvent.TRACE_START, {"message": message})
            
            if "_quick_chat" not in self.graphs:
                self.create_simple_workflow("_quick_chat", system_prompt)

            session_id = f"quick-{int(time.time())}"
            state = await self.run(session_id, "_quick_chat", message)

            response = ""
            for msg in reversed(state.messages):
                if msg.role == AgentRole.ASSISTANT:
                    response = msg.content
                    break
            
            # v3.3: 응답 정보 추적
            span.set_attribute("response.length", len(response))
            
            # v3.3: Hook 이벤트 발행
            await self.hook_manager.emit(HookEvent.TRACE_END, {"response_length": len(response)})
            
            return response

    def create_simple_workflow(self, name: str, system_prompt: str = "You are a helpful assistant.") -> Graph:
        """간단한 대화 워크플로우 생성"""
        graph = self.create_graph(name)

        agent = SimpleAgent(
            name="assistant",
            system_prompt=system_prompt,
            model=self.config.model,
            temperature=self.config.temperature,
            max_tokens=self.config.max_tokens,
            enable_streaming=self.config.enable_streaming,
            event_bus=self.event_bus,
            service_id=self.config.deployment_name
        )

        graph.add_node(Node("assistant", agent))
        graph.set_start("assistant")
        graph.set_end("assistant")

        return graph

    def create_router_workflow(self, name: str, routes: dict[str, dict[str, str]]) -> Graph:
        """라우팅 워크플로우 생성"""
        graph = self.create_graph(name)

        router = RouterAgent(
            name="router",
            system_prompt="Classify user intent accurately.",
            model=self.config.model,
            routes={k: f"{k}_agent" for k in routes.keys()},
            event_bus=self.event_bus,
            service_id=self.config.deployment_name
        )
        graph.add_node(Node("router", router))
        graph.set_start("router")

        for route_name, route_config in routes.items():
            agent = SimpleAgent(
                name=f"{route_name}_agent",
                system_prompt=route_config.get("prompt", f"You handle {route_name} inquiries."),
                model=self.config.model,
                event_bus=self.event_bus,
                service_id=self.config.deployment_name
            )
            graph.add_node(Node(f"{route_name}_agent", agent))
            graph.set_end(f"{route_name}_agent")

        return graph

    def create_skilled_agent(
        self,
        name: str,
        skills: list[str] | None = None,
        base_prompt: str = "",
        auto_detect_skills: bool = True
    ) -> SimpleAgent:
        """Skills 기반 에이전트 생성"""
        skill_objects = []
        if skills:
            for skill_name in skills:
                skill = self.skill_manager.get_skill(skill_name)
                if skill:
                    skill_objects.append(skill)
                else:
                    logging.warning(f"스킬을 찾을 수 없습니다: {skill_name}")

        system_prompt = self.skill_manager.build_system_prompt(
            skill_objects,
            base_prompt=base_prompt,
            include_full=True
        )

        agent = SimpleAgent(
            name=name,
            system_prompt=system_prompt,
            model=self.config.model,
            temperature=self.config.temperature,
            max_tokens=self.config.max_tokens,
            enable_streaming=self.config.enable_streaming,
            event_bus=self.event_bus,
            service_id=self.config.deployment_name
        )

        agent._auto_detect_skills = auto_detect_skills
        agent._skill_manager = self.skill_manager

        return agent

    def create_skill_workflow(
        self,
        name: str,
        skills: list[str],
        base_prompt: str = "You are a helpful assistant."
    ) -> Graph:
        """Skills 기반 워크플로우 생성"""
        graph = self.create_graph(name)

        agent = self.create_skilled_agent(
            name="skilled_assistant",
            skills=skills,
            base_prompt=base_prompt
        )

        graph.add_node(Node("skilled_assistant", agent))
        graph.set_start("skilled_assistant")
        graph.set_end("skilled_assistant")

        return graph

    async def smart_chat(
        self,
        message: str,
        base_prompt: str = "You are a helpful assistant.",
        max_skills: int = 2
    ) -> str:
        """스마트 질의응답 - 쿼리에 맞는 스킬 자동 활성화"""
        matched_skills = self.skill_manager.match_skills(
            message,
            threshold=0.2,
            max_skills=max_skills
        )

        if matched_skills:
            skill_names = [s.name for s in matched_skills]
            logging.info(f"🎯 매칭된 스킬: {', '.join(skill_names)}")

        workflow_name = f"_smart_chat_{int(time.time())}"
        self.create_skill_workflow(
            workflow_name,
            skills=[s.name for s in matched_skills],
            base_prompt=base_prompt
        )

        session_id = f"smart-{int(time.time())}"
        state = await self.run(session_id, workflow_name, message)

        for msg in reversed(state.messages):
            if msg.role == AgentRole.ASSISTANT:
                return msg.content
        return ""

    def create_team_workflow(
        self,
        team_config: TeamConfiguration,
        name: str | None = None
    ) -> Graph:
        """Team 기반 워크플로우 생성 (Microsoft Pattern)"""
        workflow_name = name or f"team_{team_config.name}"
        graph = self.create_graph(workflow_name)

        factory = AgentFactory(framework=self)
        agents = factory.create_team(team_config)

        if team_config.orchestration_mode == "supervisor":
            supervisor = SupervisorAgent(
                name="team_supervisor",
                system_prompt=f"You supervise the {team_config.name} team. {team_config.description}",
                model=self.config.model,
                sub_agents=list(agents.values()),
                max_rounds=team_config.max_rounds,
                event_bus=self.event_bus,
                service_id=self.config.deployment_name
            )
            graph.add_node(Node("team_supervisor", supervisor))
            graph.set_start("team_supervisor")
            graph.set_end("team_supervisor")

        elif team_config.orchestration_mode == "sequential":
            agent_list = list(agents.items())
            for i, (agent_name, agent) in enumerate(agent_list):
                node = Node(agent_name, agent)
                graph.add_node(node)

                if i == 0:
                    graph.set_start(agent_name)
                if i == len(agent_list) - 1:
                    graph.set_end(agent_name)
                if i > 0:
                    prev_name = agent_list[i-1][0]
                    graph.add_edge(prev_name, agent_name)

        else:
            first_agent = list(agents.values())[0] if agents else None
            if first_agent:
                graph.add_node(Node(first_agent.name, first_agent))
                graph.set_start(first_agent.name)
                graph.set_end(first_agent.name)

        return graph

    def create_orchestration_manager(
        self,
        team_config: TeamConfiguration,
        require_plan_approval: bool = False,
        ws_callback: Callable | None = None
    ) -> OrchestrationManager:
        """OrchestrationManager 생성 (Microsoft Pattern)"""
        return OrchestrationManager(
            team_config=team_config,
            framework=self,
            kernel=self.kernel,
            require_plan_approval=require_plan_approval,
            rai_validator=RAIValidator(),
            ws_callback=ws_callback
        )

    def create_agent_factory(self) -> AgentFactory:
        """AgentFactory 인스턴스 생성"""
        return AgentFactory(framework=self)

    def create_proxy_agent(
        self,
        name: str = "clarifier",
        clarification_callback: Callable | None = None
    ) -> ProxyAgent:
        """ProxyAgent 생성 (Microsoft Pattern)"""
        return ProxyAgent(
            name=name,
            system_prompt="You help clarify user requests when needed.",
            model=self.config.model,
            event_bus=self.event_bus,
            service_id=self.config.deployment_name,
            clarification_callback=clarification_callback
        )

    def create_rai_validator(self, strict_mode: bool = False) -> RAIValidator:
        """RAI 검증기 생성"""
        return RAIValidator(strict_mode=strict_mode)

    def create_graph(self, name: str) -> Graph:
        """워크플로우 그래프 생성"""
        graph = Graph(name)
        self.graphs[name] = graph
        logging.info(f"🎨 그래프 생성: {name}")
        return graph

    def register_mcp_tool(self, tool: MCPTool) -> None:
        """MCP 도구 등록"""
        self.mcp_tools[tool.name] = tool
        logging.info(f"🔧 MCP 도구 등록: {tool.name}")

    async def run(
        self,
        session_id: str,
        workflow_name: str,
        user_message: str = "",
        restore_from_checkpoint: bool = False,
        checkpoint_tag: str | None = None
    ) -> AgentState:
        """워크플로우 실행"""
        # 상태 복원
        if restore_from_checkpoint:
            state = await self.state_manager.restore_checkpoint(session_id, tag=checkpoint_tag)
            if not state:
                logging.warning("⚠️ 체크포인트 복원 실패, 새 세션 시작")
                state = None
        else:
            state = await self.state_manager.load_state(session_id)

        if not state:
            state = AgentState(session_id=session_id, workflow_name=workflow_name)
            logging.info(f"🆕 새 세션 시작: {session_id}")

        if user_message:
            state.add_message(AgentRole.USER, user_message)
            if self.event_bus:
                await self.event_bus.publish(AgentEvent(
                    event_type=EventType.MESSAGE_RECEIVED,
                    data={"content": user_message}
                ))

        graph = self.graphs.get(workflow_name)
        if not graph:
            raise ValueError(f"워크플로우 '{workflow_name}'를 찾을 수 없습니다.")

        start_time = time.time()
        self.global_metrics["total_executions"] += 1

        try:
            if self.tracer:
                with self.tracer.start_as_current_span("workflow_execution") as span:
                    span.set_attribute("session_id", session_id)
                    span.set_attribute("workflow_name", workflow_name)
                    state = await graph.execute(state, self.kernel)
                    span.set_attribute("status", state.execution_status.value)
                    span.set_attribute("iterations", state.metrics.get("total_iterations", 0))
            else:
                state = await graph.execute(state, self.kernel)

            execution_time = (time.time() - start_time) * 1000
            state.metrics["execution_time_ms"] = execution_time
            state.metrics["success"] = state.execution_status == ExecutionStatus.COMPLETED

        except Exception as e:
            logging.error(f"❌ 워크플로우 실행 오류: {e}")
            self.global_metrics["total_failures"] += 1
            state.execution_status = ExecutionStatus.FAILED
            state.metadata["error"] = str(e)

        await self.state_manager.save_state(state)

        if state.execution_status == ExecutionStatus.COMPLETED:
            await self.state_manager.save_checkpoint(state, tag="auto")

        return state

    async def approve_pending_request(
        self,
        session_id: str,
        request_id: int,
        approved: bool
    ) -> AgentState:
        """대기 중인 승인 요청 처리"""
        state = await self.state_manager.load_state(session_id)
        if not state:
            raise ValueError(f"세션 '{session_id}'를 찾을 수 없습니다.")

        if request_id >= len(state.pending_approvals):
            raise ValueError(f"승인 요청 #{request_id}가 존재하지 않습니다.")

        approval = state.pending_approvals[request_id]
        approval["status"] = ApprovalStatus.APPROVED if approved else ApprovalStatus.REJECTED
        approval["approved_at"] = datetime.now(timezone.utc).isoformat()

        if approved:
            state.execution_status = ExecutionStatus.RUNNING
            if self.event_bus:
                await self.event_bus.publish(AgentEvent(
                    event_type=EventType.APPROVAL_GRANTED,
                    data=approval
                ))
        else:
            state.execution_status = ExecutionStatus.FAILED
            if self.event_bus:
                await self.event_bus.publish(AgentEvent(
                    event_type=EventType.APPROVAL_DENIED,
                    data=approval
                ))

        await self.state_manager.save_state(state)
        return state

    def visualize_workflow(self, workflow_name: str) -> str:
        """워크플로우 시각화"""
        graph = self.graphs.get(workflow_name)
        if not graph:
            return f"❌ 워크플로우 '{workflow_name}'를 찾을 수 없습니다."
        return graph.visualize()

    def get_workflow_stats(self, workflow_name: str) -> dict[str, Any]:
        """워크플로우 통계"""
        graph = self.graphs.get(workflow_name)
        if not graph:
            return {}
        return graph.get_statistics()

    def get_global_metrics(self) -> dict[str, Any]:
        """전역 메트릭"""
        return {
            **self.global_metrics,
            "total_workflows": len(self.graphs),
            "total_mcp_tools": len(self.mcp_tools),
            "uptime_seconds": (
                datetime.now(timezone.utc) -
                datetime.fromisoformat(self.global_metrics["start_time"])
            ).total_seconds()
        }

    async def cleanup(self) -> None:
        """리소스 정리"""
        logging.info("🧹 프레임워크 정리 시작")

        for tool in self.mcp_tools.values():
            await tool.disconnect()
        
        # v3.4 Extensions 정리
        if self.extensions:
            await self.extensions.cleanup()

        logging.info("✅ 프레임워크 정리 완료")

    # ========================================================================
    # v4.0 팩토리 메서드 — 새로운 핵심 기능 편의 접근
    # ========================================================================

    def create_responses_client(self, config: 'ResponseConfig' | None = None) -> ResponsesClient:
        """
        v4.0 Responses API 클라이언트 생성

        사용법:
            client = framework.create_responses_client()
            response = await client.create("질문입니다")
        """
        from .responses_api import ResponseConfig as RC
        return ResponsesClient(config=config)

    def create_video_generator(self) -> VideoGenerator:
        """
        v4.0 비디오 생성기 생성

        사용법:
            gen = framework.create_video_generator()
            result = await gen.generate("A sunset over the ocean")
        """
        return VideoGenerator()

    def create_image_generator(self) -> ImageGenerator:
        """
        v4.0 이미지 생성기 생성

        사용법:
            gen = framework.create_image_generator()
            result = await gen.generate("A futuristic city")
        """
        return ImageGenerator()

    def create_open_weight_adapter(self, default_endpoint: str | None = None) -> OpenWeightAdapter:
        """
        v4.0 오픈 가중치 모델 어댑터 생성

        사용법:
            adapter = framework.create_open_weight_adapter()
            response = await adapter.generate(model="gpt-oss-120b", prompt="Hello!")
        """
        return OpenWeightAdapter(default_endpoint=default_endpoint)

    def create_universal_bridge(self) -> UniversalAgentBridge:
        """
        v4.0 Universal Agent Bridge 생성

        사용법:
            bridge = framework.create_universal_bridge()
            bridge.register("semantic-kernel", SemanticKernelAgentBridge())
            result = await bridge.run(framework="semantic-kernel", task="분석해줘")
        """
        return UniversalAgentBridge()

    def get_bridge(self, protocol: str) -> Any:
        """
        v4.0 프레임워크 브릿지 인스턴스 반환

        지원 프로토콜: semantic-kernel, openai-agents,
                      google-adk, crewai, ag2, ms-agent, a2a

        사용법:
            bridge = framework.get_bridge("crewai")
            result = await bridge.run(task="데이터를 분석해줘")
        """
        # 지연 임포트 — 필요한 브릿지만 로드하여 시작 시간 최적화
        from .sk_agent_bridge import SemanticKernelAgentBridge
        from .openai_agents_bridge import OpenAIAgentsBridge
        from .google_adk_bridge import GoogleADKBridge
        from .crewai_bridge import CrewAIBridge
        from .ag2_bridge import AG2Bridge
        from .ms_agent_bridge import MicrosoftAgentBridge
        from .a2a_bridge import A2ABridge

        bridge_map: dict[str, type] = {
            "semantic-kernel": SemanticKernelAgentBridge,
            "openai-agents": OpenAIAgentsBridge,
            "google-adk": GoogleADKBridge,
            "crewai": CrewAIBridge,
            "ag2": AG2Bridge,
            "ms-agent": MicrosoftAgentBridge,
            "a2a": A2ABridge,
        }
        cls = bridge_map.get(protocol)
        if cls is None:
            raise ConfigurationError(
                f"지원하지 않는 브릿지 프로토콜: '{protocol}'. "
                f"지원 목록: {', '.join(bridge_map)}"
            )
        return cls()

# ============================================================================
# 간편 사용 함수
# ============================================================================

async def quick_run(message: str, system_prompt: str = "You are a helpful assistant.") -> str:
    """
    가장 간단한 사용법 - 한 줄로 AI 응답 받기

    사용법:
        import asyncio
        from unified_agent import quick_run

        response = asyncio.run(quick_run("파이썬이란 무엇인가요?"))
        print(response)
    """
    framework = UnifiedAgentFramework.create()
    return await framework.quick_chat(message, system_prompt)

def create_framework(
    model: str = None,
    temperature: float = 0.7,
    **kwargs
) -> UnifiedAgentFramework:
    """
    프레임워크 간편 생성

    사용법:
        from unified_agent import create_framework

        framework = create_framework(model="gpt-4o", temperature=0.5)
    """
    config = FrameworkConfig.from_env()
    if model is not None:
        config.model = model
    config.temperature = temperature

    for key, value in kwargs.items():
        if hasattr(config, key):
            setattr(config, key, value)

    return UnifiedAgentFramework.create(config)
