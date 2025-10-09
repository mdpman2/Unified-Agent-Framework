"""
Unified Agent Framework - 최고의 Agent 프레임워크들의 장점 통합
- Microsoft AutoGen: Multi-agent 협업
- Semantic Kernel: 플러그인 시스템
- Microsoft Agent Framework: 표준 프로토콜, 관찰성
- LangGraph: 상태 기반 그래프, 조건부 라우팅

pip install semantic-kernel python-dotenv redis opentelemetry-api opentelemetry-sdk pydantic
"""

import os
import asyncio
import json
import logging
from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Any, Callable, Set
from datetime import datetime, timezone
from enum import Enum
from collections import defaultdict

from dotenv import load_dotenv
from pydantic import BaseModel, Field

# Semantic Kernel
from semantic_kernel import Kernel
from semantic_kernel.connectors.ai.chat_completion_client_base import ChatCompletionClientBase
from semantic_kernel.connectors.ai.function_choice_behavior import FunctionChoiceBehavior
from semantic_kernel.connectors.ai.open_ai.prompt_execution_settings.azure_chat_prompt_execution_settings import AzureChatPromptExecutionSettings
from semantic_kernel.connectors.ai.open_ai.services.azure_chat_completion import AzureChatCompletion
from semantic_kernel.contents.chat_history import ChatHistory

# OpenTelemetry
from opentelemetry import trace
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import ConsoleSpanExporter, BatchSpanProcessor
from opentelemetry.sdk.resources import Resource


# ============================================================================
# 핵심 데이터 모델
# ============================================================================

class AgentRole(str, Enum):
    """Agent 역할"""
    ASSISTANT = "assistant"
    USER = "user"
    SYSTEM = "system"
    FUNCTION = "function"
    ROUTER = "router"
    ORCHESTRATOR = "orchestrator"


class ExecutionStatus(str, Enum):
    """실행 상태"""
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    PAUSED = "paused"
    WAITING_APPROVAL = "waiting_approval"


class Message(BaseModel):
    """메시지 모델"""
    role: AgentRole
    content: str
    metadata: Dict[str, Any] = Field(default_factory=dict)
    timestamp: str = Field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    agent_name: Optional[str] = None


class AgentState(BaseModel):
    """Agent 상태"""
    messages: List[Message] = Field(default_factory=list)
    current_node: str = "start"
    visited_nodes: List[str] = Field(default_factory=list)
    metadata: Dict[str, Any] = Field(default_factory=dict)
    session_id: str
    workflow_name: str = "default"
    execution_status: ExecutionStatus = ExecutionStatus.PENDING
    
    def add_message(self, role: AgentRole, content: str, agent_name: Optional[str] = None):
        """메시지 추가"""
        self.messages.append(Message(
            role=role,
            content=content,
            agent_name=agent_name
        ))
    
    def get_conversation_history(self, max_messages: int = 10) -> List[Message]:
        """최근 대화 기록"""
        return self.messages[-max_messages:]


class NodeResult(BaseModel):
    """노드 실행 결과"""
    node_name: str
    output: str
    next_node: Optional[str] = None
    metadata: Dict[str, Any] = Field(default_factory=dict)
    success: bool = True
    error: Optional[str] = None
    tokens_used: int = 0
    duration_ms: float = 0.0


# ============================================================================
# 메모리 저장소
# ============================================================================

class MemoryStore(ABC):
    """메모리 저장소 인터페이스"""
    
    @abstractmethod
    async def save(self, key: str, data: Dict) -> None:
        pass
    
    @abstractmethod
    async def load(self, key: str) -> Optional[Dict]:
        pass
    
    @abstractmethod
    async def delete(self, key: str) -> None:
        pass


class CachedMemoryStore(MemoryStore):
    """캐싱 메모리 저장소"""
    
    def __init__(self):
        self.data: Dict[str, Dict] = {}
        self.cache: Dict[str, Any] = {}
        self.access_count: Dict[str, int] = defaultdict(int)
    
    async def save(self, key: str, data: Dict) -> None:
        self.data[key] = {
            'data': data,
            'timestamp': datetime.now(timezone.utc).isoformat()
        }
        self.access_count[key] += 1
        
        if self.access_count[key] > 3:
            self.cache[key] = data
    
    async def load(self, key: str) -> Optional[Dict]:
        if key in self.cache:
            self.access_count[key] += 1
            return self.cache[key]
        
        if key in self.data:
            self.access_count[key] += 1
            return self.data[key]['data']
        return None
    
    async def delete(self, key: str) -> None:
        if key in self.data:
            del self.data[key]
        if key in self.cache:
            del self.cache[key]


# ============================================================================
# Agent 기본 클래스
# ============================================================================

class Agent(ABC):
    """Agent 기본 클래스"""
    
    def __init__(
        self,
        name: str,
        role: AgentRole = AgentRole.ASSISTANT,
        system_prompt: str = "You are a helpful AI assistant.",
        model: str = "gpt-4o-mini",
        temperature: float = 0.7,
        max_tokens: int = 1000
    ):
        self.name = name
        self.role = role
        self.system_prompt = system_prompt
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        
        self.execution_settings = AzureChatPromptExecutionSettings(
            temperature=temperature,
            max_tokens=max_tokens,
            service_id=model
        )
    
    @abstractmethod
    async def execute(self, state: AgentState, kernel: Kernel) -> NodeResult:
        """Agent 실행"""
        pass
    
    async def _get_llm_response(self, kernel: Kernel, messages: List[Message]) -> str:
        """LLM 응답 가져오기"""
        chat_completion = kernel.get_service(
            service_id=self.model,
            type=ChatCompletionClientBase
        )
        
        history = ChatHistory()
        history.add_system_message(self.system_prompt)
        
        for msg in messages:
            if msg.role == AgentRole.USER:
                history.add_user_message(msg.content)
            elif msg.role == AgentRole.ASSISTANT:
                history.add_assistant_message(msg.content)
        
        settings = self.execution_settings
        settings.function_choice_behavior = None
        
        response = await chat_completion.get_chat_message_content(
            chat_history=history,
            settings=settings,
            kernel=kernel
        )
        
        return str(response)


class SimpleAgent(Agent):
    """단순 대화 Agent"""
    
    async def execute(self, state: AgentState, kernel: Kernel) -> NodeResult:
        start_time = asyncio.get_event_loop().time()
        
        try:
            recent_messages = state.get_conversation_history(max_messages=5)
            response = await self._get_llm_response(kernel, recent_messages)
            state.add_message(AgentRole.ASSISTANT, response, self.name)
            
            duration_ms = (asyncio.get_event_loop().time() - start_time) * 1000
            
            return NodeResult(
                node_name=self.name,
                output=response,
                success=True,
                duration_ms=duration_ms
            )
        except Exception as e:
            logging.error(f"Agent {self.name} 실행 실패: {e}")
            return NodeResult(
                node_name=self.name,
                output="",
                success=False,
                error=str(e)
            )


class RouterAgent(Agent):
    """라우팅 Agent"""
    
    def __init__(self, *args, routes: Dict[str, str], **kwargs):
        super().__init__(*args, role=AgentRole.ROUTER, **kwargs)
        self.routes = routes
    
    async def execute(self, state: AgentState, kernel: Kernel) -> NodeResult:
        start_time = asyncio.get_event_loop().time()
        
        try:
            recent_messages = state.get_conversation_history(max_messages=3)
            last_message = recent_messages[-1].content if recent_messages else ""
            
            routes_list = ', '.join(self.routes.keys())
            classification_prompt = f"Classify the user's intent from the following message into one of these categories: {routes_list}\n\nUser message: {last_message}\n\nRespond with ONLY the category name."
            
            temp_messages = [Message(role=AgentRole.USER, content=classification_prompt)]
            intent = await self._get_llm_response(kernel, temp_messages)
            intent = intent.strip().lower()
            
            next_node = self.routes.get(intent, list(self.routes.values())[0])
            duration_ms = (asyncio.get_event_loop().time() - start_time) * 1000
            
            logging.info(f"🔀 Router: Intent '{intent}' -> Next '{next_node}'")
            
            return NodeResult(
                node_name=self.name,
                output=f"Routing to: {next_node}",
                next_node=next_node,
                success=True,
                duration_ms=duration_ms,
                metadata={"intent": intent}
            )
        except Exception as e:
            logging.error(f"Router 실행 실패: {e}")
            return NodeResult(
                node_name=self.name,
                output="",
                next_node=list(self.routes.values())[0],
                success=False,
                error=str(e)
            )


class OrchestratorAgent(Agent):
    """Multi-agent 오케스트레이터"""
    
    def __init__(self, *args, agents: List[Agent], **kwargs):
        super().__init__(*args, role=AgentRole.ORCHESTRATOR, **kwargs)
        self.agents = {agent.name: agent for agent in agents}
        self.max_rounds = 3
    
    async def execute(self, state: AgentState, kernel: Kernel) -> NodeResult:
        start_time = asyncio.get_event_loop().time()
        
        try:
            responses = []
            
            for round_num in range(self.max_rounds):
                for agent_name, agent in self.agents.items():
                    logging.info(f"🤝 Round {round_num + 1}: {agent_name} 실행")
                    
                    result = await agent.execute(state, kernel)
                    responses.append(f"[{agent_name}]: {result.output}")
                    
                    if "TERMINATE" in result.output.upper():
                        break
                
                if responses and "TERMINATE" in responses[-1].upper():
                    break
            
            final_output = "\n\n".join(responses)
            duration_ms = (asyncio.get_event_loop().time() - start_time) * 1000
            
            return NodeResult(
                node_name=self.name,
                output=final_output,
                success=True,
                duration_ms=duration_ms,
                metadata={"rounds": round_num + 1}
            )
        except Exception as e:
            logging.error(f"Orchestrator 실행 실패: {e}")
            return NodeResult(
                node_name=self.name,
                output="",
                success=False,
                error=str(e)
            )


# ============================================================================
# 그래프 기반 워크플로우
# ============================================================================

class Node:
    """워크플로우 노드"""
    
    def __init__(self, name: str, agent: Agent, edges: Optional[Dict[str, str]] = None):
        self.name = name
        self.agent = agent
        self.edges = edges or {}
    
    async def execute(self, state: AgentState, kernel: Kernel) -> NodeResult:
        logging.info(f"📍 노드 실행: {self.name}")
        result = await self.agent.execute(state, kernel)
        
        if not result.next_node and self.edges:
            result.next_node = self.edges.get("default", None)
        
        state.visited_nodes.append(self.name)
        return result


class Graph:
    """워크플로우 그래프"""
    
    def __init__(self, name: str = "workflow"):
        self.name = name
        self.nodes: Dict[str, Node] = {}
        self.start_node: Optional[str] = None
        self.end_nodes: Set[str] = set()
    
    def add_node(self, node: Node):
        self.nodes[node.name] = node
        logging.info(f"✅ 노드 추가: {node.name}")
    
    def add_edge(self, from_node: str, to_node: str, condition: str = "default"):
        if from_node not in self.nodes:
            raise ValueError(f"노드 '{from_node}'가 존재하지 않습니다.")
        self.nodes[from_node].edges[condition] = to_node
        logging.info(f"✅ 엣지 추가: {from_node} -> {to_node}")
    
    def set_start(self, node_name: str):
        self.start_node = node_name
        logging.info(f"✅ 시작 노드: {node_name}")
    
    def set_end(self, node_name: str):
        self.end_nodes.add(node_name)
        logging.info(f"✅ 종료 노드: {node_name}")
    
    async def execute(self, state: AgentState, kernel: Kernel, max_iterations: int = 10) -> AgentState:
        if not self.start_node:
            raise ValueError("시작 노드가 설정되지 않았습니다.")
        
        current_node = self.start_node
        iterations = 0
        
        logging.info(f"🚀 워크플로우 시작: {self.name}")
        state.execution_status = ExecutionStatus.RUNNING
        
        while current_node and iterations < max_iterations:
            iterations += 1
            state.current_node = current_node
            
            node = self.nodes.get(current_node)
            if not node:
                logging.error(f"❌ 노드 '{current_node}'를 찾을 수 없습니다.")
                break
            
            result = await node.execute(state, kernel)
            # 수정: .dict() -> .model_dump()
            state.metadata[f"{current_node}_result"] = result.model_dump()
            
            if not result.success:
                logging.error(f"❌ 노드 실행 실패: {result.error}")
                state.execution_status = ExecutionStatus.FAILED
                break
            
            if current_node in self.end_nodes:
                logging.info(f"✅ 워크플로우 완료")
                state.execution_status = ExecutionStatus.COMPLETED
                break
            
            current_node = result.next_node
            
            if not current_node:
                state.execution_status = ExecutionStatus.COMPLETED
                break
        
        if iterations >= max_iterations:
            logging.warning(f"⚠️ 최대 반복 도달")
            state.execution_status = ExecutionStatus.FAILED
        
        return state
    
    def visualize(self) -> str:
        """그래프 시각화 (Mermaid 형식)"""
        mermaid_start = "```"
        mermaid_end = "```"
        
        lines = [mermaid_start, "graph TD"]
        
        for node_name in self.nodes:
            if node_name == self.start_node:
                shape = "([START])"
            elif node_name in self.end_nodes:
                shape = "[END]"
            else:
                shape = f"[{node_name}]"
            lines.append(f"    {node_name}{shape}")
        
        for node_name, node in self.nodes.items():
            for condition, target in node.edges.items():
                label = f"|{condition}|" if condition != "default" else ""
                lines.append(f"    {node_name} --{label}--> {target}")
        
        lines.append(mermaid_end)
        return "\n".join(lines)


# ============================================================================
# 상태 관리
# ============================================================================

class StateManager:
    """상태 관리자"""
    
    def __init__(self, memory_store: MemoryStore, checkpoint_dir: Optional[str] = None):
        self.memory_store = memory_store
        self.checkpoint_dir = checkpoint_dir
        
        if checkpoint_dir and not os.path.exists(checkpoint_dir):
            os.makedirs(checkpoint_dir)
    
    async def save_state(self, state: AgentState):
        # 수정: .dict() -> .model_dump()
        await self.memory_store.save(f"state:{state.session_id}", state.model_dump())
    
    async def load_state(self, session_id: str) -> Optional[AgentState]:
        data = await self.memory_store.load(f"state:{session_id}")
        if data:
            return AgentState(**data)
        return None
    
    async def save_checkpoint(self, state: AgentState) -> str:
        if not self.checkpoint_dir:
            raise ValueError("체크포인트 디렉토리 미설정")
        
        timestamp = datetime.now(timezone.utc).isoformat().replace(':', '-')
        checkpoint_file = os.path.join(
            self.checkpoint_dir,
            f"{state.session_id}_{timestamp}.json"
        )
        
        # 수정: .dict() -> .model_dump()
        with open(checkpoint_file, 'w', encoding='utf-8') as f:
            json.dump(state.model_dump(), f, ensure_ascii=False, indent=2)
        
        logging.info(f"✅ 체크포인트 저장: {checkpoint_file}")
        return checkpoint_file
    
    async def restore_checkpoint(self, session_id: str) -> Optional[AgentState]:
        if not self.checkpoint_dir:
            return None
        
        checkpoints = [
            f for f in os.listdir(self.checkpoint_dir)
            if f.startswith(session_id) and f.endswith('.json')
        ]
        
        if not checkpoints:
            return None
        
        latest = os.path.join(self.checkpoint_dir, sorted(checkpoints)[-1])
        
        with open(latest, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        logging.info(f"✅ 체크포인트 복원: {latest}")
        return AgentState(**data)


# ============================================================================
# 통합 프레임워크
# ============================================================================

class UnifiedAgentFramework:
    """통합 Agent 프레임워크"""
    
    def __init__(
        self,
        kernel: Kernel,
        memory_store: Optional[MemoryStore] = None,
        checkpoint_dir: str = "./checkpoints",
        enable_telemetry: bool = True
    ):
        self.kernel = kernel
        self.memory_store = memory_store or CachedMemoryStore()
        self.state_manager = StateManager(self.memory_store, checkpoint_dir)
        self.graphs: Dict[str, Graph] = {}
        
        if enable_telemetry:
            self.tracer = trace.get_tracer(__name__)
        else:
            self.tracer = None
    
    def create_graph(self, name: str) -> Graph:
        graph = Graph(name)
        self.graphs[name] = graph
        return graph
    
    async def run(
        self,
        session_id: str,
        workflow_name: str,
        user_message: str,
        restore_from_checkpoint: bool = False
    ) -> AgentState:
        
        if restore_from_checkpoint:
            state = await self.state_manager.restore_checkpoint(session_id)
            if not state:
                state = None
        else:
            state = await self.state_manager.load_state(session_id)
        
        if not state:
            state = AgentState(session_id=session_id, workflow_name=workflow_name)
        
        if user_message:
            state.add_message(AgentRole.USER, user_message)
        
        graph = self.graphs.get(workflow_name)
        if not graph:
            raise ValueError(f"워크플로우 '{workflow_name}'를 찾을 수 없습니다.")
        
        if self.tracer:
            with self.tracer.start_as_current_span("workflow_execution") as span:
                span.set_attribute("session_id", session_id)
                span.set_attribute("workflow_name", workflow_name)
                state = await graph.execute(state, self.kernel)
                span.set_attribute("status", state.execution_status.value)
        else:
            state = await graph.execute(state, self.kernel)
        
        await self.state_manager.save_state(state)
        await self.state_manager.save_checkpoint(state)
        
        return state
    
    def visualize_workflow(self, workflow_name: str) -> str:
        graph = self.graphs.get(workflow_name)
        if not graph:
            return f"워크플로우 '{workflow_name}'를 찾을 수 없습니다."
        return graph.visualize()


# ============================================================================
# OpenTelemetry
# ============================================================================

def setup_telemetry(service_name: str = "UnifiedAgentFramework", enable_console: bool = False):
    try:
        resource = Resource.create({"service.name": service_name})
        provider = TracerProvider(resource=resource)
        
        if enable_console:
            processor = BatchSpanProcessor(ConsoleSpanExporter())
            provider.add_span_processor(processor)
        
        trace.set_tracer_provider(provider)
        logging.info(f"✅ OpenTelemetry 설정: {service_name}")
    except Exception as e:
        logging.warning(f"⚠️ OpenTelemetry 설정 실패: {e}")


# ============================================================================
# 메인
# ============================================================================

async def main():
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("semantic_kernel").setLevel(logging.WARNING)
    
    setup_telemetry("UnifiedAgentFramework", enable_console=False)
    
    load_dotenv()
    api_key = os.getenv("AZURE_OPENAI_API_KEY")
    endpoint = os.getenv("AZURE_OPENAI_ENDPOINT")
    deployment_name = os.getenv("AZURE_OPENAI_DEPLOYMENT_NAME")
    
    if not all([api_key, endpoint, deployment_name]):
        raise ValueError("필수 환경 변수 미설정")
    
    print(f"✅ 엔드포인트: {endpoint}")
    print(f"✅ 배포: {deployment_name}")
    
    kernel = Kernel()
    chat_service = AzureChatCompletion(
        deployment_name=deployment_name,
        api_key=api_key,
        endpoint=endpoint,
        service_id="gpt-4o-mini",
        api_version="2024-08-01-preview"
    )
    kernel.add_service(chat_service)
    
    framework = UnifiedAgentFramework(kernel=kernel, checkpoint_dir="./checkpoints", enable_telemetry=True)
    
    # 단순 대화
    simple_graph = framework.create_graph("simple_chat")
    assistant = SimpleAgent(name="assistant", system_prompt="You are a helpful AI assistant.", model="gpt-4o-mini")
    simple_graph.add_node(Node("assistant", assistant))
    simple_graph.set_start("assistant")
    simple_graph.set_end("assistant")
    
    # 라우팅
    routing_graph = framework.create_graph("routing_workflow")
    
    router = RouterAgent(
        name="router",
        system_prompt="Classify user intent",
        model="gpt-4o-mini",
        routes={"order": "order_agent", "support": "support_agent", "general": "general_agent"}
    )
    
    order_agent = SimpleAgent(name="order_agent", system_prompt="You are an order specialist.", model="gpt-4o-mini")
    support_agent = SimpleAgent(name="support_agent", system_prompt="You are a support specialist.", model="gpt-4o-mini")
    general_agent = SimpleAgent(name="general_agent", system_prompt="You are a general assistant.", model="gpt-4o-mini")
    
    routing_graph.add_node(Node("router", router))
    routing_graph.add_node(Node("order_agent", order_agent))
    routing_graph.add_node(Node("support_agent", support_agent))
    routing_graph.add_node(Node("general_agent", general_agent))
    routing_graph.set_start("router")
    routing_graph.set_end("order_agent")
    routing_graph.set_end("support_agent")
    routing_graph.set_end("general_agent")
    
    print("\n" + "="*60)
    print("Routing Workflow 시각화")
    print("="*60)
    print(framework.visualize_workflow("routing_workflow"))
    
    print("\n" + "="*60)
    print("Unified Agent Framework")
    print("명령어: exit, checkpoint, restore, visualize, switch [name]")
    print("="*60 + "\n")
    
    session_id = "session-123"
    current_workflow = "simple_chat"
    
    while True:
        user_input = input(f"[{current_workflow}] User > ")
        
        if user_input.lower() == "exit":
            print("종료")
            break
        elif user_input.lower() == "checkpoint":
            state = await framework.state_manager.load_state(session_id)
            if state:
                await framework.state_manager.save_checkpoint(state)
                print("✅ 체크포인트 저장")
            continue
        elif user_input.lower() == "restore":
            state = await framework.state_manager.restore_checkpoint(session_id)
            print("✅ 복원 완료" if state else "❌ 복원 실패")
            continue
        elif user_input.lower() == "visualize":
            print(framework.visualize_workflow(current_workflow))
            continue
        elif user_input.lower().startswith("switch "):
            workflow_name = user_input.split(" ", 1)[1]
            if workflow_name in framework.graphs:
                current_workflow = workflow_name
                print(f"✅ 전환: {workflow_name}")
            continue
        
        try:
            state = await framework.run(session_id=session_id, workflow_name=current_workflow, user_message=user_input)
            
            if state.messages:
                last = state.messages[-1]
                print(f"\n[{last.agent_name or 'AI'}] > {last.content}")
                print(f"[상태] 노드: {state.current_node}, 방문: {' -> '.join(state.visited_nodes)}, 상태: {state.execution_status.value}\n")
        except Exception as e:
            logging.error(f"오류: {e}", exc_info=True)
            print(f"❌ 오류: {e}")
    
    print("✅ 종료 완료")


if __name__ == "__main__":
    asyncio.run(main())
