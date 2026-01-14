#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Unified Agent Framework - 에이전트 모듈

Agent 기본 클래스 및 다양한 에이전트 구현체들
"""

import re
import json
import time
import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Dict, List, Optional, Any, Callable

from semantic_kernel import Kernel
from semantic_kernel.connectors.ai.chat_completion_client_base import ChatCompletionClientBase
from semantic_kernel.connectors.ai.function_choice_behavior import FunctionChoiceBehavior
from semantic_kernel.contents.chat_history import ChatHistory
from semantic_kernel.contents.streaming_chat_message_content import StreamingChatMessageContent

from .config import DEFAULT_LLM_MODEL, create_execution_settings
from .models import (
    AgentRole, AgentState, Message, NodeResult, ExecutionStatus,
    ApprovalStatus, WebSocketMessageType, StreamingMessage
)
from .events import EventType, AgentEvent, EventBus
from .tools import ApprovalRequiredAIFunction
from .utils import CircuitBreaker

__all__ = [
    "Agent",
    "SimpleAgent",
    "ApprovalAgent",
    "RouterAgent",
    "ProxyAgent",
    "InvestigationPlan",
    "SupervisorAgent",
]


# ============================================================================
# Agent 기본 클래스
# ============================================================================

class Agent(ABC):
    """
    Agent 기본 클래스

    주요 기능:
    1. enable_streaming: 스트리밍 응답 지원
    2. event_bus: 이벤트 발행
    3. circuit_breaker: 회로 차단기 통합
    4. 메트릭 추적: total_executions, total_tokens, total_duration_ms
    """

    def __init__(
        self,
        name: str,
        role: AgentRole = AgentRole.ASSISTANT,
        system_prompt: str = "You are a helpful AI assistant.",
        model: str = DEFAULT_LLM_MODEL,
        temperature: float = 0.7,
        max_tokens: int = 1000,
        enable_streaming: bool = False,
        event_bus: Optional[EventBus] = None,
        service_id: Optional[str] = None
    ):
        self.name = name
        self.role = role
        self.system_prompt = system_prompt
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.enable_streaming = enable_streaming
        self.event_bus = event_bus
        self.service_id = service_id

        # 회로 차단기
        self.circuit_breaker = CircuitBreaker(failure_threshold=3, timeout=30.0)

        # 메트릭
        self.total_executions = 0
        self.total_tokens = 0
        self.total_duration_ms = 0.0

    @abstractmethod
    async def execute(self, state: AgentState, kernel: Kernel) -> NodeResult:
        """에이전트 실행 로직 - 각 에이전트가 구현"""
        pass

    async def _get_llm_response(
        self,
        kernel: Kernel,
        messages: List[Message],
        streaming: bool = False
    ) -> str:
        """LLM 응답 가져오기"""
        chat_history = ChatHistory(system_message=self.system_prompt)

        for msg in messages:
            if msg.role == AgentRole.USER:
                chat_history.add_user_message(msg.content)
            elif msg.role == AgentRole.ASSISTANT:
                chat_history.add_assistant_message(msg.content)

        # 실행 설정
        settings = create_execution_settings(
            model=self.model,
            temperature=self.temperature,
            max_tokens=self.max_tokens,
            service_id=self.service_id
        )

        # 서비스 가져오기
        chat_service = kernel.get_service(type=ChatCompletionClientBase)

        if streaming or self.enable_streaming:
            return await self._stream_response(chat_service, chat_history, settings)
        else:
            response = await chat_service.get_chat_message_content(
                chat_history=chat_history,
                settings=settings
            )
            return str(response) if response else ""

    async def _stream_response(
        self,
        chat_service: ChatCompletionClientBase,
        chat_history: ChatHistory,
        settings
    ) -> str:
        """스트리밍 응답 처리"""
        full_response = []
        async for chunk in chat_service.get_streaming_chat_message_content(
            chat_history=chat_history,
            settings=settings
        ):
            if isinstance(chunk, StreamingChatMessageContent):
                content = str(chunk)
                full_response.append(content)
                print(content, end='', flush=True)
        print()
        return "".join(full_response)

    async def _emit_event(self, event_type: EventType, data: Dict[str, Any]):
        """이벤트 발행"""
        if self.event_bus:
            event = AgentEvent(
                event_type=event_type,
                agent_name=self.name,
                data=data
            )
            await self.event_bus.publish(event)


# ============================================================================
# SimpleAgent - 단순 대화 에이전트
# ============================================================================

class SimpleAgent(Agent):
    """
    단순 대화 Agent

    주요 기능:
    1. 이벤트 발행 (AGENT_STARTED, AGENT_COMPLETED, AGENT_FAILED)
    2. 회로 차단기를 통한 호출
    3. 메트릭 수집 (total_executions, total_duration_ms)
    """

    async def execute(self, state: AgentState, kernel: Kernel) -> NodeResult:
        start_time = time.time()

        await self._emit_event(EventType.AGENT_STARTED, {"node": self.name})

        try:
            recent_messages = state.get_conversation_history(max_messages=5)

            response = await self.circuit_breaker.call(
                self._get_llm_response,
                kernel,
                recent_messages,
                self.enable_streaming
            )

            state.add_message(AgentRole.ASSISTANT, response, self.name)

            duration_ms = (time.time() - start_time) * 1000

            self.total_executions += 1
            self.total_duration_ms += duration_ms

            await self._emit_event(EventType.AGENT_COMPLETED, {
                "node": self.name,
                "duration_ms": duration_ms
            })

            return NodeResult(
                node_name=self.name,
                output=response,
                success=True,
                duration_ms=duration_ms
            )
        except Exception as e:
            logging.error(f"❌ Agent {self.name} 실행 실패: {e}")

            await self._emit_event(EventType.AGENT_FAILED, {
                "node": self.name,
                "error": str(e)
            })

            return NodeResult(
                node_name=self.name,
                output="",
                success=False,
                error=str(e)
            )


# ============================================================================
# ApprovalAgent - 승인 필요 에이전트
# ============================================================================

class ApprovalAgent(Agent):
    """
    승인이 필요한 작업을 수행하는 Agent

    Human-in-the-loop 패턴 구현

    사용 시나리오:
    - 데이터 삭제 작업
    - 결제 처리
    - 중요 설정 변경
    - 외부 API 호출
    """

    def __init__(self, *args, approval_function: ApprovalRequiredAIFunction, **kwargs):
        super().__init__(*args, **kwargs)
        self.approval_function = approval_function

    async def execute(self, state: AgentState, kernel: Kernel) -> NodeResult:
        start_time = time.time()

        try:
            recent_messages = state.get_conversation_history(max_messages=3)
            last_message = recent_messages[-1].content if recent_messages else ""

            approval_result = await self.approval_function.execute(input=last_message)

            if approval_result["status"] == ApprovalStatus.PENDING:
                state.add_pending_approval(approval_result)
                await self._emit_event(EventType.APPROVAL_REQUESTED, approval_result)

                return NodeResult(
                    node_name=self.name,
                    output=f"승인 대기 중: {approval_result['description']}",
                    success=True,
                    requires_approval=True,
                    approval_data=approval_result,
                    duration_ms=(time.time() - start_time) * 1000
                )
            else:
                result = approval_result.get("result", "")
                state.add_message(AgentRole.ASSISTANT, str(result), self.name)

                return NodeResult(
                    node_name=self.name,
                    output=str(result),
                    success=True,
                    duration_ms=(time.time() - start_time) * 1000
                )

        except Exception as e:
            logging.error(f"❌ ApprovalAgent 실행 실패: {e}")
            return NodeResult(
                node_name=self.name,
                output="",
                success=False,
                error=str(e)
            )


# ============================================================================
# RouterAgent - 라우팅 에이전트
# ============================================================================

class RouterAgent(Agent):
    """
    라우팅 Agent

    주요 기능:
    1. default_route 파라미터
    2. routing_history 추적
    3. 메타데이터에 confidence 추가
    """

    def __init__(self, *args, routes: Dict[str, str],
                 default_route: Optional[str] = None, **kwargs):
        super().__init__(*args, role=AgentRole.ROUTER, **kwargs)
        self.routes = routes
        self.default_route = default_route or list(routes.values())[0] if routes else None
        self.routing_history: List[Dict[str, Any]] = []

    async def execute(self, state: AgentState, kernel: Kernel) -> NodeResult:
        start_time = time.time()

        try:
            recent_messages = state.get_conversation_history(max_messages=3)
            last_message = recent_messages[-1].content if recent_messages else ""

            routes_list = ', '.join(self.routes.keys())
            classification_prompt = f"""Classify the user's intent into one of these categories: {routes_list}

User message: {last_message}

Respond with ONLY the category name (one word)."""

            temp_messages = [Message(role=AgentRole.USER, content=classification_prompt)]
            intent = await self._get_llm_response(kernel, temp_messages)
            intent = intent.strip().lower()

            next_node = self.routes.get(intent, self.default_route)
            duration_ms = (time.time() - start_time) * 1000

            routing_record = {
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "message": last_message,
                "intent": intent,
                "next_node": next_node
            }
            self.routing_history.append(routing_record)

            logging.info(f"🔀 Router: '{intent}' -> '{next_node}'")

            return NodeResult(
                node_name=self.name,
                output=f"라우팅: {next_node} (인텐트: {intent})",
                next_node=next_node,
                success=True,
                duration_ms=duration_ms,
                metadata={"intent": intent, "confidence": 0.95}
            )
        except Exception as e:
            logging.error(f"❌ Router 실행 실패: {e}")
            return NodeResult(
                node_name=self.name,
                output="",
                next_node=self.default_route,
                success=False,
                error=str(e)
            )


# ============================================================================
# ProxyAgent - 사용자 명확화 요청 에이전트 (Microsoft Pattern)
# ============================================================================

class ProxyAgent(Agent):
    """
    ProxyAgent - 사용자 명확화 요청 에이전트 (Microsoft Pattern)

    작업을 진행하기 전에 사용자에게 추가 정보나 명확화를 요청합니다.

    사용 시나리오:
    - 모호한 요청의 명확화
    - 중요 결정 전 사용자 확인
    - 추가 정보 수집
    - 복잡한 옵션 중 선택 요청
    """

    def __init__(
        self,
        *args,
        clarification_callback: Optional[Callable] = None,
        max_wait_seconds: int = 300,
        auto_proceed_on_timeout: bool = False,
        **kwargs
    ):
        super().__init__(*args, **kwargs)
        self.clarification_callback = clarification_callback
        self.max_wait_seconds = max_wait_seconds
        self.auto_proceed_on_timeout = auto_proceed_on_timeout
        self.pending_clarifications: List[Dict[str, Any]] = []

    async def request_clarification(
        self,
        question: str,
        options: Optional[List[str]] = None,
        context: str = "",
        required: bool = True
    ) -> Dict[str, Any]:
        """사용자에게 명확화 요청"""
        clarification_request = {
            "id": f"clarify-{int(time.time()*1000)}",
            "question": question,
            "options": options,
            "context": context,
            "required": required,
            "status": "pending",
            "requested_at": datetime.now(timezone.utc).isoformat(),
            "response": None
        }

        self.pending_clarifications.append(clarification_request)
        return clarification_request

    async def execute(self, state: AgentState, kernel: Kernel) -> NodeResult:
        """ProxyAgent 실행"""
        start_time = time.time()

        try:
            recent_messages = state.get_conversation_history(max_messages=3)
            last_message = recent_messages[-1].content if recent_messages else ""

            analysis_prompt = f"""Analyze if the following request needs clarification.

User request: {last_message}

If clarification is needed, respond with:
{{
    "needs_clarification": true,
    "question": "the clarification question",
    "options": ["option1", "option2"] or null,
    "reason": "why clarification is needed"
}}

If no clarification is needed, respond with:
{{
    "needs_clarification": false
}}
"""
            temp_messages = [Message(role=AgentRole.USER, content=analysis_prompt)]
            response = await self._get_llm_response(kernel, temp_messages)

            try:
                json_match = re.search(r'\{[^{}]*\}', response, re.DOTALL)
                if json_match:
                    analysis = json.loads(json_match.group())
                else:
                    analysis = {"needs_clarification": False}
            except json.JSONDecodeError:
                analysis = {"needs_clarification": False}

            duration_ms = (time.time() - start_time) * 1000

            if analysis.get("needs_clarification", False):
                clarification = await self.request_clarification(
                    question=analysis.get("question", "추가 정보가 필요합니다."),
                    options=analysis.get("options"),
                    context=analysis.get("reason", "")
                )

                await self._emit_event(
                    EventType.APPROVAL_REQUESTED,
                    {"clarification": clarification}
                )

                ws_message = StreamingMessage(
                    type=WebSocketMessageType.USER_CLARIFICATION_NEEDED,
                    content=clarification["question"],
                    agent_name=self.name,
                    session_id=state.session_id,
                    metadata={"clarification_id": clarification["id"], "options": clarification["options"]}
                )

                return NodeResult(
                    node_name=self.name,
                    output=f"명확화 필요: {clarification['question']}",
                    success=True,
                    requires_approval=True,
                    approval_data=clarification,
                    duration_ms=duration_ms,
                    metadata={
                        "clarification_request": clarification,
                        "ws_message": ws_message.model_dump()
                    }
                )
            else:
                return NodeResult(
                    node_name=self.name,
                    output="명확화 불필요 - 진행합니다.",
                    success=True,
                    duration_ms=duration_ms
                )

        except Exception as e:
            logging.error(f"❌ ProxyAgent 실행 실패: {e}")
            return NodeResult(
                node_name=self.name,
                output="",
                success=False,
                error=str(e)
            )

    async def provide_response(self, clarification_id: str, response: str) -> bool:
        """명확화 응답 제공"""
        for clarification in self.pending_clarifications:
            if clarification["id"] == clarification_id:
                clarification["response"] = response
                clarification["status"] = "answered"
                clarification["answered_at"] = datetime.now(timezone.utc).isoformat()
                return True
        return False


# ============================================================================
# InvestigationPlan & SupervisorAgent
# ============================================================================

@dataclass(slots=True)
class InvestigationPlan:
    """
    Investigation Plan - 멀티 에이전트 조사 계획

    참조: amazon-bedrock-agentcore-samples/SRE-agent/supervisor.py
    """
    steps: List[str]
    agents_sequence: List[str]
    complexity: str = "simple"
    auto_execute: bool = True
    reasoning: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "steps": self.steps,
            "agents_sequence": self.agents_sequence,
            "complexity": self.complexity,
            "auto_execute": self.auto_execute,
            "reasoning": self.reasoning
        }


class SupervisorAgent(Agent):
    """
    Supervisor Agent - 여러 Agent를 감독하고 조율

    주요 기능:
    1. Investigation Plan 생성 및 실행
    2. 라운드 기반 협업 (max_rounds)
    3. 조기 종료 조건 ("TERMINATE" 키워드)
    4. 상세한 실행 로그 (execution_log)
    5. 응답 집계 (aggregate_responses)
    6. 메모리 컨텍스트 통합 (memory_hook)
    """

    def __init__(
        self,
        *args,
        sub_agents: List[Agent],
        max_rounds: int = 3,
        memory_hook: Optional[Any] = None,
        auto_approve_simple: bool = True,
        **kwargs
    ):
        super().__init__(*args, role=AgentRole.SUPERVISOR, **kwargs)
        self.sub_agents = {agent.name: agent for agent in sub_agents}
        self.max_rounds = max_rounds
        self.memory_hook = memory_hook
        self.auto_approve_simple = auto_approve_simple
        self.execution_log: List[Dict[str, Any]] = []
        self.investigation_history: List[InvestigationPlan] = []

    async def create_investigation_plan(
        self,
        state: AgentState,
        kernel: Kernel
    ) -> InvestigationPlan:
        """Investigation Plan 생성"""
        agent_names = list(self.sub_agents.keys())
        agent_descriptions = ", ".join([
            f"{name}: {agent.system_prompt[:100]}..."
            for name, agent in self.sub_agents.items()
        ])

        query = state.messages[-1].content if state.messages else ""

        planning_prompt = f"""You are a Supervisor Agent. Create an investigation plan for the following query.

Query: {query}

Available Agents: {agent_descriptions}

Respond with:
1. Steps to execute (numbered list)
2. Agent sequence (comma-separated agent names)
3. Complexity assessment (simple/complex)
4. Brief reasoning

Format your response as:
STEPS: step1, step2, step3
AGENTS: agent1, agent2
COMPLEXITY: simple
REASONING: explanation
"""
        temp_messages = [Message(role=AgentRole.USER, content=planning_prompt)]
        response = await self._get_llm_response(kernel, temp_messages)

        # 응답 파싱
        steps = []
        agents_sequence = agent_names[:2]
        complexity = "simple"
        reasoning = ""

        for line in response.split('\n'):
            line_upper = line.upper().strip()
            if line_upper.startswith('STEPS:'):
                steps = [s.strip() for s in line.split(':', 1)[1].split(',')]
            elif line_upper.startswith('AGENTS:'):
                agents_sequence = [a.strip() for a in line.split(':', 1)[1].split(',')]
            elif line_upper.startswith('COMPLEXITY:'):
                complexity = line.split(':', 1)[1].strip().lower()
            elif line_upper.startswith('REASONING:'):
                reasoning = line.split(':', 1)[1].strip()

        plan = InvestigationPlan(
            steps=steps or ["Execute query"],
            agents_sequence=[a for a in agents_sequence if a in self.sub_agents],
            complexity=complexity,
            auto_execute=self.auto_approve_simple and complexity == "simple",
            reasoning=reasoning
        )

        self.investigation_history.append(plan)
        return plan

    async def execute(self, state: AgentState, kernel: Kernel) -> NodeResult:
        """Supervisor 실행"""
        start_time = time.time()

        await self._emit_event(EventType.AGENT_STARTED, {
            "agent": self.name,
            "sub_agents": list(self.sub_agents.keys())
        })

        try:
            plan = await self.create_investigation_plan(state, kernel)
            logging.info(f"📋 Investigation Plan: {plan.to_dict()}")

            all_responses = []
            round_count = 0

            for agent_name in plan.agents_sequence:
                if round_count >= self.max_rounds:
                    break

                agent = self.sub_agents.get(agent_name)
                if not agent:
                    continue

                logging.info(f"▶️ Round {round_count + 1}: Executing {agent_name}")
                round_start = time.time()

                result = await agent.execute(state, kernel)

                self.execution_log.append({
                    "round": round_count + 1,
                    "agent": agent_name,
                    "success": result.success,
                    "output": result.output[:200],
                    "duration_ms": (time.time() - round_start) * 1000
                })

                if result.success:
                    all_responses.append(f"[{agent_name}]: {result.output}")
                    if "TERMINATE" in result.output.upper():
                        logging.info("🛑 종료 조건 감지")
                        break

                round_count += 1

            aggregated = await self._aggregate_responses(kernel, all_responses)
            state.add_message(AgentRole.ASSISTANT, aggregated, self.name)

            duration_ms = (time.time() - start_time) * 1000

            await self._emit_event(EventType.AGENT_COMPLETED, {
                "agent": self.name,
                "rounds": round_count,
                "duration_ms": duration_ms
            })

            return NodeResult(
                node_name=self.name,
                output=aggregated,
                success=True,
                duration_ms=duration_ms,
                metadata={
                    "plan": plan.to_dict(),
                    "rounds_executed": round_count,
                    "execution_log": self.execution_log[-round_count:]
                }
            )
        except Exception as e:
            logging.error(f"❌ Supervisor 실행 실패: {e}")
            return NodeResult(
                node_name=self.name,
                output="",
                success=False,
                error=str(e)
            )

    async def _aggregate_responses(
        self,
        kernel: Kernel,
        responses: List[str]
    ) -> str:
        """응답 집계"""
        if not responses:
            return "No responses collected."

        if len(responses) == 1:
            return responses[0]

        aggregation_prompt = f"""Summarize and synthesize these responses:

{chr(10).join(responses)}

Provide a coherent, comprehensive summary."""

        temp_messages = [Message(role=AgentRole.USER, content=aggregation_prompt)]
        return await self._get_llm_response(kernel, temp_messages)
