#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Unified Agent Framework - 오케스트레이션 모듈

팀 기반 멀티 에이전트 오케스트레이션 관리
Microsoft Multi-Agent-Custom-Automation-Engine 패턴 구현
"""

import re
import json
import time
import logging
from datetime import datetime, timezone
from typing import Dict, List, Optional, Any, Callable, TYPE_CHECKING

from semantic_kernel import Kernel

from .models import (
    AgentRole, AgentState, Message, PlanStep, MPlan,
    PlanStepStatus, TeamConfiguration, WebSocketMessageType, StreamingMessage
)
from .agents import Agent
from .utils import StructuredLogger, RAIValidator

if TYPE_CHECKING:
    from .framework import UnifiedAgentFramework

__all__ = [
    "OrchestrationManager",
    "AgentFactory",
]


# ============================================================================
# AgentFactory - 에이전트 생성 팩토리
# ============================================================================

class AgentFactory:
    """
    에이전트 팩토리 - 다양한 타입의 에이전트 생성

    TeamConfiguration에서 에이전트 팀을 생성합니다.
    """

    def __init__(self, framework: Optional['UnifiedAgentFramework'] = None):
        """
        팩토리 초기화

        Args:
            framework: UnifiedAgentFramework 인스턴스 (선택)
        """
        self.framework = framework
        self._logger = StructuredLogger("agent_factory")

    def create_team(self, config: TeamConfiguration) -> Dict[str, Agent]:
        """
        팀 설정에서 에이전트 딕셔너리 생성

        Args:
            config: 팀 설정

        Returns:
            에이전트 이름 -> Agent 매핑 딕셔너리
        """
        from .agents import SimpleAgent

        agents: Dict[str, Agent] = {}

        for agent_config in config.agents:
            agent = SimpleAgent(
                name=agent_config.name,
                role=agent_config.role,
                system_prompt=agent_config.system_prompt or f"You are {agent_config.name}",
                model=agent_config.model,
                temperature=agent_config.temperature
            )
            agents[agent_config.name] = agent
            self._logger.info("Agent created", name=agent_config.name, role=agent_config.role.value)

        return agents


# ============================================================================
# OrchestrationManager - 팀 기반 오케스트레이션
# ============================================================================

class OrchestrationManager:
    """
    OrchestrationManager - 팀 기반 오케스트레이션 (Microsoft Pattern)

    참조: Multi-Agent-Custom-Automation-Engine-Solution-Accelerator/orchestration/orchestration_manager.py

    멀티 에이전트 팀의 실행을 관리하고 조율합니다.
    MPlan을 사용한 구조화된 계획 실행을 지원합니다.

    사용법:
        # 오케스트레이터 생성
        orchestrator = OrchestrationManager(
            team_config=team_config,
            framework=framework,
            require_plan_approval=True
        )

        # 작업 실행
        result = await orchestrator.execute_task(
            task="연구 보고서 작성",
            session_id="session-123"
        )

        # 계획 승인 (필요한 경우)
        if orchestrator.current_plan.requires_approval:
            orchestrator.approve_plan()
            result = await orchestrator.continue_execution()

    오케스트레이션 모드:
    - supervisor: Supervisor 에이전트가 조율
    - sequential: 순차 실행
    - parallel: 병렬 실행
    - round_robin: 라운드 로빈
    """

    def __init__(
        self,
        team_config: TeamConfiguration,
        framework: Optional['UnifiedAgentFramework'] = None,
        kernel: Optional[Kernel] = None,
        require_plan_approval: bool = False,
        rai_validator: Optional[RAIValidator] = None,
        ws_callback: Optional[Callable] = None
    ):
        """
        오케스트레이션 매니저 초기화

        Args:
            team_config: 팀 설정
            framework: UnifiedAgentFramework 인스턴스
            kernel: Semantic Kernel 인스턴스
            require_plan_approval: 계획 승인 필요 여부
            rai_validator: RAI 검증기
            ws_callback: WebSocket 콜백 함수
        """
        self.team_config = team_config
        self.framework = framework
        self.kernel = kernel or (framework.kernel if framework else None)
        self.require_plan_approval = require_plan_approval or team_config.require_plan_approval
        self.rai_validator = rai_validator or RAIValidator()
        self.ws_callback = ws_callback

        self._logger = StructuredLogger("orchestration_manager")
        self._factory = AgentFactory(framework=framework)
        self._agents: Dict[str, Agent] = {}
        self.current_plan: Optional[MPlan] = None
        self.execution_history: List[Dict[str, Any]] = []

        # 에이전트 생성
        self._initialize_agents()

    def _initialize_agents(self):
        """팀 설정에서 에이전트들 초기화"""
        self._agents = self._factory.create_team(self.team_config)
        self._logger.info(
            "Agents initialized",
            count=len(self._agents),
            names=list(self._agents.keys())
        )

    async def _send_ws_message(self, message: StreamingMessage):
        """WebSocket 메시지 전송"""
        if self.ws_callback:
            try:
                await self.ws_callback(message)
            except Exception as e:
                self._logger.error("WebSocket callback failed", error=str(e))

    async def create_plan(self, task: str, state: AgentState) -> MPlan:
        """
        작업에 대한 실행 계획 생성

        Args:
            task: 실행할 작업 설명
            state: 현재 에이전트 상태

        Returns:
            생성된 MPlan
        """
        agent_descriptions = "\n".join([
            f"- {a.name}: {a.description}"
            for a in self.team_config.agents
        ])

        planning_prompt = f"""Create an execution plan for the following task.

Task: {task}

Available Agents:
{agent_descriptions}

Create a structured plan with steps and agent assignments.
Respond in JSON format:
{{
    "name": "plan name",
    "description": "plan description",
    "complexity": "simple" or "moderate" or "complex",
    "steps": [
        {{"index": 0, "description": "step description", "agent_name": "agent_name", "depends_on": []}},
        ...
    ],
    "reasoning": "why this plan"
}}
"""
        # LLM을 사용하여 계획 생성 (첫 번째 에이전트 사용)
        first_agent = list(self._agents.values())[0] if self._agents else None
        if first_agent and self.kernel:
            temp_messages = [Message(role=AgentRole.USER, content=planning_prompt)]
            response = await first_agent._get_llm_response(self.kernel, temp_messages)

            try:
                json_match = re.search(r'\{.*\}', response, re.DOTALL)
                if json_match:
                    plan_data = json.loads(json_match.group())
                else:
                    plan_data = self._create_default_plan(task)
            except json.JSONDecodeError:
                plan_data = self._create_default_plan(task)
        else:
            plan_data = self._create_default_plan(task)

        # MPlan 생성
        steps = [
            PlanStep(
                index=s.get("index", i),
                description=s.get("description", f"Step {i}"),
                agent_name=s.get("agent_name", list(self._agents.keys())[0] if self._agents else "unknown"),
                depends_on=s.get("depends_on", [])
            )
            for i, s in enumerate(plan_data.get("steps", []))
        ]

        plan = MPlan(
            name=plan_data.get("name", "execution_plan"),
            description=plan_data.get("description", task),
            steps=steps,
            complexity=plan_data.get("complexity", "simple"),
            reasoning=plan_data.get("reasoning", ""),
            requires_approval=self.require_plan_approval
        )

        self.current_plan = plan
        self._logger.info(
            "Plan created",
            name=plan.name,
            steps=len(plan.steps),
            complexity=plan.complexity
        )

        # WebSocket 알림
        await self._send_ws_message(StreamingMessage(
            type=WebSocketMessageType.PLAN_CREATED,
            content=plan.to_summary(),
            metadata={"plan": plan.model_dump()}
        ))

        return plan

    def _create_default_plan(self, task: str) -> Dict[str, Any]:
        """기본 계획 생성"""
        agent_names = list(self._agents.keys())
        return {
            "name": "default_plan",
            "description": task,
            "complexity": "simple",
            "steps": [
                {
                    "index": 0,
                    "description": task,
                    "agent_name": agent_names[0] if agent_names else "unknown",
                    "depends_on": []
                }
            ],
            "reasoning": "Default single-step plan"
        }

    def approve_plan(self, approved_by: str = "user"):
        """현재 계획 승인"""
        if self.current_plan:
            self.current_plan.approve(approved_by)
            self._logger.info("Plan approved", plan=self.current_plan.name)

    def reject_plan(self, reason: str = ""):
        """현재 계획 거부"""
        if self.current_plan:
            self.current_plan.reject(reason)
            self._logger.info("Plan rejected", plan=self.current_plan.name, reason=reason)

    async def execute_task(
        self,
        task: str,
        session_id: str,
        auto_approve: bool = False
    ) -> Dict[str, Any]:
        """
        작업 실행

        Args:
            task: 실행할 작업
            session_id: 세션 ID
            auto_approve: 자동 승인 여부

        Returns:
            실행 결과
        """
        state = AgentState(session_id=session_id, workflow_name=self.team_config.name)
        state.add_message(AgentRole.USER, task)

        # 작업 시작 알림
        await self._send_ws_message(StreamingMessage(
            type=WebSocketMessageType.START_TASK,
            content=task,
            session_id=session_id
        ))

        # 계획 생성
        plan = await self.create_plan(task, state)

        # 승인 필요 여부 확인
        if plan.requires_approval and not auto_approve:
            if not (self.team_config.auto_approve_simple and plan.complexity == "simple"):
                await self._send_ws_message(StreamingMessage(
                    type=WebSocketMessageType.PLAN_APPROVAL_REQUESTED,
                    content=plan.to_summary(),
                    session_id=session_id,
                    metadata={"plan_id": plan.id}
                ))
                return {
                    "status": "awaiting_approval",
                    "plan": plan.model_dump(),
                    "message": "계획 승인이 필요합니다."
                }

        # 계획 실행
        return await self._execute_plan(plan, state)

    async def _execute_plan(
        self,
        plan: MPlan,
        state: AgentState
    ) -> Dict[str, Any]:
        """계획 단계별 실행"""
        results = []
        plan.status = PlanStepStatus.IN_PROGRESS

        while True:
            # 다음 실행 가능한 단계 가져오기
            next_steps = plan.get_next_steps()
            if not next_steps:
                break

            for step in next_steps:
                # 단계 시작 알림
                await self._send_ws_message(StreamingMessage(
                    type=WebSocketMessageType.PLAN_STEP_STARTED,
                    content=step.description,
                    agent_name=step.agent_name,
                    step_index=step.index,
                    total_steps=len(plan.steps)
                ))

                step.status = PlanStepStatus.IN_PROGRESS

                # 에이전트 실행
                agent = self._agents.get(step.agent_name)
                if not agent:
                    step.status = PlanStepStatus.FAILED
                    step.error = f"Agent not found: {step.agent_name}"
                    continue

                try:
                    start_time = time.time()

                    # 상태에 현재 단계 정보 추가
                    state.add_message(
                        AgentRole.SYSTEM,
                        f"Execute step: {step.description}"
                    )

                    result = await agent.execute(state, self.kernel)

                    # RAI 검증
                    rai_result = self.rai_validator.validate(result.output)
                    if not rai_result.is_safe:
                        self._logger.warning(
                            "RAI validation failed",
                            agent=step.agent_name,
                            reason=rai_result.reason
                        )
                        # 안전하지 않은 출력 필터링
                        result.output = f"[콘텐츠 필터링됨: {rai_result.reason}]"

                    duration_ms = (time.time() - start_time) * 1000
                    plan.complete_step(step.index, result.output, duration_ms)

                    results.append({
                        "step_index": step.index,
                        "agent": step.agent_name,
                        "output": result.output,
                        "duration_ms": duration_ms,
                        "success": result.success
                    })

                    # 단계 완료 알림
                    await self._send_ws_message(StreamingMessage(
                        type=WebSocketMessageType.PLAN_STEP_COMPLETED,
                        content=result.output,
                        agent_name=step.agent_name,
                        step_index=step.index,
                        total_steps=len(plan.steps),
                        progress=plan.get_progress()
                    ))

                except Exception as e:
                    plan.fail_step(step.index, str(e))
                    self._logger.error(
                        "Step execution failed",
                        step=step.index,
                        agent=step.agent_name,
                        error=str(e)
                    )
                    results.append({
                        "step_index": step.index,
                        "agent": step.agent_name,
                        "error": str(e),
                        "success": False
                    })

            # 실패한 단계가 있으면 중단
            if plan.status == PlanStepStatus.FAILED:
                break

        # 완료 알림
        await self._send_ws_message(StreamingMessage(
            type=WebSocketMessageType.TASK_COMPLETE,
            content=f"작업 완료: {plan.name}",
            progress=1.0,
            metadata={
                "plan_status": plan.status.value,
                "total_steps": len(plan.steps),
                "completed_steps": len([s for s in plan.steps if s.status == PlanStepStatus.COMPLETED])
            }
        ))

        # 실행 기록 저장
        execution_record = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "plan": plan.model_dump(),
            "results": results,
            "status": plan.status.value
        }
        self.execution_history.append(execution_record)

        return {
            "status": "completed" if plan.status == PlanStepStatus.COMPLETED else "failed",
            "plan": plan.model_dump(),
            "results": results
        }

    async def continue_execution(self) -> Dict[str, Any]:
        """승인 후 계획 실행 계속"""
        if not self.current_plan:
            return {"status": "error", "message": "No plan to continue"}

        if self.current_plan.approval_status != "approved":
            return {"status": "error", "message": "Plan not approved"}

        state = AgentState(
            session_id=f"continue-{int(time.time())}",
            workflow_name=self.team_config.name
        )
        return await self._execute_plan(self.current_plan, state)
