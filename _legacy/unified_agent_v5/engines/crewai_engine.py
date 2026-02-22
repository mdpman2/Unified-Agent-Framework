#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Unified Agent Framework v5 — CrewAI Engine

================================================================================
v4.1 대응: crewai_bridge.py + ag2_bridge.py + agents.py(SupervisorAgent) 대체
축소 이유: 멀티 에이전트 협업에서 AutoGen(AG2)보다 CrewAI가 API 안정성 높음.
          AutoGen은 v0.2→v0.4→AgentOS로 API가 너무 자주 변경됨.
          v4.1의 5종 에이전트(Simple/Router/Supervisor/Proxy/Approval)를
          CrewAI의 역할 기반 협업으로 대체.
================================================================================

CrewAI 기반 멀티 에이전트 협업 엔진.
설치: pip install crewai

사용법:
    >>> result = await run_agent(
    ...     "시장 분석 보고서를 작성해줘",
    ...     engine="crewai",
    ...     crew_agents=[
    ...         {"role": "Researcher", "goal": "데이터 수집"},
    ...         {"role": "Analyst", "goal": "데이터 분석"},
    ...         {"role": "Writer", "goal": "보고서 작성"},
    ...     ]
    ... )
"""

from __future__ import annotations

import asyncio
import logging
import time
from typing import Any, AsyncIterator

from ..types import AgentResult, StreamChunk
from ..tools import Tool
from ..callback import CallbackHandler, fire_callbacks

__all__ = ["CrewAIEngine"]

logger = logging.getLogger(__name__)

try:
    from crewai import Agent as CrewAgent, Task as CrewTask, Crew, Process
    HAS_CREWAI = True
except ImportError:
    HAS_CREWAI = False


class CrewAIEngine:
    """
    CrewAI Engine — 멀티 에이전트 협업

    v4.1 대응: crewai_bridge + ag2_bridge + agents.py(SupervisorAgent) 통합
    축소 이유: AutoGen 대신 CrewAI 채택 (API 안정성).
              역할 기반 협업(Researcher, Analyst, Writer 등)이
              복잡한 작업을 직관적으로 수행.
    """

    def __init__(self, **kwargs):
        if not HAS_CREWAI:
            raise ImportError(
                "CrewAI is not installed. "
                "Install with: pip install crewai"
            )

    async def run(
        self,
        messages: list[dict[str, Any]],
        model: str,
        tools: list[Tool] | None = None,
        callbacks: list[CallbackHandler] | None = None,
        **kwargs: Any,
    ) -> AgentResult:
        """
        CrewAI 실행

        kwargs로 전달 가능한 추가 파라미터:
            crew_agents: list[dict] — 에이전트 정의
                [{"role": "Researcher", "goal": "...", "backstory": "..."}]
            crew_tasks: list[dict] — 태스크 정의
                [{"description": "...", "agent_role": "Researcher"}]
            process: str — "sequential" 또는 "hierarchical"
        """
        start_time = time.time()
        callbacks = callbacks or []

        if tools:
            logger.warning(
                "CrewAI engine received %d tool(s) but CrewAI manages tools internally. "
                "Use crew_agents[].tools to assign tools to specific agents.",
                len(tools),
            )

        # 사용자 메시지에서 태스크 추출
        user_task = ""
        for msg in reversed(messages):
            if msg.get("role") == "user":
                user_task = msg.get("content", "")
                break

        await fire_callbacks(callbacks, "on_llm_start", model, messages)

        # 에이전트 정의
        crew_agent_defs = kwargs.get("crew_agents", [
            {
                "role": "Default Agent",
                "goal": "Complete the given task",
                "backstory": "You are a helpful AI assistant.",
            }
        ])

        agents = []
        for agent_def in crew_agent_defs:
            agent = CrewAgent(
                role=agent_def.get("role", "Assistant"),
                goal=agent_def.get("goal", "Complete the task"),
                backstory=agent_def.get("backstory", "You are helpful."),
                llm=model,
                verbose=agent_def.get("verbose", False),
            )
            agents.append(agent)

        # 태스크 정의
        crew_task_defs = kwargs.get("crew_tasks", [])
        if not crew_task_defs:
            # 기본: 첫 번째 에이전트에게 전체 태스크 할당
            tasks = [
                CrewTask(
                    description=user_task,
                    expected_output="Detailed result",
                    agent=agents[0],
                )
            ]
        else:
            tasks = []
            agent_map = {a.role: a for a in agents}
            for task_def in crew_task_defs:
                agent_role = task_def.get("agent_role", agents[0].role)
                tasks.append(
                    CrewTask(
                        description=task_def.get("description", user_task),
                        expected_output=task_def.get("expected_output", "Detailed result"),
                        agent=agent_map.get(agent_role, agents[0]),
                    )
                )

        # Crew 구성 및 실행
        process_type = kwargs.get("process", "sequential")
        crew = Crew(
            agents=agents,
            tasks=tasks,
            process=Process.hierarchical if process_type == "hierarchical" else Process.sequential,
            verbose=kwargs.get("verbose", False),
        )

        # CrewAI는 동기 실행이므로 executor 사용
        loop = asyncio.get_running_loop()
        try:
            result = await loop.run_in_executor(None, crew.kickoff)
        except Exception as e:
            logger.error("CrewAI kickoff failed: %s", e)
            await fire_callbacks(callbacks, "on_agent_error", e)
            return AgentResult(
                content=f"CrewAI execution failed: {e}",
                model=model,
                engine="crewai",
                duration_ms=(time.time() - start_time) * 1000,
            )

        content = str(result)
        duration_ms = (time.time() - start_time) * 1000

        await fire_callbacks(callbacks, "on_llm_end", content, {})

        return AgentResult(
            content=content,
            model=model,
            engine="crewai",
            duration_ms=duration_ms,
            metadata={
                "agents": [a.role for a in agents],
                "tasks": len(tasks),
                "process": process_type,
            },
        )

    async def stream(
        self,
        messages: list[dict[str, Any]],
        model: str,
        tools: list[Tool] | None = None,
        callbacks: list[CallbackHandler] | None = None,
        **kwargs: Any,
    ) -> AsyncIterator[StreamChunk]:
        """CrewAI는 네이티브 스트리밍을 지원하지 않으므로 결과를 한번에 반환"""
        result = await self.run(messages, model, tools, callbacks, **kwargs)
        yield StreamChunk(content=result.content, is_final=True)
