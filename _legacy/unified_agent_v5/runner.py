#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Unified Agent Framework v5 — Runner

================================================================================
v4.1 대응: framework.py(875줄) + orchestration.py + workflow.py
          + agents.py(931줄) + events.py 통합
축소 이유: v4.1은 UnifiedAgentFramework.create() → validate() →
          create_simple_workflow() → run() 등 복잡한 설정 단계가 필요했음.
          5종 에이전트(Simple/Router/Supervisor/Proxy/Approval) +
          EventBus + Graph + OrchestrationManager를
          "run_agent() 한 줄 + Runner 클래스"로 단순화.
          "만드는 것"은 엔진(LangChain, CrewAI)에 맡기고,
          "실행하고 추적하는 것"에 집중.
================================================================================

설계 철학:
    - "만드는 것"은 LangChain 등에 맡기고
    - "실행하는 것"에 집중
    - 한 줄: result = await run_agent("질문")

사용법:
    >>> # 가장 간단한 사용
    >>> from unified_agent_v5 import run_agent
    >>> result = await run_agent("파이썬 피보나치 함수 작성해줘")
    >>> print(result.content)

    >>> # 엔진 선택
    >>> result = await run_agent("RAG 검색", engine="langchain")

    >>> # CrewAI 멀티 에이전트
    >>> result = await run_agent(
    ...     "시장 분석 보고서",
    ...     engine="crewai",
    ...     crew_agents=[
    ...         {"role": "Researcher", "goal": "데이터 수집"},
    ...         {"role": "Writer", "goal": "보고서 작성"},
    ...     ]
    ... )

    >>> # 전체 설정 커스텀
    >>> runner = Runner(config=AgentConfig(model="gpt-5.2", engine="direct"))
    >>> result = await runner.run("질문")
"""

from __future__ import annotations

import logging
import time
from typing import Any, AsyncIterator

from .types import AgentResult, StreamChunk
from .config import AgentConfig, Settings
from .memory import Memory
from .tools import Tool
from .callback import CallbackHandler, CompositeCallbackHandler
from .engines.base import get_engine

__all__ = ["run_agent", "stream_agent", "Runner"]

logger = logging.getLogger(__name__)


# ============================================================================
# run_agent() — 최상위 함수 (가장 쉬운 진입점)
# ============================================================================

async def run_agent(
    task: str,
    *,
    # 엔진 선택
    engine: str = "",
    model: str = "",
    # 시스템 프롬프트
    system_prompt: str = "You are a helpful assistant.",
    # 도구
    tools: list[Tool] | None = None,
    # 대화 히스토리 (이전 대화 이어가기)
    memory: Memory | None = None,
    # 모니터링 콜백
    callbacks: list[CallbackHandler] | None = None,
    # LLM 파라미터
    temperature: float | None = None,
    max_tokens: int | None = None,
    # 스트리밍 (False면 동기, True면 stream_agent 사용)
    stream: bool = False,
    # 추가 설정 (엔진별 파라미터)
    **kwargs: Any,
) -> AgentResult:
    """
    에이전트 실행 — 한 줄로 끝내는 핵심 함수

    v4.1 대응: UnifiedAgentFramework.create() + validate() +
              create_simple_workflow() + run() → 한 줄로 단순화.
    축소 이유: 복잡한 설정 단계 없이 즉시 실행 가능하도록.
              AgentConfig 자동 생성 + Runner 자동 생성.

    Args:
        task: 실행할 태스크/질문
        engine: 엔진 선택 ("direct", "langchain", "crewai")
        model: LLM 모델 (미지정 시 Settings.DEFAULT_MODEL)
        system_prompt: 시스템 프롬프트
        tools: 사용할 도구 리스트
        memory: 대화 히스토리 (이전 대화 이어가기)
        callbacks: 모니터링 콜백 (OTEL, Logging 등)
        temperature: LLM temperature
        max_tokens: 최대 토큰 수
        stream: 스트리밍 모드 (True이면 stream_agent 권장)
        **kwargs: 엔진별 추가 파라미터

    Returns:
        AgentResult: 실행 결과 (content, usage, tool_calls 등)

    사용 예시:
        >>> result = await run_agent("안녕하세요!")
        >>> print(result.content)
        '안녕하세요! 무엇을 도와드릴까요?'

        >>> print(result.usage)
        {'input_tokens': 12, 'output_tokens': 15, 'total_tokens': 27}
    """
    config = AgentConfig(
        model=model or Settings.DEFAULT_MODEL,
        engine=engine or Settings.DEFAULT_ENGINE,
        system_prompt=system_prompt,
        temperature=temperature,
        max_tokens=max_tokens,
        stream=stream,
        callbacks=callbacks or [],
    )

    runner = Runner(config=config)
    return await runner.run(task, tools=tools, memory=memory, **kwargs)


async def stream_agent(
    task: str,
    *,
    engine: str = "",
    model: str = "",
    system_prompt: str = "You are a helpful assistant.",
    tools: list[Tool] | None = None,
    memory: Memory | None = None,
    callbacks: list[CallbackHandler] | None = None,
    temperature: float | None = None,
    max_tokens: int | None = None,
    **kwargs: Any,
) -> AsyncIterator[StreamChunk]:
    """
    에이전트 스트리밍 실행

    사용법:
        >>> async for chunk in stream_agent("긴 답변이 필요한 질문"):
        ...     print(chunk.content, end="", flush=True)
    """
    config = AgentConfig(
        model=model or Settings.DEFAULT_MODEL,
        engine=engine or Settings.DEFAULT_ENGINE,
        system_prompt=system_prompt,
        temperature=temperature,
        max_tokens=max_tokens,
        callbacks=callbacks or [],
    )
    runner = Runner(config=config)
    async for chunk in runner.stream(task, tools=tools, memory=memory, **kwargs):
        yield chunk


# ============================================================================
# Runner — 에이전트 실행 관리자
# ============================================================================

class Runner:
    """
    에이전트 Runner — 실행과 추적에 집중

    v4.1 대응: UnifiedAgentFramework + WorkflowManager + OrchestrationManager
              + 5종 Agent 클래스 통합
    축소 이유: v4.1의 "프레임워크 객체 생성 → 워크플로 정의 → 그래프 연결
              → 오케스트레이션 → 실행" 단계를
              "Runner.run(task) 한 줄"로 단순화.
              "만드는 것"은 엔진(LangChain, CrewAI)에 맡기고,
              "만들어진 에이전트를 쉽게 실행하고, 결과를 추적하는" 역할.

    사용법:
        >>> runner = Runner()
        >>> result = await runner.run("질문")
        >>>
        >>> # 설정 커스텀
        >>> runner = Runner(config=AgentConfig(
        ...     model="gpt-5.2",
        ...     engine="langchain",
        ...     system_prompt="You are a Python expert."
        ... ))
        >>> result = await runner.run("데코레이터 설명해줘")
        >>>
        >>> # 대화 이어가기
        >>> memory = Memory(system_prompt="You are helpful.")
        >>> r1 = await runner.run("내 이름은 철수야", memory=memory)
        >>> r2 = await runner.run("내 이름이 뭐였지?", memory=memory)
    """

    def __init__(self, config: AgentConfig | None = None):
        self.config = config or AgentConfig()

    def _build_engine_kwargs(self, **extra: Any) -> dict[str, Any]:
        """엔진에 전달할 공통 kwargs 구성 (run/stream 공유)"""
        return {
            "temperature": self.config.temperature,
            "max_tokens": self.config.max_tokens,
            "max_tool_rounds": self.config.max_tool_rounds,
            "azure_endpoint": self.config.azure_endpoint,
            "azure_api_key": self.config.azure_api_key,
            "azure_deployment": self.config.azure_deployment,
            "azure_api_version": self.config.azure_api_version,
            "openai_api_key": self.config.openai_api_key,
            "openai_base_url": self.config.openai_base_url,
            **extra,
        }

    async def run(
        self,
        task: str,
        *,
        tools: list[Tool] | None = None,
        memory: Memory | None = None,
        callbacks: list[CallbackHandler] | None = None,
        **kwargs: Any,
    ) -> AgentResult:
        """
        태스크 실행 — Runner 핵심 로직

        v4.1의 복잡한 5단계(config → validate → workflow → graph → run)를
        단순한 순서로 대체:
            1. 메모리에서 메시지 히스토리 구성
            2. 콜백에 시작 알림
            3. 엔진으로 LLM 호출
            4. 결과를 메모리에 저장
            5. 콜백에 완료 알림
        """
        start_time = time.time()

        # 1. 메모리 구성
        if memory is None:
            memory = Memory(system_prompt=self.config.system_prompt)

        memory.add_user(task)
        messages = memory.get_messages()

        # 2. 콜백 구성
        all_callbacks = list(self.config.callbacks) + (callbacks or [])
        composite = CompositeCallbackHandler(all_callbacks) if all_callbacks else None

        if composite:
            await composite.on_agent_start(task, self.config)

        # 3. 엔진 실행
        try:
            engine = get_engine(self.config.engine)
            engine_kwargs = self._build_engine_kwargs(**kwargs)

            result = await engine.run(
                messages=messages,
                model=self.config.model,
                tools=tools,
                callbacks=all_callbacks,
                **engine_kwargs,
            )

            # 4. 결과를 메모리에 저장
            memory.add_assistant(result.content)

            # 실행 시간 업데이트
            result.duration_ms = (time.time() - start_time) * 1000

            # 5. 콜백 완료
            if composite:
                await composite.on_agent_end(result)

            return result

        except Exception as e:
            if composite:
                await composite.on_agent_error(e)
            raise

    async def stream(
        self,
        task: str,
        *,
        tools: list[Tool] | None = None,
        memory: Memory | None = None,
        callbacks: list[CallbackHandler] | None = None,
        **kwargs: Any,
    ) -> AsyncIterator[StreamChunk]:
        """스트리밍 실행 (콜백 지원)"""
        start_time = time.time()

        if memory is None:
            memory = Memory(system_prompt=self.config.system_prompt)

        memory.add_user(task)
        messages = memory.get_messages()

        # 콜백 구성
        all_callbacks = list(self.config.callbacks) + (callbacks or [])
        composite = CompositeCallbackHandler(all_callbacks) if all_callbacks else None

        if composite:
            await composite.on_agent_start(task, self.config)

        engine = get_engine(self.config.engine)
        engine_kwargs = self._build_engine_kwargs(**kwargs)

        try:
            collected_content: list[str] = []
            async for chunk in engine.stream(
                messages=messages,
                model=self.config.model,
                tools=tools,
                callbacks=all_callbacks,
                **engine_kwargs,
            ):
                collected_content.append(chunk.content)
                yield chunk

            # 스트리밍 완료 후 메모리에 저장
            full_content = "".join(collected_content)
            if full_content:
                memory.add_assistant(full_content)

            # 콜백 완료
            if composite:
                duration_ms = (time.time() - start_time) * 1000
                result = AgentResult(
                    content=full_content,
                    model=self.config.model,
                    engine=self.config.engine,
                    duration_ms=duration_ms,
                )
                await composite.on_agent_end(result)

        except Exception as e:
            if composite:
                await composite.on_agent_error(e)
            raise
