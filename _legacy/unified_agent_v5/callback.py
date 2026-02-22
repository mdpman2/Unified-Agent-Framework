#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Unified Agent Framework v5 — Callback / Observability

================================================================================
v4.1 대응: tracer.py(851줄) + observability.py(721줄) + hooks.py
          + reward.py + agent_store.py + events.py 통합
축소 이유: v4.1은 자체 Tracer/Dashboard/MetricsCollector/AlertManager/
          RewardTracker/EventBus를 구현했으나, 이 자체 도구들이
          프레임워크보다 무거워지는 문제 발생.
          전문 도구(Azure Monitor, Datadog, Jaeger)가 이미 존재하므로,
          프레임워크는 "Export만 하는 어댑터"를 제공하는 것이 올바른 역할.
          OTEL(OpenTelemetry) 표준을 채택하여 하나의 CallbackHandler로
          어떤 관찰성 도구에든 연결 가능.
================================================================================

설계 원칙:
    - CallbackHandler 하나만 잘 만들어두면 어디든 연결 가능
    - 프레임워크 자체는 데이터를 저장하지 않음 (Export only)
    - OpenTelemetry 표준 준수
    - 커스텀 핸들러로 Arize/WandB/LangSmith 등 연동 가능

사용법:
    >>> from unified_agent_v5 import OTelCallbackHandler, run_agent
    >>> cb = OTelCallbackHandler(endpoint="http://localhost:4318")
    >>> result = await run_agent("질문", callbacks=[cb])
"""

from __future__ import annotations

import asyncio
import logging
from typing import Any

from .types import AgentResult, ToolCall, ToolResult

__all__ = [
    "CallbackHandler",
    "OTelCallbackHandler",
    "CompositeCallbackHandler",
    "LoggingCallbackHandler",
    "fire_callbacks",
]

logger = logging.getLogger(__name__)

# 로그 태스크 문자열 최대 출력 길이
_MAX_LOG_TASK_LEN = 80


# ============================================================================
# fire_callbacks — 엔진에서 공통 사용하는 콜백 유틸리티
# ============================================================================

async def fire_callbacks(
    callbacks: list[CallbackHandler],
    method: str,
    *args: Any,
    **kwargs: Any,
) -> None:
    """콜백 리스트에 안전하게 이벤트 발송 (에러 격리, 순차 실행)

    엔진 코드에서 반복되는 ``for cb in callbacks`` 패턴을 대체.
    각 핸들러의 예외는 로깅만 하고 전파하지 않습니다.

    Note:
        순차 실행으로 콜백 간 실행 순서를 보장합니다.
        동시 실행이 필요하면 CompositeCallbackHandler를 사용하세요.

    사용법:
        >>> await fire_callbacks(callbacks, "on_llm_start", model, messages)
    """
    for cb in callbacks:
        try:
            await getattr(cb, method)(*args, **kwargs)
        except Exception as e:
            logger.warning(f"Callback {type(cb).__name__}.{method} error: {e}")


# ============================================================================
# CallbackHandler — 인터페이스 (기본 구현 제공)
# ============================================================================

class CallbackHandler:
    """
    콜백 핸들러 인터페이스 (ABC)

    v4.1 대응: Tracer + ObservabilityPipeline + EventBus + LifecycleHooks 통합
    축소 이유: v4.1의 복잡한 라이프사이클 훅을 7개 비동기 콜백으로 단순화.
              모든 이벤트를 이 인터페이스로 받아서
              원하는 관찰성 도구(OTEL, LangSmith, Datadog 등)로 Export.

    구현 예시:
        class MyDatadogHandler(CallbackHandler):
            async def on_llm_start(self, model, messages, **kwargs):
                datadog.start_span("llm.call", {"model": model})
            async def on_llm_end(self, result, **kwargs):
                datadog.finish_span({"tokens": result.usage})
    """

    async def on_agent_start(self, task: str, config: Any = None, **kwargs) -> None:
        """에이전트 실행 시작"""
        pass

    async def on_agent_end(self, result: AgentResult, **kwargs) -> None:
        """에이전트 실행 완료"""
        pass

    async def on_agent_error(self, error: Exception, **kwargs) -> None:
        """에이전트 실행 오류"""
        pass

    async def on_llm_start(self, model: str, messages: list[dict], **kwargs) -> None:
        """LLM 호출 시작"""
        pass

    async def on_llm_end(self, content: str, usage: dict[str, int] | None = None, **kwargs) -> None:
        """LLM 호출 완료"""
        pass

    async def on_tool_start(self, tool_call: ToolCall, **kwargs) -> None:
        """도구 호출 시작"""
        pass

    async def on_tool_end(self, tool_result: ToolResult, **kwargs) -> None:
        """도구 호출 완료"""
        pass


# ============================================================================
# LoggingCallbackHandler — 기본 로깅
# ============================================================================

class LoggingCallbackHandler(CallbackHandler):
    """
    Python logging 기반 콜백 핸들러 (디버깅용)

    v4.1 대응: DebugTracer + ConsoleLogger 통합
    축소 이유: 디버깅은 표준 logging으로 충분. 별도 클래스 불필요.

    사용법:
        >>> cb = LoggingCallbackHandler(level=logging.DEBUG)
        >>> result = await run_agent("질문", callbacks=[cb])
    """

    def __init__(self, level: int = logging.INFO):
        self.level = level
        self._logger = logging.getLogger("unified_agent_v5.callback")

    async def on_agent_start(self, task: str, config: Any = None, **kwargs) -> None:
        display = task[:_MAX_LOG_TASK_LEN] + "..." if len(task) > _MAX_LOG_TASK_LEN else task
        self._logger.log(self.level, f"[Agent Start] task={display}")

    async def on_agent_end(self, result: AgentResult, **kwargs) -> None:
        self._logger.log(
            self.level,
            f"[Agent End] engine={result.engine} model={result.model} "
            f"tokens={result.usage.get('total_tokens', 0)} "
            f"duration={result.duration_ms:.0f}ms"
        )

    async def on_agent_error(self, error: Exception, **kwargs) -> None:
        self._logger.error(f"[Agent Error] {type(error).__name__}: {error}")

    async def on_llm_start(self, model: str, messages: list[dict], **kwargs) -> None:
        self._logger.log(self.level, f"[LLM Start] model={model} messages={len(messages)}")

    async def on_llm_end(self, content: str, usage: dict[str, int] | None = None, **kwargs) -> None:
        tokens = usage.get("total_tokens", "?") if usage else "?"
        self._logger.log(self.level, f"[LLM End] tokens={tokens} content_len={len(content)}")

    async def on_tool_start(self, tool_call: ToolCall, **kwargs) -> None:
        self._logger.log(self.level, f"[Tool Start] {tool_call.name}({tool_call.arguments})")

    async def on_tool_end(self, tool_result: ToolResult, **kwargs) -> None:
        status = "ERROR" if tool_result.is_error else "OK"
        self._logger.log(self.level, f"[Tool End] {tool_result.name} -> {status}")


# ============================================================================
# OTelCallbackHandler — OpenTelemetry 표준 Export
# ============================================================================

class OTelCallbackHandler(CallbackHandler):
    """
    OpenTelemetry 표준 콜백 핸들러

    v4.1 대응: Tracer(851줄) + ObservabilityPipeline(721줄) 대체
    축소 이유: v4.1의 자체 Span/Trace/Dashboard/Alert/Metrics 구현을
              OTEL 표준 SDK로 완전 대체. BatchSpanProcessor + OTLP Export로
              Azure Monitor, Datadog, Jaeger 등 전문 도구에 위임.
              프레임워크는 데이터를 저장/시각화하지 않음.

    사용법:
        >>> cb = OTelCallbackHandler(
        ...     service_name="my-agent",
        ...     endpoint="http://localhost:4318"  # OTLP endpoint
        ... )
        >>> result = await run_agent("질문", callbacks=[cb])

    연동 가능 대상:
        - Azure Monitor / Application Insights (OTEL Collector 경유)
        - Datadog (OTEL Collector 또는 Datadog Agent)
        - Jaeger / Zipkin / Grafana Tempo (OTLP 직접)
        - LangSmith (OTEL Export 지원 시)
        - Arize (OTEL Export 지원 시)
    """

    def __init__(
        self,
        service_name: str = "unified-agent-v5",
        endpoint: str = "",
    ):
        self.service_name = service_name
        self.endpoint = endpoint
        self._tracer = None
        self._spans: dict[str, Any] = {}
        self._llm_span_stack: list[Any] = []  # LIFO stack for nested/concurrent LLM spans
        self._agent_span_stack: list[Any] = []  # LIFO stack for nested agent spans
        self._initialized = False

    def _ensure_init(self) -> None:
        """OTEL 초기화 (lazy)"""
        if self._initialized:
            return
        try:
            from opentelemetry import trace
            from opentelemetry.sdk.trace import TracerProvider
            from opentelemetry.sdk.resources import Resource

            resource = Resource.create({"service.name": self.service_name})
            provider = TracerProvider(resource=resource)

            if self.endpoint:
                from opentelemetry.sdk.trace.export import BatchSpanProcessor
                from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter
                exporter = OTLPSpanExporter(endpoint=self.endpoint)
                provider.add_span_processor(BatchSpanProcessor(exporter))

            trace.set_tracer_provider(provider)
            self._tracer = trace.get_tracer("unified-agent-v5")
            self._initialized = True
        except ImportError:
            logger.warning(
                "OpenTelemetry not installed. "
                "Install with: pip install opentelemetry-api opentelemetry-sdk "
                "opentelemetry-exporter-otlp"
            )
            self._initialized = True  # Don't retry

    async def on_agent_start(self, task: str, config: Any = None, **kwargs) -> None:
        self._ensure_init()
        if not self._tracer:
            return
        span = self._tracer.start_span("agent.run")
        span.set_attribute("agent.task", task[:200])
        span.set_attribute("agent.engine", getattr(config, "engine", "unknown"))
        span.set_attribute("agent.model", getattr(config, "model", "unknown"))
        self._agent_span_stack.append(span)

    async def on_agent_end(self, result: AgentResult, **kwargs) -> None:
        span = self._agent_span_stack.pop() if self._agent_span_stack else None
        if span:
            span.set_attribute("agent.tokens.total", result.usage.get("total_tokens", 0))
            span.set_attribute("agent.duration_ms", result.duration_ms)
            span.set_attribute("agent.engine", result.engine)
            span.end()

    async def on_agent_error(self, error: Exception, **kwargs) -> None:
        span = self._agent_span_stack.pop() if self._agent_span_stack else None
        if span:
            span.set_attribute("error", True)
            span.set_attribute("error.message", str(error))
            span.end()

    async def on_llm_start(self, model: str, messages: list[dict], **kwargs) -> None:
        self._ensure_init()
        if not self._tracer:
            return
        span = self._tracer.start_span("llm.call")
        span.set_attribute("llm.model", model)
        span.set_attribute("llm.messages_count", len(messages))
        self._llm_span_stack.append(span)

    async def on_llm_end(self, content: str, usage: dict[str, int] | None = None, **kwargs) -> None:
        span = self._llm_span_stack.pop() if self._llm_span_stack else None
        if span:
            if usage:
                span.set_attribute("llm.tokens.input", usage.get("input_tokens", 0))
                span.set_attribute("llm.tokens.output", usage.get("output_tokens", 0))
                span.set_attribute("llm.tokens.total", usage.get("total_tokens", 0))
            span.end()

    async def on_tool_start(self, tool_call: ToolCall, **kwargs) -> None:
        self._ensure_init()
        if not self._tracer:
            return
        span = self._tracer.start_span(f"tool.{tool_call.name}")
        span.set_attribute("tool.name", tool_call.name)
        self._spans[f"tool.{tool_call.id}"] = span

    async def on_tool_end(self, tool_result: ToolResult, **kwargs) -> None:
        span = self._spans.pop(f"tool.{tool_result.tool_call_id}", None)
        if span:
            span.set_attribute("tool.is_error", tool_result.is_error)
            span.end()


# ============================================================================
# CompositeCallbackHandler — 여러 핸들러 통합
# ============================================================================

class CompositeCallbackHandler(CallbackHandler):
    """
    여러 CallbackHandler를 동시에 실행

    v4.1 대응: EventBus(다대다 pub/sub) → 단순 리스트 순회로 대체
    축소 이유: EventBus의 토픽/구독/필터 기능은 실무 미사용.
              handlers 리스트를 순회하며 호출하는 것으로 충분.

    사용법:
        >>> composite = CompositeCallbackHandler([
        ...     LoggingCallbackHandler(),
        ...     OTelCallbackHandler(endpoint="http://localhost:4318"),
        ... ])
        >>> result = await run_agent("질문", callbacks=[composite])
    """

    def __init__(self, handlers: list[CallbackHandler] | None = None):
        self.handlers = handlers or []

    def add(self, handler: CallbackHandler) -> None:
        self.handlers.append(handler)

    async def _dispatch(self, method: str, *args: Any, **kwargs: Any) -> None:
        """모든 핸들러에 이벤트 동시 발송 (독립 핸들러 → asyncio.gather)"""
        if not self.handlers:
            return

        async def _safe_call(h: CallbackHandler) -> None:
            try:
                await getattr(h, method)(*args, **kwargs)
            except Exception as e:
                logger.warning(f"Callback error in {type(h).__name__}.{method}: {e}")

        await asyncio.gather(*(_safe_call(h) for h in self.handlers))

    async def on_agent_start(self, task: str, config: Any = None, **kwargs) -> None:
        await self._dispatch("on_agent_start", task, config, **kwargs)

    async def on_agent_end(self, result: AgentResult, **kwargs) -> None:
        await self._dispatch("on_agent_end", result, **kwargs)

    async def on_agent_error(self, error: Exception, **kwargs) -> None:
        await self._dispatch("on_agent_error", error, **kwargs)

    async def on_llm_start(self, model: str, messages: list[dict], **kwargs) -> None:
        await self._dispatch("on_llm_start", model, messages, **kwargs)

    async def on_llm_end(self, content: str, usage: dict[str, int] | None = None, **kwargs) -> None:
        await self._dispatch("on_llm_end", content, usage, **kwargs)

    async def on_tool_start(self, tool_call: ToolCall, **kwargs) -> None:
        await self._dispatch("on_tool_start", tool_call, **kwargs)

    async def on_tool_end(self, tool_result: ToolResult, **kwargs) -> None:
        await self._dispatch("on_tool_end", tool_result, **kwargs)
