#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Unified Agent Framework v6 — Observability (OpenTelemetry)

================================================================================
Microsoft Agent Framework 1.0.0-rc1 호환 관측성 모듈

OpenTelemetry 기반 분산 추적을 제공합니다.
GenAI Semantic Conventions를 준수하며,
콘솔 출력 · OTLP 내보내기 · Azure Monitor를 지원합니다.

v5에는 없던 새 기능:
    - OpenTelemetry 네이티브 연동 (자체 Tracer 제거)
    - Azure Monitor 직접 연동
    - GenAI Semantic Conventions 준수
    - TracerProvider 자동 설정

핵심 함수:
    - configure_tracing() : OTEL TracerProvider + Exporter 설정
    - get_tracer()        : 현재 설정된 트레이서 인스턴스 반환

사용법:
    >>> from unified_agent_v6 import configure_tracing
    >>> configure_tracing(service_name="my-agent", enable_console=True)
================================================================================
"""

from __future__ import annotations

import logging
from typing import Any

__all__ = [
    "configure_tracing",
    "get_tracer",
]

logger = logging.getLogger("agent_framework")

_tracer = None


def configure_tracing(
    *,
    service_name: str = "agent-framework",
    enable_console: bool = False,
    otlp_endpoint: str | None = None,
    azure_monitor_connection_string: str | None = None,
) -> Any:
    """
    OpenTelemetry 트레이싱 설정.

    TracerProvider를 생성하고 지정된 Exporter를 등록합니다.
    여러 Exporter를 동시에 등록할 수 있습니다.

    Args:
        service_name: OTel resource의 service.name (기본: "agent-framework")
        enable_console: 콘솔 출력 활성화
        otlp_endpoint: OTLP gRPC 내보내기 엔드포인트
        azure_monitor_connection_string: Azure Monitor 연결 문자열

    Returns:
        설정된 Tracer 인스턴스 (또는 opentelemetry 미설치 시 None)

    사용법:
        >>> # 콘솔 출력
        >>> configure_tracing(enable_console=True)
        >>>
        >>> # OTLP 내보내기
        >>> configure_tracing(otlp_endpoint="http://localhost:4317")
        >>>
        >>> # Azure Monitor
        >>> configure_tracing(azure_monitor_connection_string="InstrumentationKey=...")
    """
    global _tracer

    try:
        from opentelemetry import trace
        from opentelemetry.sdk.trace import TracerProvider
        from opentelemetry.sdk.resources import Resource

        resource = Resource.create({"service.name": service_name})
        provider = TracerProvider(resource=resource)

        if enable_console:
            from opentelemetry.sdk.trace.export import (
                SimpleSpanProcessor,
                ConsoleSpanExporter,
            )
            provider.add_span_processor(SimpleSpanProcessor(ConsoleSpanExporter()))

        if otlp_endpoint:
            try:
                from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import OTLPSpanExporter
                from opentelemetry.sdk.trace.export import BatchSpanProcessor
                exporter = OTLPSpanExporter(endpoint=otlp_endpoint)
                provider.add_span_processor(BatchSpanProcessor(exporter))
            except ImportError:
                logger.warning("opentelemetry-exporter-otlp not installed. OTLP tracing disabled.")

        if azure_monitor_connection_string:
            try:
                from azure.monitor.opentelemetry.exporter import AzureMonitorTraceExporter
                from opentelemetry.sdk.trace.export import BatchSpanProcessor
                exporter = AzureMonitorTraceExporter(
                    connection_string=azure_monitor_connection_string
                )
                provider.add_span_processor(BatchSpanProcessor(exporter))
            except ImportError:
                logger.warning("azure-monitor-opentelemetry-exporter not installed.")

        trace.set_tracer_provider(provider)
        _tracer = provider.get_tracer("agent_framework")
        logger.info("OpenTelemetry tracing configured: %s", service_name)
        return _tracer

    except ImportError:
        logger.info("opentelemetry not installed. Tracing disabled.")
        return None


def get_tracer() -> Any:
    """현재 설정된 트레이서 반환."""
    global _tracer
    if _tracer is None:
        try:
            from opentelemetry import trace
            _tracer = trace.get_tracer("agent_framework")
        except ImportError:
            pass
    return _tracer
