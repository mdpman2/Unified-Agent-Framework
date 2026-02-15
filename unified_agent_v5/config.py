#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Unified Agent Framework v5 — Configuration

================================================================================
v4.1 대응: config.py (668줄) → 160줄로 축소 (76% ↓)
축소 이유: v4.1의 54개 모델 목록, 8개 클래스(FrameworkConfig, ModelConfig,
          ObservabilityConfig, SecurityConfig, RetryConfig, CacheConfig,
          BridgeConfig, ExtensionsConfig), 20+ 검증 로직을
          실무 필수 항목만으로 축소.
          환경변수 자동 로드 + __post_init__ 폴백으로 설정 작업 최소화.
================================================================================

v5 설정 구조:
    - Settings: 전역 설정 (클래스 변수, 환경변수 자동 로드)
    - AgentConfig: 에이전트 인스턴스별 설정 (dataclass)
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Any

from dotenv import load_dotenv

load_dotenv()

__all__ = ["Settings", "AgentConfig"]


class Settings:
    """
    전역 설정 (클래스 변수로 관리)

    v4.1 대응: FrameworkConfig + ObservabilityConfig + CacheConfig 등 8개 클래스 통합
    축소 이유: 실무에서 변경하는 설정은 model/engine/api_key 정도.
              나머지는 환경변수 자동 로드로 충분.

    사용법:
        >>> Settings.DEFAULT_MODEL = "gpt-5.2"
        >>> Settings.DEFAULT_ENGINE = "direct"
    """

    # 기본 모델 / 엔진
    DEFAULT_MODEL: str = os.getenv("AGENT_MODEL", "gpt-5.2")
    DEFAULT_ENGINE: str = os.getenv("AGENT_ENGINE", "direct")

    # Azure OpenAI 설정
    AZURE_ENDPOINT: str = os.getenv("AZURE_OPENAI_ENDPOINT", "")
    AZURE_API_KEY: str = os.getenv("AZURE_OPENAI_API_KEY", "")
    AZURE_DEPLOYMENT: str = os.getenv("AZURE_OPENAI_DEPLOYMENT", "")
    AZURE_API_VERSION: str = os.getenv("AZURE_API_VERSION", "2025-12-01-preview")

    # OpenAI 직접 호출 설정
    OPENAI_API_KEY: str = os.getenv("OPENAI_API_KEY", "")
    OPENAI_BASE_URL: str = os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1")

    # LLM 파라미터
    DEFAULT_TEMPERATURE: float = float(os.getenv("AGENT_TEMPERATURE", "0.7"))
    DEFAULT_MAX_TOKENS: int = int(os.getenv("AGENT_MAX_TOKENS", "4096"))

    # 스트리밍
    ENABLE_STREAMING: bool = os.getenv("AGENT_STREAMING", "false").lower() == "true"

    # OTEL
    OTEL_ENDPOINT: str = os.getenv("OTEL_EXPORTER_OTLP_ENDPOINT", "")
    OTEL_SERVICE_NAME: str = os.getenv("OTEL_SERVICE_NAME", "unified-agent-v5")

    # Reasoning 모델 (temperature 미지원) — 환경변수로 재정의 가능
    REASONING_MODELS: frozenset[str] = frozenset(
        os.getenv("AGENT_REASONING_MODELS", "").split(",")
        if os.getenv("AGENT_REASONING_MODELS")
        else {
            "o1", "o1-mini", "o1-preview", "o3", "o3-mini", "o3-pro", "o4-mini",
            "gpt-5", "gpt-5.1", "gpt-5.2", "gpt-5-pro",
            "gpt-5.1-codex", "gpt-5.2-codex",
            "deepseek-r1", "deepseek-r1-0528",
        }
    )

    @classmethod
    def supports_temperature(cls, model: str) -> bool:
        """모델이 temperature 파라미터를 지원하는지 확인"""
        return model not in cls.REASONING_MODELS


@dataclass(slots=True)
class AgentConfig:
    """
    에이전트 인스턴스별 설정

    v4.1 대응: UnifiedAgentConfig(50+ 필드) → 실무 필수 15개 필드로 축소
    축소 이유: 미설정 항목은 Settings 폴백, __post_init__으로 자동 채움.
              사용자는 AgentConfig(model="gpt-5.2") 한 줄로 충분.

    사용법:
        >>> config = AgentConfig(model="gpt-5.2", system_prompt="You are helpful.")
        >>> config = AgentConfig.from_env()  # 환경변수에서 로드
    """
    # 모델 설정
    model: str = ""
    system_prompt: str = "You are a helpful assistant."
    temperature: float | None = None
    max_tokens: int | None = None

    # 엔진 선택 (direct, langchain, crewai)
    engine: str = ""

    # Azure 설정 (미설정 시 Settings 사용)
    azure_endpoint: str = ""
    azure_api_key: str = ""
    azure_deployment: str = ""
    azure_api_version: str = ""

    # OpenAI 설정
    openai_api_key: str = ""
    openai_base_url: str = ""

    # 스트리밍
    stream: bool = False

    # 최대 도구 호출 라운드 (무한루프 방지)
    max_tool_rounds: int = 10

    # 콜백 (OTEL 등)
    callbacks: list[Any] = field(default_factory=list)

    # 추가 설정
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        """기본값을 Settings에서 채움"""
        # 문자열 필드 폴백 (DRY)
        _str_defaults = {
            "model": Settings.DEFAULT_MODEL,
            "engine": Settings.DEFAULT_ENGINE,
            "azure_endpoint": Settings.AZURE_ENDPOINT,
            "azure_api_key": Settings.AZURE_API_KEY,
            "azure_deployment": Settings.AZURE_DEPLOYMENT,
            "azure_api_version": Settings.AZURE_API_VERSION,
            "openai_api_key": Settings.OPENAI_API_KEY,
            "openai_base_url": Settings.OPENAI_BASE_URL,
        }
        for attr, default in _str_defaults.items():
            if not getattr(self, attr):
                object.__setattr__(self, attr, default)

        # temperature / max_tokens 폴백
        if self.temperature is None:
            temp = Settings.DEFAULT_TEMPERATURE if Settings.supports_temperature(self.model) else None
            object.__setattr__(self, "temperature", temp)
        if self.max_tokens is None:
            object.__setattr__(self, "max_tokens", Settings.DEFAULT_MAX_TOKENS)

        # 입력 검증
        if self.max_tool_rounds < 1:
            raise ValueError(f"max_tool_rounds must be >= 1, got {self.max_tool_rounds}")

    @classmethod
    def from_env(cls) -> AgentConfig:
        """환경변수에서 자동 로드"""
        return cls()

    def get_api_key(self) -> str:
        """사용 가능한 API 키 반환 (Azure 우선)"""
        return self.azure_api_key or self.openai_api_key

    def get_base_url(self) -> str:
        """사용 가능한 Base URL 반환"""
        if self.azure_endpoint:
            return self.azure_endpoint
        return self.openai_base_url
