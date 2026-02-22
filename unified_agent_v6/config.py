#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Unified Agent Framework v6 — Configuration

================================================================================
Microsoft Agent Framework 1.0.0-rc1 호환 설정 관리

환경변수 기반으로 에이전트 설정을 로드하는 모듈입니다.
python-dotenv를 통해 .env 파일을 자동으로 로드하며,
OpenAI / Azure OpenAI / 에이전트 동작 설정을 TypedDict로 관리합니다.

핵심 함수/타입:
    - AgentConfig : 환경변수 기반 설정 TypedDict
    - load_config : .env 로드 + 환경변수 파싱 → AgentConfig 반환
    - get_env     : 환경변수 안전 조회

v5 → v6 변경 요약:
    - Settings dataclass → AgentConfig TypedDict + 환경변수 로딩
    - 설정 로딩 방식 간소화
================================================================================
"""

from __future__ import annotations

import logging
import os
from typing import Any, TypedDict

__all__ = [
    "AgentConfig",
    "load_config",
    "get_env",
]

logger = logging.getLogger("agent_framework")


class AgentConfig(TypedDict, total=False):
    """
    에이전트 설정 — 환경변수 기반 TypedDict.

    load_config()로 자동 로드되며, IDE 자동완성과
    타입 안전성을 제공합니다.

    지원 환경변수:
        - OPENAI_API_KEY, OPENAI_CHAT_MODEL_ID, OPENAI_BASE_URL
        - AZURE_OPENAI_API_KEY, AZURE_OPENAI_ENDPOINT, AZURE_OPENAI_DEPLOYMENT
        - AGENT_TEMPERATURE, AGENT_MAX_TOKENS, AGENT_MAX_TOOL_ROUNDS
        - AGENT_STREAM, AGENT_LOG_LEVEL
    """

    # OpenAI 설정
    openai_api_key: str
    openai_model: str
    openai_base_url: str

    # Azure OpenAI 설정
    azure_openai_api_key: str
    azure_openai_endpoint: str
    azure_openai_deployment: str
    azure_openai_api_version: str

    # 에이전트 동작 설정
    temperature: float
    max_tokens: int
    max_tool_rounds: int
    stream: bool

    # 로깅
    log_level: str

    # 기타 커스텀 설정
    extra: dict[str, Any]


def load_config(
    *,
    env_file: str | None = None,
    defaults: dict[str, Any] | None = None,
) -> AgentConfig:
    """
    환경변수에서 설정을 로드합니다.

    .env 파일이 있으면 자동으로 로드합니다 (python-dotenv 필요).

    사용법:
        >>> config = load_config()
        >>> print(config.get("openai_model", "gpt-5.2"))
    """
    # .env 파일 로드
    try:
        from dotenv import load_dotenv
        load_dotenv(env_file or ".env", override=False)
    except ImportError:
        pass

    config: AgentConfig = {}  # type: ignore

    # OpenAI
    if val := os.getenv("OPENAI_API_KEY"):
        config["openai_api_key"] = val
    config["openai_model"] = os.getenv("OPENAI_CHAT_MODEL_ID", os.getenv("OPENAI_MODEL", "gpt-5.2"))
    if val := os.getenv("OPENAI_BASE_URL"):
        config["openai_base_url"] = val

    # Azure OpenAI
    if val := os.getenv("AZURE_OPENAI_API_KEY"):
        config["azure_openai_api_key"] = val
    if val := os.getenv("AZURE_OPENAI_ENDPOINT"):
        config["azure_openai_endpoint"] = val
    if val := os.getenv("AZURE_OPENAI_DEPLOYMENT"):
        config["azure_openai_deployment"] = val
    config["azure_openai_api_version"] = os.getenv("AZURE_OPENAI_API_VERSION", "2025-12-01-preview")

    # 에이전트 동작 설정
    config["temperature"] = float(os.getenv("AGENT_TEMPERATURE", "0.7"))
    config["max_tokens"] = int(os.getenv("AGENT_MAX_TOKENS", "4096"))
    config["max_tool_rounds"] = int(os.getenv("AGENT_MAX_TOOL_ROUNDS", "10"))
    config["stream"] = os.getenv("AGENT_STREAM", "false").lower() == "true"

    # 로깅
    config["log_level"] = os.getenv("AGENT_LOG_LEVEL", "INFO")

    # defaults 병합
    if defaults:
        for k, v in defaults.items():
            if k not in config:
                config[k] = v  # type: ignore

    return config


def get_env(key: str, default: str = "") -> str:
    """환경변수 값을 안전하게 가져옵니다."""
    return os.getenv(key, default)
