#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Unified Agent Framework - 설정 모듈

전역 설정 및 프레임워크 구성을 관리합니다.
"""

import os
import logging
from dataclasses import dataclass, field
from typing import Optional

from dotenv import load_dotenv
from semantic_kernel.connectors.ai.open_ai.prompt_execution_settings.azure_chat_prompt_execution_settings import (
    AzureChatPromptExecutionSettings
)

from .exceptions import ConfigurationError

__all__ = [
    "Settings",
    "FrameworkConfig",
    "DEFAULT_LLM_MODEL",
    "DEFAULT_API_VERSION",
    "SUPPORTED_MODELS",
    "O_SERIES_MODELS",
    "MODELS_WITHOUT_TEMPERATURE",
    "supports_temperature",
    "create_execution_settings",
]


class Settings:
    """
    프레임워크 전역 설정 - 모든 설정을 한 곳에서 관리

    사용법:
        # 모델 변경
        Settings.DEFAULT_MODEL = "gpt-4.1"

        # 설정 확인
        print(Settings.DEFAULT_MODEL)
    """

    # ─────────────────────────────────────────────────────────────────────────
    # LLM 모델 설정
    # ─────────────────────────────────────────────────────────────────────────
    DEFAULT_MODEL: str = "gpt-5.2"           # 기본 모델
    DEFAULT_API_VERSION: str = "2024-08-01-preview"  # API 버전
    DEFAULT_TEMPERATURE: float = 0.7         # 기본 Temperature (GPT-4 계열만)
    DEFAULT_MAX_TOKENS: int = 1000           # 기본 최대 토큰 수

    # ─────────────────────────────────────────────────────────────────────────
    # 지원 모델 목록
    # ─────────────────────────────────────────────────────────────────────────
    SUPPORTED_MODELS: list = [
        # GPT-4 계열
        "gpt-4", "gpt-4o", "gpt-4o-mini", "gpt-4.1", "gpt-4.1-mini", "gpt-4.1-nano",
        # GPT-5 계열
        "gpt-5", "gpt-5.1", "gpt-5.2",
        # o-시리즈 (Reasoning)
        "o1", "o1-mini", "o1-preview", "o3", "o3-mini", "o4-mini"
    ]

    # Temperature 미지원 모델 (자동으로 temperature 파라미터 제외)
    MODELS_WITHOUT_TEMPERATURE: list = [
        "gpt-5", "gpt-5.1", "gpt-5.2",
        "o1", "o1-mini", "o1-preview", "o3", "o3-mini", "o4-mini"
    ]

    # ─────────────────────────────────────────────────────────────────────────
    # 프레임워크 설정
    # ─────────────────────────────────────────────────────────────────────────
    CHECKPOINT_DIR: str = "./checkpoints"    # 체크포인트 저장 경로
    ENABLE_TELEMETRY: bool = True            # OpenTelemetry 활성화
    ENABLE_EVENTS: bool = True               # 이벤트 시스템 활성화
    ENABLE_STREAMING: bool = False           # 스트리밍 응답 활성화
    MAX_CACHE_SIZE: int = 100                # 메모리 캐시 최대 크기

    # ─────────────────────────────────────────────────────────────────────────
    # Memory 설정 (AWS AgentCore 패턴)
    # ─────────────────────────────────────────────────────────────────────────
    ENABLE_MEMORY_HOOKS: bool = True         # Memory Hook 활성화
    MEMORY_NAMESPACE: str = "/conversation"  # 메모리 네임스페이스
    MAX_MEMORY_TURNS: int = 20               # 최대 대화 턴 수
    SESSION_TTL_HOURS: int = 24              # 세션 만료 시간 (시간)

    # ─────────────────────────────────────────────────────────────────────────
    # Supervisor 설정 (SRE Agent 패턴)
    # ─────────────────────────────────────────────────────────────────────────
    AUTO_APPROVE_SIMPLE_PLANS: bool = True   # 간단한 계획 자동 승인
    MAX_SUPERVISOR_ROUNDS: int = 5           # Supervisor 최대 라운드

    # ─────────────────────────────────────────────────────────────────────────
    # 로깅 설정
    # ─────────────────────────────────────────────────────────────────────────
    LOG_LEVEL: str = "INFO"                  # 로그 레벨
    LOG_FILE: str = "agent_framework.log"    # 로그 파일 경로


# 하위 호환성을 위한 전역 변수 (Settings 클래스 참조)
DEFAULT_LLM_MODEL = Settings.DEFAULT_MODEL
DEFAULT_API_VERSION = Settings.DEFAULT_API_VERSION
SUPPORTED_MODELS = Settings.SUPPORTED_MODELS
MODELS_WITHOUT_TEMPERATURE = Settings.MODELS_WITHOUT_TEMPERATURE
O_SERIES_MODELS = Settings.MODELS_WITHOUT_TEMPERATURE  # o-시리즈 모델 (temperature 미지원)


@dataclass
class FrameworkConfig:
    """
    프레임워크 설정 - Settings 클래스의 값을 기본값으로 사용

    사용법:
        # 기본 설정 사용 (Settings 클래스 값 적용)
        config = FrameworkConfig()

        # 커스텀 설정
        config = FrameworkConfig(
            model="gpt-4o",
            temperature=0.5,
            checkpoint_dir="./my_checkpoints"
        )

        # 환경변수에서 자동 로드
        config = FrameworkConfig.from_env()
    """
    # LLM 설정 - Settings 클래스 참조
    model: str = field(default_factory=lambda: Settings.DEFAULT_MODEL)
    api_version: str = field(default_factory=lambda: Settings.DEFAULT_API_VERSION)
    temperature: float = field(default_factory=lambda: Settings.DEFAULT_TEMPERATURE)
    max_tokens: int = field(default_factory=lambda: Settings.DEFAULT_MAX_TOKENS)

    # Azure 설정 (환경변수에서 로드)
    api_key: Optional[str] = None
    endpoint: Optional[str] = None
    deployment_name: Optional[str] = None

    # 프레임워크 설정 - Settings 클래스 참조
    checkpoint_dir: str = field(default_factory=lambda: Settings.CHECKPOINT_DIR)
    enable_telemetry: bool = field(default_factory=lambda: Settings.ENABLE_TELEMETRY)
    enable_events: bool = field(default_factory=lambda: Settings.ENABLE_EVENTS)
    enable_streaming: bool = field(default_factory=lambda: Settings.ENABLE_STREAMING)
    max_cache_size: int = field(default_factory=lambda: Settings.MAX_CACHE_SIZE)

    # Memory 설정 - Settings 클래스 참조
    enable_memory_hooks: bool = field(default_factory=lambda: Settings.ENABLE_MEMORY_HOOKS)
    memory_namespace: str = field(default_factory=lambda: Settings.MEMORY_NAMESPACE)
    max_memory_turns: int = field(default_factory=lambda: Settings.MAX_MEMORY_TURNS)
    session_ttl_hours: int = field(default_factory=lambda: Settings.SESSION_TTL_HOURS)

    # Supervisor 설정 - Settings 클래스 참조
    auto_approve_simple_plans: bool = field(default_factory=lambda: Settings.AUTO_APPROVE_SIMPLE_PLANS)
    max_supervisor_rounds: int = field(default_factory=lambda: Settings.MAX_SUPERVISOR_ROUNDS)

    # 로깅 설정 - Settings 클래스 참조
    log_level: str = field(default_factory=lambda: Settings.LOG_LEVEL)
    log_file: Optional[str] = field(default_factory=lambda: Settings.LOG_FILE)

    @classmethod
    def from_env(cls, dotenv_path: Optional[str] = None) -> 'FrameworkConfig':
        """
        환경변수에서 설정 로드

        지원하는 환경변수 (우선순위 순서):
        - API Key: AZURE_OPENAI_API_KEY
        - Endpoint: AZURE_OPENAI_ENDPOINT
        - Deployment: AZURE_OPENAI_DEPLOYMENT
        - API Version: AZURE_OPENAI_API_VERSION (기본: 2024-08-01-preview)
        """
        load_dotenv(dotenv_path)

        # API Key
        api_key = os.getenv("AZURE_OPENAI_API_KEY")

        # Endpoint
        endpoint = os.getenv("AZURE_OPENAI_ENDPOINT")

        # Deployment Name
        deployment_name = os.getenv("AZURE_OPENAI_DEPLOYMENT")

        # 환경변수 값에서 따옴표와 공백 제거 (Windows .env 파일 문제 해결)
        if api_key:
            api_key = api_key.strip().strip('"').strip("'").strip()
        if endpoint:
            endpoint = endpoint.strip().strip('"').strip("'").strip()
        if deployment_name:
            deployment_name = deployment_name.strip().strip('"').strip("'").strip()

        return cls(
            api_key=api_key,
            endpoint=endpoint,
            deployment_name=deployment_name,
            model=os.getenv("AZURE_OPENAI_MODEL", Settings.DEFAULT_MODEL),
            api_version=os.getenv("AZURE_OPENAI_API_VERSION", Settings.DEFAULT_API_VERSION),
        )

    def validate(self) -> bool:
        """설정 유효성 검증"""
        missing = []
        if not self.api_key:
            missing.append("api_key (AZURE_OPENAI_API_KEY)")
        if not self.endpoint:
            missing.append("endpoint (AZURE_OPENAI_ENDPOINT)")
        if not self.deployment_name:
            missing.append("deployment_name (AZURE_OPENAI_DEPLOYMENT)")

        if missing:
            raise ConfigurationError(
                f"❌ 필수 설정이 누락되었습니다:\n" +
                "\n".join(f"  - {m}" for m in missing) +
                "\n\n💡 .env 파일을 생성하거나 환경변수를 설정하세요."
            )
        return True


def supports_temperature(model: str) -> bool:
    """
    모델이 temperature 파라미터를 지원하는지 확인

    Args:
        model: 모델 이름 (예: 'gpt-4.1', 'gpt-5', 'o1')

    Returns:
        bool: temperature 지원 여부

    Note:
        GPT-5, o1, o3 계열 모델은 temperature를 지원하지 않습니다.
    """
    model_lower = model.lower()
    return model_lower not in Settings.MODELS_WITHOUT_TEMPERATURE and \
           not any(model_lower.startswith(prefix) for prefix in ("gpt-5", "o1", "o3", "o4"))


def create_execution_settings(
    model: str,
    temperature: float = 0.7,
    max_tokens: int = 1000,
    service_id: Optional[str] = None,
    **kwargs
) -> AzureChatPromptExecutionSettings:
    """
    모델에 따라 적절한 실행 설정 생성

    Args:
        model: 모델 이름
        temperature: 온도 설정 (지원하는 모델에만 적용)
        max_tokens: 최대 토큰 수
        service_id: 서비스 ID (없으면 model 사용)
        **kwargs: 추가 설정

    Returns:
        AzureChatPromptExecutionSettings 인스턴스
    """
    settings_kwargs = {
        "max_tokens": max_tokens,
        "service_id": service_id or model,
        **kwargs
    }

    # Temperature 지원 모델에만 temperature 추가
    if supports_temperature(model):
        settings_kwargs["temperature"] = temperature
    else:
        logging.info(f"ℹ️ 모델 '{model}'은(는) temperature를 지원하지 않습니다. 해당 파라미터를 생략합니다.")

    return AzureChatPromptExecutionSettings(**settings_kwargs)
