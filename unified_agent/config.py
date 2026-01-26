#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Unified Agent Framework - 설정 모듈 (Configuration Module)

================================================================================
📁 파일 위치: unified_agent/config.py
📋 역할: 전역 설정 및 프레임워크 구성 관리
📅 최종 업데이트: 2026년 1월
================================================================================

🎯 주요 기능:
    1. Settings 클래스 - 전역 설정 관리 (모델, API 버전, 기능 토글 등)
    2. FrameworkConfig - 인스턴스 단위 설정 (환경변수 자동 로드 지원)
    3. 모델 유틸리티 함수 - temperature 지원, 멀티모달, 컨텍스트 윈도우 확인
    4. 실행 설정 생성 - AzureChatPromptExecutionSettings 인스턴스 생성

🔧 지원 모델 (2026년 1월 기준):
    - OpenAI: GPT-5.2, GPT-5.1 Codex, GPT-4.1, o4-mini, o3
    - Anthropic: Claude Opus 4.5, Claude Sonnet 4.5 (Microsoft Foundry)
    - xAI: Grok-4, Grok-4 Fast Reasoning (Microsoft Foundry)
    - DeepSeek: V3.2, R1-0528
    - Meta: Llama 4 Maverick, Llama 4 Scout
    - Microsoft: Phi-4, Phi-4 Reasoning

📌 사용 예시:
    >>> from unified_agent.config import Settings, FrameworkConfig
    >>>
    >>> # 전역 모델 변경
    >>> Settings.DEFAULT_MODEL = "gpt-5.2"
    >>>
    >>> # 환경변수에서 설정 로드
    >>> config = FrameworkConfig.from_env()
    >>> config.validate()  # 필수 설정 검증

⚠️ 주의사항:
    - Reasoning 모델(o1, o3, o4, GPT-5 기본)은 temperature를 지원하지 않습니다.
    - 환경변수 설정 시 따옴표와 공백에 주의하세요.
    - LARGE_CONTEXT_MODELS는 100K+ 토큰을 지원합니다.

🔗 관련 문서:
    - Azure OpenAI: https://learn.microsoft.com/azure/ai-services/openai/
    - Microsoft Agent Framework: https://github.com/microsoft/agent-framework
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
    "is_multimodal_model",
    "is_large_context_model",
    "get_model_context_window",
    "create_execution_settings",
]


class Settings:
    """
    프레임워크 전역 설정 클래스 (Singleton-like Pattern)

    ================================================================================
    📋 역할: 모든 전역 설정을 한 곳에서 중앙 집중 관리
    📅 최종 업데이트: 2026년 1월
    ================================================================================

    🎯 주요 기능:
        - LLM 모델 설정 (기본 모델, API 버전, 온도, 토큰 등)
        - 지원 모델 목록 관리 (SUPPORTED_MODELS, MODELS_WITHOUT_TEMPERATURE)
        - 프레임워크 기능 토글 (스트리밍, 텔레메트리, 이벤트 등)
        - Memory 시스템 설정 (훅, 네임스페이스, TTL 등)
        - MCP (Model Context Protocol) 설정
        - Multi-Agent 오케스트레이션 설정
        - RAI (Responsible AI) 설정

    📌 사용 예시:
        >>> # 기본 모델 변경 (런타임)
        >>> Settings.DEFAULT_MODEL = "claude-opus-4-5"
        >>>
        >>> # 스트리밍 비활성화
        >>> Settings.ENABLE_STREAMING = False
        >>>
        >>> # MCP 설정 조정
        >>> Settings.MCP_APPROVAL_MODE = "always"  # 모든 MCP 호출에 승인 필요
        >>>
        >>> # 지원 모델 확인
        >>> print(Settings.SUPPORTED_MODELS)

    🔧 2026년 1월 업데이트 내역:
        ✅ GPT-5.2 시리즈: gpt-5.2, gpt-5.2-chat, gpt-5.2-codex (최신)
        ✅ GPT-5.1 Codex: gpt-5.1-codex, gpt-5.1-codex-mini, gpt-5.1-codex-max
        ✅ o4-mini: o3-mini 후속 Reasoning 모델
        ✅ Claude 4.5: claude-opus-4-5, claude-sonnet-4-5 (Microsoft Foundry)
        ✅ Grok-4: grok-4, grok-4-fast-reasoning, grok-4-fast-non-reasoning
        ✅ DeepSeek: V3.2, V3.2-speciale, R1-0528 (Reasoning)
        ✅ Llama 4: llama-4-maverick-17b, llama-4-scout-17b (Meta)
        ✅ Phi-4: phi-4, phi-4-reasoning, phi-4-multimodal-instruct

    ⚠️ 주의사항:
        - 클래스 변수이므로 모든 인스턴스에서 공유됩니다.
        - 스레드 안전하지 않으므로 멀티스레드 환경에서는 주의가 필요합니다.
        - 환경변수를 통한 설정은 FrameworkConfig.from_env()를 사용하세요.

    🔗 참고:
        - FrameworkConfig: 인스턴스 단위 설정
        - supports_temperature(): 모델별 temperature 지원 확인
        - create_execution_settings(): 실행 설정 생성
    """

    # ─────────────────────────────────────────────────────────────────────────
    # LLM 모델 설정 (2026년 최신)
    # ─────────────────────────────────────────────────────────────────────────
    DEFAULT_MODEL: str = "gpt-5.2"           # 기본 모델 (2026년 최신)
    DEFAULT_API_VERSION: str = "2025-12-01-preview"  # API 버전 (최신)
    DEFAULT_TEMPERATURE: float = 0.7         # 기본 Temperature (GPT-4 계열만)
    DEFAULT_MAX_TOKENS: int = 4096           # 기본 최대 토큰 수 (증가)
    DEFAULT_CONTEXT_WINDOW: int = 200000     # 기본 컨텍스트 윈도우

    # ─────────────────────────────────────────────────────────────────────────
    # 지원 모델 목록 (2026년 1월 기준)
    # ─────────────────────────────────────────────────────────────────────────
    SUPPORTED_MODELS: list = [
        # GPT-4 계열 (Legacy)
        "gpt-4", "gpt-4o", "gpt-4o-mini", "gpt-4.1", "gpt-4.1-mini", "gpt-4.1-nano",
        # GPT-5 계열 (2025년 출시)
        "gpt-5", "gpt-5-mini", "gpt-5-nano", "gpt-5-chat", "gpt-5-pro",
        # GPT-5.1 계열
        "gpt-5.1", "gpt-5.1-chat", "gpt-5.1-codex", "gpt-5.1-codex-mini", "gpt-5.1-codex-max",
        # GPT-5.2 계열 (2026년 최신)
        "gpt-5.2", "gpt-5.2-chat", "gpt-5.2-codex",
        # o-시리즈 (Reasoning Models)
        "o1", "o1-mini", "o1-preview", "o3", "o3-mini", "o3-pro", "o4-mini",
        # Claude 시리즈 (Anthropic - Microsoft Foundry 지원)
        "claude-opus-4-5", "claude-sonnet-4-5", "claude-haiku-4-5", "claude-opus-4-1",
        # Grok 시리즈 (xAI - Microsoft Foundry 지원)
        "grok-4", "grok-4-fast-reasoning", "grok-4-fast-non-reasoning",
        "grok-3", "grok-3-mini", "grok-code-fast-1",
        # DeepSeek 시리즈
        "deepseek-v3.2", "deepseek-v3.2-speciale", "deepseek-v3.1", "deepseek-r1-0528", "deepseek-r1",
        # Meta Llama 4 시리즈
        "llama-4-maverick-17b-128e-instruct-fp8", "llama-4-scout-17b-16e-instruct",
        "llama-3.3-70b-instruct",
        # Microsoft Phi 시리즈
        "phi-4", "phi-4-reasoning", "phi-4-mini-reasoning", "phi-4-multimodal-instruct",
        # Mistral 시리즈
        "mistral-large-3", "mistral-medium-2505", "mistral-small-2503",
        # 기타
        "codex-mini", "computer-use-preview", "gpt-oss-120b"
    ]

    # Temperature 미지원 모델 (자동으로 temperature 파라미터 제외)
    # Reasoning 모델 및 일부 특수 모델
    MODELS_WITHOUT_TEMPERATURE: list = [
        # GPT-5 Reasoning 계열
        "gpt-5", "gpt-5.1", "gpt-5.2", "gpt-5-pro",
        "gpt-5.1-codex", "gpt-5.1-codex-mini", "gpt-5.1-codex-max", "gpt-5.2-codex",
        # o-시리즈 (모두 Reasoning)
        "o1", "o1-mini", "o1-preview", "o3", "o3-mini", "o3-pro", "o4-mini",
        # DeepSeek Reasoning
        "deepseek-r1", "deepseek-r1-0528",
        # Phi Reasoning
        "phi-4-reasoning", "phi-4-mini-reasoning",
        # Codex 특수 모델
        "codex-mini"
    ]

    # 대용량 컨텍스트 모델 (100K+ tokens)
    LARGE_CONTEXT_MODELS: list = [
        "gpt-5.2", "gpt-5.2-codex", "gpt-5.1", "gpt-5.1-codex", "gpt-5.1-codex-max",
        "gpt-4.1", "gpt-4.1-mini", "claude-opus-4-5", "claude-sonnet-4-5",
        "grok-4-fast-reasoning", "llama-4-scout-17b-16e-instruct"
    ]

    # Multimodal 모델 (이미지/오디오 입력 지원)
    MULTIMODAL_MODELS: list = [
        "gpt-5.2", "gpt-5.2-chat", "gpt-5.1", "gpt-5.1-chat", "gpt-5",
        "gpt-4o", "gpt-4o-mini", "gpt-4.1",
        "claude-opus-4-5", "claude-sonnet-4-5", "claude-haiku-4-5",
        "grok-4", "grok-4-fast-reasoning",
        "phi-4-multimodal-instruct", "computer-use-preview"
    ]

    # ─────────────────────────────────────────────────────────────────────────
    # 프레임워크 설정
    # ─────────────────────────────────────────────────────────────────────────
    CHECKPOINT_DIR: str = "./checkpoints"    # 체크포인트 저장 경로
    ENABLE_TELEMETRY: bool = True            # OpenTelemetry 활성화
    ENABLE_EVENTS: bool = True               # 이벤트 시스템 활성화
    ENABLE_STREAMING: bool = True            # 스트리밍 응답 활성화 (기본 활성화)
    MAX_CACHE_SIZE: int = 500                # 메모리 캐시 최대 크기 (증가)
    ENABLE_PARALLEL_TOOLS: bool = True       # 병렬 도구 호출 활성화
    MAX_PARALLEL_TOOL_CALLS: int = 5         # 최대 병렬 도구 호출 수

    # ─────────────────────────────────────────────────────────────────────────
    # Memory 설정 (Microsoft Agent Framework 패턴)
    # ─────────────────────────────────────────────────────────────────────────
    ENABLE_MEMORY_HOOKS: bool = True         # Memory Hook 활성화
    MEMORY_NAMESPACE: str = "/conversation"  # 메모리 네임스페이스
    MAX_MEMORY_TURNS: int = 50               # 최대 대화 턴 수 (증가)
    SESSION_TTL_HOURS: int = 72              # 세션 만료 시간 (증가)
    ENABLE_SEMANTIC_MEMORY: bool = True      # 시맨틱 메모리 활성화
    MEMORY_EMBEDDING_MODEL: str = "text-embedding-3-large"  # 임베딩 모델

    # ─────────────────────────────────────────────────────────────────────────
    # MCP (Model Context Protocol) 설정 - 2026 최신
    # ─────────────────────────────────────────────────────────────────────────
    ENABLE_MCP: bool = True                  # MCP 활성화
    MCP_AUTO_CONNECT: bool = True            # MCP 자동 연결
    MCP_RECONNECT_ATTEMPTS: int = 3          # MCP 재연결 시도 횟수
    MCP_REQUEST_TIMEOUT: int = 30            # MCP 요청 타임아웃 (초)
    MCP_APPROVAL_MODE: str = "selective"     # MCP 승인 모드 (always/never/selective)

    # ─────────────────────────────────────────────────────────────────────────
    # Multi-Agent Orchestration 설정 (Microsoft Agent Framework 패턴)
    # ─────────────────────────────────────────────────────────────────────────
    AUTO_APPROVE_SIMPLE_PLANS: bool = True   # 간단한 계획 자동 승인
    MAX_SUPERVISOR_ROUNDS: int = 10          # Supervisor 최대 라운드 (증가)
    ORCHESTRATION_MODE: str = "adaptive"     # 오케스트레이션 모드 (supervisor/sequential/parallel/adaptive)
    ENABLE_HANDOFF: bool = True              # 에이전트 간 Handoff 활성화
    MAX_CONCURRENT_AGENTS: int = 5           # 최대 동시 에이전트 수
    ENABLE_REFLECTION: bool = True           # 반성(Reflection) 패턴 활성화

    # ─────────────────────────────────────────────────────────────────────────
    # RAI (Responsible AI) 설정
    # ─────────────────────────────────────────────────────────────────────────
    ENABLE_RAI_VALIDATION: bool = True       # RAI 검증 활성화
    RAI_STRICT_MODE: bool = False            # RAI 엄격 모드
    RAI_CONTENT_SAFETY_LEVEL: str = "medium" # 콘텐츠 안전 레벨 (low/medium/high)
    ENABLE_PII_DETECTION: bool = True        # PII 감지 활성화

    # ─────────────────────────────────────────────────────────────────────────
    # 로깅 및 트레이싱 설정
    # ─────────────────────────────────────────────────────────────────────────
    LOG_LEVEL: str = "INFO"                  # 로그 레벨
    LOG_FILE: str = "agent_framework.log"    # 로그 파일 경로
    ENABLE_TRACE_LOGGING: bool = True        # 트레이스 로깅 활성화
    TRACE_EXPORT_ENDPOINT: str = ""          # 트레이스 내보내기 엔드포인트


# 하위 호환성을 위한 전역 변수 (Settings 클래스 참조)
DEFAULT_LLM_MODEL = Settings.DEFAULT_MODEL
DEFAULT_API_VERSION = Settings.DEFAULT_API_VERSION
SUPPORTED_MODELS = Settings.SUPPORTED_MODELS
MODELS_WITHOUT_TEMPERATURE = Settings.MODELS_WITHOUT_TEMPERATURE
O_SERIES_MODELS = Settings.MODELS_WITHOUT_TEMPERATURE  # o-시리즈 모델 (temperature 미지원)


@dataclass
class FrameworkConfig:
    """
    프레임워크 인스턴스 설정 (Dataclass)

    ================================================================================
    📋 역할: 개별 인스턴스 단위의 설정 관리 (Settings 클래스 값을 기본값으로 사용)
    📅 최종 업데이트: 2026년 1월
    ================================================================================

    🎯 주요 기능:
        - LLM 설정: 모델, API 버전, 온도, 최대 토큰
        - Azure 설정: API 키, 엔드포인트, 배포명 (환경변수 자동 로드)
        - 프레임워크 설정: 체크포인트, 텔레메트리, 이벤트, 스트리밍
        - Memory 설정: 훅 활성화, 네임스페이스, 최대 턴 수, TTL
        - Supervisor 설정: 자동 승인, 최대 라운드
        - 로깅 설정: 레벨, 파일 경로

    📌 사용 예시:
        >>> # 방법 1: 기본 설정 (Settings 클래스 값 사용)
        >>> config = FrameworkConfig()
        >>>
        >>> # 방법 2: 커스텀 설정
        >>> config = FrameworkConfig(
        ...     model="gpt-5.2",
        ...     temperature=0.5,
        ...     enable_streaming=True,
        ...     checkpoint_dir="./my_checkpoints"
        ... )
        >>>
        >>> # 방법 3: 환경변수에서 자동 로드 (권장)
        >>> config = FrameworkConfig.from_env()
        >>> config.validate()  # 필수 설정 검증 (api_key, endpoint, deployment_name)
        >>>
        >>> # 방법 4: .env 파일 경로 지정
        >>> config = FrameworkConfig.from_env(dotenv_path="./production.env")

    🔧 지원 환경변수:
        - AZURE_OPENAI_API_KEY: Azure OpenAI API 키 (필수)
        - AZURE_OPENAI_ENDPOINT: Azure OpenAI 엔드포인트 URL (필수)
        - AZURE_OPENAI_DEPLOYMENT: 모델 배포명 (필수)
        - AZURE_OPENAI_API_VERSION: API 버전 (선택, 기본: 2025-12-01-preview)
        - AZURE_OPENAI_MODEL: 모델명 (선택, 기본: gpt-5.2)

    ⚠️ 주의사항:
        - validate() 메서드로 필수 설정 확인 필수
        - Windows .env 파일의 따옴표/공백 자동 처리됨
        - dataclass이므로 불변성을 보장하지 않습니다.

    🔗 참고:
        - Settings: 전역 설정 (클래스 변수)
        - ConfigurationError: 설정 오류 시 발생하는 예외
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
    모델의 temperature 파라미터 지원 여부 확인

    ================================================================================
    📋 역할: 주어진 모델이 temperature 파라미터를 지원하는지 확인
    📅 최종 업데이트: 2026년 1월
    ================================================================================

    🎯 기능 설명:
        temperature는 LLM 출력의 무작위성을 조절하는 파라미터입니다.
        - 0.0: 결정론적 (항상 동일한 출력)
        - 1.0: 높은 무작위성 (창의적 출력)

        Reasoning 모델(o1, o3, o4, GPT-5 기본 등)은 내부적으로 추론 과정을
        사용하므로 temperature 파라미터를 지원하지 않습니다.
        이러한 모델에 temperature를 전달하면 API 오류가 발생합니다.

    Args:
        model (str): 모델 이름
            예: 'gpt-4.1', 'gpt-5.2-chat', 'o3', 'claude-opus-4-5'

    Returns:
        bool: temperature 지원 여부
            - True: temperature 파라미터 사용 가능
            - False: temperature 파라미터 사용 불가 (API 호출 시 제외 필요)

    📌 사용 예시:
        >>> supports_temperature("gpt-4o")  # True (Chat 모델)
        >>> supports_temperature("gpt-5.2-chat")  # True (Chat 모델)
        >>> supports_temperature("gpt-5.2")  # False (Reasoning 모델)
        >>> supports_temperature("o4-mini")  # False (Reasoning 모델)
        >>> supports_temperature("claude-opus-4-5")  # True (Claude)

    🔧 Temperature 미지원 모델 (2026년 1월 기준):
        - GPT-5 Reasoning: gpt-5, gpt-5.1, gpt-5.2, gpt-5-pro
        - GPT-5 Codex: gpt-5.1-codex, gpt-5.2-codex (코드 특화)
        - o-시리즈 전체: o1, o1-mini, o3, o3-mini, o3-pro, o4-mini
        - DeepSeek Reasoning: deepseek-r1, deepseek-r1-0528
        - Phi Reasoning: phi-4-reasoning, phi-4-mini-reasoning
        - Codex 특수: codex-mini

    ⚠️ 주의사항:
        - 'chat' 접미사가 있는 모델은 temperature 지원 (예: gpt-5.2-chat)
        - 새로운 모델 추가 시 Settings.MODELS_WITHOUT_TEMPERATURE 업데이트 필요

    🔗 참고:
        - create_execution_settings(): 자동으로 이 함수를 사용하여 설정 생성
        - Settings.MODELS_WITHOUT_TEMPERATURE: 미지원 모델 목록
    """
    model_lower = model.lower()

    # chat 모델은 temperature 지원
    if "chat" in model_lower:
        return True

    # 명시적으로 temperature 미지원 모델 확인
    if model_lower in [m.lower() for m in Settings.MODELS_WITHOUT_TEMPERATURE]:
        return False

    # Reasoning 모델 계열 패턴 확인
    reasoning_prefixes = (
        "gpt-5", "o1", "o3", "o4",
        "deepseek-r", "phi-4-reasoning", "phi-4-mini-reasoning",
        "codex"
    )
    return not any(model_lower.startswith(prefix) for prefix in reasoning_prefixes)


def is_multimodal_model(model: str) -> bool:
    """
    모델의 멀티모달 (이미지/오디오/비디오 입력) 지원 여부 확인

    ================================================================================
    📋 역할: 주어진 모델이 텍스트 외 입력(이미지, 오디오 등)을 처리할 수 있는지 확인
    📅 최종 업데이트: 2026년 1월
    ================================================================================

    🎯 기능 설명:
        멀티모달 모델은 텍스트뿐만 아니라 이미지, 오디오, 비디오 등
        다양한 형태의 입력을 처리할 수 있습니다.

        지원 입력 유형:
        - 이미지: JPEG, PNG, GIF, WebP
        - 오디오: MP3, WAV (일부 모델)
        - 비디오: MP4 (일부 모델, 예: phi-4-multimodal-instruct)

    Args:
        model (str): 모델 이름

    Returns:
        bool: 멀티모달 지원 여부
            - True: 이미지/오디오 입력 처리 가능
            - False: 텍스트만 처리 가능

    📌 사용 예시:
        >>> is_multimodal_model("gpt-5.2")  # True
        >>> is_multimodal_model("gpt-5.2-codex")  # False (코드 특화)
        >>> is_multimodal_model("claude-opus-4-5")  # True
        >>> is_multimodal_model("o3")  # False

    🔧 멀티모달 지원 모델 (2026년 1월 기준):
        - OpenAI: gpt-5.2, gpt-5.2-chat, gpt-5.1, gpt-5.1-chat, gpt-5, gpt-4o
        - Anthropic: claude-opus-4-5, claude-sonnet-4-5, claude-haiku-4-5
        - xAI: grok-4, grok-4-fast-reasoning
        - Microsoft: phi-4-multimodal-instruct
        - 특수: computer-use-preview (화면 캡처 입력)
    """
    return model.lower() in [m.lower() for m in Settings.MULTIMODAL_MODELS]


def is_large_context_model(model: str) -> bool:
    """
    모델의 대용량 컨텍스트 (100K+ 토큰) 지원 여부 확인

    ================================================================================
    📋 역할: 주어진 모델이 100,000 토큰 이상의 컨텍스트를 처리할 수 있는지 확인
    📅 최종 업데이트: 2026년 1월
    ================================================================================

    🎯 기능 설명:
        대용량 컨텍스트 모델은 긴 문서, 코드베이스, 대화 기록 등을
        한 번에 처리할 수 있습니다.

        일반적인 컨텍스트 크기:
        - 표준: 8K ~ 32K 토큰
        - 대용량: 100K ~ 200K 토큰
        - 초대용량: 400K+ 토큰 (GPT-5.2, GPT-4.1)
        - 극대용량: 1M ~ 10M 토큰 (GPT-4.1, Llama 4 Scout)

    Args:
        model (str): 모델 이름

    Returns:
        bool: 대용량 컨텍스트 지원 여부 (100K+ 토큰)

    📌 사용 예시:
        >>> is_large_context_model("gpt-5.2")  # True (400K)
        >>> is_large_context_model("gpt-4o")  # False (128K)
        >>> is_large_context_model("gpt-4.1")  # True (1M)

    🔧 대용량 컨텍스트 모델 (2026년 1월 기준):
        - 400K: gpt-5.2, gpt-5.2-codex, gpt-5.1, gpt-5.1-codex-max
        - 200K: claude-opus-4-5, claude-sonnet-4-5, o3, o4-mini
        - 1M: gpt-4.1, gpt-4.1-mini, gpt-4.1-nano
        - 2M: grok-4-fast-reasoning
        - 10M: llama-4-scout-17b-16e-instruct (최대)
    """
    return model.lower() in [m.lower() for m in Settings.LARGE_CONTEXT_MODELS]


def get_model_context_window(model: str) -> int:
    """
    모델의 컨텍스트 윈도우 크기 (토큰 수) 반환

    ================================================================================
    📋 역할: 주어진 모델이 처리할 수 있는 최대 토큰 수 반환
    📅 최종 업데이트: 2026년 1월
    ================================================================================

    🎯 기능 설명:
        컨텍스트 윈도우는 모델이 한 번에 처리할 수 있는 입력과 출력의
        총 토큰 수를 의미합니다. 이 값을 초과하면 오류가 발생하거나
        이전 내용이 잘립니다.

        토큰 ≈ 단어 비율 (영어 기준):
        - 1 토큰 ≈ 0.75 단어 (또는 4자)
        - 100K 토큰 ≈ 75,000 단어 ≈ 300페이지

    Args:
        model (str): 모델 이름

    Returns:
        int: 컨텍스트 윈도우 크기 (토큰 수)
             알 수 없는 모델은 Settings.DEFAULT_CONTEXT_WINDOW 반환

    📌 사용 예시:
        >>> get_model_context_window("gpt-5.2")  # 400000
        >>> get_model_context_window("gpt-4.1")  # 1000000
        >>> get_model_context_window("llama-4-scout-17b-16e-instruct")  # 10000000
        >>> get_model_context_window("unknown-model")  # 200000 (기본값)

    🔧 컨텍스트 윈도우 목록 (2026년 1월 기준):
        - 128K: gpt-5.2-chat, gpt-5.1-chat
        - 200K: gpt-5, claude-opus-4-5, o3, o4-mini (기본값)
        - 400K: gpt-5.2, gpt-5.1-codex, gpt-5-pro
        - 1M: gpt-4.1 시리즈
        - 2M: grok-4-fast-reasoning
        - 10M: llama-4-scout-17b-16e-instruct (최대)
    """
    model_lower = model.lower()

    # 2026년 최신 모델 컨텍스트 윈도우
    context_windows = {
        # GPT-5.2 시리즈
        "gpt-5.2": 400000,
        "gpt-5.2-codex": 400000,
        "gpt-5.2-chat": 128000,
        # GPT-5.1 시리즈
        "gpt-5.1": 400000,
        "gpt-5.1-codex": 400000,
        "gpt-5.1-codex-max": 400000,
        "gpt-5.1-codex-mini": 400000,
        "gpt-5.1-chat": 128000,
        # GPT-5 시리즈
        "gpt-5": 200000,
        "gpt-5-pro": 400000,
        # GPT-4.1 시리즈
        "gpt-4.1": 1000000,
        "gpt-4.1-mini": 1000000,
        "gpt-4.1-nano": 1000000,
        # o-시리즈
        "o3": 200000,
        "o4-mini": 200000,
        # Claude 시리즈
        "claude-opus-4-5": 200000,
        "claude-sonnet-4-5": 200000,
        # Grok 시리즈
        "grok-4-fast-reasoning": 2000000,
        # Llama 시리즈
        "llama-4-scout-17b-16e-instruct": 10000000,
    }

    return context_windows.get(model_lower, Settings.DEFAULT_CONTEXT_WINDOW)


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
