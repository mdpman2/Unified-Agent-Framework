#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Unified Agent Framework - Enterprise Edition
Microsoft Agent Framework 패턴 통합 (MCP, Approval, Streaming 지원)
+ Anthropic Skills 시스템 통합

============================================================================
🚀 빠른 시작 가이드
============================================================================

1. 환경변수 설정 (.env 파일):
   AZURE_OPENAI_API_KEY=your-api-key
   AZURE_OPENAI_ENDPOINT=https://your-endpoint.openai.azure.com/
   AZURE_OPENAI_DEPLOYMENT=your-deployment-name

2. 가장 간단한 사용법:
   ```python
   import asyncio
   from Semantic_agent_framework import quick_run

   response = asyncio.run(quick_run("파이썬이란 무엇인가요?"))
   print(response)
   ```

3. 프레임워크 직접 사용:
   ```python
   import asyncio
   from Semantic_agent_framework import UnifiedAgentFramework

   async def main():
       # 프레임워크 생성 (환경변수 자동 로드)
       framework = UnifiedAgentFramework.create()

       # 빠른 질의응답
       response = await framework.quick_chat("안녕하세요!")
       print(response)

       # 워크플로우 생성 및 실행
       framework.create_simple_workflow("my_bot", "너는 친절한 AI야.")
       state = await framework.run("session-1", "my_bot", "질문입니다")

   asyncio.run(main())
   ```

4. Skills 시스템 사용:
   ```python
   from Semantic_agent_framework import Skill, SkillManager

   # 스킬 생성
   coding_skill = Skill(
       name="python-expert",
       description="Python 코딩 전문가. 코드 작성, 디버깅, 최적화 요청 시 사용.",
       instructions='''
       ## 역할
       Python 전문 개발자로서 코드를 작성합니다.

       ## 가이드라인
       - PEP 8 스타일 가이드 준수
       - 타입 힌트 사용
       - 명확한 docstring 작성
       ''',
       triggers=["python", "코딩", "프로그래밍", "코드"]
   )

   # 프레임워크에 스킬 등록
   framework.skill_manager.register_skill(coding_skill)

   # 스킬 기반 에이전트 생성
   agent = framework.create_skilled_agent("coder", skills=["python-expert"])
   ```

============================================================================
주요 기능
============================================================================
1. MCP (Model Context Protocol) 서버 통합 - 외부 도구 연동
2. Human-in-the-loop 승인 시스템 - 민감한 작업 승인 필요
3. 스트리밍 응답 지원 - 실시간 토큰 출력
4. 재시도 로직 및 회로 차단기 패턴 - 장애 격리
5. 비동기 이벤트 시스템 - Pub-Sub 패턴
6. 향상된 메모리 관리 - LRU 캐시
7. Supervisor Agent 패턴 - 멀티 에이전트 협업
8. 조건부 라우팅 및 루프 지원 - 동적 워크플로우
9. 버전 관리 및 롤백 - 상태 복원
10. 상세 메트릭 및 성능 모니터링
11. Anthropic Skills 시스템 - 모듈화된 전문 지식 관리 (NEW!)

============================================================================
필요 패키지
============================================================================
pip install semantic-kernel python-dotenv opentelemetry-api opentelemetry-sdk pydantic pyyaml
"""

import os
import sys
import asyncio
import json
import logging
import re
import glob
import fnmatch
from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Any, Callable, Set, AsyncIterator, Union
from datetime import datetime, timezone
from enum import Enum
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
import time

# UTF-8 인코딩 기본 설정 (Windows 환경 지원)
if sys.stdout and hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')
if sys.stderr and hasattr(sys.stderr, 'reconfigure'):
    sys.stderr.reconfigure(encoding='utf-8')

try:
    import yaml
    YAML_AVAILABLE = True
except ImportError:
    YAML_AVAILABLE = False

from dotenv import load_dotenv
from pydantic import BaseModel, Field

# Semantic Kernel
from semantic_kernel import Kernel
from semantic_kernel.connectors.ai.chat_completion_client_base import ChatCompletionClientBase
from semantic_kernel.connectors.ai.function_choice_behavior import FunctionChoiceBehavior
from semantic_kernel.connectors.ai.open_ai.prompt_execution_settings.azure_chat_prompt_execution_settings import AzureChatPromptExecutionSettings
from semantic_kernel.connectors.ai.open_ai.services.azure_chat_completion import AzureChatCompletion
from semantic_kernel.contents.chat_history import ChatHistory
from semantic_kernel.contents.streaming_chat_message_content import StreamingChatMessageContent

# OpenTelemetry
from opentelemetry import trace
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import ConsoleSpanExporter, BatchSpanProcessor
from opentelemetry.sdk.resources import Resource


# ============================================================================
# 🎯 중앙 설정 (CENTRAL CONFIGURATION)
# ============================================================================
# 🚨 모든 설정은 여기서만 변경하세요!
# ============================================================================

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


# ============================================================================
# 설정 클래스 (Configuration Class)
# ============================================================================

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

        # API Key (AZURE_OPENAI_API_KEY 우선)
        api_key = (
            os.getenv("AZURE_OPENAI_API_KEY")
        )

        # Endpoint (AZURE_OPENAI_ENDPOINT 우선)
        endpoint = (
            os.getenv("AZURE_OPENAI_ENDPOINT")
        )

        # Deployment Name (AZURE_OPENAI_DEPLOYMENT 우선) - 값에서 따옴표/공백 제거
        deployment_name = (
            os.getenv("AZURE_OPENAI_DEPLOYMENT")
        )

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
            raise ValueError(
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
    for unsupported in MODELS_WITHOUT_TEMPERATURE:
        if unsupported in model_lower:
            return False
    return True


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


# ============================================================================
# Skills 시스템 (Anthropic Skills 패턴)
# ============================================================================

@dataclass
class SkillResource:
    """
    스킬 번들 리소스

    스킬에 포함되는 추가 리소스를 정의합니다:
    - scripts/: 실행 가능한 스크립트 (Python, Bash 등)
    - references/: 참조 문서 (마크다운, 텍스트 등)
    - assets/: 템플릿, 이미지 등 출력용 파일
    """
    resource_type: str  # 'script', 'reference', 'asset'
    name: str
    path: str
    content: Optional[str] = None
    description: Optional[str] = None


@dataclass
class Skill:
    """
    Anthropic Skills 패턴 구현

    Skills는 Claude의 능력을 확장하는 모듈화된 패키지입니다.
    특정 도메인의 지식, 워크플로우, 도구를 제공합니다.

    구조:
    ```
    skill-name/
    ├── SKILL.md (필수)
    │   ├── YAML frontmatter (name, description)
    │   └── Markdown 지침
    └── Bundled Resources (선택)
        ├── scripts/      - 실행 코드
        ├── references/   - 참조 문서
        └── assets/       - 템플릿, 아이콘 등
    ```

    사용법:
    ```python
    # 직접 생성
    skill = Skill(
        name="python-expert",
        description="Python 코딩 전문가",
        instructions="## 역할\\n파이썬 전문가로서...",
        triggers=["python", "코딩"]
    )

    # 파일에서 로드
    skill = Skill.from_file("skills/python-expert/SKILL.md")

    # 디렉토리에서 로드 (리소스 포함)
    skill = Skill.from_directory("skills/python-expert/")
    ```
    """
    name: str
    description: str
    instructions: str = ""
    triggers: List[str] = field(default_factory=list)
    resources: List[SkillResource] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    enabled: bool = True
    priority: int = 0  # 높을수록 우선순위 높음

    # Progressive Disclosure 관련
    always_loaded: bool = False  # True면 항상 컨텍스트에 포함
    max_context_lines: int = 500  # SKILL.md 최대 라인 수

    @classmethod
    def from_file(cls, filepath: str) -> 'Skill':
        """
        SKILL.md 파일에서 스킬 로드

        파일 형식:
        ```markdown
        ---
        name: skill-name
        description: 스킬 설명
        ---

        # 스킬 제목

        ## 지침
        ...
        ```
        """
        path = Path(filepath)
        if not path.exists():
            raise FileNotFoundError(f"스킬 파일을 찾을 수 없습니다: {filepath}")

        content = path.read_text(encoding='utf-8')
        return cls._parse_skill_content(content, filepath)

    @classmethod
    def from_directory(cls, dirpath: str) -> 'Skill':
        """
        스킬 디렉토리에서 스킬 로드 (리소스 포함)

        디렉토리 구조:
        ```
        skill-name/
        ├── SKILL.md
        ├── scripts/
        │   └── example.py
        ├── references/
        │   └── api_reference.md
        └── assets/
            └── template.txt
        ```
        """
        dirpath = Path(dirpath)
        skill_file = dirpath / "SKILL.md"

        if not skill_file.exists():
            raise FileNotFoundError(f"SKILL.md를 찾을 수 없습니다: {skill_file}")

        # 기본 스킬 로드
        skill = cls.from_file(str(skill_file))

        # 리소스 로드
        skill._load_resources(dirpath)

        return skill

    @classmethod
    def _parse_skill_content(cls, content: str, source: str = "") -> 'Skill':
        """SKILL.md 내용 파싱"""
        # YAML frontmatter 추출
        frontmatter = {}
        body = content

        if content.startswith('---'):
            parts = content.split('---', 2)
            if len(parts) >= 3:
                if YAML_AVAILABLE:
                    try:
                        frontmatter = yaml.safe_load(parts[1]) or {}
                    except yaml.YAMLError:
                        frontmatter = cls._parse_simple_yaml(parts[1])
                else:
                    frontmatter = cls._parse_simple_yaml(parts[1])
                body = parts[2].strip()

        name = frontmatter.get('name', Path(source).stem if source else 'unnamed-skill')
        description = frontmatter.get('description', '')

        # triggers 추출 (description에서 자동 추출 또는 명시적 지정)
        triggers = frontmatter.get('triggers', [])
        if not triggers and description:
            # description에서 주요 키워드 추출
            triggers = cls._extract_triggers(description)

        # priority 추출 (기본값: 0)
        priority = frontmatter.get('priority', 0)
        if isinstance(priority, str):
            try:
                priority = int(priority)
            except ValueError:
                priority = 0

        return cls(
            name=name,
            description=description,
            instructions=body,
            triggers=triggers,
            priority=priority,  # 🆕 priority 반영
            metadata={
                'source': source,
                'license': frontmatter.get('license', ''),
                **{k: v for k, v in frontmatter.items() if k not in ['name', 'description', 'triggers', 'license', 'priority']}
            }
        )

    @staticmethod
    def _parse_simple_yaml(text: str) -> Dict[str, Any]:
        """간단한 YAML 파싱 (yaml 라이브러리 없을 때)"""
        result = {}
        for line in text.strip().split('\n'):
            if ':' in line:
                key, value = line.split(':', 1)
                key = key.strip()
                value = value.strip()
                # 따옴표 제거
                if value.startswith('"') and value.endswith('"'):
                    value = value[1:-1]
                elif value.startswith("'") and value.endswith("'"):
                    value = value[1:-1]
                result[key] = value
        return result

    @staticmethod
    def _extract_triggers(description: str) -> List[str]:
        """설명에서 트리거 키워드 추출"""
        # 주요 키워드 패턴
        keywords = []
        # 괄호 안의 내용 추출
        parens = re.findall(r'\(([^)]+)\)', description)
        for paren in parens:
            keywords.extend([k.strip() for k in paren.split(',')])

        # 주요 단어 추출 (영문은 소문자로)
        words = re.findall(r'\b[A-Za-z가-힣]{3,}\b', description)
        stop_words = {'the', 'and', 'for', 'use', 'when', 'with', 'this', 'that', 'from', 'have', 'are'}
        keywords.extend([w.lower() for w in words if w.lower() not in stop_words][:5])

        return list(set(keywords))[:10]

    def _load_resources(self, dirpath: Path):
        """디렉토리에서 리소스 로드"""
        # Scripts
        scripts_dir = dirpath / "scripts"
        if scripts_dir.exists():
            for script_file in scripts_dir.glob("*"):
                if script_file.is_file():
                    self.resources.append(SkillResource(
                        resource_type="script",
                        name=script_file.name,
                        path=str(script_file),
                        description=f"Script: {script_file.name}"
                    ))

        # References
        refs_dir = dirpath / "references"
        if refs_dir.exists():
            for ref_file in refs_dir.glob("*"):
                if ref_file.is_file():
                    self.resources.append(SkillResource(
                        resource_type="reference",
                        name=ref_file.name,
                        path=str(ref_file),
                        description=f"Reference: {ref_file.name}"
                    ))

        # Assets
        assets_dir = dirpath / "assets"
        if assets_dir.exists():
            for asset_file in assets_dir.glob("*"):
                if asset_file.is_file() or asset_file.is_dir():
                    self.resources.append(SkillResource(
                        resource_type="asset",
                        name=asset_file.name,
                        path=str(asset_file),
                        description=f"Asset: {asset_file.name}"
                    ))

    def get_resource(self, name: str) -> Optional[SkillResource]:
        """이름으로 리소스 찾기"""
        for resource in self.resources:
            if resource.name == name:
                return resource
        return None

    def load_resource_content(self, resource: SkillResource) -> str:
        """리소스 내용 로드"""
        if resource.content:
            return resource.content

        path = Path(resource.path)
        if path.exists() and path.is_file():
            try:
                resource.content = path.read_text(encoding='utf-8')
                return resource.content
            except Exception as e:
                logging.warning(f"리소스 로드 실패: {resource.path} - {e}")
        return ""

    def matches(self, query: str) -> float:
        """
        쿼리와의 매칭 점수 계산 (0.0 ~ 1.0)

        Progressive Disclosure: 쿼리에 따라 스킬 활성화 여부 결정
        """
        query_lower = query.lower()
        score = 0.0

        # 이름 매칭 (높은 가중치)
        if self.name.lower() in query_lower:
            score += 0.5

        # 트리거 매칭
        for trigger in self.triggers:
            if trigger.lower() in query_lower:
                score += 0.3
                break

        # 설명 매칭
        desc_words = self.description.lower().split()
        query_words = query_lower.split()
        common_words = set(desc_words) & set(query_words)
        if common_words:
            score += min(len(common_words) * 0.1, 0.2)

        return min(score, 1.0)

    def get_prompt_section(self, include_full: bool = False) -> str:
        """
        프롬프트에 포함할 스킬 섹션 생성

        Progressive Disclosure 적용:
        - include_full=False: 메타데이터만 (name + description)
        - include_full=True: 전체 지침 포함
        """
        if include_full:
            return f"""
## Skill: {self.name}

**Description:** {self.description}

{self.instructions}

---
"""
        else:
            return f"- **{self.name}**: {self.description}\n"

    def to_dict(self) -> Dict[str, Any]:
        """딕셔너리로 변환"""
        return {
            "name": self.name,
            "description": self.description,
            "instructions": self.instructions,
            "triggers": self.triggers,
            "resources": [
                {"type": r.resource_type, "name": r.name, "path": r.path}
                for r in self.resources
            ],
            "metadata": self.metadata,
            "enabled": self.enabled,
            "priority": self.priority
        }


class SkillManager:
    """
    스킬 관리자 - 스킬 등록, 검색, 활성화 관리

    주요 기능:
    - 스킬 등록 및 해제
    - 쿼리 기반 스킬 매칭 (Progressive Disclosure)
    - 디렉토리에서 스킬 일괄 로드
    - 스킬 우선순위 관리

    사용법:
    ```python
    manager = SkillManager()

    # 스킬 등록
    manager.register_skill(my_skill)

    # 디렉토리에서 로드
    manager.load_skills_from_directory("./skills")

    # 쿼리에 맞는 스킬 찾기
    matched_skills = manager.match_skills("Python 코드 작성해줘")

    # 활성화된 스킬로 시스템 프롬프트 생성
    prompt = manager.build_system_prompt(matched_skills)
    ```
    """

    def __init__(self, skill_dirs: Optional[List[str]] = None):
        self.skills: Dict[str, Skill] = {}
        self.skill_history: List[Dict[str, Any]] = []  # 스킬 사용 기록

        # 기본 스킬 디렉토리에서 로드
        if skill_dirs:
            for skill_dir in skill_dirs:
                self.load_skills_from_directory(skill_dir)

    def register_skill(self, skill: Skill) -> bool:
        """스킬 등록"""
        if skill.name in self.skills:
            logging.warning(f"스킬 '{skill.name}'이 이미 존재합니다. 덮어씁니다.")

        self.skills[skill.name] = skill
        logging.info(f"✅ 스킬 등록: {skill.name}")
        return True

    def unregister_skill(self, name: str) -> bool:
        """스킬 해제"""
        if name in self.skills:
            del self.skills[name]
            logging.info(f"🗑️ 스킬 해제: {name}")
            return True
        return False

    def get_skill(self, name: str) -> Optional[Skill]:
        """이름으로 스킬 가져오기"""
        return self.skills.get(name)

    def list_skills(self, enabled_only: bool = True) -> List[Skill]:
        """등록된 스킬 목록"""
        skills = list(self.skills.values())
        if enabled_only:
            skills = [s for s in skills if s.enabled]
        return sorted(skills, key=lambda s: -s.priority)

    def load_skills_from_directory(self, dirpath: str) -> int:
        """
        디렉토리에서 스킬 일괄 로드

        디렉토리 구조:
        ```
        skills/
        ├── python-expert/
        │   └── SKILL.md
        ├── data-analyst/
        │   ├── SKILL.md
        │   └── scripts/
        └── ...
        ```
        """
        dirpath = Path(dirpath)
        if not dirpath.exists():
            logging.warning(f"스킬 디렉토리가 존재하지 않습니다: {dirpath}")
            return 0

        loaded = 0
        for skill_dir in dirpath.iterdir():
            if skill_dir.is_dir():
                skill_file = skill_dir / "SKILL.md"
                if skill_file.exists():
                    try:
                        skill = Skill.from_directory(str(skill_dir))
                        self.register_skill(skill)
                        loaded += 1
                    except Exception as e:
                        logging.error(f"스킬 로드 실패: {skill_dir} - {e}")

        logging.info(f"📦 {loaded}개 스킬 로드 완료 from {dirpath}")
        return loaded

    def match_skills(
        self,
        query: str,
        threshold: float = 0.2,
        max_skills: int = 3
    ) -> List[Skill]:
        """
        쿼리에 매칭되는 스킬 찾기

        Progressive Disclosure 구현:
        - threshold 이상의 매칭 점수를 가진 스킬만 반환
        - max_skills 개수 제한
        - always_loaded 스킬은 항상 포함
        """
        matched = []

        for skill in self.list_skills():
            if skill.always_loaded:
                matched.append((skill, 1.0))
                continue

            score = skill.matches(query)
            if score >= threshold:
                matched.append((skill, score))

        # 점수 및 우선순위로 정렬
        matched.sort(key=lambda x: (-x[1], -x[0].priority))

        result = [skill for skill, _ in matched[:max_skills]]

        # 사용 기록
        self.skill_history.append({
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "query": query,
            "matched": [s.name for s in result]
        })

        return result

    def build_system_prompt(
        self,
        skills: List[Skill],
        base_prompt: str = "",
        include_full: bool = True
    ) -> str:
        """
        스킬을 포함한 시스템 프롬프트 생성

        Progressive Disclosure:
        - 매칭된 스킬만 전체 지침 포함
        - 다른 스킬은 메타데이터만 포함 (선택적)
        """
        prompt_parts = []

        if base_prompt:
            prompt_parts.append(base_prompt)

        if skills:
            prompt_parts.append("\n# Active Skills\n")
            for skill in skills:
                prompt_parts.append(skill.get_prompt_section(include_full=include_full))

        # 사용 가능한 다른 스킬 목록 (Progressive Disclosure)
        other_skills = [s for s in self.list_skills() if s not in skills]
        if other_skills:
            prompt_parts.append("\n# Available Skills (activate by mentioning)\n")
            for skill in other_skills[:5]:  # 최대 5개만 표시
                prompt_parts.append(skill.get_prompt_section(include_full=False))

        return "\n".join(prompt_parts)

    def get_usage_stats(self) -> Dict[str, Any]:
        """스킬 사용 통계"""
        stats = defaultdict(int)
        for record in self.skill_history:
            for skill_name in record.get("matched", []):
                stats[skill_name] += 1

        return {
            "total_queries": len(self.skill_history),
            "skill_usage": dict(stats),
            "registered_skills": len(self.skills),
            "enabled_skills": len([s for s in self.skills.values() if s.enabled])
        }

    def create_skill_template(self, name: str, output_dir: str) -> str:
        """
        새 스킬 템플릿 생성

        init_skill.py 스크립트와 유사한 기능
        """
        output_path = Path(output_dir) / name
        output_path.mkdir(parents=True, exist_ok=True)

        # SKILL.md 템플릿
        skill_md = f"""---
name: {name}
description: [TODO: 이 스킬이 무엇을 하는지, 언제 사용해야 하는지 설명하세요]
---

# {name.replace('-', ' ').title()}

## Overview

[TODO: 1-2문장으로 이 스킬이 무엇을 가능하게 하는지 설명]

## When to Use

이 스킬은 다음과 같은 경우에 사용합니다:
- [TODO: 사용 시나리오 1]
- [TODO: 사용 시나리오 2]

## Instructions

[TODO: AI가 따라야 할 지침을 작성하세요]

## Examples

### Example 1
[TODO: 예시 추가]

## Resources

- scripts/: 실행 가능한 스크립트
- references/: 참조 문서
- assets/: 템플릿 및 에셋
"""

        (output_path / "SKILL.md").write_text(skill_md, encoding='utf-8')

        # 리소스 디렉토리 생성
        (output_path / "scripts").mkdir(exist_ok=True)
        (output_path / "references").mkdir(exist_ok=True)
        (output_path / "assets").mkdir(exist_ok=True)

        # 예제 스크립트
        example_script = f'''#!/usr/bin/env python3
"""
Example script for {name}
"""

def main():
    print("Hello from {name}!")

if __name__ == "__main__":
    main()
'''
        (output_path / "scripts" / "example.py").write_text(example_script, encoding='utf-8')

        logging.info(f"✅ 스킬 템플릿 생성: {output_path}")
        return str(output_path)


# 기본 스킬 디렉토리 경로 (파일 기반 로드)
BUILTIN_SKILLS_DIR = Path(__file__).parent / "skills"


# ============================================================================
# 유틸리티 & 인프라 (New)
# ============================================================================

class StructuredLogger:
    """
    JSON 형태의 구조화된 로깅
    """
    def __init__(self, name: str):
        self.logger = logging.getLogger(name)

    def info(self, message: str, **kwargs):
        self._log(logging.INFO, message, **kwargs)

    def error(self, message: str, **kwargs):
        self._log(logging.ERROR, message, **kwargs)

    def warning(self, message: str, **kwargs):
        self._log(logging.WARNING, message, **kwargs)

    def _log(self, level: int, message: str, **kwargs):
        log_data = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "message": message,
            **kwargs
        }
        # 실제 환경에서는 json.dumps 사용, 여기서는 가독성을 위해 포맷팅
        self.logger.log(level, f"[{level}] {json.dumps(log_data, ensure_ascii=False)}")

async def retry_with_backoff(
    func: Callable,
    max_retries: int = 3,
    base_delay: float = 1.0,
    max_delay: float = 60.0,
    exponential_base: float = 2.0,
    *args,
    **kwargs
) -> Any:
    """
    지수 백오프 재시도 로직
    """
    retries = 0
    while True:
        try:
            return await func(*args, **kwargs)
        except Exception as e:
            retries += 1
            if retries > max_retries:
                raise e

            delay = min(base_delay * (exponential_base ** (retries - 1)), max_delay)
            logging.warning(f"⚠️ 재시도 {retries}/{max_retries} ({delay:.2f}s 후): {e}")
            await asyncio.sleep(delay)



# ============================================================================
# 핵심 데이터 모델
# ============================================================================

class AgentRole(str, Enum):
    """
    Agent 역할 정의

    [수정] SUPERVISOR 추가 - Microsoft AutoGen 패턴
    기존: ASSISTANT, USER, SYSTEM, FUNCTION, ROUTER, ORCHESTRATOR
    추가: SUPERVISOR - 여러 에이전트를 감독하고 조율하는 역할
    """
    ASSISTANT = "assistant"
    USER = "user"
    SYSTEM = "system"
    FUNCTION = "function"
    ROUTER = "router"
    ORCHESTRATOR = "orchestrator"
    SUPERVISOR = "supervisor"  # 🆕 추가


class ExecutionStatus(str, Enum):
    """
    실행 상태 정의

    [수정] 승인 관련 상태 추가 - Human-in-the-loop 패턴
    기존: PENDING, RUNNING, COMPLETED, FAILED, PAUSED, WAITING_APPROVAL
    추가: APPROVED, REJECTED
    """
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    PAUSED = "paused"
    WAITING_APPROVAL = "waiting_approval"
    APPROVED = "approved"    # 🆕 추가
    REJECTED = "rejected"    # 🆕 추가


class ApprovalStatus(str, Enum):
    """
    승인 상태 정의

    [신규] Microsoft Agent Framework의 approval 패턴
    - PENDING: 승인 대기 중
    - APPROVED: 사용자가 승인함
    - REJECTED: 사용자가 거부함
    - AUTO_APPROVED: 자동 승인됨 (안전한 작업)
    """
    PENDING = "pending"
    APPROVED = "approved"
    REJECTED = "rejected"
    AUTO_APPROVED = "auto_approved"  # 🆕 자동 승인


class Message(BaseModel):
    """
    메시지 모델

    [수정] function_call 필드 추가
    - 함수 호출 정보를 저장하여 OpenAI Function Calling 지원
    """
    role: AgentRole
    content: str
    metadata: Dict[str, Any] = Field(default_factory=dict)
    timestamp: str = Field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    agent_name: Optional[str] = None
    function_call: Optional[Dict[str, Any]] = None  # 🆕 함수 호출 정보

    class Config:
        use_enum_values = True


class AgentState(BaseModel):
    """
    Agent 상태 - 체크포인팅 및 복원 지원

    [수정] pending_approvals, metrics 필드 추가
    - pending_approvals: 승인 대기 중인 요청 목록
    - metrics: 실행 메트릭 (시간, 토큰 등)
    """
    messages: List[Message] = Field(default_factory=list)
    current_node: str = "start"
    visited_nodes: List[str] = Field(default_factory=list)
    metadata: Dict[str, Any] = Field(default_factory=dict)
    session_id: str
    workflow_name: str = "default"
    execution_status: ExecutionStatus = ExecutionStatus.PENDING
    pending_approvals: List[Dict[str, Any]] = Field(default_factory=list)  # 🆕 승인 대기
    metrics: Dict[str, Any] = Field(default_factory=dict)  # 🆕 메트릭

    def add_message(self, role: AgentRole, content: str, agent_name: Optional[str] = None,
                   function_call: Optional[Dict[str, Any]] = None):
        """메시지 추가"""
        self.messages.append(Message(
            role=role,
            content=content,
            agent_name=agent_name,
            function_call=function_call
        ))

    def get_conversation_history(self, max_messages: int = 10) -> List[Message]:
        """최근 대화 기록"""
        return self.messages[-max_messages:]

    def add_pending_approval(self, approval_request: Dict[str, Any]):
        """
        승인 대기 요청 추가

        [신규] Human-in-the-loop 패턴 지원
        """
        self.pending_approvals.append(approval_request)
        self.execution_status = ExecutionStatus.WAITING_APPROVAL


class NodeResult(BaseModel):
    """
    노드 실행 결과

    [수정] requires_approval, approval_data 필드 추가
    - 승인이 필요한 작업인지 표시
    """
    node_name: str
    output: str
    next_node: Optional[str] = None
    metadata: Dict[str, Any] = Field(default_factory=dict)
    success: bool = True
    error: Optional[str] = None
    tokens_used: int = 0
    duration_ms: float = 0.0
    requires_approval: bool = False  # 🆕 승인 필요 여부
    approval_data: Optional[Dict[str, Any]] = None  # 🆕 승인 데이터


# ============================================================================
# AIFunction - Microsoft Agent Framework 패턴
# ============================================================================

class AIFunction(ABC):
    """
    AI Function 추상 클래스 - Microsoft Agent Framework 패턴

    [신규] OpenAI Function Calling을 위한 추상 클래스

    참조: https://github.com/microsoft/agent-framework/blob/main/python/samples/getting_started/tools/

    주요 기능:
    - get_schema(): OpenAI Function Calling 스키마 반환
    - invoke_with_metrics(): 메트릭과 함께 실행
    """

    def __init__(self, name: str, description: str, parameters: Optional[Dict[str, Any]] = None):
        self.name = name
        self.description = description
        self.parameters = parameters or {}
        self.execution_count = 0
        self.total_duration_ms = 0.0

    @abstractmethod
    async def execute(self, **kwargs) -> Any:
        """함수 실행"""
        pass

    def get_schema(self) -> Dict[str, Any]:
        """
        OpenAI Function Calling 스키마

        [신규] OpenAI API에 전달할 함수 스키마 생성
        """
        return {
            "name": self.name,
            "description": self.description,
            "parameters": self.parameters
        }

    async def invoke_with_metrics(self, **kwargs) -> tuple[Any, float]:
        """
        메트릭과 함께 실행

        [신규] 실행 시간 측정 및 메트릭 수집
        """
        start_time = time.time()
        result = await self.execute(**kwargs)
        duration_ms = (time.time() - start_time) * 1000

        self.execution_count += 1
        self.total_duration_ms += duration_ms

        return result, duration_ms


class ApprovalRequiredAIFunction(AIFunction):
    """
    Human-in-the-loop 승인이 필요한 함수

    [신규] Microsoft Agent Framework의 approval 패턴

    참조: https://github.com/microsoft/agent-framework/blob/main/python/samples/getting_started/tools/ai_tool_with_approval.py

    사용 시나리오:
    - 결제 처리
    - 데이터 삭제
    - 중요한 설정 변경
    - 외부 API 호출

    자동 승인:
    - auto_approve_threshold 설정 시 안전한 작업은 자동 승인
    - 예: 읽기 전용 작업, 낮은 금액의 결제 등
    """

    def __init__(self, base_function: AIFunction,
                 approval_callback: Optional[Callable] = None,
                 auto_approve_threshold: Optional[float] = None):
        super().__init__(
            name=f"{base_function.name}_approval_required",
            description=f"{base_function.description} (Requires Approval)",
            parameters=base_function.parameters
        )
        self.base_function = base_function
        self.approval_callback = approval_callback
        self.auto_approve_threshold = auto_approve_threshold

    async def execute(self, **kwargs) -> Dict[str, Any]:
        """승인 요청 생성"""
        approval_request = {
            "function_name": self.base_function.name,
            "arguments": kwargs,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "status": ApprovalStatus.PENDING,
            "description": self.description
        }

        # [신규] 자동 승인 임계값 확인
        if self.auto_approve_threshold and self._is_safe_operation(**kwargs):
            approval_request["status"] = ApprovalStatus.AUTO_APPROVED
            result = await self.base_function.execute(**kwargs)
            approval_request["result"] = result
            return approval_request

        # 승인 콜백 실행
        if self.approval_callback:
            approved = await self.approval_callback(approval_request)
            if approved:
                approval_request["status"] = ApprovalStatus.APPROVED
                result = await self.base_function.execute(**kwargs)
                approval_request["result"] = result
            else:
                approval_request["status"] = ApprovalStatus.REJECTED
                approval_request["result"] = "Operation rejected by user"

        return approval_request

    def _is_safe_operation(self, **kwargs) -> bool:
        """
        안전한 작업인지 확인 (예: 읽기 전용)

        [신규] 자동 승인 로직
        """
        # 읽기 전용 작업은 자동 승인 (예: get_, read_, list_ 로 시작)
        if self.base_function.name.startswith(("get_", "read_", "list_")):
            return True
        return False


# ============================================================================
# Memory Hook Provider 패턴 (Amazon Bedrock AgentCore 참조)
# ============================================================================

@dataclass
class ConversationMessage:
    """
    대화 메시지 모델 (AgentCore Memory 패턴)

    참조: https://github.com/awslabs/amazon-bedrock-agentcore-samples
    """
    content: str
    role: str  # USER, ASSISTANT, TOOL
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    agent_name: Optional[str] = None
    session_id: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


class MemoryHookProvider:
    """
    Memory Hook Provider - 자동 메모리 관리

    참조: amazon-bedrock-agentcore-samples/memory/hooks.py

    주요 기능:
    - 대화 기록 자동 저장/로드
    - 세션 기반 컨텍스트 관리
    - 네임스페이스 기반 메모리 분류

    사용법:
    ```python
    memory_hook = MemoryHookProvider(
        memory_store=memory_store,
        session_id="session-123",
        actor_id="user-456"
    )

    # 에이전트 초기화 시 컨텍스트 로드
    context = await memory_hook.on_agent_initialized(agent_name="assistant")

    # 메시지 추가 시 자동 저장
    await memory_hook.on_message_added(message, agent_name="assistant")
    ```
    """

    def __init__(
        self,
        memory_store: 'MemoryStore',
        session_id: str,
        actor_id: str,
        max_context_turns: int = 10,
        namespace: str = "/conversation"
    ):
        self.memory_store = memory_store
        self.session_id = session_id
        self.actor_id = actor_id
        self.max_context_turns = max_context_turns
        self.namespace = namespace
        self.conversation_history: List[ConversationMessage] = []
        self._logger = StructuredLogger("memory_hook")

    async def on_agent_initialized(self, agent_name: str) -> List[ConversationMessage]:
        """
        에이전트 초기화 시 최근 대화 기록 로드
        """
        try:
            key = f"{self.namespace}/{self.session_id}/history"
            data = await self.memory_store.load(key)

            if data:
                messages = data.get("messages", [])
                self.conversation_history = [
                    ConversationMessage(**msg) for msg in messages[-self.max_context_turns:]
                ]
                self._logger.info(
                    f"Loaded {len(self.conversation_history)} messages",
                    agent=agent_name,
                    session_id=self.session_id
                )

            return self.conversation_history
        except Exception as e:
            self._logger.error(f"Failed to load history: {e}")
            return []

    async def on_message_added(
        self,
        content: str,
        role: str,
        agent_name: Optional[str] = None
    ):
        """
        메시지 추가 시 자동 저장
        """
        message = ConversationMessage(
            content=content,
            role=role,
            agent_name=agent_name,
            session_id=self.session_id
        )

        self.conversation_history.append(message)

        # 저장
        try:
            key = f"{self.namespace}/{self.session_id}/history"
            await self.memory_store.save(key, {
                "messages": [{
                    "content": m.content,
                    "role": m.role,
                    "timestamp": m.timestamp.isoformat(),
                    "agent_name": m.agent_name,
                    "session_id": m.session_id
                } for m in self.conversation_history[-self.max_context_turns:]],
                "actor_id": self.actor_id,
                "updated_at": datetime.now(timezone.utc).isoformat()
            })
        except Exception as e:
            self._logger.error(f"Failed to save message: {e}")

    async def get_last_k_turns(self, k: int = 5) -> List[ConversationMessage]:
        """
        최근 k개 대화 턴 조회
        """
        return self.conversation_history[-k:]

    async def clear_session(self):
        """
        세션 데이터 삭제
        """
        key = f"{self.namespace}/{self.session_id}/history"
        await self.memory_store.delete(key)
        self.conversation_history = []
        self._logger.info("Session cleared", session_id=self.session_id)


class MemorySessionManager:
    """
    세션 기반 메모리 관리자 (AgentCore MemorySessionManager 패턴)

    참조: amazon-bedrock-agentcore-samples/memory/session_manager.py

    주요 기능:
    - 다중 세션 관리
    - 세션 간 컨텍스트 공유
    - 자동 세션 정리
    """

    def __init__(self, memory_store: 'MemoryStore', default_ttl_hours: int = 24):
        self.memory_store = memory_store
        self.default_ttl_hours = default_ttl_hours
        self._sessions: Dict[str, MemoryHookProvider] = {}
        self._logger = StructuredLogger("session_manager")

    def get_or_create_session(
        self,
        session_id: str,
        actor_id: str,
        namespace: str = "/conversation"
    ) -> MemoryHookProvider:
        """
        세션 조회 또는 생성
        """
        key = f"{actor_id}:{session_id}"

        if key not in self._sessions:
            self._sessions[key] = MemoryHookProvider(
                memory_store=self.memory_store,
                session_id=session_id,
                actor_id=actor_id,
                namespace=namespace
            )
            self._logger.info(
                "Created new session",
                session_id=session_id,
                actor_id=actor_id
            )

        return self._sessions[key]

    async def list_sessions(self, actor_id: Optional[str] = None) -> List[str]:
        """
        세션 목록 조회
        """
        sessions = []
        for key in self._sessions.keys():
            if actor_id is None or key.startswith(f"{actor_id}:"):
                sessions.append(key)
        return sessions

    async def cleanup_expired_sessions(self):
        """
        만료된 세션 정리
        """
        # 구현: TTL 기반 세션 정리
        pass


# ============================================================================
# MCP (Model Context Protocol) 통합
# ============================================================================

class MockMCPClient:
    """
    [신규] MCP 클라이언트 모의 구현 (데모용)
    """
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.tools = {
            "calculator": {
                "name": "calculator",
                "description": "Perform basic calculations",
                "parameters": {"type": "object", "properties": {"expression": {"type": "string"}}}
            },
            "web_search": {
                "name": "web_search",
                "description": "Search the web for information",
                "parameters": {"type": "object", "properties": {"query": {"type": "string"}}}
            }
        }

    async def list_tools(self) -> List[Dict[str, Any]]:
        return list(self.tools.values())

    async def call_tool(self, name: str, arguments: Dict[str, Any]) -> Any:
        if name == "calculator":
            return f"Calculated: {arguments.get('expression')} = 42 (Mock)"
        elif name == "web_search":
            return f"Search results for '{arguments.get('query')}': [Mock Result 1, Mock Result 2]"
        return f"Tool {name} executed with {arguments}"

class MCPTool:
    """
    MCP 서버와 통합하는 도구
    """

    def __init__(self, name: str, server_config: Dict[str, Any]):
        self.name = name
        self.server_config = server_config
        self.connected = False
        self.client: Optional[MockMCPClient] = None
        self.available_tools: List[Dict[str, Any]] = []

    async def connect(self):
        """
        MCP 서버 연결
        """
        try:
            logging.info(f"🔌 MCP 서버 연결 시도: {self.name}")
            # 실제 구현에서는 mcp.Client 사용
            self.client = MockMCPClient(self.server_config)
            self.available_tools = await self.client.list_tools()
            self.connected = True
            logging.info(f"✅ MCP 서버 연결 성공: {self.name}")
        except Exception as e:
            logging.error(f"❌ MCP 서버 연결 실패: {e}")
            raise

    async def disconnect(self):
        """MCP 서버 연결 해제"""
        if self.connected:
            logging.info(f"🔌 MCP 서버 연결 해제: {self.name}")
            self.connected = False
            self.client = None

    async def get_available_tools(self) -> List[Dict[str, Any]]:
        """사용 가능한 도구 목록"""
        if not self.connected:
            await self.connect()
        return self.available_tools

    async def invoke_tool(self, tool_name: str, **kwargs) -> Any:
        """MCP 도구 호출"""
        if not self.connected:
            raise RuntimeError("MCP 서버가 연결되지 않았습니다")

        logging.info(f"🛠️ MCP 도구 호출: {tool_name}")
        return await self.client.call_tool(tool_name, kwargs)


# ============================================================================
# 회로 차단기 패턴
# ============================================================================

class CircuitBreaker:
    """
    회로 차단기 - 장애 전파 방지

    [신규] 마이크로서비스 아키텍처의 핵심 패턴

    상태 전환:
    1. CLOSED (정상): 모든 요청 허용
    2. OPEN (차단): 실패 임계값 도달, 모든 요청 차단
    3. HALF_OPEN (반개방): 타임아웃 후 일부 요청 허용하여 테스트

    주요 파라미터:
    - failure_threshold: 연속 실패 임계값 (기본 5회)
    - timeout: OPEN 상태 유지 시간 (기본 60초)

    사용 시나리오:
    - 외부 API 호출
    - 데이터베이스 쿼리
    - LLM API 호출
    """

    def __init__(self, failure_threshold: int = 5, timeout: float = 60.0):
        self.failure_threshold = failure_threshold
        self.timeout = timeout
        self.failure_count = 0
        self.last_failure_time: Optional[float] = None
        self.state = "CLOSED"  # CLOSED, OPEN, HALF_OPEN

    async def call(self, func: Callable, *args, **kwargs):
        """
        회로 차단기를 통한 함수 호출

        [신규] 장애 격리 및 빠른 실패
        """
        if self.state == "OPEN":
            if time.time() - self.last_failure_time > self.timeout:
                self.state = "HALF_OPEN"
                logging.info("🔄 회로 차단기: HALF_OPEN 상태")
            else:
                raise RuntimeError("회로 차단기가 OPEN 상태입니다")

        try:
            result = await func(*args, **kwargs)
            if self.state == "HALF_OPEN":
                self.state = "CLOSED"
                self.failure_count = 0
                logging.info("✅ 회로 차단기: CLOSED 상태 복구")
            return result
        except Exception as e:
            self.failure_count += 1
            self.last_failure_time = time.time()

            if self.failure_count >= self.failure_threshold:
                self.state = "OPEN"
                logging.error(f"❌ 회로 차단기: OPEN 상태 ({self.failure_count} 실패)")

            raise e


# ============================================================================
# 메모리 저장소 - 향상된 버전
# ============================================================================

class MemoryStore(ABC):
    """
    메모리 저장소 인터페이스

    [수정] list_keys 메서드 추가
    """

    @abstractmethod
    async def save(self, key: str, data: Dict) -> None:
        pass

    @abstractmethod
    async def load(self, key: str) -> Optional[Dict]:
        pass

    @abstractmethod
    async def delete(self, key: str) -> None:
        pass

    @abstractmethod
    async def list_keys(self, pattern: str = "*") -> List[str]:
        """[신규] 키 목록 조회"""
        pass


class CachedMemoryStore(MemoryStore):
    """
    캐싱 메모리 저장소 - LRU 캐시

    [수정] LRU (Least Recently Used) 캐시 알고리즘 적용

    기존 vs 고도화:
    - 기존: 단순 접근 횟수 기반 캐싱
    - 고도화: LRU 알고리즘 + max_cache_size + access_order 추적

    LRU 캐시 장점:
    - 메모리 사용량 제한 (max_cache_size)
    - 최근 사용 데이터 우선 유지
    - 오래된 데이터 자동 제거
    """

    def __init__(self, max_cache_size: int = 100):
        self.data: Dict[str, Dict] = {}
        self.cache: Dict[str, Any] = {}
        self.access_count: Dict[str, int] = defaultdict(int)
        self.max_cache_size = max_cache_size  # 🆕 최대 캐시 크기
        self.access_order: List[str] = []  # 🆕 LRU 순서 추적

    async def save(self, key: str, data: Dict) -> None:
        self.data[key] = {
            'data': data,
            'timestamp': datetime.now(timezone.utc).isoformat(),
            'version': self.data.get(key, {}).get('version', 0) + 1  # 🆕 버전 관리
        }
        self.access_count[key] += 1

        # 자주 접근하는 데이터는 캐시에 저장
        if self.access_count[key] > 3:
            self._add_to_cache(key, data)

    async def load(self, key: str) -> Optional[Dict]:
        if key in self.cache:
            self._update_access_order(key)  # 🆕 LRU 순서 업데이트
            self.access_count[key] += 1
            return self.cache[key]

        if key in self.data:
            self.access_count[key] += 1
            return self.data[key]['data']
        return None

    async def delete(self, key: str) -> None:
        if key in self.data:
            del self.data[key]
        if key in self.cache:
            del self.cache[key]
            self.access_order.remove(key)  # 🆕 순서에서도 제거

    async def list_keys(self, pattern: str = "*") -> List[str]:
        """
        키 목록 반환 (간단한 패턴 매칭)

        [신규] 와일드카드 패턴 지원
        """
        if pattern == "*":
            return list(self.data.keys())
        # 간단한 와일드카드 지원
        import fnmatch
        return [k for k in self.data.keys() if fnmatch.fnmatch(k, pattern)]

    def _add_to_cache(self, key: str, data: Any):
        """
        LRU 캐시에 추가

        [신규] LRU 알고리즘 구현
        """
        if len(self.cache) >= self.max_cache_size:
            # 가장 오래된 항목 제거 (LRU)
            oldest_key = self.access_order.pop(0)
            del self.cache[oldest_key]

        self.cache[key] = data
        self._update_access_order(key)

    def _update_access_order(self, key: str):
        """
        접근 순서 업데이트

        [신규] LRU 순서 추적
        """
        if key in self.access_order:
            self.access_order.remove(key)
        self.access_order.append(key)


# ============================================================================
# 이벤트 시스템
# ============================================================================

class EventType(str, Enum):
    """
    이벤트 타입

    [신규] Pub-Sub 패턴을 위한 이벤트 타입 정의

    10가지 이벤트 타입:
    - Agent 생명주기: STARTED, COMPLETED, FAILED
    - Node 생명주기: NODE_STARTED, NODE_COMPLETED
    - 승인 관련: APPROVAL_REQUESTED, APPROVAL_GRANTED, APPROVAL_DENIED
    - 메시지: MESSAGE_RECEIVED, MESSAGE_SENT
    """
    AGENT_STARTED = "agent_started"
    AGENT_COMPLETED = "agent_completed"
    AGENT_FAILED = "agent_failed"
    NODE_STARTED = "node_started"
    NODE_COMPLETED = "node_completed"
    APPROVAL_REQUESTED = "approval_requested"
    APPROVAL_GRANTED = "approval_granted"
    APPROVAL_DENIED = "approval_denied"
    MESSAGE_RECEIVED = "message_received"
    MESSAGE_SENT = "message_sent"


class AgentEvent(BaseModel):
    """
    Agent 이벤트

    [신규] 이벤트 데이터 모델
    """
    event_type: EventType
    timestamp: str = Field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    agent_name: Optional[str] = None
    node_name: Optional[str] = None
    data: Dict[str, Any] = Field(default_factory=dict)


class EventBus:
    """
    이벤트 버스

    [신규] Pub-Sub 패턴 구현

    주요 기능:
    - subscribe(): 이벤트 구독
    - publish(): 이벤트 발행
    - get_event_history(): 이벤트 히스토리 조회

    사용 시나리오:
    - 로깅 및 모니터링
    - 알림 전송
    - 메트릭 수집
    - 워크플로우 조율

    예시:
    async def on_approval_requested(event):
        await send_slack_notification(event.data)

    event_bus.subscribe(EventType.APPROVAL_REQUESTED, on_approval_requested)
    """

    def __init__(self):
        self.subscribers: Dict[EventType, List[Callable]] = defaultdict(list)
        self.event_history: List[AgentEvent] = []

    def subscribe(self, event_type: EventType, handler: Callable):
        """이벤트 구독"""
        self.subscribers[event_type].append(handler)
        logging.info(f"📢 이벤트 구독: {event_type}")

    async def publish(self, event: AgentEvent):
        """이벤트 발행"""
        self.event_history.append(event)

        handlers = self.subscribers.get(event.event_type, [])
        for handler in handlers:
            try:
                if asyncio.iscoroutinefunction(handler):
                    await handler(event)
                else:
                    handler(event)
            except Exception as e:
                logging.error(f"❌ 이벤트 핸들러 오류: {e}")

    def get_event_history(self, event_type: Optional[EventType] = None,
                         limit: int = 100) -> List[AgentEvent]:
        """이벤트 히스토리 조회"""
        if event_type:
            filtered = [e for e in self.event_history if e.event_type == event_type]
            return filtered[-limit:]
        return self.event_history[-limit:]


# ============================================================================
# Agent 기본 클래스 - 향상된 버전
# ============================================================================

class Agent(ABC):
    """
    Agent 기본 클래스

    [수정] 여러 기능 추가
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
        model: str = DEFAULT_LLM_MODEL,  # 🆕 중앙 설정 사용
        temperature: float = 0.7,
        max_tokens: int = 1000,
        enable_streaming: bool = False,  # 🆕 스트리밍 옵션
        event_bus: Optional[EventBus] = None,  # 🆕 이벤트 버스
        circuit_breaker: Optional[CircuitBreaker] = None,  # 🆕 회로 차단기
        service_id: Optional[str] = None  # 🆕 서비스 ID (없으면 model 사용)
    ):
        self.name = name
        self.role = role
        self.system_prompt = system_prompt
        self.model = model
        self.service_id = service_id or model  # 🆕 service_id 저장
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.enable_streaming = enable_streaming
        self.event_bus = event_bus
        self.circuit_breaker = circuit_breaker or CircuitBreaker()

        # 🆕 모델에 따라 temperature 지원 여부 확인
        self.execution_settings = create_execution_settings(
            model=model,
            temperature=temperature,
            max_tokens=max_tokens,
            service_id=self.service_id
        )

        # 🆕 구조화된 로거
        self.logger = StructuredLogger(f"agent.{name}")

        # 🆕 메트릭
        self.total_executions = 0
        self.total_tokens = 0
        self.total_duration_ms = 0.0

    @abstractmethod
    async def execute(self, state: AgentState, kernel: Kernel) -> NodeResult:
        """Agent 실행"""
        pass

    async def _get_llm_response(self, kernel: Kernel, messages: List[Message],
                               use_streaming: bool = False) -> str:
        """
        LLM 응답 가져오기

        [수정] use_streaming 파라미터 추가
        """
        chat_completion = kernel.get_service(
            service_id=self.service_id,  # 🆕 service_id 사용
            type=ChatCompletionClientBase
        )

        history = ChatHistory()
        history.add_system_message(self.system_prompt)

        for msg in messages:
            if msg.role == AgentRole.USER:
                history.add_user_message(msg.content)
            elif msg.role == AgentRole.ASSISTANT:
                history.add_assistant_message(msg.content)

        settings = self.execution_settings
        settings.function_choice_behavior = None

        # 🆕 스트리밍 지원
        if use_streaming and self.enable_streaming:
            return await self._get_streaming_response(chat_completion, history, settings, kernel)
        else:
            # 🆕 재시도 로직 적용
            response = await retry_with_backoff(
                chat_completion.get_chat_message_content,
                max_retries=3,
                chat_history=history,
                settings=settings,
                kernel=kernel
            )
            return str(response)

    async def _get_streaming_response(self, chat_completion, history, settings, kernel) -> str:
        """
        스트리밍 응답 처리

        [신규] 실시간 토큰 단위 출력

        장점:
        - 긴 응답의 경우 사용자 경험 향상
        - 응답 대기 시간 감소
        - 실시간 피드백
        """
        full_response = []

        async for chunk in chat_completion.get_streaming_chat_message_contents(
            chat_history=history,
            settings=settings,
            kernel=kernel
        ):
            if chunk:
                content = str(chunk)
                full_response.append(content)
                # 실시간 출력 (옵션)
                print(content, end="", flush=True)

        print()  # 줄바꿈
        return "".join(full_response)

    async def _emit_event(self, event_type: EventType, data: Dict[str, Any]):
        """
        이벤트 발행

        [신규] EventBus를 통한 이벤트 발행
        """
        if self.event_bus:
            event = AgentEvent(
                event_type=event_type,
                agent_name=self.name,
                data=data
            )
            await self.event_bus.publish(event)


class SimpleAgent(Agent):
    """
    단순 대화 Agent - 향상된 버전

    [수정] 개선사항:
    1. 이벤트 발행 (AGENT_STARTED, AGENT_COMPLETED, AGENT_FAILED)
    2. 회로 차단기를 통한 호출
    3. 메트릭 수집 (total_executions, total_duration_ms)
    """

    async def execute(self, state: AgentState, kernel: Kernel) -> NodeResult:
        start_time = time.time()

        # 🆕 이벤트 발행
        await self._emit_event(EventType.AGENT_STARTED, {"node": self.name})

        try:
            recent_messages = state.get_conversation_history(max_messages=5)

            # 🆕 회로 차단기를 통한 호출
            response = await self.circuit_breaker.call(
                self._get_llm_response,
                kernel,
                recent_messages,
                self.enable_streaming
            )

            state.add_message(AgentRole.ASSISTANT, response, self.name)

            duration_ms = (time.time() - start_time) * 1000

            # 🆕 메트릭 업데이트
            self.total_executions += 1
            self.total_duration_ms += duration_ms

            # 🆕 완료 이벤트
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

            # 🆕 실패 이벤트
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


class ApprovalAgent(Agent):
    """
    승인이 필요한 작업을 수행하는 Agent

    [신규] Human-in-the-loop 패턴 구현

    참조: https://github.com/microsoft/agent-framework/blob/main/python/samples/getting_started/tools/ai_tool_with_approval.py

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
            # 사용자 입력에서 파라미터 추출
            recent_messages = state.get_conversation_history(max_messages=3)
            last_message = recent_messages[-1].content if recent_messages else ""

            # 승인 요청 생성
            approval_result = await self.approval_function.execute(input=last_message)

            if approval_result["status"] == ApprovalStatus.PENDING:
                # 승인 대기 상태
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
                # 승인됨 또는 자동 승인
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


class RouterAgent(Agent):
    """
    라우팅 Agent - 향상된 버전

    [수정] 개선사항:
    1. default_route 파라미터 추가
    2. routing_history 추적 (인텐트 분류 히스토리)
    3. 메타데이터에 confidence 추가
    """

    def __init__(self, *args, routes: Dict[str, str],
                 default_route: Optional[str] = None, **kwargs):
        super().__init__(*args, role=AgentRole.ROUTER, **kwargs)
        self.routes = routes
        self.default_route = default_route or list(routes.values())[0] if routes else None  # 🆕 기본 경로
        self.routing_history: List[Dict[str, Any]] = []  # 🆕 라우팅 히스토리

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

            # 🆕 라우팅 히스토리 저장
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
                metadata={"intent": intent, "confidence": 0.95}  # 🆕 신뢰도 추가
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


@dataclass
class InvestigationPlan:
    """
    Investigation Plan - 멀티 에이전트 조사 계획

    참조: amazon-bedrock-agentcore-samples/SRE-agent/supervisor.py
    """
    steps: List[str]
    agents_sequence: List[str]
    complexity: str = "simple"  # simple, complex
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

    개선된 패턴 (Amazon Bedrock AgentCore + Microsoft AutoGen 통합):
    - Investigation Plan 기반 체계적 실행
    - 메모리 컨텍스트 통합
    - 상세한 실행 추적 및 집계

    참조:
    - amazon-bedrock-agentcore-samples/SRE-agent/supervisor.py
    - Microsoft AutoGen의 GroupChat 패턴

    주요 기능:
    1. Investigation Plan 생성 및 실행
    2. 라운드 기반 협업 (max_rounds)
    3. 조기 종료 조건 ("TERMINATE" 키워드)
    4. 상세한 실행 로그 (execution_log)
    5. 응답 집계 (aggregate_responses)
    6. 메모리 컨텍스트 통합 (memory_hook)

    사용 시나리오:
    - Research Agent + Writer Agent 협업
    - Diagnostic + Remediation + Prevention 협업 (SRE 패턴)
    - 복잡한 multi-step 작업
    """

    def __init__(
        self,
        *args,
        sub_agents: List[Agent],
        max_rounds: int = 3,
        memory_hook: Optional['MemoryHookProvider'] = None,
        auto_approve_simple: bool = True,  # 간단한 계획 자동 실행
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
        """
        Investigation Plan 생성 (SRE Agent 패턴)

        쿼리를 분석하여 최적의 에이전트 실행 순서를 결정합니다.
        """
        agent_names = list(self.sub_agents.keys())
        agent_descriptions = ", ".join([
            f"{name}: {agent.system_prompt[:100]}..."
            for name, agent in self.sub_agents.items()
        ])

        query = state.messages[-1].content if state.messages else ""

        planning_prompt = f"""You are a supervisor planning an investigation.

Available Agents:
{agent_descriptions}

User Query: {query}

Create a plan with:
1. Steps to execute
2. Agent sequence (from: {', '.join(agent_names)})
3. Complexity (simple if ≤3 steps, complex otherwise)

Respond in JSON format:
{{
  "steps": ["step1", "step2"],
  "agents_sequence": ["agent1", "agent2"],
  "complexity": "simple",
  "reasoning": "brief explanation"
}}"""

        temp_messages = [Message(role=AgentRole.USER, content=planning_prompt)]
        response = await self._get_llm_response(kernel, temp_messages)

        try:
            # JSON 파싱
            import re
            json_match = re.search(r'\{[^{}]*\}', response, re.DOTALL)
            if json_match:
                plan_data = json.loads(json_match.group())
            else:
                plan_data = {
                    "steps": ["Execute query"],
                    "agents_sequence": [agent_names[0]] if agent_names else [],
                    "complexity": "simple",
                    "reasoning": "Default single-step plan"
                }
        except json.JSONDecodeError:
            plan_data = {
                "steps": ["Execute query"],
                "agents_sequence": [agent_names[0]] if agent_names else [],
                "complexity": "simple",
                "reasoning": "Fallback plan"
            }

        plan = InvestigationPlan(
            steps=plan_data.get("steps", []),
            agents_sequence=plan_data.get("agents_sequence", []),
            complexity=plan_data.get("complexity", "simple"),
            auto_execute=plan_data.get("complexity", "simple") == "simple" and self.auto_approve_simple,
            reasoning=plan_data.get("reasoning", "")
        )

        self.investigation_history.append(plan)
        logging.info(f"📋 Investigation Plan: {len(plan.steps)} steps, complexity={plan.complexity}")

        return plan

    async def aggregate_responses(
        self,
        responses: List[Dict[str, Any]],
        state: AgentState,
        kernel: Kernel
    ) -> str:
        """
        다중 에이전트 응답 집계 (SRE Agent 패턴)
        """
        if not responses:
            return "No responses to aggregate."

        responses_text = "\n\n".join([
            f"[{r['agent']}]:\n{r['output']}" for r in responses
        ])

        aggregation_prompt = f"""Summarize the following agent responses into a cohesive answer:

{responses_text}

Provide a clear, unified response that synthesizes all findings."""

        temp_messages = [Message(role=AgentRole.USER, content=aggregation_prompt)]
        return await self._get_llm_response(kernel, temp_messages)

    async def execute(self, state: AgentState, kernel: Kernel) -> NodeResult:
        start_time = time.time()

        try:
            # Investigation Plan 생성
            plan = await self.create_investigation_plan(state, kernel)

            responses = []
            current_round = 0

            # Agent 이름 목록
            agent_names = list(self.sub_agents.keys())
            agent_list_str = ", ".join(agent_names)

            while current_round < self.max_rounds:
                current_round += 1
                logging.info(f"🎯 Supervisor Round {current_round}/{self.max_rounds}")

                # 1. 다음 실행할 Agent 결정 (LLM 사용)
                history_text = "\n".join(responses[-3:]) if responses else "No history yet."

                decision_prompt = f"""
You are a Supervisor managing these agents: {agent_list_str}.
Current goal: {state.messages[-1].content if state.messages else 'Unknown'}

Recent history:
{history_text}

Decide the next step:
1. Select the next agent to act (respond with agent name).
2. If the task is complete, respond with "TERMINATE".

Respond with ONLY the agent name or "TERMINATE".
"""
                temp_messages = [Message(role=AgentRole.SYSTEM, content=decision_prompt)]
                decision = await self._get_llm_response(kernel, temp_messages)
                decision = decision.strip()

                logging.info(f"🤔 Supervisor Decision: {decision}")

                if "TERMINATE" in decision.upper():
                    logging.info("✅ Supervisor decided to terminate.")
                    break

                # 선택된 Agent 실행
                selected_agent_name = None
                for name in agent_names:
                    if name.lower() in decision.lower():
                        selected_agent_name = name
                        break

                if not selected_agent_name:
                    # 매칭 실패 시 기본적으로 첫 번째 또는 라운드 로빈 등 대안 필요
                    # 여기서는 로깅 후 계속 진행 (혹은 종료)
                    logging.warning(f"⚠️ Unknown agent selected: {decision}. Stopping.")
                    break

                agent = self.sub_agents[selected_agent_name]
                logging.info(f"  ➤ {selected_agent_name} 실행 중...")

                result = await agent.execute(state, kernel)

                # 🆕 실행 로그 기록
                execution_record = {
                    "round": current_round,
                    "agent": selected_agent_name,
                    "output": result.output,
                    "success": result.success,
                    "duration_ms": result.duration_ms
                }
                self.execution_log.append(execution_record)

                if result.success:
                    response_text = f"[Round {current_round} - {selected_agent_name}]\n{result.output}"
                    responses.append(response_text)
                    # 상태에 중간 결과 추가 (선택 사항)
                    # state.add_message(AgentRole.FUNCTION, result.output, selected_agent_name)

                # Agent가 명시적으로 종료 요청한 경우
                if "TERMINATE" in result.output.upper():
                    logging.info(f"✅ 조기 종료 요청 by {selected_agent_name}")
                    break

            # 응답 집계 (SRE Agent 패턴)
            if responses and len(responses) > 1:
                aggregated = await self.aggregate_responses(
                    self.execution_log, state, kernel
                )
                final_output = aggregated
            else:
                final_output = "\n\n".join(responses)

            duration_ms = (time.time() - start_time) * 1000

            # 최종 요약
            summary = f"Supervisor 실행 완료: {current_round}라운드"
            state.add_message(AgentRole.SUPERVISOR, summary, self.name)

            # Memory Hook 저장 (있는 경우)
            if self.memory_hook:
                await self.memory_hook.on_message_added(
                    content=final_output,
                    role="ASSISTANT",
                    agent_name=self.name
                )

            return NodeResult(
                node_name=self.name,
                output=final_output,
                success=True,
                duration_ms=duration_ms,
                metadata={
                    "rounds": current_round,
                    "agents": len(self.sub_agents),
                    "execution_log": self.execution_log,
                    "investigation_plan": plan.to_dict() if plan else None
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


# ============================================================================
# 그래프 기반 워크플로우 - 향상된 버전
# ============================================================================

class Node:
    """
    워크플로우 노드

    [수정] condition_func 파라미터 추가
    - 조건부 라우팅 지원 (LangGraph 패턴)
    """

    def __init__(self, name: str, agent: Agent,
                 edges: Optional[Dict[str, str]] = None,
                 condition_func: Optional[Callable] = None):  # 🆕 조건 함수
        self.name = name
        self.agent = agent
        self.edges = edges or {}
        self.condition_func = condition_func
        self.execution_count = 0  # 🆕 실행 횟수 추적

    async def execute(self, state: AgentState, kernel: Kernel) -> NodeResult:
        logging.info(f"📍 노드 실행: {self.name} (#{self.execution_count + 1})")

        result = await self.agent.execute(state, kernel)
        self.execution_count += 1

        # 🆕 조건부 라우팅
        if not result.next_node and self.edges:
            if self.condition_func:
                # 조건 함수로 다음 노드 결정
                next_node = await self.condition_func(state, result)
                result.next_node = self.edges.get(next_node, self.edges.get("default"))
            else:
                result.next_node = self.edges.get("default", None)

        state.visited_nodes.append(self.name)
        return result


class Graph:
    """
    워크플로우 그래프 - 조건부 라우팅 및 루프 지원

    [수정] 여러 기능 추가:
    1. loop_nodes: 루프 가능한 노드 집합
    2. add_conditional_edge(): 조건부 엣지 추가
    3. 무한 루프 방지 로직
    4. 상세한 실행 로그
    5. get_statistics(): 그래프 통계
    """

    def __init__(self, name: str = "workflow"):
        self.name = name
        self.nodes: Dict[str, Node] = {}
        self.start_node: Optional[str] = None
        self.end_nodes: Set[str] = set()
        self.loop_nodes: Set[str] = set()  # 🆕 루프 가능 노드

    def add_node(self, node: Node, allow_loop: bool = False):  # 🆕 allow_loop 파라미터
        """
        노드 추가

        [수정] allow_loop 파라미터로 루프 허용 여부 지정
        """
        self.nodes[node.name] = node
        if allow_loop:
            self.loop_nodes.add(node.name)
        logging.info(f"✅ 노드 추가: {node.name}")

    def add_edge(self, from_node: str, to_node: str, condition: str = "default"):
        if from_node not in self.nodes:
            raise ValueError(f"노드 '{from_node}'가 존재하지 않습니다.")
        self.nodes[from_node].edges[condition] = to_node
        logging.info(f"✅ 엣지 추가: {from_node} --[{condition}]--> {to_node}")

    def add_conditional_edge(self, from_node: str, condition_func: Callable):
        """
        조건부 엣지 추가

        [신규] LangGraph의 조건부 라우팅 패턴

        사용 예시:
        async def route_by_complexity(state, result):
            if "simple" in result.output.lower():
                return "simple"
            return "complex"

        graph.add_conditional_edge("analyzer", route_by_complexity)
        """
        if from_node not in self.nodes:
            raise ValueError(f"노드 '{from_node}'가 존재하지 않습니다.")
        self.nodes[from_node].condition_func = condition_func
        logging.info(f"✅ 조건부 엣지 추가: {from_node}")

    def set_start(self, node_name: str):
        self.start_node = node_name
        logging.info(f"✅ 시작 노드: {node_name}")

    def set_end(self, node_name: str):
        self.end_nodes.add(node_name)
        logging.info(f"✅ 종료 노드: {node_name}")

    async def execute(self, state: AgentState, kernel: Kernel,
                     max_iterations: int = 10) -> AgentState:
        """
        그래프 실행

        [수정] 개선사항:
        1. 승인 대기 처리
        2. 무한 루프 방지 (loop_nodes 체크)
        3. 상세한 로그 출력
        4. 실행 메트릭 수집
        """
        if not self.start_node:
            raise ValueError("시작 노드가 설정되지 않았습니다.")

        current_node = self.start_node
        iterations = 0

        logging.info(f"\n{'='*60}")
        logging.info(f"🚀 워크플로우 시작: {self.name}")
        logging.info(f"{'='*60}")
        state.execution_status = ExecutionStatus.RUNNING

        while current_node and iterations < max_iterations:
            iterations += 1
            state.current_node = current_node

            logging.info(f"\n▶️ Iteration {iterations}: {current_node}")

            node = self.nodes.get(current_node)
            if not node:
                logging.error(f"❌ 노드 '{current_node}'를 찾을 수 없습니다.")
                state.execution_status = ExecutionStatus.FAILED
                break

            # 🆕 무한 루프 방지 (같은 노드 재방문 체크)
            if current_node in state.visited_nodes and current_node not in self.loop_nodes:
                logging.warning(f"⚠️ 노드 재방문 감지: {current_node}")

            result = await node.execute(state, kernel)
            state.metadata[f"{current_node}_result"] = result.model_dump()

            # 🆕 승인 대기 처리
            if result.requires_approval:
                logging.info(f"⏸️ 승인 대기: {current_node}")
                state.execution_status = ExecutionStatus.WAITING_APPROVAL
                return state

            if not result.success:
                logging.error(f"❌ 노드 실행 실패: {result.error}")
                state.execution_status = ExecutionStatus.FAILED
                break

            # 종료 조건
            if current_node in self.end_nodes:
                logging.info(f"\n{'='*60}")
                logging.info(f"✅ 워크플로우 완료: {self.name}")
                logging.info(f"{'='*60}")
                state.execution_status = ExecutionStatus.COMPLETED
                break

            current_node = result.next_node

            if not current_node:
                state.execution_status = ExecutionStatus.COMPLETED
                break

        if iterations >= max_iterations:
            logging.warning(f"⚠️ 최대 반복 도달 ({max_iterations})")
            state.execution_status = ExecutionStatus.FAILED

        # 🆕 실행 통계
        state.metrics["total_iterations"] = iterations
        state.metrics["visited_nodes"] = len(state.visited_nodes)
        state.metrics["workflow_name"] = self.name

        return state

    def visualize(self) -> str:
        """
        그래프 시각화 (Mermaid 형식)

        [수정] loop_nodes 표시 개선
        """
        lines = []
        lines.append("```")
        lines.append("graph TD")

        # 노드 정의
        for node_name, node in self.nodes.items():
            if node_name == self.start_node:
                shape = f"{node_name}([🎬 START: {node_name}])"
            elif node_name in self.end_nodes:
                shape = f"{node_name}[🏁 END: {node_name}]"
            elif node_name in self.loop_nodes:  # 🆕 루프 노드 표시
                shape = f"{node_name}{{🔄 {node_name}}}"
            else:
                shape = f"{node_name}[{node_name}]"

            lines.append(f"    {shape}")

        # 엣지 정의
        for node_name, node in self.nodes.items():
            for condition, target in node.edges.items():
                if condition == "default":
                    lines.append(f"    {node_name} --> {target}")
                else:
                    lines.append(f"    {node_name} -->|{condition}| {target}")

        lines.append("```")
        return "\n".join(lines)

    def get_statistics(self) -> Dict[str, Any]:
        """
        그래프 통계

        [신규] 워크플로우 실행 통계
        """
        return {
            "name": self.name,
            "total_nodes": len(self.nodes),
            "start_node": self.start_node,
            "end_nodes": list(self.end_nodes),
            "loop_nodes": list(self.loop_nodes),
            "total_edges": sum(len(node.edges) for node in self.nodes.values()),
            "node_execution_counts": {
                name: node.execution_count
                for name, node in self.nodes.items()
            }
        }


# ============================================================================
# 상태 관리 - 향상된 버전
# ============================================================================

class StateManager:
    """
    상태 관리자 - 버전 관리 및 롤백 지원

    [수정] 여러 기능 추가:
    1. 버전 관리 (state_versions)
    2. load_state(version): 특정 버전 로드
    3. save_checkpoint(tag): 태그와 함께 체크포인트 저장
    4. restore_checkpoint(tag): 특정 태그 복원
    5. list_checkpoints(): 체크포인트 목록
    6. rollback(steps): 이전 상태로 롤백
    """

    def __init__(self, memory_store: MemoryStore, checkpoint_dir: Optional[str] = None):
        self.memory_store = memory_store
        self.checkpoint_dir = checkpoint_dir
        self.state_versions: Dict[str, List[str]] = defaultdict(list)  # 🆕 버전 추적

        if checkpoint_dir and not os.path.exists(checkpoint_dir):
            os.makedirs(checkpoint_dir)

    async def save_state(self, state: AgentState):
        """
        상태 저장

        [수정] 버전 추적 추가
        """
        state_dict = state.model_dump()
        await self.memory_store.save(f"state:{state.session_id}", state_dict)

        # 🆕 버전 추적
        version_key = f"state:{state.session_id}:v{len(self.state_versions[state.session_id])}"
        await self.memory_store.save(version_key, state_dict)
        self.state_versions[state.session_id].append(version_key)

    async def load_state(self, session_id: str, version: Optional[int] = None) -> Optional[AgentState]:
        """
        상태 로드 (특정 버전 지원)

        [수정] version 파라미터 추가
        """
        if version is not None:
            # 🆕 특정 버전 로드
            version_key = f"state:{session_id}:v{version}"
            data = await self.memory_store.load(version_key)
        else:
            # 최신 버전 로드
            data = await self.memory_store.load(f"state:{session_id}")

        if data:
            return AgentState(**data)
        return None

    async def save_checkpoint(self, state: AgentState, tag: Optional[str] = None) -> str:
        """
        체크포인트 저장

        [수정] tag 파라미터 추가
        """
        if not self.checkpoint_dir:
            raise ValueError("체크포인트 디렉토리 미설정")

        timestamp = datetime.now(timezone.utc).isoformat().replace(':', '-').replace('.', '-')
        tag_suffix = f"_{tag}" if tag else ""  # 🆕 태그 접미사
        checkpoint_file = os.path.join(
            self.checkpoint_dir,
            f"{state.session_id}_{timestamp}{tag_suffix}.json"
        )

        with open(checkpoint_file, 'w', encoding='utf-8') as f:
            json.dump(state.model_dump(), f, ensure_ascii=False, indent=2)

        logging.info(f"💾 체크포인트 저장: {checkpoint_file}")
        return checkpoint_file

    async def restore_checkpoint(self, session_id: str, tag: Optional[str] = None) -> Optional[AgentState]:
        """
        체크포인트 복원

        [수정] tag 파라미터 추가
        """
        if not self.checkpoint_dir:
            return None

        checkpoints = [
            f for f in os.listdir(self.checkpoint_dir)
            if f.startswith(session_id) and f.endswith('.json')
        ]

        # 🆕 태그 필터링
        if tag:
            checkpoints = [f for f in checkpoints if tag in f]

        if not checkpoints:
            return None

        latest = os.path.join(self.checkpoint_dir, sorted(checkpoints)[-1])

        with open(latest, 'r', encoding='utf-8') as f:
            data = json.load(f)

        logging.info(f"📂 체크포인트 복원: {latest}")
        return AgentState(**data)

    async def list_checkpoints(self, session_id: str) -> List[str]:
        """
        체크포인트 목록

        [신규] 저장된 체크포인트 목록 조회
        """
        if not self.checkpoint_dir or not os.path.exists(self.checkpoint_dir):
            return []

        checkpoints = [
            f for f in os.listdir(self.checkpoint_dir)
            if f.startswith(session_id) and f.endswith('.json')
        ]
        return sorted(checkpoints)

    async def rollback(self, session_id: str, steps: int = 1) -> Optional[AgentState]:
        """
        이전 상태로 롤백

        [신규] 버전 기반 롤백

        사용 예시:
        # 1단계 이전으로 롤백
        state = await state_manager.rollback(session_id, steps=1)

        # 3단계 이전으로 롤백
        state = await state_manager.rollback(session_id, steps=3)
        """
        versions = self.state_versions.get(session_id, [])
        if len(versions) < steps:
            logging.warning(f"⚠️ 롤백 불가: {steps}단계 이전 버전 없음")
            return None

        target_version = len(versions) - steps - 1
        return await self.load_state(session_id, version=target_version)


# ============================================================================
# 통합 프레임워크 - Enterprise Edition
# ============================================================================

class UnifiedAgentFramework:
    """
    통합 Agent 프레임워크 - Enterprise Edition

    간편한 사용법:
    ```python
    # 1. 가장 간단한 방법 (환경변수에서 자동 로드)
    framework = UnifiedAgentFramework.create()

    # 2. 설정 객체 사용
    config = FrameworkConfig.from_env()
    framework = UnifiedAgentFramework.create(config)

    # 3. 빠른 질의응답
    response = await framework.quick_chat("안녕하세요!")

    # 4. 워크플로우 실행
    state = await framework.run("session-1", "simple_chat", "질문입니다")

    # 5. Skills 기반 에이전트 (NEW!)
    agent = framework.create_skilled_agent("coder", skills=["python-expert"])
    ```

    주요 기능:
    - MCP 도구 관리
    - 이벤트 시스템 (Pub-Sub)
    - 전역 메트릭 수집
    - 체크포인트 및 롤백
    - Human-in-the-loop 승인
    - Skills 시스템 (Anthropic 패턴)
    """

    def __init__(
        self,
        kernel: Kernel,
        config: Optional[FrameworkConfig] = None,
        memory_store: Optional[MemoryStore] = None,
        checkpoint_dir: str = "./checkpoints",
        enable_telemetry: bool = True,
        enable_events: bool = True,
        skill_dirs: Optional[List[str]] = None,  # Skills 디렉토리
        load_builtin_skills: bool = True  # 기본 스킬 로드
    ):
        self.kernel = kernel
        self.config = config or FrameworkConfig()
        self.memory_store = memory_store or CachedMemoryStore(max_cache_size=self.config.max_cache_size)
        self.state_manager = StateManager(self.memory_store, checkpoint_dir)
        self.graphs: Dict[str, Graph] = {}
        self.mcp_tools: Dict[str, MCPTool] = {}
        self.event_bus = EventBus() if enable_events else None

        # Skills 시스템 초기화
        self.skill_manager = SkillManager(skill_dirs)
        if load_builtin_skills:
            self._load_builtin_skills()

        if enable_telemetry:
            self.tracer = trace.get_tracer(__name__)
        else:
            self.tracer = None

        self.global_metrics = {
            "total_workflows": 0,
            "total_executions": 0,
            "total_failures": 0,
            "start_time": datetime.now(timezone.utc).isoformat()
        }

    def _load_builtin_skills(self):
        """
        기본 제공 스킬 로드 (SKILL.md 파일 기반)

        skills/ 디렉토리에서 SKILL.md 파일을 읽어 스킬을 로드합니다.
        """
        if BUILTIN_SKILLS_DIR.exists():
            loaded = self.skill_manager.load_skills_from_directory(str(BUILTIN_SKILLS_DIR))
            logging.info(f"📚 SKILL.md 기반 스킬 {loaded}개 로드 완료 (from {BUILTIN_SKILLS_DIR})")
        else:
            logging.warning(f"⚠️ 기본 스킬 디렉토리가 없습니다: {BUILTIN_SKILLS_DIR}")
            logging.info("💡 'skills' 디렉토리를 생성하고 SKILL.md 파일을 추가하세요.")

    @classmethod
    def create(
        cls,
        config: Optional[FrameworkConfig] = None,
        skill_dirs: Optional[List[str]] = None,
        load_builtin_skills: bool = True
    ) -> 'UnifiedAgentFramework':
        """
        프레임워크 간편 생성 (권장)

        사용법:
        ```python
        # 환경변수에서 자동 로드
        framework = UnifiedAgentFramework.create()

        # 커스텀 설정 + 스킬 디렉토리
        framework = UnifiedAgentFramework.create(
            skill_dirs=["./my_skills", "./team_skills"]
        )
        ```
        """
        if config is None:
            config = FrameworkConfig.from_env()

        config.validate()

        # Kernel 초기화
        kernel = Kernel()
        chat_service = AzureChatCompletion(
            deployment_name=config.deployment_name,
            api_key=config.api_key,
            endpoint=config.endpoint,
            service_id=config.deployment_name,  # deployment_name과 동일하게 설정
            api_version=config.api_version
        )
        kernel.add_service(chat_service)

        return cls(
            kernel=kernel,
            config=config,
            checkpoint_dir=config.checkpoint_dir,
            enable_telemetry=config.enable_telemetry,
            enable_events=config.enable_events,
            skill_dirs=skill_dirs,
            load_builtin_skills=load_builtin_skills
        )

    async def quick_chat(self, message: str, system_prompt: str = "You are a helpful assistant.") -> str:
        """
        빠른 질의응답 (워크플로우 없이)

        사용법:
        ```python
        response = await framework.quick_chat("파이썬이란 무엇인가요?")
        print(response)
        ```
        """
        # 임시 워크플로우가 없으면 생성
        if "_quick_chat" not in self.graphs:
            self.create_simple_workflow("_quick_chat", system_prompt)

        session_id = f"quick-{int(time.time())}"
        state = await self.run(session_id, "_quick_chat", message)

        # 마지막 어시스턴트 메시지 반환
        for msg in reversed(state.messages):
            if msg.role == AgentRole.ASSISTANT:
                return msg.content
        return ""

    def create_simple_workflow(self, name: str, system_prompt: str = "You are a helpful assistant.") -> Graph:
        """
        간단한 대화 워크플로우 생성

        사용법:
        ```python
        workflow = framework.create_simple_workflow("my_assistant", "너는 한국어 선생님이야.")
        ```
        """
        graph = self.create_graph(name)

        agent = SimpleAgent(
            name="assistant",
            system_prompt=system_prompt,
            model=self.config.model,
            temperature=self.config.temperature,
            max_tokens=self.config.max_tokens,
            enable_streaming=self.config.enable_streaming,
            event_bus=self.event_bus,
            service_id=self.config.deployment_name  # 🆕 deployment_name 사용
        )

        graph.add_node(Node("assistant", agent))
        graph.set_start("assistant")
        graph.set_end("assistant")

        return graph

    def create_router_workflow(
        self,
        name: str,
        routes: Dict[str, Dict[str, str]]
    ) -> Graph:
        """
        라우팅 워크플로우 생성

        사용법:
        ```python
        workflow = framework.create_router_workflow(
            "customer_service",
            routes={
                "order": {"prompt": "주문 전문가입니다."},
                "support": {"prompt": "기술 지원 전문가입니다."},
                "general": {"prompt": "일반 상담원입니다."}
            }
        )
        ```
        """
        graph = self.create_graph(name)

        # 라우터 생성
        router = RouterAgent(
            name="router",
            system_prompt="Classify user intent accurately.",
            model=self.config.model,
            routes={k: f"{k}_agent" for k in routes.keys()},
            event_bus=self.event_bus,
            service_id=self.config.deployment_name  # 🆕 deployment_name 사용
        )
        graph.add_node(Node("router", router))
        graph.set_start("router")

        # 각 라우트별 에이전트 생성
        for route_name, route_config in routes.items():
            agent = SimpleAgent(
                name=f"{route_name}_agent",
                system_prompt=route_config.get("prompt", f"You handle {route_name} inquiries."),
                model=self.config.model,
                event_bus=self.event_bus,
                service_id=self.config.deployment_name  # 🆕 deployment_name 사용
            )
            graph.add_node(Node(f"{route_name}_agent", agent))
            graph.set_end(f"{route_name}_agent")

        return graph

    def create_skilled_agent(
        self,
        name: str,
        skills: Optional[List[str]] = None,
        base_prompt: str = "",
        auto_detect_skills: bool = True
    ) -> SimpleAgent:
        """
        Skills 기반 에이전트 생성

        사용법:
        ```python
        # 특정 스킬 지정
        agent = framework.create_skilled_agent(
            "coder",
            skills=["python-expert", "api-developer"]
        )

        # 자동 스킬 감지 (쿼리 기반)
        agent = framework.create_skilled_agent(
            "assistant",
            auto_detect_skills=True
        )
        ```
        """
        # 스킬 가져오기
        skill_objects = []
        if skills:
            for skill_name in skills:
                skill = self.skill_manager.get_skill(skill_name)
                if skill:
                    skill_objects.append(skill)
                else:
                    logging.warning(f"스킬을 찾을 수 없습니다: {skill_name}")

        # 시스템 프롬프트 구성
        system_prompt = self.skill_manager.build_system_prompt(
            skill_objects,
            base_prompt=base_prompt,
            include_full=True
        )

        agent = SimpleAgent(
            name=name,
            system_prompt=system_prompt,
            model=self.config.model,
            temperature=self.config.temperature,
            max_tokens=self.config.max_tokens,
            enable_streaming=self.config.enable_streaming,
            event_bus=self.event_bus,
            service_id=self.config.deployment_name  # 🆕 deployment_name 사용
        )

        # 자동 스킬 감지 메타데이터 추가
        agent._auto_detect_skills = auto_detect_skills
        agent._skill_manager = self.skill_manager

        return agent

    def create_skill_workflow(
        self,
        name: str,
        skills: List[str],
        base_prompt: str = "You are a helpful assistant."
    ) -> Graph:
        """
        Skills 기반 워크플로우 생성

        사용법:
        ```python
        workflow = framework.create_skill_workflow(
            "data_pipeline",
            skills=["python-expert", "data-analyst"],
            base_prompt="데이터 처리 전문가입니다."
        )
        ```
        """
        graph = self.create_graph(name)

        agent = self.create_skilled_agent(
            name="skilled_assistant",
            skills=skills,
            base_prompt=base_prompt
        )

        graph.add_node(Node("skilled_assistant", agent))
        graph.set_start("skilled_assistant")
        graph.set_end("skilled_assistant")

        return graph

    async def smart_chat(
        self,
        message: str,
        base_prompt: str = "You are a helpful assistant.",
        max_skills: int = 2
    ) -> str:
        """
        스마트 질의응답 - 쿼리에 맞는 스킬 자동 활성화

        Progressive Disclosure 적용:
        - 메시지 분석하여 관련 스킬 자동 매칭
        - 매칭된 스킬의 지침을 시스템 프롬프트에 포함

        사용법:
        ```python
        # 자동으로 python-expert 스킬이 활성화됨
        response = await framework.smart_chat("파이썬으로 웹 크롤러 만들어줘")
        ```
        """
        # 스킬 매칭
        matched_skills = self.skill_manager.match_skills(
            message,
            threshold=0.2,
            max_skills=max_skills
        )

        if matched_skills:
            skill_names = [s.name for s in matched_skills]
            logging.info(f"🎯 매칭된 스킬: {', '.join(skill_names)}")

        # 동적 워크플로우 생성
        workflow_name = f"_smart_chat_{int(time.time())}"
        self.create_skill_workflow(
            workflow_name,
            skills=[s.name for s in matched_skills],
            base_prompt=base_prompt
        )

        session_id = f"smart-{int(time.time())}"
        state = await self.run(session_id, workflow_name, message)

        # 마지막 어시스턴트 메시지 반환
        for msg in reversed(state.messages):
            if msg.role == AgentRole.ASSISTANT:
                return msg.content
        return ""

    def create_graph(self, name: str) -> Graph:
        """워크플로우 그래프 생성"""
        graph = Graph(name)
        self.graphs[name] = graph
        logging.info(f"🎨 그래프 생성: {name}")
        return graph

    def register_mcp_tool(self, tool: MCPTool):
        """
        MCP 도구 등록

        [신규] MCP 서버 연동
        """
        self.mcp_tools[tool.name] = tool
        logging.info(f"🔧 MCP 도구 등록: {tool.name}")

    async def run(
        self,
        session_id: str,
        workflow_name: str,
        user_message: str = "",
        restore_from_checkpoint: bool = False,
        checkpoint_tag: Optional[str] = None  # 🆕 태그 지원
    ) -> AgentState:
        """
        워크플로우 실행

        [수정] 개선사항:
        1. checkpoint_tag 파라미터 추가
        2. 실행 메트릭 수집
        3. 자동 체크포인트 (완료 시)
        4. 에러 핸들링 강화
        """

        # 상태 복원
        if restore_from_checkpoint:
            state = await self.state_manager.restore_checkpoint(session_id, tag=checkpoint_tag)
            if not state:
                logging.warning("⚠️ 체크포인트 복원 실패, 새 세션 시작")
                state = None
        else:
            state = await self.state_manager.load_state(session_id)

        if not state:
            state = AgentState(session_id=session_id, workflow_name=workflow_name)
            logging.info(f"🆕 새 세션 시작: {session_id}")

        if user_message:
            state.add_message(AgentRole.USER, user_message)
            # 🆕 이벤트 발행
            if self.event_bus:
                await self.event_bus.publish(AgentEvent(
                    event_type=EventType.MESSAGE_RECEIVED,
                    data={"content": user_message}
                ))

        graph = self.graphs.get(workflow_name)
        if not graph:
            raise ValueError(f"워크플로우 '{workflow_name}'를 찾을 수 없습니다.")

        # 실행
        start_time = time.time()
        self.global_metrics["total_executions"] += 1

        try:
            if self.tracer:
                with self.tracer.start_as_current_span("workflow_execution") as span:
                    span.set_attribute("session_id", session_id)
                    span.set_attribute("workflow_name", workflow_name)
                    state = await graph.execute(state, self.kernel)
                    span.set_attribute("status", state.execution_status.value)
                    span.set_attribute("iterations", state.metrics.get("total_iterations", 0))
            else:
                state = await graph.execute(state, self.kernel)

            # 🆕 실행 메트릭 저장
            execution_time = (time.time() - start_time) * 1000
            state.metrics["execution_time_ms"] = execution_time
            state.metrics["success"] = state.execution_status == ExecutionStatus.COMPLETED

        except Exception as e:
            logging.error(f"❌ 워크플로우 실행 오류: {e}")
            self.global_metrics["total_failures"] += 1
            state.execution_status = ExecutionStatus.FAILED
            state.metadata["error"] = str(e)

        # 상태 저장
        await self.state_manager.save_state(state)

        # 🆕 자동 체크포인트 (완료 시)
        if state.execution_status == ExecutionStatus.COMPLETED:
            await self.state_manager.save_checkpoint(state, tag="auto")

        return state

    async def approve_pending_request(self, session_id: str, request_id: int,
                                     approved: bool) -> AgentState:
        """
        대기 중인 승인 요청 처리

        [신규] Human-in-the-loop 승인 처리
        """
        state = await self.state_manager.load_state(session_id)
        if not state:
            raise ValueError(f"세션 '{session_id}'를 찾을 수 없습니다.")

        if request_id >= len(state.pending_approvals):
            raise ValueError(f"승인 요청 #{request_id}가 존재하지 않습니다.")

        approval = state.pending_approvals[request_id]
        approval["status"] = ApprovalStatus.APPROVED if approved else ApprovalStatus.REJECTED
        approval["approved_at"] = datetime.now(timezone.utc).isoformat()

        if approved:
            # 승인됨 - 워크플로우 계속 실행
            state.execution_status = ExecutionStatus.RUNNING
            if self.event_bus:
                await self.event_bus.publish(AgentEvent(
                    event_type=EventType.APPROVAL_GRANTED,
                    data=approval
                ))
        else:
            # 거부됨
            state.execution_status = ExecutionStatus.FAILED
            if self.event_bus:
                await self.event_bus.publish(AgentEvent(
                    event_type=EventType.APPROVAL_DENIED,
                    data=approval
                ))

        await self.state_manager.save_state(state)
        return state

    def visualize_workflow(self, workflow_name: str) -> str:
        """워크플로우 시각화"""
        graph = self.graphs.get(workflow_name)
        if not graph:
            return f"❌ 워크플로우 '{workflow_name}'를 찾을 수 없습니다."
        return graph.visualize()

    def get_workflow_stats(self, workflow_name: str) -> Dict[str, Any]:
        """
        워크플로우 통계

        [신규] 그래프 실행 통계
        """
        graph = self.graphs.get(workflow_name)
        if not graph:
            return {}
        return graph.get_statistics()

    def get_global_metrics(self) -> Dict[str, Any]:
        """
        전역 메트릭

        [신규] 프레임워크 전체 메트릭
        """
        return {
            **self.global_metrics,
            "total_workflows": len(self.graphs),
            "total_mcp_tools": len(self.mcp_tools),
            "uptime_seconds": (
                datetime.now(timezone.utc) -
                datetime.fromisoformat(self.global_metrics["start_time"])
            ).total_seconds()
        }

    async def cleanup(self):
        """
        리소스 정리

        [신규] 프레임워크 종료 시 리소스 해제
        """
        logging.info("🧹 프레임워크 정리 시작")

        # MCP 도구 연결 해제
        for tool in self.mcp_tools.values():
            await tool.disconnect()

        logging.info("✅ 프레임워크 정리 완료")


# ============================================================================
# OpenTelemetry 설정
# ============================================================================

def setup_telemetry(service_name: str = "UnifiedAgentFramework",
                   enable_console: bool = False):
    """OpenTelemetry 설정"""
    try:
        resource = Resource.create({"service.name": service_name})
        provider = TracerProvider(resource=resource)

        if enable_console:
            processor = BatchSpanProcessor(ConsoleSpanExporter())
            provider.add_span_processor(processor)

        trace.set_tracer_provider(provider)
        logging.info(f"✅ OpenTelemetry 설정: {service_name}")
    except Exception as e:
        logging.warning(f"⚠️ OpenTelemetry 설정 실패: {e}")


# ============================================================================
# 데모 함수들 - 학습용 4가지 데모
# ============================================================================

async def demo_simple_chat(framework: UnifiedAgentFramework):
    """
    데모 1: 단순 대화 Agent
    """
    print("\n" + "="*60)
    print("📚 데모 1: 단순 대화 Agent")
    print("="*60)

    # 간편 메서드 사용
    framework.create_simple_workflow(
        "simple_chat",
        "You are a helpful AI assistant. Answer questions clearly and concisely."
    )

    print("\n워크플로우 구조:")
    print(framework.visualize_workflow("simple_chat"))


async def demo_routing_workflow(framework: UnifiedAgentFramework):
    """
    데모 2: 라우팅 워크플로우 (인텐트 기반)
    """
    print("\n" + "="*60)
    print("📚 데모 2: 인텐트 기반 라우팅")
    print("="*60)

    # 간편 메서드 사용
    framework.create_router_workflow(
        "routing_workflow",
        routes={
            "order": {"prompt": "You are an order specialist. Help with ordering and purchases."},
            "support": {"prompt": "You are a support specialist. Help troubleshoot and resolve issues."},
            "general": {"prompt": "You are a general assistant. Answer various questions."}
        }
    )

    print("\n워크플로우 구조:")
    print(framework.visualize_workflow("routing_workflow"))


async def demo_supervisor_workflow(framework: UnifiedAgentFramework):
    """
    데모 3: Supervisor 패턴 (멀티 에이전트 협업)
    """
    print("\n" + "="*60)
    print("📚 데모 3: Supervisor Multi-Agent 협업")
    print("="*60)

    graph = framework.create_graph("supervisor_workflow")
    config = framework.config

    # Sub-agents
    research_agent = SimpleAgent(
        name="researcher",
        system_prompt="You are a research specialist. Gather and analyze information.",
        model=config.model,
        event_bus=framework.event_bus,
        service_id=config.deployment_name  # 🆕 deployment_name 사용
    )

    writer_agent = SimpleAgent(
        name="writer",
        system_prompt="You are a content writer. Create clear, engaging content.",
        model=config.model,
        event_bus=framework.event_bus,
        service_id=config.deployment_name  # 🆕 deployment_name 사용
    )

    # Supervisor
    supervisor = SupervisorAgent(
        name="supervisor",
        system_prompt="Coordinate research and writing tasks.",
        model=config.model,
        sub_agents=[research_agent, writer_agent],
        max_rounds=2,
        event_bus=framework.event_bus,
        service_id=config.deployment_name  # 🆕 deployment_name 사용
    )

    graph.add_node(Node("supervisor", supervisor))
    graph.set_start("supervisor")
    graph.set_end("supervisor")

    print("\n워크플로우 구조:")
    print(framework.visualize_workflow("supervisor_workflow"))


async def demo_conditional_workflow(framework: UnifiedAgentFramework):
    """
    데모 4: 조건부 라우팅 (복잡도 기반 분기)
    """
    print("\n" + "="*60)
    print("📚 데모 4: 조건부 라우팅 및 루프")
    print("="*60)

    graph = framework.create_graph("conditional_workflow")
    config = framework.config

    # Agents
    analyzer = SimpleAgent(
        name="analyzer",
        system_prompt="Analyze the complexity of the user's question. Respond with SIMPLE or COMPLEX.",
        model=config.model,
        event_bus=framework.event_bus,
        service_id=config.deployment_name  # 🆕 deployment_name 사용
    )

    simple_handler = SimpleAgent(
        name="simple_handler",
        system_prompt="Answer simple questions directly and briefly.",
        model=config.model,
        event_bus=framework.event_bus,
        service_id=config.deployment_name  # 🆕 deployment_name 사용
    )

    complex_handler = SimpleAgent(
        name="complex_handler",
        system_prompt="Provide detailed, comprehensive answers to complex questions.",
        model=config.model,
        max_tokens=2000,
        event_bus=framework.event_bus,
        service_id=config.deployment_name  # 🆕 deployment_name 사용
    )

    # 조건부 라우팅 함수
    async def route_by_complexity(state: AgentState, result: NodeResult) -> str:
        """복잡도에 따라 라우팅"""
        output_lower = result.output.lower()
        return "simple" if "simple" in output_lower else "complex"

    # Build Graph
    analyzer_node = Node(
        "analyzer",
        analyzer,
        edges={"simple": "simple_handler", "complex": "complex_handler"}
    )
    analyzer_node.condition_func = route_by_complexity

    graph.add_node(analyzer_node)
    graph.add_node(Node("simple_handler", simple_handler))
    graph.add_node(Node("complex_handler", complex_handler))

    graph.set_start("analyzer")
    graph.set_end("simple_handler")
    graph.set_end("complex_handler")

    print("\n워크플로우 구조:")
    print(framework.visualize_workflow("conditional_workflow"))


# ============================================================================
# 메인 함수 - 향상된 CLI
# ============================================================================

async def main():
    """
    메인 실행 함수 - 인터랙티브 데모

    실행 방법:
        python Semantic-agent_framework.py

    필수 환경변수 (.env 파일):
        AZURE_OPENAI_API_KEY=your-api-key
        AZURE_OPENAI_ENDPOINT =https://your-endpoint.openai.azure.com/
        AZURE_OPENAI_DEPLOYMENT=your-deployment-name
    """
    # 로깅 설정
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler("agent_framework.log", encoding='utf-8'),
            logging.StreamHandler()
        ]
    )
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("semantic_kernel").setLevel(logging.WARNING)

    # OpenTelemetry 설정
    setup_telemetry("UnifiedAgentFramework-Enterprise", enable_console=False)

    print("\n" + "="*60)
    print("🚀 Unified Agent Framework - Enterprise Edition")
    print("="*60)

    try:
        # 프레임워크 간편 생성
        framework = UnifiedAgentFramework.create()
        config = framework.config

        print(f"✅ 엔드포인트: {config.endpoint}")
        print(f"✅ 모델: {config.deployment_name}")
        print("="*60)

    except ValueError as e:
        print(str(e))
        print("\n💡 .env 파일 예시:")
        print("OPEN_AI_KEY_5=your-api-key")
        print("OPEN_AI_ENDPOINT_5=https://your-endpoint.openai.azure.com/")
        print("AZURE_OPENAI_DEPLOYMENT=your-deployment-name")
        return

    # 이벤트 리스너 등록
    if framework.event_bus:
        async def log_event(event: AgentEvent):
            logging.info(f"📢 이벤트: {event.event_type.value} - {event.agent_name or 'System'}")

        framework.event_bus.subscribe(EventType.AGENT_STARTED, log_event)
        framework.event_bus.subscribe(EventType.AGENT_COMPLETED, log_event)
        framework.event_bus.subscribe(EventType.APPROVAL_REQUESTED, log_event)

    # 데모 워크플로우 생성
    await demo_simple_chat(framework)
    await demo_routing_workflow(framework)
    await demo_supervisor_workflow(framework)
    await demo_conditional_workflow(framework)

    # 인터랙티브 세션
    print("\n" + "="*60)
    print("💬 인터랙티브 모드")
    print("="*60)
    print("명령어:")
    print("  exit          - 종료")
    print("  quick         - 빠른 질의응답 (예: quick 안녕하세요)")
    print("  smart         - 스킬 자동 감지 질의응답 (예: smart 파이썬 코드 작성)")
    print("  model         - 모델 변경 (예: model gpt-5, model list)")
    print("  skills        - 스킬 관리 (예: skills list, skills info python-expert)")
    print("  switch        - 워크플로우 전환 (예: switch routing_workflow)")
    print("  list          - 사용 가능한 워크플로우 목록")
    print("  visualize     - 현재 워크플로우 시각화")
    print("  stats         - 워크플로우 통계")
    print("  metrics       - 전역 메트릭")
    print("  events        - 이벤트 히스토리")
    print("  checkpoint    - 체크포인트 저장")
    print("  restore       - 체크포인트 복원")
    print("  rollback      - 이전 상태로 롤백")
    print("="*60 + "\n")

    session_id = f"session-{int(time.time())}"
    current_workflow = "simple_chat"

    try:
        while True:
            try:
                user_input = input(f"\n[{current_workflow}] User > ").strip()
            except EOFError:
                break

            if not user_input:
                continue

            # 명령어 처리
            cmd = user_input.lower().split()[0] if user_input else ""
            args = user_input.split()[1:] if len(user_input.split()) > 1 else []

            if cmd == "exit":
                print("\n👋 종료합니다...")
                break

            elif cmd == "model":
                # 모델 변경
                subcmd = args[0].lower() if args else "info"

                if subcmd == "list":
                    print("\n📋 지원하는 모델 목록:")
                    print("\n  [GPT-4 계열]")
                    for m in ["gpt-4", "gpt-4o", "gpt-4o-mini", "gpt-4.1", "gpt-4.1-mini", "gpt-4.1-nano"]:
                        marker = "👉" if m == framework.config.model else "  "
                        temp_info = "(temp ✓)" if supports_temperature(m) else "(temp ✗)"
                        print(f"  {marker} {m} {temp_info}")
                    print("\n  [GPT-5 계열] - NEW!")
                    for m in ["gpt-5", "gpt-5.1", "gpt-5.2"]:
                        marker = "👉" if m == framework.config.model else "  "
                        temp_info = "(temp ✓)" if supports_temperature(m) else "(temp ✗)"
                        print(f"  {marker} {m} {temp_info}")
                    print("\n  [o-시리즈 (Reasoning)]")
                    for m in ["o1", "o1-mini", "o1-preview", "o3", "o3-mini", "o4-mini"]:
                        marker = "👉" if m == framework.config.model else "  "
                        temp_info = "(temp ✓)" if supports_temperature(m) else "(temp ✗)"
                        print(f"  {marker} {m} {temp_info}")
                    print("\n  ※ (temp ✗) = temperature 파라미터 미지원")

                elif subcmd == "info":
                    print(f"\n📊 현재 모델 정보:")
                    print(f"   모델: {framework.config.model}")
                    print(f"   배포명: {framework.config.deployment_name}")
                    print(f"   Temperature 지원: {'예' if supports_temperature(framework.config.model) else '아니오'}")
                    print(f"   Temperature: {framework.config.temperature}")
                    print(f"   Max Tokens: {framework.config.max_tokens}")

                elif subcmd in SUPPORTED_MODELS:
                    old_model = framework.config.model
                    framework.config.model = subcmd
                    framework.config.deployment_name = subcmd

                    # 커널 재생성
                    framework.kernel = framework._create_kernel()

                    temp_info = "" if supports_temperature(subcmd) else " (temperature 미지원)"
                    print(f"\n✅ 모델 변경: {old_model} → {subcmd}{temp_info}")

                    # 워크플로우 재생성 (새 모델 적용)
                    await demo_simple_chat(framework)
                    print(f"   워크플로우 업데이트 완료")

                else:
                    print(f"\n❌ 알 수 없는 모델: {subcmd}")
                    print("   'model list'로 지원하는 모델을 확인하세요.")
                continue

            elif cmd == "quick":
                # 빠른 질의응답
                message = " ".join(args) if args else input("질문: ")
                print("\n⏳ 처리 중...")
                response = await framework.quick_chat(message)
                print(f"\n[AI] > {response}")
                continue

            elif cmd == "checkpoint":
                tag = args[0] if args else None
                state = await framework.state_manager.load_state(session_id)
                if state:
                    checkpoint_file = await framework.state_manager.save_checkpoint(state, tag=tag)
                    print(f"✅ 체크포인트 저장: {checkpoint_file}")
                else:
                    print("❌ 저장할 상태가 없습니다")
                continue

            elif cmd == "restore":
                tag = args[0] if args else None
                state = await framework.state_manager.restore_checkpoint(session_id, tag=tag)
                if state:
                    print(f"✅ 체크포인트 복원 완료")
                else:
                    print("❌ 복원할 체크포인트가 없습니다")
                continue

            elif cmd == "rollback":
                steps = int(args[0]) if args else 1
                state = await framework.state_manager.rollback(session_id, steps=steps)
                if state:
                    print(f"✅ {steps}단계 롤백 완료")
                else:
                    print("❌ 롤백 실패")
                continue

            elif cmd == "visualize":
                print("\n" + framework.visualize_workflow(current_workflow))
                continue

            elif cmd == "switch":
                if args:
                    workflow_name = args[0]
                    if workflow_name in framework.graphs:
                        current_workflow = workflow_name
                        print(f"✅ 워크플로우 전환: {workflow_name}")
                    else:
                        print(f"❌ 워크플로우 '{workflow_name}'를 찾을 수 없습니다")
                        print(f"   사용 가능: {', '.join(framework.graphs.keys())}")
                else:
                    print("❌ 워크플로우 이름을 지정하세요 (예: switch simple_chat)")
                continue

            elif cmd == "stats":
                stats = framework.get_workflow_stats(current_workflow)
                print("\n📊 워크플로우 통계:")
                print(json.dumps(stats, indent=2, ensure_ascii=False))
                continue

            elif cmd == "metrics":
                metrics = framework.get_global_metrics()
                print("\n📈 전역 메트릭:")
                print(json.dumps(metrics, indent=2, ensure_ascii=False))
                continue

            elif cmd == "events":
                event_type = args[0] if args else None

                if framework.event_bus:
                    if event_type:
                        try:
                            et = EventType(event_type)
                            events = framework.event_bus.get_event_history(event_type=et, limit=10)
                        except ValueError:
                            print(f"❌ 잘못된 이벤트 타입: {event_type}")
                            print(f"   가능한 타입: {', '.join(e.value for e in EventType)}")
                            continue
                    else:
                        events = framework.event_bus.get_event_history(limit=10)

                    print(f"\n📜 최근 이벤트 ({len(events)}개):")
                    for event in events:
                        print(f"  - {event.timestamp}: {event.event_type.value} ({event.agent_name or 'System'})")
                else:
                    print("❌ 이벤트 시스템이 비활성화되어 있습니다")
                continue

            elif cmd == "list":
                print("\n📋 사용 가능한 워크플로우:")
                for name in framework.graphs.keys():
                    marker = "👉" if name == current_workflow else "  "
                    print(f"{marker} {name}")
                continue

            elif cmd == "skills":
                # 스킬 관련 명령어
                subcmd = args[0] if args else "list"

                if subcmd == "list":
                    print("\n📚 등록된 스킬:")
                    for skill in framework.skill_manager.list_skills():
                        status = "✅" if skill.enabled else "❌"
                        print(f"  {status} {skill.name}: {skill.description[:50]}...")

                elif subcmd == "info" and len(args) > 1:
                    skill_name = args[1]
                    skill = framework.skill_manager.get_skill(skill_name)
                    if skill:
                        print(f"\n📖 스킬: {skill.name}")
                        print(f"   설명: {skill.description}")
                        print(f"   트리거: {', '.join(skill.triggers)}")
                        print(f"   리소스: {len(skill.resources)}개")
                        print(f"   우선순위: {skill.priority}")
                    else:
                        print(f"❌ 스킬을 찾을 수 없습니다: {skill_name}")

                elif subcmd == "stats":
                    stats = framework.skill_manager.get_usage_stats()
                    print("\n📊 스킬 사용 통계:")
                    print(json.dumps(stats, indent=2, ensure_ascii=False))

                elif subcmd == "create" and len(args) > 1:
                    skill_name = args[1]
                    output_dir = args[2] if len(args) > 2 else "./skills"
                    path = framework.skill_manager.create_skill_template(skill_name, output_dir)
                    print(f"✅ 스킬 템플릿 생성: {path}")

                elif subcmd == "load" and len(args) > 1:
                    skill_dir = args[1]
                    count = framework.skill_manager.load_skills_from_directory(skill_dir)
                    print(f"✅ {count}개 스킬 로드 완료")

                else:
                    print("\n💡 스킬 명령어:")
                    print("  skills list           - 등록된 스킬 목록")
                    print("  skills info <name>    - 스킬 상세 정보")
                    print("  skills stats          - 스킬 사용 통계")
                    print("  skills create <name>  - 새 스킬 템플릿 생성")
                    print("  skills load <dir>     - 디렉토리에서 스킬 로드")
                continue

            elif cmd == "smart":
                # 스마트 질의응답 (스킬 자동 감지)
                message = " ".join(args) if args else input("질문: ")
                print("\n⏳ 스킬 매칭 및 처리 중...")
                response = await framework.smart_chat(message)
                print(f"\n[AI] > {response}")
                continue

            elif cmd == "help":
                print("\n💡 도움말:")
                print("  일반 텍스트를 입력하면 현재 워크플로우로 처리됩니다.")
                print("  'quick 질문' 형식으로 빠른 질의응답이 가능합니다.")
                print("  'smart 질문' 형식으로 스킬 자동 감지 질의응답이 가능합니다.")
                print("  'skills' 명령어로 스킬을 관리할 수 있습니다.")
                continue

            # 일반 메시지 처리
            try:
                print("\n⏳ 처리 중...")
                state = await framework.run(
                    session_id=session_id,
                    workflow_name=current_workflow,
                    user_message=user_input
                )

                # 응답 출력
                if state.messages:
                    last_message = state.messages[-1]
                    print(f"\n[{last_message.agent_name or 'AI'}] > {last_message.content}")

                # 상태 정보
                print(f"\n📍 상태: {state.execution_status.value}")
                print(f"📊 노드: {state.current_node}")
                print(f"📈 방문: {' → '.join(state.visited_nodes[-5:])}")

                if state.metrics:
                    exec_time = state.metrics.get('execution_time_ms', 0)
                    iterations = state.metrics.get('total_iterations', 0)
                    print(f"⏱️ 실행 시간: {exec_time:.2f}ms ({iterations} iterations)")

                # 승인 대기 처리
                if state.execution_status == ExecutionStatus.WAITING_APPROVAL:
                    print("\n⏸️ 승인 대기 중:")
                    for i, approval in enumerate(state.pending_approvals):
                        print(f"  [{i}] {approval.get('description', 'N/A')}")
                        print(f"      Arguments: {approval.get('arguments', {})}")

                    approve_input = input("\n승인하시겠습니까? (y/n): ").strip().lower()
                    approved = approve_input == 'y'

                    state = await framework.approve_pending_request(
                        session_id,
                        request_id=0,
                        approved=approved
                    )
                    print(f"\n{'✅ 승인됨' if approved else '❌ 거부됨'}")

            except Exception as e:
                logging.error(f"❌ 실행 오류: {e}", exc_info=True)
                print(f"\n❌ 오류: {e}")

    finally:
        # 정리
        await framework.cleanup()
        print("\n✅ 프레임워크 종료 완료")


# ============================================================================
# 간편 사용 함수 (모듈로 import 시 활용)
# ============================================================================

async def quick_run(message: str, system_prompt: str = "You are a helpful assistant.") -> str:
    """
    가장 간단한 사용법 - 한 줄로 AI 응답 받기

    사용법:
    ```python
    import asyncio
    from Semantic_agent_framework import quick_run

    response = asyncio.run(quick_run("파이썬이란 무엇인가요?"))
    print(response)
    ```
    """
    framework = UnifiedAgentFramework.create()
    return await framework.quick_chat(message, system_prompt)


def create_framework(
    model: str = None,  # None이면 DEFAULT_LLM_MODEL 사용
    temperature: float = 0.7,
    **kwargs
) -> UnifiedAgentFramework:
    """
    프레임워크 간편 생성

    사용법:
    ```python
    from Semantic_agent_framework import create_framework

    framework = create_framework(model="gpt-4o", temperature=0.5)
    ```
    """
    config = FrameworkConfig.from_env()
    if model is not None:
        config.model = model
    config.temperature = temperature

    for key, value in kwargs.items():
        if hasattr(config, key):
            setattr(config, key, value)

    return UnifiedAgentFramework.create(config)


if __name__ == "__main__":
    asyncio.run(main())
