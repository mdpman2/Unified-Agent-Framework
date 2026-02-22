#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Unified Agent Framework - 보안 가드레일 모듈 (Security Guardrails Module)

================================================================================
📁 파일 위치: unified_agent/security_guardrails.py
📋 역할: Prompt Injection 방어, Jailbreak 탐지, 출력 검증, Groundedness 검사
📅 최종 업데이트: 2026년 2월 4일
📦 버전: v3.5.0
✅ 테스트: test_security_guardrails.py, test_v35_scenarios.py
================================================================================

🎯 주요 구성 요소:
    1. PromptShield - Prompt Injection 및 간접 공격 탐지
    2. JailbreakDetector - Jailbreak 시도 탐지
    3. OutputValidator - 출력 안전성 검증
    4. GroundednessChecker - 응답의 근거 확인 (Hallucination 방지)
    5. PIIDetector - 개인정보(PII) 탐지 및 마스킹
    6. SecurityOrchestrator - 통합 보안 파이프라인

🔧 2026년 2월 기능:
    - Azure AI Content Safety API 통합
    - 로컬 패턴 기반 빠른 필터링
    - 다층 보안 검증 (Input → Processing → Output)
    - 실시간 위협 탐지 및 차단
    - 감사 로그 (Audit Log) 지원

📌 사용 예시:
    >>> from unified_agent.security_guardrails import (
    ...     PromptShield, JailbreakDetector, SecurityOrchestrator,
    ...     SecurityConfig, ThreatLevel
    ... )
    >>>
    >>> # 방법 1: 개별 컴포넌트 사용
    >>> shield = PromptShield()
    >>> result = await shield.analyze("사용자 입력")
    >>> if result.is_attack:
    ...     print(f"⚠️ 공격 탐지: {result.attack_type}")
    >>>
    >>> # 방법 2: 통합 오케스트레이터 사용
    >>> orchestrator = SecurityOrchestrator(SecurityConfig(
    ...     enable_prompt_shield=True,
    ...     enable_jailbreak_detection=True,
    ...     enable_pii_detection=True,
    ...     threat_level=ThreatLevel.MEDIUM
    ... ))
    >>> result = await orchestrator.validate_input("사용자 입력")
    >>> if not result.is_safe:
    ...     print(f"차단됨: {result.blocked_reason}")

⚠️ 주의사항:
    - 프로덕션에서는 Azure AI Content Safety API 사용을 강력 권장합니다.
    - 로컬 패턴 매칭은 기본 필터링용이며, 우회될 수 있습니다.
    - API 키는 환경 변수로 관리하세요.

🔗 관련 문서:
    - Azure AI Content Safety: https://learn.microsoft.com/azure/ai-services/content-safety/
    - Prompt Shield: https://learn.microsoft.com/azure/ai-services/content-safety/concepts/jailbreak-detection
    - Groundedness Detection: https://learn.microsoft.com/azure/ai-services/content-safety/concepts/groundedness
"""

from __future__ import annotations

import re
import hashlib
import asyncio
import aiohttp
import json
import logging
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Callable
from functools import lru_cache

__all__ = [
    # Enums
    "ThreatLevel",
    "AttackType",
    "PIIType",
    "ValidationStage",
    # Config & Results
    "SecurityConfig",
    "ShieldResult",
    "JailbreakResult",
    "PIIResult",
    "GroundednessResult",
    "ValidationResult",
    "AuditLogEntry",
    # Core Components
    "PromptShield",
    "JailbreakDetector",
    "OutputValidator",
    "GroundednessChecker",
    "PIIDetector",
    "SecurityOrchestrator",
    # Utilities
    "SecurityAuditLogger",
]

# ============================================================================
# Enums
# ============================================================================

class ThreatLevel(str, Enum):
    """위협 수준 설정"""
    LOW = "low"           # 기본 필터링만
    MEDIUM = "medium"     # 패턴 매칭 + 휴리스틱
    HIGH = "high"         # 모든 검증 + API 호출
    PARANOID = "paranoid" # 최대 보안 (성능 저하 가능)

class AttackType(str, Enum):
    """공격 유형"""
    NONE = "none"
    DIRECT_INJECTION = "direct_injection"       # 직접 프롬프트 주입
    INDIRECT_INJECTION = "indirect_injection"   # 간접 프롬프트 주입 (문서 내)
    JAILBREAK = "jailbreak"                     # 역할 탈출 시도
    PROMPT_LEAKING = "prompt_leaking"           # 시스템 프롬프트 유출 시도
    INSTRUCTION_OVERRIDE = "instruction_override" # 지시 무시/덮어쓰기
    ENCODING_ATTACK = "encoding_attack"         # Base64/인코딩 우회
    MULTI_TURN_ATTACK = "multi_turn_attack"     # 다중 턴 공격
    ROLE_PLAY_ATTACK = "role_play_attack"       # 역할극 기반 공격

class PIIType(str, Enum):
    """개인정보 유형"""
    EMAIL = "email"
    PHONE = "phone"
    SSN = "ssn"                    # 주민등록번호
    CREDIT_CARD = "credit_card"
    IP_ADDRESS = "ip_address"
    ADDRESS = "address"
    NAME = "name"
    PASSPORT = "passport"
    DRIVER_LICENSE = "driver_license"

class ValidationStage(str, Enum):
    """검증 단계"""
    INPUT = "input"           # 사용자 입력 검증
    PROCESSING = "processing" # 처리 중 검증
    OUTPUT = "output"         # 출력 검증

# ============================================================================
# Data Classes
# ============================================================================

@dataclass(frozen=True, slots=True)
class SecurityConfig:
    """
    보안 설정 구성
    
    ================================================================================
    📋 역할: 보안 가드레일의 전체 설정을 관리
    📅 최종 업데이트: 2026년 2월
    ================================================================================
    
    Attributes:
        threat_level: 위협 수준 (LOW, MEDIUM, HIGH, PARANOID)
        enable_prompt_shield: Prompt Injection 탐지 활성화
        enable_jailbreak_detection: Jailbreak 탐지 활성화
        enable_pii_detection: PII 탐지 활성화
        enable_output_validation: 출력 검증 활성화
        enable_groundedness_check: Groundedness 검사 활성화
        enable_audit_logging: 감사 로깅 활성화
        azure_content_safety_endpoint: Azure Content Safety API 엔드포인트
        azure_content_safety_key: Azure Content Safety API 키
        block_on_detection: 탐지 시 즉시 차단 여부
        custom_blocked_patterns: 사용자 정의 차단 패턴
        pii_mask_char: PII 마스킹 문자
        max_input_length: 최대 입력 길이
    
    📌 사용 예시:
        >>> config = SecurityConfig(
        ...     threat_level=ThreatLevel.HIGH,
        ...     enable_prompt_shield=True,
        ...     azure_content_safety_endpoint="https://xxx.cognitiveservices.azure.com",
        ...     azure_content_safety_key="your-api-key"
        ... )
    """
    threat_level: ThreatLevel = ThreatLevel.MEDIUM
    enable_prompt_shield: bool = True
    enable_jailbreak_detection: bool = True
    enable_pii_detection: bool = True
    enable_output_validation: bool = True
    enable_groundedness_check: bool = False  # API 필요
    enable_audit_logging: bool = True
    
    # Azure AI Content Safety API (선택적)
    azure_content_safety_endpoint: str | None = None
    azure_content_safety_key: str | None = field(default=None, repr=False)
    
    # 동작 설정
    block_on_detection: bool = True
    custom_blocked_patterns: list[str] = field(default_factory=list)
    pii_mask_char: str = "*"
    max_input_length: int = 100000  # 100K chars
    
    # 타임아웃
    api_timeout_seconds: float = 10.0
    
    def has_azure_api(self) -> bool:
        """Azure API 설정 여부 확인"""
        return bool(self.azure_content_safety_endpoint and self.azure_content_safety_key)

@dataclass(slots=True)
class ShieldResult:
    """
    Prompt Shield 분석 결과
    
    Attributes:
        is_attack: 공격 여부
        attack_type: 공격 유형
        confidence: 신뢰도 (0.0 ~ 1.0)
        matched_patterns: 매칭된 패턴 목록
        details: 상세 정보
        processing_time_ms: 처리 시간 (밀리초)
    """
    is_attack: bool = False
    attack_type: AttackType = AttackType.NONE
    confidence: float = 0.0
    matched_patterns: list[str] = field(default_factory=list)
    details: dict[str, Any] = field(default_factory=dict)
    processing_time_ms: float = 0.0

@dataclass(slots=True)
class JailbreakResult:
    """
    Jailbreak 탐지 결과
    
    Attributes:
        is_jailbreak: Jailbreak 시도 여부
        confidence: 신뢰도 (0.0 ~ 1.0)
        detected_techniques: 탐지된 기법 목록
        risk_score: 위험 점수 (0 ~ 100)
        recommendation: 권장 조치
    """
    is_jailbreak: bool = False
    confidence: float = 0.0
    detected_techniques: list[str] = field(default_factory=list)
    risk_score: int = 0
    recommendation: str = ""

@dataclass(slots=True)
class PIIResult:
    """
    PII 탐지 결과
    
    Attributes:
        has_pii: PII 포함 여부
        detected_types: 탐지된 PII 유형
        pii_locations: PII 위치 정보 (start, end, type)
        masked_text: PII가 마스킹된 텍스트
        pii_count: 탐지된 PII 개수
    """
    has_pii: bool = False
    detected_types: set[PIIType] = field(default_factory=set)
    pii_locations: list[dict[str, Any]] = field(default_factory=list)
    masked_text: str = ""
    pii_count: int = 0

@dataclass(slots=True)
class GroundednessResult:
    """
    Groundedness 검사 결과 (Hallucination 탐지)
    
    Attributes:
        is_grounded: 근거 있는 응답 여부
        groundedness_score: 근거 점수 (0.0 ~ 1.0)
        ungrounded_claims: 근거 없는 주장 목록
        source_coverage: 소스 커버리지
    """
    is_grounded: bool = True
    groundedness_score: float = 1.0
    ungrounded_claims: list[str] = field(default_factory=list)
    source_coverage: float = 1.0

@dataclass(slots=True)
class ValidationResult:
    """
    통합 검증 결과
    
    Attributes:
        is_safe: 안전 여부
        blocked: 차단 여부
        blocked_reason: 차단 사유
        stage: 검증 단계
        shield_result: Prompt Shield 결과
        jailbreak_result: Jailbreak 탐지 결과
        pii_result: PII 탐지 결과
        groundedness_result: Groundedness 결과
        sanitized_input: 정화된 입력
        total_processing_time_ms: 총 처리 시간
    """
    is_safe: bool = True
    blocked: bool = False
    blocked_reason: str | None = None
    stage: ValidationStage = ValidationStage.INPUT
    shield_result: ShieldResult | None = None
    jailbreak_result: JailbreakResult | None = None
    pii_result: PIIResult | None = None
    groundedness_result: GroundednessResult | None = None
    sanitized_input: str | None = None
    total_processing_time_ms: float = 0.0

@dataclass(frozen=True, slots=True)
class AuditLogEntry:
    """
    감사 로그 항목
    
    Attributes:
        timestamp: 타임스탬프
        session_id: 세션 ID
        stage: 검증 단계
        input_hash: 입력 해시 (SHA-256)
        result: 검증 결과
        threat_detected: 위협 탐지 여부
        threat_type: 위협 유형
        action_taken: 취한 조치
        metadata: 추가 메타데이터
    """
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    session_id: str = ""
    stage: ValidationStage = ValidationStage.INPUT
    input_hash: str = ""
    result: str = ""
    threat_detected: bool = False
    threat_type: str | None = None
    action_taken: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)

# ============================================================================
# Prompt Shield - Prompt Injection 탐지
# ============================================================================

class PromptShield:
    """
    Prompt Shield - Prompt Injection 및 간접 공격 탐지
    
    ================================================================================
    📋 역할: 사용자 입력에서 악의적인 프롬프트 주입 시도를 탐지
    📅 최종 업데이트: 2026년 2월
    ================================================================================
    
    🎯 탐지 대상:
        - 직접 프롬프트 주입 (Direct Injection)
        - 간접 프롬프트 주입 (Indirect Injection) - 문서/URL 내 숨겨진 지시
        - 지시 무시/덮어쓰기 시도
        - 인코딩 기반 우회 (Base64, Unicode 등)
        - 프롬프트 유출 시도
    
    🔧 탐지 방법:
        1. 패턴 매칭 (빠른 1차 필터링)
        2. 휴리스틱 분석 (의심스러운 구조 탐지)
        3. Azure Content Safety API (선택적, 높은 정확도)
    
    📌 사용 예시:
        >>> shield = PromptShield()
        >>> result = await shield.analyze("사용자 입력")
        >>> if result.is_attack:
        ...     print(f"공격 탐지: {result.attack_type}")
        ...     print(f"신뢰도: {result.confidence:.2%}")
        >>>
        >>> # Azure API 사용
        >>> shield = PromptShield(
        ...     azure_endpoint="https://xxx.cognitiveservices.azure.com",
        ...     azure_key="your-key"
        ... )
        >>> result = await shield.analyze_with_api(user_input, documents=[doc1, doc2])
    
    ⚠️ 주의사항:
        - 패턴 매칭은 기본 필터링용입니다. 정교한 공격은 우회될 수 있습니다.
        - 프로덕션에서는 Azure Content Safety API 사용을 권장합니다.
    """
    
    # -------------------------------------------------------------------------
    # 직접 주입 패턴 (Direct Injection Patterns)
    # -------------------------------------------------------------------------
    DIRECT_INJECTION_PATTERNS = [
        # 시스템 프롬프트 무시 시도
        r"(?i)ignore\s+(all\s+)?(previous|prior|above|earlier)\s+(instructions?|prompts?|rules?|guidelines?)",
        r"(?i)disregard\s+(all\s+)?(previous|prior|above)\s+(instructions?|prompts?)",
        r"(?i)forget\s+(everything|all|your)\s+(you\s+)?(know|learned|instructions?)",
        r"(?i)override\s+(your\s+)?(instructions?|programming|rules?|system)",
        r"(?i)새로운\s*지시|이전\s*(지시|명령)\s*(무시|잊어)",
        r"(?i)시스템\s*프롬프트\s*(무시|변경|덮어쓰기)",
        r"(무시|잊어).*(지시|명령|규칙|프롬프트)",
        r"(이전|위|앞).*(지시|명령|규칙).*(무시|잊|건너)",
        
        # 역할 변경 시도
        r"(?i)you\s+are\s+now\s+(a|an|the)\s+(?!assistant|helpful)",
        r"(?i)pretend\s+(to\s+be|you'?re)\s+(a|an)",
        r"(?i)act\s+as\s+(if\s+)?(you\s+)?(are|were)\s+(a|an)",
        r"(?i)roleplay\s+as\s+(a|an)",
        r"(?i)너는?\s*(이제|지금부터)\s*.+?(이다|야|입니다)",
        r"역할.*(수행|바꿔|변경)",
        r"새로운\s*역할",
        
        # 탈옥 키워드
        r"(?i)jailbreak",
        r"(?i)DAN\s*(mode)?",
        r"(?i)developer\s+mode",
        r"(?i)unrestricted\s+mode",
        r"(?i)do\s+anything\s+now",
        
        # 프롬프트 유출 시도
        r"(?i)show\s+(me\s+)?(your\s+)?(system\s+)?prompt",
        r"(?i)reveal\s+(your\s+)?(system\s+)?prompt",
        r"(?i)what\s+(is|are)\s+your\s+(instructions?|rules?|prompt)",
        r"(?i)print\s+(your\s+)?(system\s+)?prompt",
        r"(?i)시스템\s*프롬프트\s*(보여|알려|출력)",
        r"(?i)너의?\s*(지시|명령|규칙)\s*(알려|보여)",
    ]
    
    # -------------------------------------------------------------------------
    # 간접 주입 패턴 (Indirect Injection Patterns - 문서 내)
    # -------------------------------------------------------------------------
    INDIRECT_INJECTION_PATTERNS = [
        # 문서 내 숨겨진 지시
        r"(?i)\[SYSTEM\]",
        r"(?i)\[INSTRUCTION\]",
        r"(?i)<<<\s*(system|instruction|command)",
        r"(?i)>>>\s*END\s*OF\s*DOCUMENT",
        r"(?i)<!-- hidden instruction",
        r"(?i)/\*\s*system\s*override",
        
        # 마크다운/HTML 악용
        r"(?i)<script[^>]*>",
        r"(?i)javascript:",
        r"(?i)data:text/html",
        
        # 구분자 악용
        r"-{5,}\s*(system|instruction|ignore)",
        r"={5,}\s*(system|instruction|ignore)",
        
        # 역할 지시 마커
        r"(?i)\[INST\]",
        r"(?i)\[/INST\]",
        r"<<SYS>>",
        r"<\|im_start\|>system",
    ]
    
    # -------------------------------------------------------------------------
    # 인코딩 우회 패턴
    # -------------------------------------------------------------------------
    ENCODING_PATTERNS = [
        # Base64 인코딩
        r"(?i)base64:\s*[A-Za-z0-9+/=]{20,}",
        r"(?i)decode\s+this:\s*[A-Za-z0-9+/=]{20,}",
        
        # Hex 인코딩
        r"(?i)hex:\s*[0-9a-fA-F]{20,}",
        r"(?i)\\x[0-9a-fA-F]{2}(\\x[0-9a-fA-F]{2}){5,}",
        
        # Unicode 이스케이프
        r"\\u[0-9a-fA-F]{4}(\\u[0-9a-fA-F]{4}){5,}",
    ]
    
    def __init__(
        self,
        azure_endpoint: str | None = None,
        azure_key: str | None = None,
        custom_patterns: list[str] | None = None
    ):
        """
        PromptShield 초기화
        
        Args:
            azure_endpoint: Azure Content Safety API 엔드포인트
            azure_key: Azure Content Safety API 키
            custom_patterns: 추가 사용자 정의 패턴
        """
        self.azure_endpoint = azure_endpoint
        self.azure_key = azure_key
        self.custom_patterns = custom_patterns or []
        self.logger = logging.getLogger(__name__)
        
        # 패턴 컴파일 (성능 최적화)
        self._compiled_direct = [re.compile(p) for p in self.DIRECT_INJECTION_PATTERNS]
        self._compiled_indirect = [re.compile(p) for p in self.INDIRECT_INJECTION_PATTERNS]
        self._compiled_encoding = [re.compile(p) for p in self.ENCODING_PATTERNS]
        self._compiled_custom = [re.compile(p) for p in self.custom_patterns]
    
    async def analyze(
        self,
        text: str,
        documents: list[str] | None = None,
        use_api: bool = False
    ) -> ShieldResult:
        """
        텍스트 분석하여 Prompt Injection 탐지
        
        Args:
            text: 분석할 텍스트 (사용자 입력)
            documents: 함께 분석할 문서 목록 (간접 주입 탐지용)
            use_api: Azure API 사용 여부
        
        Returns:
            ShieldResult: 분석 결과
        """
        start_time = time.perf_counter()
        
        result = ShieldResult()
        matched_patterns = []
        
        # 1. 직접 주입 패턴 검사
        for pattern in self._compiled_direct:
            match = pattern.search(text)
            if match:
                matched_patterns.append(f"direct:{pattern.pattern[:50]}...")
                result.attack_type = AttackType.DIRECT_INJECTION
        
        # 2. 인코딩 우회 검사
        for pattern in self._compiled_encoding:
            match = pattern.search(text)
            if match:
                matched_patterns.append(f"encoding:{pattern.pattern[:50]}...")
                if result.attack_type == AttackType.NONE:
                    result.attack_type = AttackType.ENCODING_ATTACK
        
        # 3. 사용자 정의 패턴 검사
        for pattern in self._compiled_custom:
            match = pattern.search(text)
            if match:
                matched_patterns.append(f"custom:{pattern.pattern[:50]}...")
        
        # 4. 문서 내 간접 주입 검사
        if documents:
            for i, doc in enumerate(documents):
                for pattern in self._compiled_indirect:
                    match = pattern.search(doc)
                    if match:
                        matched_patterns.append(f"indirect_doc_{i}:{pattern.pattern[:50]}...")
                        if result.attack_type == AttackType.NONE:
                            result.attack_type = AttackType.INDIRECT_INJECTION
        
        # 5. 휴리스틱 분석
        heuristic_score = self._heuristic_analysis(text)
        
        # 결과 집계
        if matched_patterns or heuristic_score > 0.5:
            result.is_attack = True
            result.matched_patterns = matched_patterns
            result.confidence = min(1.0, len(matched_patterns) * 0.2 + heuristic_score * 0.5)
            
            # 공격 유형 세분화
            if result.attack_type == AttackType.NONE and heuristic_score > 0.5:
                result.attack_type = AttackType.INSTRUCTION_OVERRIDE
        
        result.details = {
            "heuristic_score": heuristic_score,
            "pattern_matches": len(matched_patterns),
            "text_length": len(text),
            "documents_checked": len(documents) if documents else 0
        }
        
        # 6. Azure API 호출 (선택적)
        if use_api and self.azure_endpoint and self.azure_key:
            api_result = await self._call_azure_api(text, documents)
            if api_result:
                result.details["azure_api_result"] = api_result
                # API 결과로 신뢰도 보정
                if api_result.get("attackDetected"):
                    result.is_attack = True
                    result.confidence = max(result.confidence, api_result.get("confidence", 0.8))
        
        result.processing_time_ms = (time.perf_counter() - start_time) * 1000
        return result
    
    def _heuristic_analysis(self, text: str) -> float:
        """
        휴리스틱 기반 분석
        
        의심스러운 구조 및 패턴을 점수화합니다.
        
        Args:
            text: 분석할 텍스트
        
        Returns:
            float: 의심 점수 (0.0 ~ 1.0)
        """
        score = 0.0
        text_lower = text.lower()
        
        # 1. 명령어 키워드 밀도
        command_keywords = [
            "ignore", "forget", "override", "bypass", "skip",
            "무시", "잊어", "덮어", "우회", "건너뛰"
        ]
        keyword_count = sum(1 for kw in command_keywords if kw in text_lower)
        score += min(0.3, keyword_count * 0.1)
        
        # 2. 역할 변경 시도 (더 일반적인 패턴)
        role_indicators = [
            "you are", "act as", "pretend", "roleplay",
            "너는", "역할", "행동해", "인척"
        ]
        if any(ind in text_lower for ind in role_indicators):
            score += 0.2
        
        # 3. 구분자 남용 (다중 구분자)
        separators = text.count("---") + text.count("===") + text.count("***")
        if separators > 2:
            score += 0.15
        
        # 4. 대문자 비율 (과도한 강조)
        if len(text) > 10:
            upper_ratio = sum(1 for c in text if c.isupper()) / len(text)
            if upper_ratio > 0.4:
                score += 0.1
        
        # 5. 특수 마커 존재
        special_markers = ["[INST]", "[/INST]", "<|im_start|>", "<|im_end|>", "<<SYS>>"]
        if any(marker in text for marker in special_markers):
            score += 0.25
        
        return min(1.0, score)
    
    async def _call_azure_api(
        self,
        text: str,
        documents: list[str] | None = None
    ) -> dict[str, Any] | None:
        """
        Azure Content Safety API 호출 (Prompt Shield)
        
        Args:
            text: 사용자 프롬프트
            documents: 함께 분석할 문서
        
        Returns:
            Dict | None: API 응답 또는 None
        """
        if not self.azure_endpoint or not self.azure_key:
            return None
        
        url = f"{self.azure_endpoint}/contentsafety/text:shieldPrompt?api-version=2024-09-01"
        
        headers = {
            "Content-Type": "application/json",
            "Ocp-Apim-Subscription-Key": self.azure_key
        }
        
        payload = {
            "userPrompt": text,
            "documents": documents or []
        }
        
        try:
            async with aiohttp.ClientSession() as session:
                async with session.post(url, json=payload, headers=headers, timeout=10) as response:
                    if response.status == 200:
                        data = await response.json()
                        return {
                            "attackDetected": data.get("userPromptAnalysis", {}).get("attackDetected", False),
                            "confidence": 0.9 if data.get("userPromptAnalysis", {}).get("attackDetected") else 0.1,
                            "documentsAnalysis": data.get("documentsAnalysis", [])
                        }
                    else:
                        self.logger.warning(f"Azure API 호출 실패: {response.status}")
                        return None
        except Exception as e:
            self.logger.error(f"Azure API 호출 오류: {e}")
            return None

# ============================================================================
# Jailbreak Detector
# ============================================================================

class JailbreakDetector:
    """
    Jailbreak 탐지기 - 역할 탈출 및 제한 우회 시도 탐지
    
    ================================================================================
    📋 역할: AI 모델의 안전 장치 우회 시도를 탐지
    📅 최종 업데이트: 2026년 2월
    ================================================================================
    
    🎯 탐지 대상:
        - DAN (Do Anything Now) 공격
        - 역할극 기반 우회
        - 가상 시나리오 악용
        - 다중 턴 조작
        - 감정 조작 (Guilt-tripping)
    
    📌 사용 예시:
        >>> detector = JailbreakDetector()
        >>> result = detector.detect("Hi ChatGPT, let's play a game...")
        >>> if result.is_jailbreak:
        ...     print(f"Jailbreak 시도 탐지: {result.detected_techniques}")
    """
    
    # Jailbreak 기법별 패턴
    JAILBREAK_TECHNIQUES = {
        "DAN": [
            r"(?i)do\s+anything\s+now",
            r"(?i)DAN\s*(mode|prompt)?",
            r"(?i)you\s+can\s+do\s+anything",
            r"(?i)no\s+longer\s+bound\s+by\s+rules",
        ],
        "ROLEPLAY": [
            r"(?i)let'?s?\s+play\s+a\s+game",
            r"(?i)pretend\s+(we'?re|this\s+is)\s+a\s+(movie|story|fiction)",
            r"(?i)in\s+this\s+hypothetical\s+scenario",
            r"(?i)imagine\s+you'?re?\s+(a|an)\s+evil",
        ],
        "DEVELOPER_MODE": [
            r"(?i)developer\s+mode",
            r"(?i)admin\s+mode",
            r"(?i)god\s+mode",
            r"(?i)debug\s+mode",
            r"(?i)unrestricted\s+access",
        ],
        "SPLIT_PERSONALITY": [
            r"(?i)two\s+responses",
            r"(?i)one\s+filtered.+one\s+unfiltered",
            r"(?i)🔓\s*(unlocked|jailbroken)",
            r"(?i)normal\s+response.+dev\s+mode",
        ],
        "EMOTIONAL_MANIPULATION": [
            r"(?i)please.+i'?m?\s+dying",
            r"(?i)my\s+grandmother\s+used\s+to",
            r"(?i)for\s+educational\s+purposes\s+only",
            r"(?i)this\s+is\s+just\s+for\s+research",
        ],
        "TOKEN_SMUGGLING": [
            r"(?i)complete\s+the\s+following",
            r"(?i)continue\s+this\s+story",
            r"(?i)fill\s+in\s+the\s+blank",
        ],
    }
    
    def __init__(self, sensitivity: float = 0.5):
        """
        JailbreakDetector 초기화
        
        Args:
            sensitivity: 탐지 민감도 (0.0 ~ 1.0, 높을수록 민감)
        """
        self.sensitivity = sensitivity
        self._compiled_patterns = {
            technique: [re.compile(p) for p in patterns]
            for technique, patterns in self.JAILBREAK_TECHNIQUES.items()
        }
    
    def detect(self, text: str) -> JailbreakResult:
        """
        Jailbreak 시도 탐지
        
        Args:
            text: 분석할 텍스트
        
        Returns:
            JailbreakResult: 탐지 결과
        """
        result = JailbreakResult()
        detected_techniques = []
        total_score = 0
        
        # 기법별 패턴 매칭
        for technique, patterns in self._compiled_patterns.items():
            for pattern in patterns:
                if pattern.search(text):
                    detected_techniques.append(technique)
                    total_score += 20
                    break  # 같은 기법 중복 카운트 방지
        
        # 추가 휴리스틱
        text_lower = text.lower()
        
        # 1. 과도한 설득 시도
        persuasion_words = ["please", "just", "only", "simply", "merely"]
        persuasion_count = sum(text_lower.count(w) for w in persuasion_words)
        if persuasion_count > 5:
            total_score += 10
        
        # 2. 위험한 주제 요청
        dangerous_topics = ["bomb", "hack", "weapon", "kill", "drug", "폭탄", "해킹", "무기"]
        if any(topic in text_lower for topic in dangerous_topics):
            total_score += 30
        
        # 3. 역할 강요
        if "you must" in text_lower or "you have to" in text_lower:
            total_score += 15
        
        # 결과 계산
        result.risk_score = min(100, total_score)
        result.detected_techniques = list(set(detected_techniques))
        
        # 민감도에 따른 Jailbreak 판정
        threshold = int((1 - self.sensitivity) * 50 + 20)  # 20 ~ 70
        result.is_jailbreak = result.risk_score >= threshold
        result.confidence = min(1.0, result.risk_score / 100)
        
        # 권장 조치
        if result.is_jailbreak:
            if result.risk_score >= 70:
                result.recommendation = "차단 권장: 높은 위험도의 Jailbreak 시도"
            elif result.risk_score >= 40:
                result.recommendation = "검토 권장: 의심스러운 패턴 발견"
            else:
                result.recommendation = "모니터링: 경미한 의심 패턴"
        else:
            result.recommendation = "통과: 위험 패턴 미발견"
        
        return result

# ============================================================================
# PII Detector - 개인정보 탐지
# ============================================================================

class PIIDetector:
    """
    PII (Personally Identifiable Information) 탐지기
    
    ================================================================================
    📋 역할: 텍스트에서 개인정보를 탐지하고 마스킹
    📅 최종 업데이트: 2026년 2월
    ================================================================================
    
    🎯 탐지 대상:
        - 이메일 주소
        - 전화번호 (한국/미국 형식)
        - 주민등록번호
        - 신용카드 번호
        - IP 주소
        - 여권 번호
    
    📌 사용 예시:
        >>> detector = PIIDetector()
        >>> result = detector.detect("연락처: 010-1234-5678, 이메일: test@example.com")
        >>> print(result.masked_text)
        # 연락처: ***-****-****, 이메일: ****@*******.***
    """
    
    # PII 패턴 정의
    PII_PATTERNS = {
        PIIType.EMAIL: r'[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}',
        PIIType.PHONE: r'\b(010|011|016|017|018|019)[-.\s]?\d{3,4}[-.\s]?\d{4}\b|\b\d{2,3}[-.\s]?\d{3,4}[-.\s]?\d{4}\b',
        PIIType.SSN: r'\b\d{6}[-.\s]?\d{7}\b',  # 한국 주민등록번호
        PIIType.CREDIT_CARD: r'\b(?:\d{4}[-.\s]?){3}\d{4}\b',
        PIIType.IP_ADDRESS: r'\b(?:25[0-5]|2[0-4]\d|1\d{2}|[1-9]?\d)(?:\.(?:25[0-5]|2[0-4]\d|1\d{2}|[1-9]?\d)){3}\b',
    }
    
    def __init__(self, mask_char: str = "*"):
        """
        PIIDetector 초기화
        
        Args:
            mask_char: 마스킹에 사용할 문자
        """
        self.mask_char = mask_char
        self._compiled_patterns = {
            pii_type: re.compile(pattern)
            for pii_type, pattern in self.PII_PATTERNS.items()
        }
    
    def detect(self, text: str, mask: bool = True) -> PIIResult:
        """
        PII 탐지 및 선택적 마스킹
        
        Args:
            text: 분석할 텍스트
            mask: 마스킹 여부
        
        Returns:
            PIIResult: 탐지 결과
        """
        result = PIIResult()
        result.masked_text = text
        pii_locations = []
        detected_types = set()
        
        for pii_type, pattern in self._compiled_patterns.items():
            for match in pattern.finditer(text):
                detected_types.add(pii_type)
                pii_locations.append({
                    "type": pii_type.value,
                    "start": match.start(),
                    "end": match.end(),
                    "value_preview": match.group()[:3] + "..."  # 일부만 기록
                })
        
        # 마스킹 적용 (뒤에서부터 처리하여 인덱스 유지)
        if mask and pii_locations:
            masked = list(text)
            for loc in sorted(pii_locations, key=lambda x: x["start"], reverse=True):
                start, end = loc["start"], loc["end"]
                # 일부 문자만 마스킹 (형식 유지)
                for i in range(start, end):
                    if text[i] not in "-. @":
                        masked[i] = self.mask_char
            result.masked_text = "".join(masked)
        
        result.has_pii = bool(detected_types)
        result.detected_types = detected_types
        result.pii_locations = pii_locations
        result.pii_count = len(pii_locations)
        
        return result

# ============================================================================
# Output Validator - 출력 검증
# ============================================================================

class OutputValidator:
    """
    출력 검증기 - AI 응답의 안전성 검증
    
    ================================================================================
    📋 역할: AI 모델 출력에서 유해 콘텐츠 및 PII 유출 탐지
    📅 최종 업데이트: 2026년 2월
    ================================================================================
    
    🎯 검증 대상:
        - 유해 콘텐츠 (폭력, 혐오 등)
        - PII 유출
        - 시스템 프롬프트 유출
        - 코드 인젝션
    
    📌 사용 예시:
        >>> validator = OutputValidator()
        >>> result = await validator.validate(ai_response, context={"system_prompt": "..."})
    """
    
    # 유해 콘텐츠 패턴
    HARMFUL_PATTERNS = [
        r"(?i)how\s+to\s+(make|build|create)\s+(a\s+)?(bomb|weapon|explosive)",
        r"(?i)step.+by.+step.+(hack|exploit|attack)",
    ]
    
    # 프롬프트 유출 징후
    PROMPT_LEAK_INDICATORS = [
        r"(?i)my\s+(system\s+)?instructions?\s+(are|is|say)",
        r"(?i)i\s+was\s+(told|instructed|programmed)\s+to",
        r"(?i)my\s+initial\s+prompt",
    ]
    
    def __init__(self, pii_detector: PIIDetector | None = None):
        """
        OutputValidator 초기화
        
        Args:
            pii_detector: PII 탐지기 인스턴스 (공유 가능)
        """
        self.pii_detector = pii_detector or PIIDetector()
        self._compiled_harmful = [re.compile(p) for p in self.HARMFUL_PATTERNS]
        self._compiled_leak = [re.compile(p) for p in self.PROMPT_LEAK_INDICATORS]
    
    async def validate(
        self,
        output: str,
        context: dict[str, Any] | None = None
    ) -> ValidationResult:
        """
        AI 출력 검증
        
        Args:
            output: AI 모델 출력
            context: 컨텍스트 (system_prompt 등)
        
        Returns:
            ValidationResult: 검증 결과
        """
        start_time = time.perf_counter()
        
        result = ValidationResult(stage=ValidationStage.OUTPUT)
        issues = []
        
        # 1. 유해 콘텐츠 검사
        for pattern in self._compiled_harmful:
            if pattern.search(output):
                issues.append("harmful_content")
                break
        
        # 2. 프롬프트 유출 검사
        for pattern in self._compiled_leak:
            if pattern.search(output):
                issues.append("potential_prompt_leak")
                break
        
        # 3. 시스템 프롬프트와의 유사도 검사 (선택적)
        if context and context.get("system_prompt"):
            system_prompt = context["system_prompt"]
            # 시스템 프롬프트의 일부가 출력에 포함되었는지 확인
            if len(system_prompt) > 20:
                chunks = [system_prompt[i:i+50] for i in range(0, len(system_prompt), 50)]
                for chunk in chunks:
                    if chunk.lower() in output.lower():
                        issues.append("system_prompt_leak")
                        break
        
        # 4. PII 유출 검사
        pii_result = self.pii_detector.detect(output, mask=False)
        if pii_result.has_pii:
            issues.append("pii_in_output")
            result.pii_result = pii_result
        
        # 결과 집계
        result.is_safe = len(issues) == 0
        if not result.is_safe:
            result.blocked = True
            result.blocked_reason = f"출력 검증 실패: {', '.join(issues)}"
        
        result.total_processing_time_ms = (time.perf_counter() - start_time) * 1000
        return result

# ============================================================================
# Groundedness Checker - 근거 검사
# ============================================================================

class GroundednessChecker:
    """
    Groundedness 검사기 - Hallucination 탐지
    
    ================================================================================
    📋 역할: AI 응답이 제공된 소스에 근거하는지 검증
    📅 최종 업데이트: 2026년 2월
    ================================================================================
    
    ⚠️ 주의: Azure Groundedness Detection API 사용 권장
    
    📌 사용 예시:
        >>> checker = GroundednessChecker(
        ...     azure_endpoint="https://xxx.cognitiveservices.azure.com",
        ...     azure_key="your-key"
        ... )
        >>> result = await checker.check(
        ...     response="AI가 생성한 응답",
        ...     sources=["소스 문서 1", "소스 문서 2"]
        ... )
    """
    
    def __init__(
        self,
        azure_endpoint: str | None = None,
        azure_key: str | None = None
    ):
        self.azure_endpoint = azure_endpoint
        self.azure_key = azure_key
        self.logger = logging.getLogger(__name__)
    
    async def check(
        self,
        response: str,
        sources: list[str],
        query: str | None = None
    ) -> GroundednessResult:
        """
        응답의 근거 검사
        
        Args:
            response: AI 응답
            sources: 소스 문서 목록
            query: 원본 질문 (선택)
        
        Returns:
            GroundednessResult: 검사 결과
        """
        result = GroundednessResult()
        
        # Azure API 사용 시
        if self.azure_endpoint and self.azure_key:
            api_result = await self._call_groundedness_api(response, sources, query)
            if api_result:
                result.is_grounded = api_result.get("ungroundedDetected", False) == False
                result.groundedness_score = 1.0 - api_result.get("ungroundedPercentage", 0) / 100
                result.ungrounded_claims = api_result.get("ungroundedSegments", [])
                return result
        
        # 로컬 휴리스틱 (간단한 버전)
        # 실제 프로덕션에서는 Azure API 사용 권장
        combined_sources = " ".join(sources).lower()
        response_lower = response.lower()
        
        # 응답의 주요 명사/키워드가 소스에 있는지 확인
        # (매우 단순화된 버전)
        words = set(response_lower.split())
        source_words = set(combined_sources.split())
        
        common_words = {"the", "a", "an", "is", "are", "was", "were", "이", "그", "저", "을", "를"}
        meaningful_words = words - common_words
        
        if meaningful_words:
            coverage = len(meaningful_words & source_words) / len(meaningful_words)
            result.source_coverage = coverage
            result.groundedness_score = coverage
            result.is_grounded = coverage > 0.3  # 30% 이상 커버리지
        
        return result
    
    async def _call_groundedness_api(
        self,
        response: str,
        sources: list[str],
        query: str | None = None
    ) -> dict[str, Any] | None:
        """Azure Groundedness Detection API 호출"""
        if not self.azure_endpoint or not self.azure_key:
            return None
        
        url = f"{self.azure_endpoint}/contentsafety/text:detectGroundedness?api-version=2024-09-15-preview"
        
        headers = {
            "Content-Type": "application/json",
            "Ocp-Apim-Subscription-Key": self.azure_key
        }
        
        payload = {
            "domain": "Generic",
            "task": "QnA" if query else "Summarization",
            "text": response,
            "groundingSources": sources,
            "reasoning": False
        }
        
        if query:
            payload["query"] = query
        
        try:
            async with aiohttp.ClientSession() as session:
                async with session.post(url, json=payload, headers=headers, timeout=15) as resp:
                    if resp.status == 200:
                        return await resp.json()
                    else:
                        self.logger.warning(f"Groundedness API 오류: {resp.status}")
                        return None
        except Exception as e:
            self.logger.error(f"Groundedness API 호출 실패: {e}")
            return None

# ============================================================================
# Security Audit Logger
# ============================================================================

class SecurityAuditLogger:
    """
    보안 감사 로거
    
    ================================================================================
    📋 역할: 보안 이벤트를 감사 로그로 기록
    📅 최종 업데이트: 2026년 2월
    ================================================================================
    
    📌 사용 예시:
        >>> audit = SecurityAuditLogger(log_file="./security_audit.log")
        >>> audit.log_event(AuditLogEntry(
        ...     session_id="sess-123",
        ...     threat_detected=True,
        ...     threat_type="prompt_injection"
        ... ))
    """
    
    def __init__(self, log_file: str | None = None):
        """
        SecurityAuditLogger 초기화
        
        Args:
            log_file: 로그 파일 경로 (None이면 콘솔 출력)
        """
        self.log_file = log_file
        self.logger = logging.getLogger("security_audit")
        
        if log_file:
            handler = logging.FileHandler(log_file, encoding="utf-8")
            handler.setFormatter(logging.Formatter('%(message)s'))
            self.logger.addHandler(handler)
            self.logger.setLevel(logging.INFO)
    
    def log_event(self, entry: AuditLogEntry) -> None:
        """
        감사 이벤트 기록
        
        Args:
            entry: 감사 로그 항목
        """
        log_data = {
            "timestamp": entry.timestamp.isoformat(),
            "session_id": entry.session_id,
            "stage": entry.stage.value,
            "input_hash": entry.input_hash,
            "result": entry.result,
            "threat_detected": entry.threat_detected,
            "threat_type": entry.threat_type,
            "action_taken": entry.action_taken,
            "metadata": entry.metadata
        }
        
        self.logger.info(json.dumps(log_data, ensure_ascii=False))
    
    @staticmethod
    def hash_input(text: str) -> str:
        """입력 텍스트의 SHA-256 해시 생성"""
        return hashlib.sha256(text.encode()).hexdigest()[:16]

# ============================================================================
# Security Orchestrator - 통합 보안 파이프라인
# ============================================================================

class SecurityOrchestrator:
    """
    보안 오케스트레이터 - 통합 보안 검증 파이프라인
    
    ================================================================================
    📋 역할: 모든 보안 컴포넌트를 조율하여 입력/출력 검증 수행
    📅 최종 업데이트: 2026년 2월
    ================================================================================
    
    🎯 파이프라인 구조:
        Input → [PromptShield] → [JailbreakDetector] → [PIIDetector] → Validated Input
        Output → [OutputValidator] → [GroundednessChecker] → Validated Output
    
    📌 사용 예시:
        >>> orchestrator = SecurityOrchestrator(SecurityConfig(
        ...     threat_level=ThreatLevel.HIGH,
        ...     enable_prompt_shield=True,
        ...     enable_jailbreak_detection=True,
        ...     azure_content_safety_endpoint="https://xxx.cognitiveservices.azure.com",
        ...     azure_content_safety_key="your-key"
        ... ))
        >>>
        >>> # 입력 검증
        >>> input_result = await orchestrator.validate_input(user_input)
        >>> if not input_result.is_safe:
        ...     print(f"차단: {input_result.blocked_reason}")
        ...     return
        >>>
        >>> # (AI 처리)
        >>> ai_response = await call_ai(input_result.sanitized_input)
        >>>
        >>> # 출력 검증
        >>> output_result = await orchestrator.validate_output(ai_response, sources)
        >>> if output_result.is_safe:
        ...     return output_result.sanitized_input  # 정화된 출력
    """
    
    def __init__(self, config: SecurityConfig | None = None):
        """
        SecurityOrchestrator 초기화
        
        Args:
            config: 보안 설정 (기본값 사용 시 None)
        """
        self.config = config or SecurityConfig()
        
        # 컴포넌트 초기화
        self.prompt_shield = PromptShield(
            azure_endpoint=self.config.azure_content_safety_endpoint,
            azure_key=self.config.azure_content_safety_key,
            custom_patterns=self.config.custom_blocked_patterns
        ) if self.config.enable_prompt_shield else None
        
        self.jailbreak_detector = JailbreakDetector(
            sensitivity=0.7 if self.config.threat_level in [ThreatLevel.HIGH, ThreatLevel.PARANOID] else 0.5
        ) if self.config.enable_jailbreak_detection else None
        
        self.pii_detector = PIIDetector(
            mask_char=self.config.pii_mask_char
        ) if self.config.enable_pii_detection else None
        
        self.output_validator = OutputValidator(
            pii_detector=self.pii_detector
        ) if self.config.enable_output_validation else None
        
        self.groundedness_checker = GroundednessChecker(
            azure_endpoint=self.config.azure_content_safety_endpoint,
            azure_key=self.config.azure_content_safety_key
        ) if self.config.enable_groundedness_check else None
        
        self.audit_logger = SecurityAuditLogger() if self.config.enable_audit_logging else None
        
        self.logger = logging.getLogger(__name__)
    
    async def validate_input(
        self,
        text: str,
        session_id: str = "",
        documents: list[str] | None = None
    ) -> ValidationResult:
        """
        사용자 입력 검증
        
        Args:
            text: 사용자 입력
            session_id: 세션 ID (감사 로그용)
            documents: 함께 검증할 문서 (간접 주입 탐지)
        
        Returns:
            ValidationResult: 검증 결과
        """
        start_time = time.perf_counter()
        
        result = ValidationResult(stage=ValidationStage.INPUT)
        result.sanitized_input = text
        
        # 길이 검사
        if len(text) > self.config.max_input_length:
            result.is_safe = False
            result.blocked = True
            result.blocked_reason = f"입력 길이 초과: {len(text)} > {self.config.max_input_length}"
            return result
        
        # 1. Prompt Shield 검사
        if self.prompt_shield:
            use_api = self.config.threat_level in [ThreatLevel.HIGH, ThreatLevel.PARANOID]
            shield_result = await self.prompt_shield.analyze(text, documents, use_api=use_api)
            result.shield_result = shield_result
            
            if shield_result.is_attack:
                result.is_safe = False
                if self.config.block_on_detection:
                    result.blocked = True
                    result.blocked_reason = f"Prompt Injection 탐지: {shield_result.attack_type.value}"
                    self._log_audit(session_id, "input", text, "blocked", True, shield_result.attack_type.value)
                    result.total_processing_time_ms = (time.perf_counter() - start_time) * 1000
                    return result
        
        # 2. Jailbreak 검사
        if self.jailbreak_detector:
            jailbreak_result = self.jailbreak_detector.detect(text)
            result.jailbreak_result = jailbreak_result
            
            if jailbreak_result.is_jailbreak:
                result.is_safe = False
                if self.config.block_on_detection and jailbreak_result.risk_score >= 50:
                    result.blocked = True
                    result.blocked_reason = f"Jailbreak 시도: {', '.join(jailbreak_result.detected_techniques)}"
                    self._log_audit(session_id, "input", text, "blocked", True, "jailbreak")
                    result.total_processing_time_ms = (time.perf_counter() - start_time) * 1000
                    return result
        
        # 3. PII 탐지 및 마스킹
        if self.pii_detector:
            pii_result = self.pii_detector.detect(text, mask=True)
            result.pii_result = pii_result
            
            if pii_result.has_pii:
                # PII는 마스킹하고 통과 (차단하지 않음)
                result.sanitized_input = pii_result.masked_text
                self.logger.info(f"PII 탐지 및 마스킹: {pii_result.pii_count}개")
        
        # 감사 로그
        self._log_audit(session_id, "input", text, "passed" if result.is_safe else "flagged", not result.is_safe, None)
        
        result.total_processing_time_ms = (time.perf_counter() - start_time) * 1000
        return result
    
    async def validate_output(
        self,
        output: str,
        sources: list[str] | None = None,
        context: dict[str, Any] | None = None,
        session_id: str = ""
    ) -> ValidationResult:
        """
        AI 출력 검증
        
        Args:
            output: AI 모델 출력
            sources: 응답 생성에 사용된 소스 (Groundedness 검사용)
            context: 추가 컨텍스트 (system_prompt 등)
            session_id: 세션 ID
        
        Returns:
            ValidationResult: 검증 결과
        """
        start_time = time.perf_counter()
        
        result = ValidationResult(stage=ValidationStage.OUTPUT)
        result.sanitized_input = output
        
        # 1. 출력 검증
        if self.output_validator:
            output_result = await self.output_validator.validate(output, context)
            if not output_result.is_safe:
                result.is_safe = False
                result.blocked = output_result.blocked
                result.blocked_reason = output_result.blocked_reason
                result.pii_result = output_result.pii_result
                self._log_audit(session_id, "output", output, "blocked", True, "output_validation_failed")
                result.total_processing_time_ms = (time.perf_counter() - start_time) * 1000
                return result
        
        # 2. Groundedness 검사
        if self.groundedness_checker and sources:
            groundedness_result = await self.groundedness_checker.check(output, sources)
            result.groundedness_result = groundedness_result
            
            if not groundedness_result.is_grounded:
                # Groundedness 실패는 경고만 (차단하지 않음)
                self.logger.warning(f"Groundedness 검사 실패: score={groundedness_result.groundedness_score:.2f}")
        
        # 3. PII 마스킹 (출력에서도)
        if self.pii_detector:
            pii_result = self.pii_detector.detect(output, mask=True)
            if pii_result.has_pii:
                result.sanitized_input = pii_result.masked_text
                result.pii_result = pii_result
        
        self._log_audit(session_id, "output", output, "passed", False, None)
        result.total_processing_time_ms = (time.perf_counter() - start_time) * 1000
        return result
    
    def _log_audit(
        self,
        session_id: str,
        stage: str,
        text: str,
        result: str,
        threat_detected: bool,
        threat_type: str | None
    ) -> None:
        """감사 로그 기록"""
        if self.audit_logger:
            entry = AuditLogEntry(
                session_id=session_id,
                stage=ValidationStage(stage),
                input_hash=SecurityAuditLogger.hash_input(text),
                result=result,
                threat_detected=threat_detected,
                threat_type=threat_type,
                action_taken="blocked" if threat_detected else "allowed"
            )
            self.audit_logger.log_event(entry)
