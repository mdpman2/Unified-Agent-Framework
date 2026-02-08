#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Unified Agent Framework - Structured Output 모듈 (Structured Output Module)

================================================================================
📁 파일 위치: unified_agent/structured_output.py
📋 역할: GPT-5.2 Structured Outputs를 활용한 JSON Schema 강제 출력
📅 최종 업데이트: 2026년 2월 4일
📦 버전: v3.5.0
✅ 테스트: test_new_modules.py, test_v35_scenarios.py
================================================================================

🎯 주요 구성 요소:
    1. OutputSchema - JSON Schema 정의 클래스
    2. StructuredOutputConfig - 구조화된 출력 설정
    3. StructuredOutputParser - 출력 파싱 및 검증
    4. StructuredOutputValidator - 스키마 검증기
    5. @structured_output - 데코레이터를 통한 자동 적용
    6. StructuredOutputClient - OpenAI API 연동 클라이언트

🔧 2026년 2월 기능:
    - OpenAI GPT-5.2 Structured Outputs 네이티브 지원
    - JSON Schema Draft 2020-12 호환
    - Pydantic 모델 자동 변환
    - 스트리밍 + 구조화된 출력 동시 지원
    - 재시도 및 폴백 로직
    - 부분 출력 파싱 (Partial Parsing)

📌 사용 예시:
    >>> from unified_agent.structured_output import (
    ...     StructuredOutput, OutputSchema, structured_output
    ... )
    >>>
    >>> # 방법 1: 데코레이터 사용
    >>> @structured_output(schema={
    ...     "type": "object",
    ...     "properties": {
    ...         "summary": {"type": "string"},
    ...         "confidence": {"type": "number", "minimum": 0, "maximum": 1}
    ...     },
    ...     "required": ["summary", "confidence"]
    ... })
    >>> async def analyze(text: str) -> dict:
    ...     return await llm_call(text)
    >>>
    >>> # 방법 2: Pydantic 모델 사용
    >>> from pydantic import BaseModel
    >>> class AnalysisResult(BaseModel):
    ...     summary: str
    ...     confidence: float
    ...     sources: list[str]
    >>>
    >>> client = StructuredOutputClient()
    >>> result = await client.generate(prompt, response_model=AnalysisResult)

⚠️ 주의사항:
    - Structured Outputs는 GPT-4o 이상, GPT-5 계열에서 지원됩니다.
    - 복잡한 스키마는 토큰 사용량이 증가할 수 있습니다.
    - additionalProperties: false를 사용하면 엄격한 검증이 적용됩니다.

🔗 관련 문서:
    - OpenAI Structured Outputs: https://platform.openai.com/docs/guides/structured-outputs
    - JSON Schema: https://json-schema.org/
"""

from __future__ import annotations

import json
import asyncio
import logging
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from functools import wraps
from typing import (
    Any,
    Callable,
    Generic,
    Type,
    TypeVar,
    get_type_hints,
)

try:
    from pydantic import BaseModel, ValidationError
    PYDANTIC_AVAILABLE = True
except ImportError:
    PYDANTIC_AVAILABLE = False
    BaseModel = None
    ValidationError = None

try:
    from openai import AsyncOpenAI
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False
    AsyncOpenAI = None

__all__ = [
    # Enums
    "SchemaFormat",
    "OutputMode",
    "ValidationLevel",
    # Config & Schema
    "OutputSchema",
    "StructuredOutputConfig",
    # Results
    "StructuredOutputResult",
    "ValidationError",
    "ParseError",
    # Core Components
    "StructuredOutputParser",
    "StructuredOutputValidator",
    "StructuredOutputClient",
    # Decorators
    "structured_output",
    # Utilities
    "pydantic_to_schema",
    "schema_to_pydantic",
    "infer_schema_from_example",
]

# Type variable for generic model support
T = TypeVar("T")

# ============================================================================
# Enums
# ============================================================================

class SchemaFormat(str, Enum):
    """스키마 포맷"""
    JSON_SCHEMA = "json_schema"      # JSON Schema Draft 2020-12
    PYDANTIC = "pydantic"            # Pydantic 모델
    TYPESCRIPT = "typescript"         # TypeScript 인터페이스 (변환용)

class OutputMode(str, Enum):
    """출력 모드"""
    STRICT = "strict"               # 엄격한 스키마 준수 (additionalProperties: false)
    FLEXIBLE = "flexible"           # 유연한 스키마 (추가 필드 허용)
    PARTIAL = "partial"             # 부분 출력 허용 (스트리밍용)

class ValidationLevel(str, Enum):
    """검증 수준"""
    NONE = "none"                   # 검증 없음
    SCHEMA_ONLY = "schema_only"     # 스키마 검증만
    SEMANTIC = "semantic"           # 의미적 검증 포함
    FULL = "full"                   # 전체 검증 (타입 + 범위 + 의미)

# ============================================================================
# Data Classes
# ============================================================================

@dataclass(frozen=True, slots=True)
class OutputSchema:
    """
    JSON Schema 정의 클래스
    
    ================================================================================
    📋 역할: 구조화된 출력을 위한 JSON Schema 래퍼
    📅 최종 업데이트: 2026년 2월
    ================================================================================
    
    Attributes:
        name: 스키마 이름
        schema: JSON Schema 딕셔너리
        description: 스키마 설명
        strict: 엄격 모드 (additionalProperties: false)
        examples: 예시 출력
    
    📌 사용 예시:
        >>> schema = OutputSchema(
        ...     name="analysis_result",
        ...     schema={
        ...         "type": "object",
        ...         "properties": {
        ...             "summary": {"type": "string", "description": "요약"},
        ...             "score": {"type": "number", "minimum": 0, "maximum": 100}
        ...         },
        ...         "required": ["summary", "score"]
        ...     },
        ...     strict=True
        ... )
    """
    name: str
    schema: dict[str, Any]
    description: str = ""
    strict: bool = True
    examples: list[dict[str, Any]] = field(default_factory=list)
    
    def __post_init__(self):
        """스키마 후처리"""
        # strict 모드일 때 additionalProperties 설정
        if self.strict and "additionalProperties" not in self.schema:
            self.schema["additionalProperties"] = False
    
    def to_openai_format(self) -> dict[str, Any]:
        """
        OpenAI API 포맷으로 변환
        
        Returns:
            Dict: OpenAI response_format용 딕셔너리
        """
        return {
            "type": "json_schema",
            "json_schema": {
                "name": self.name,
                "description": self.description,
                "schema": self.schema,
                "strict": self.strict
            }
        }
    
    def validate(self, data: dict[str, Any]) -> bool:
        """
        데이터가 스키마를 준수하는지 검증
        
        Args:
            data: 검증할 데이터
        
        Returns:
            bool: 유효 여부
        """
        try:
            # 기본 타입 검증
            if self.schema.get("type") == "object":
                if not isinstance(data, dict):
                    return False
                
                # 필수 필드 검증
                required = self.schema.get("required", [])
                for field_name in required:
                    if field_name not in data:
                        return False
                
                # 프로퍼티 검증
                properties = self.schema.get("properties", {})
                for key, value in data.items():
                    if key in properties:
                        prop_schema = properties[key]
                        if not self._validate_property(value, prop_schema):
                            return False
                    elif self.strict:
                        return False  # 추가 프로퍼티 불허
            
            return True
        except (TypeError, KeyError, ValueError) as e:
            logger.debug(f"[스키마 검증 실패] {e}")
            return False
    
    def _validate_property(self, value: Any, prop_schema: dict[str, Any]) -> bool:
        """프로퍼티 검증"""
        prop_type = prop_schema.get("type")
        
        if prop_type == "string":
            if not isinstance(value, str):
                return False
            # enum 검증
            if "enum" in prop_schema and value not in prop_schema["enum"]:
                return False
            # 길이 검증
            if "minLength" in prop_schema and len(value) < prop_schema["minLength"]:
                return False
            if "maxLength" in prop_schema and len(value) > prop_schema["maxLength"]:
                return False
                
        elif prop_type == "number" or prop_type == "integer":
            if not isinstance(value, (int, float)):
                return False
            if prop_type == "integer" and not isinstance(value, int):
                return False
            # 범위 검증
            if "minimum" in prop_schema and value < prop_schema["minimum"]:
                return False
            if "maximum" in prop_schema and value > prop_schema["maximum"]:
                return False
                
        elif prop_type == "boolean":
            if not isinstance(value, bool):
                return False
                
        elif prop_type == "array":
            if not isinstance(value, list):
                return False
            # 아이템 검증
            if "items" in prop_schema:
                for item in value:
                    if not self._validate_property(item, prop_schema["items"]):
                        return False
            # 길이 검증
            if "minItems" in prop_schema and len(value) < prop_schema["minItems"]:
                return False
            if "maxItems" in prop_schema and len(value) > prop_schema["maxItems"]:
                return False
                
        elif prop_type == "object":
            if not isinstance(value, dict):
                return False
        
        return True
    
    @classmethod
    def from_pydantic(cls, model: Type, name: str | None = None) -> "OutputSchema":
        """
        Pydantic 모델에서 OutputSchema 생성
        
        Args:
            model: Pydantic 모델 클래스
            name: 스키마 이름 (기본값: 모델 이름)
        
        Returns:
            OutputSchema: 변환된 스키마
        """
        if not PYDANTIC_AVAILABLE:
            raise ImportError("Pydantic is required for this feature")
        
        schema = model.model_json_schema()
        return cls(
            name=name or model.__name__,
            schema=schema,
            description=model.__doc__ or "",
            strict=True
        )

@dataclass(frozen=True, slots=True)
class StructuredOutputConfig:
    """
    구조화된 출력 설정
    
    Attributes:
        mode: 출력 모드 (strict/flexible/partial)
        validation_level: 검증 수준
        max_retries: 최대 재시도 횟수
        fallback_to_unstructured: 실패 시 비구조화 출력 폴백
        parse_partial: 부분 JSON 파싱 시도
        timeout_seconds: 타임아웃
    """
    mode: OutputMode = OutputMode.STRICT
    validation_level: ValidationLevel = ValidationLevel.FULL
    max_retries: int = 2
    fallback_to_unstructured: bool = False
    parse_partial: bool = True
    timeout_seconds: float = 30.0
    
    # 모델 설정
    model: str = "gpt-5.2"
    temperature: float | None = None  # Structured Outputs에서는 보통 낮은 값 권장

@dataclass(slots=True)
class StructuredOutputResult(Generic[T]):
    """
    구조화된 출력 결과
    
    Attributes:
        success: 성공 여부
        data: 파싱된 데이터
        raw_output: 원본 출력
        validation_errors: 검증 오류 목록
        retries: 재시도 횟수
        processing_time_ms: 처리 시간
    """
    success: bool
    data: T | None = None
    raw_output: str = ""
    validation_errors: list[str] = field(default_factory=list)
    retries: int = 0
    processing_time_ms: float = 0.0
    model_used: str = ""
    tokens_used: int = 0

# ============================================================================
# Exceptions
# ============================================================================

class StructuredOutputError(Exception):
    """구조화된 출력 관련 기본 예외"""
    pass

class SchemaValidationError(StructuredOutputError):
    """스키마 검증 오류"""
    def __init__(self, message: str, errors: list[str] = None):
        super().__init__(message)
        self.errors = errors or []

class ParseError(StructuredOutputError):
    """JSON 파싱 오류"""
    def __init__(self, message: str, raw_output: str = ""):
        super().__init__(message)
        self.raw_output = raw_output

# ============================================================================
# Parser
# ============================================================================

class StructuredOutputParser:
    """
    구조화된 출력 파서
    
    ================================================================================
    📋 역할: LLM 출력을 구조화된 데이터로 파싱
    📅 최종 업데이트: 2026년 2월
    ================================================================================
    
    📌 사용 예시:
        >>> parser = StructuredOutputParser()
        >>> result = parser.parse('{"name": "test", "value": 42}', schema)
    """
    
    def __init__(self, config: StructuredOutputConfig | None = None):
        self.config = config or StructuredOutputConfig()
        self.logger = logging.getLogger(__name__)
    
    def parse(
        self,
        output: str,
        schema: OutputSchema | None = None
    ) -> StructuredOutputResult:
        """
        출력 파싱
        
        Args:
            output: LLM 출력 문자열
            schema: 검증용 스키마 (선택)
        
        Returns:
            StructuredOutputResult: 파싱 결과
        """
        start_time = time.perf_counter()
        
        result = StructuredOutputResult(success=False, raw_output=output)
        
        try:
            # JSON 추출 시도
            json_str = self._extract_json(output)
            
            if json_str:
                data = json.loads(json_str)
                result.data = data
                
                # 스키마 검증
                if schema:
                    if schema.validate(data):
                        result.success = True
                    else:
                        result.validation_errors.append("Schema validation failed")
                else:
                    result.success = True
            else:
                result.validation_errors.append("No valid JSON found in output")
                
        except json.JSONDecodeError as e:
            result.validation_errors.append(f"JSON parse error: {str(e)}")
            
            # 부분 파싱 시도
            if self.config.parse_partial:
                partial_data = self._parse_partial_json(output)
                if partial_data:
                    result.data = partial_data
                    result.validation_errors.append("Partial JSON parsed")
                    
        except Exception as e:
            result.validation_errors.append(f"Parse error: {str(e)}")
        
        result.processing_time_ms = (time.perf_counter() - start_time) * 1000
        return result
    
    def _extract_json(self, text: str) -> str | None:
        """
        텍스트에서 JSON 추출
        
        코드 블록, 중괄호 블록 등에서 JSON 추출 시도
        """
        import re
        
        # 1. 코드 블록에서 추출
        code_block_pattern = r'```(?:json)?\s*\n?([\s\S]*?)\n?```'
        matches = re.findall(code_block_pattern, text)
        for match in matches:
            try:
                json.loads(match.strip())
                return match.strip()
            except json.JSONDecodeError:
                continue
        
        # 2. 중괄호/대괄호 블록 추출
        # 가장 외부의 JSON 객체/배열 찾기
        brace_count = 0
        bracket_count = 0
        start_idx = -1
        
        for i, char in enumerate(text):
            if char == '{':
                if brace_count == 0 and bracket_count == 0:
                    start_idx = i
                brace_count += 1
            elif char == '}':
                brace_count -= 1
                if brace_count == 0 and bracket_count == 0 and start_idx >= 0:
                    json_str = text[start_idx:i+1]
                    try:
                        json.loads(json_str)
                        return json_str
                    except json.JSONDecodeError:
                        start_idx = -1
            elif char == '[':
                if brace_count == 0 and bracket_count == 0:
                    start_idx = i
                bracket_count += 1
            elif char == ']':
                bracket_count -= 1
                if brace_count == 0 and bracket_count == 0 and start_idx >= 0:
                    json_str = text[start_idx:i+1]
                    try:
                        json.loads(json_str)
                        return json_str
                    except json.JSONDecodeError:
                        start_idx = -1
        
        return None
    
    def _parse_partial_json(self, text: str) -> dict[str, Any] | None:
        """
        불완전한 JSON 파싱 시도
        
        스트리밍 등으로 중간에 끊긴 JSON 처리
        """
        import re
        
        # 불완전한 JSON 수정 시도
        # 마지막 쉼표 제거
        text = re.sub(r',\s*$', '', text)
        
        # 닫히지 않은 중괄호/대괄호 추가
        open_braces = text.count('{') - text.count('}')
        open_brackets = text.count('[') - text.count(']')
        
        if open_braces > 0 or open_brackets > 0:
            text = text.rstrip()
            if text.endswith(','):
                text = text[:-1]
            text += ']' * open_brackets + '}' * open_braces
        
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            return None

# ============================================================================
# Validator
# ============================================================================

class StructuredOutputValidator:
    """
    구조화된 출력 검증기
    
    ================================================================================
    📋 역할: 스키마 및 의미적 검증 수행
    📅 최종 업데이트: 2026년 2월
    ================================================================================
    """
    
    def __init__(self, schema: OutputSchema):
        self.schema = schema
        self.logger = logging.getLogger(__name__)
    
    def validate(
        self,
        data: dict[str, Any],
        level: ValidationLevel = ValidationLevel.FULL
    ) -> tuple[bool, list[str]]:
        """
        데이터 검증
        
        Args:
            data: 검증할 데이터
            level: 검증 수준
        
        Returns:
            tuple[bool, list[str]]: (유효 여부, 오류 목록)
        """
        errors = []
        
        if level == ValidationLevel.NONE:
            return True, []
        
        # 스키마 검증
        if not self.schema.validate(data):
            errors.append("Schema validation failed")
        
        if level == ValidationLevel.SCHEMA_ONLY:
            return len(errors) == 0, errors
        
        # 타입 및 범위 검증 (FULL)
        type_errors = self._validate_types(data, self.schema.schema)
        errors.extend(type_errors)
        
        return len(errors) == 0, errors
    
    def _validate_types(
        self,
        data: Any,
        schema: dict[str, Any],
        path: str = ""
    ) -> list[str]:
        """타입 검증"""
        errors = []
        
        schema_type = schema.get("type")
        
        if schema_type == "object" and isinstance(data, dict):
            properties = schema.get("properties", {})
            for key, prop_schema in properties.items():
                if key in data:
                    sub_path = f"{path}.{key}" if path else key
                    sub_errors = self._validate_types(data[key], prop_schema, sub_path)
                    errors.extend(sub_errors)
        
        elif schema_type == "array" and isinstance(data, list):
            items_schema = schema.get("items", {})
            for i, item in enumerate(data):
                sub_path = f"{path}[{i}]"
                sub_errors = self._validate_types(item, items_schema, sub_path)
                errors.extend(sub_errors)
        
        elif schema_type == "string":
            if not isinstance(data, str):
                errors.append(f"{path}: Expected string, got {type(data).__name__}")
        
        elif schema_type == "number":
            if not isinstance(data, (int, float)):
                errors.append(f"{path}: Expected number, got {type(data).__name__}")
        
        elif schema_type == "integer":
            if not isinstance(data, int) or isinstance(data, bool):
                errors.append(f"{path}: Expected integer, got {type(data).__name__}")
        
        elif schema_type == "boolean":
            if not isinstance(data, bool):
                errors.append(f"{path}: Expected boolean, got {type(data).__name__}")
        
        return errors

# ============================================================================
# Client
# ============================================================================

class StructuredOutputClient:
    """
    구조화된 출력 클라이언트
    
    ================================================================================
    📋 역할: OpenAI API를 통한 구조화된 출력 생성
    📅 최종 업데이트: 2026년 2월
    ================================================================================
    
    📌 사용 예시:
        >>> client = StructuredOutputClient()
        >>> 
        >>> # 스키마 사용
        >>> schema = OutputSchema(
        ...     name="analysis",
        ...     schema={"type": "object", "properties": {...}}
        ... )
        >>> result = await client.generate("분석해주세요", schema=schema)
        >>> 
        >>> # Pydantic 모델 사용
        >>> class Analysis(BaseModel):
        ...     summary: str
        ...     score: float
        >>> result = await client.generate("분석해주세요", response_model=Analysis)
    """
    
    def __init__(
        self,
        config: StructuredOutputConfig | None = None,
        api_key: str | None = None,
        base_url: str | None = None
    ):
        self.config = config or StructuredOutputConfig()
        self.parser = StructuredOutputParser(self.config)
        self.logger = logging.getLogger(__name__)
        
        # OpenAI 클라이언트 초기화
        if OPENAI_AVAILABLE:
            kwargs = {}
            if api_key:
                kwargs["api_key"] = api_key
            if base_url:
                kwargs["base_url"] = base_url
            self._client = AsyncOpenAI(**kwargs) if kwargs else AsyncOpenAI()
        else:
            self._client = None
    
    async def generate(
        self,
        prompt: str,
        schema: OutputSchema | None = None,
        response_model: Type[T] | None = None,
        system_prompt: str | None = None,
        **kwargs
    ) -> StructuredOutputResult[T]:
        """
        구조화된 출력 생성
        
        Args:
            prompt: 사용자 프롬프트
            schema: OutputSchema 인스턴스
            response_model: Pydantic 모델 클래스 (schema 대신 사용)
            system_prompt: 시스템 프롬프트
            **kwargs: 추가 OpenAI API 파라미터
        
        Returns:
            StructuredOutputResult: 생성 결과
        """
        start_time = time.perf_counter()
        
        if not self._client:
            raise RuntimeError("OpenAI client not available. Install openai package.")
        
        # Pydantic 모델을 스키마로 변환
        if response_model and not schema:
            schema = OutputSchema.from_pydantic(response_model)
        
        if not schema:
            raise ValueError("Either schema or response_model must be provided")
        
        # 메시지 구성
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})
        
        # API 호출
        result = StructuredOutputResult[T](success=False)
        retries = 0
        
        while retries <= self.config.max_retries:
            try:
                response = await self._client.chat.completions.create(
                    model=kwargs.get("model", self.config.model),
                    messages=messages,
                    response_format=schema.to_openai_format(),
                    temperature=kwargs.get("temperature", self.config.temperature),
                    **{k: v for k, v in kwargs.items() if k not in ["model", "temperature"]}
                )
                
                content = response.choices[0].message.content
                result.raw_output = content
                result.model_used = response.model
                result.tokens_used = response.usage.total_tokens if response.usage else 0
                
                # 파싱
                parse_result = self.parser.parse(content, schema)
                
                if parse_result.success:
                    result.success = True
                    
                    # Pydantic 모델로 변환
                    if response_model and PYDANTIC_AVAILABLE:
                        result.data = response_model.model_validate(parse_result.data)
                    else:
                        result.data = parse_result.data
                    
                    break
                else:
                    result.validation_errors = parse_result.validation_errors
                    retries += 1
                    
            except Exception as e:
                result.validation_errors.append(str(e))
                retries += 1
                self.logger.warning(f"Retry {retries}/{self.config.max_retries}: {e}")
                await asyncio.sleep(1)  # 재시도 전 대기
        
        result.retries = retries
        result.processing_time_ms = (time.perf_counter() - start_time) * 1000
        
        return result
    
    async def generate_stream(
        self,
        prompt: str,
        schema: OutputSchema,
        system_prompt: str | None = None,
        **kwargs
    ):
        """
        스트리밍 구조화된 출력 생성
        
        Args:
            prompt: 사용자 프롬프트
            schema: OutputSchema 인스턴스
            system_prompt: 시스템 프롬프트
        
        Yields:
            str: 스트리밍 청크
        """
        if not self._client:
            raise RuntimeError("OpenAI client not available")
        
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})
        
        stream = await self._client.chat.completions.create(
            model=kwargs.get("model", self.config.model),
            messages=messages,
            response_format=schema.to_openai_format(),
            stream=True,
            **{k: v for k, v in kwargs.items() if k not in ["model"]}
        )
        
        async for chunk in stream:
            if chunk.choices[0].delta.content:
                yield chunk.choices[0].delta.content

# ============================================================================
# Decorator
# ============================================================================

def structured_output(
    schema: dict[str, Any] | None = None,
    response_model: Type | None = None,
    name: str = "response",
    strict: bool = True,
    config: StructuredOutputConfig | None = None
):
    """
    구조화된 출력 데코레이터
    
    ================================================================================
    📋 역할: 함수에 구조화된 출력 검증 자동 적용
    📅 최종 업데이트: 2026년 2월
    ================================================================================
    
    📌 사용 예시:
        >>> @structured_output(schema={
        ...     "type": "object",
        ...     "properties": {
        ...         "result": {"type": "string"},
        ...         "confidence": {"type": "number"}
        ...     },
        ...     "required": ["result", "confidence"]
        ... })
        >>> async def analyze(text: str) -> dict:
        ...     response = await llm.generate(text)
        ...     return response
        >>>
        >>> # Pydantic 모델 사용
        >>> @structured_output(response_model=AnalysisResult)
        >>> async def analyze(text: str) -> AnalysisResult:
        ...     ...
    
    Args:
        schema: JSON Schema 딕셔너리
        response_model: Pydantic 모델 클래스
        name: 스키마 이름
        strict: 엄격 모드
        config: 설정 객체
    """
    def decorator(func: Callable) -> Callable:
        # 스키마 준비
        if response_model:
            output_schema = OutputSchema.from_pydantic(response_model)
        elif schema:
            output_schema = OutputSchema(name=name, schema=schema, strict=strict)
        else:
            raise ValueError("Either schema or response_model must be provided")
        
        parser = StructuredOutputParser(config or StructuredOutputConfig())
        
        @wraps(func)
        async def async_wrapper(*args, **kwargs):
            result = await func(*args, **kwargs)
            
            # 문자열 결과면 파싱
            if isinstance(result, str):
                parse_result = parser.parse(result, output_schema)
                if parse_result.success:
                    if response_model and PYDANTIC_AVAILABLE:
                        return response_model.model_validate(parse_result.data)
                    return parse_result.data
                else:
                    raise SchemaValidationError(
                        "Output validation failed",
                        parse_result.validation_errors
                    )
            
            # 딕셔너리면 검증만
            elif isinstance(result, dict):
                if output_schema.validate(result):
                    if response_model and PYDANTIC_AVAILABLE:
                        return response_model.model_validate(result)
                    return result
                else:
                    raise SchemaValidationError("Schema validation failed")
            
            # Pydantic 모델이면 통과
            elif PYDANTIC_AVAILABLE and isinstance(result, BaseModel):
                return result
            
            return result
        
        @wraps(func)
        def sync_wrapper(*args, **kwargs):
            result = func(*args, **kwargs)
            
            if isinstance(result, str):
                parse_result = parser.parse(result, output_schema)
                if parse_result.success:
                    if response_model and PYDANTIC_AVAILABLE:
                        return response_model.model_validate(parse_result.data)
                    return parse_result.data
                else:
                    raise SchemaValidationError(
                        "Output validation failed",
                        parse_result.validation_errors
                    )
            
            elif isinstance(result, dict):
                if output_schema.validate(result):
                    if response_model and PYDANTIC_AVAILABLE:
                        return response_model.model_validate(result)
                    return result
                else:
                    raise SchemaValidationError("Schema validation failed")
            
            return result
        
        # 비동기/동기 함수 구분
        if asyncio.iscoroutinefunction(func):
            return async_wrapper
        else:
            return sync_wrapper
    
    return decorator

# ============================================================================
# Utilities
# ============================================================================

def pydantic_to_schema(model: Type) -> OutputSchema:
    """
    Pydantic 모델을 OutputSchema로 변환
    
    Args:
        model: Pydantic 모델 클래스
    
    Returns:
        OutputSchema: 변환된 스키마
    """
    return OutputSchema.from_pydantic(model)

def schema_to_pydantic(schema: OutputSchema) -> Type | None:
    """
    OutputSchema를 Pydantic 모델로 변환 (실험적)
    
    Args:
        schema: OutputSchema 인스턴스
    
    Returns:
        Type: 동적 생성된 Pydantic 모델
    """
    if not PYDANTIC_AVAILABLE:
        return None
    
    from pydantic import create_model
    
    properties = schema.schema.get("properties", {})
    required = set(schema.schema.get("required", []))
    
    fields = {}
    for name, prop in properties.items():
        python_type = _json_type_to_python(prop.get("type", "string"))
        default = ... if name in required else None
        fields[name] = (python_type, default)
    
    return create_model(schema.name, **fields)

def _json_type_to_python(json_type: str) -> Type:
    """JSON 타입을 Python 타입으로 변환"""
    type_mapping = {
        "string": str,
        "number": float,
        "integer": int,
        "boolean": bool,
        "array": list,
        "object": dict,
        "null": type(None),
    }
    return type_mapping.get(json_type, Any)

def infer_schema_from_example(example: dict[str, Any], name: str = "inferred") -> OutputSchema:
    """
    예시 데이터에서 스키마 추론
    
    Args:
        example: 예시 JSON 데이터
        name: 스키마 이름
    
    Returns:
        OutputSchema: 추론된 스키마
    """
    def infer_type(value: Any) -> dict[str, Any]:
        if isinstance(value, str):
            return {"type": "string"}
        elif isinstance(value, bool):
            return {"type": "boolean"}
        elif isinstance(value, int):
            return {"type": "integer"}
        elif isinstance(value, float):
            return {"type": "number"}
        elif isinstance(value, list):
            if value:
                items_schema = infer_type(value[0])
            else:
                items_schema = {}
            return {"type": "array", "items": items_schema}
        elif isinstance(value, dict):
            properties = {}
            for k, v in value.items():
                properties[k] = infer_type(v)
            return {
                "type": "object",
                "properties": properties,
                "required": list(value.keys())
            }
        else:
            return {}
    
    schema = infer_type(example)
    return OutputSchema(name=name, schema=schema, examples=[example])
