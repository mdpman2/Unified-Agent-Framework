#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Unified Agent Framework v6 — Tools & Decorators

================================================================================
Microsoft Agent Framework 1.0.0-rc1 호환 도구 시스템

함수를 AI 에이전트의 도구로 변환하는 시스템입니다.
@tool 데코레이터로 함수를 FunctionTool로 자동 변환하며,
Python 타입 힌트에서 OpenAI function calling JSON Schema를 생성합니다.

핵심 클래스/함수:
    - @tool           : 함수 → FunctionTool 변환 데코레이터
    - FunctionTool    : 도구 클래스 (name, description, func, schema, invoke)
    - normalize_tools : 다양한 형식을 list[FunctionTool]로 정규화

v5 → v6 변경 요약:
    - @mcp_tool 데코레이터 → @tool 데코레이터
    - Tool dataclass → FunctionTool 클래스
    - 자동 스키마 생성 + 비동기 호출 지원

성능 최적화:
    - asyncio.to_thread(): 동기 함수의 비동기 실행 (deprecated get_event_loop() 대체)
    - types.UnionType 지원: Python 3.10+ 파이프 문법 (X | Y) 스키마 변환
    - functools.update_wrapper: @tool 데코레이터가 원본 함수 메타데이터 보존
================================================================================
"""

from __future__ import annotations

import asyncio
import inspect
import json
import logging
import types as _types
import functools
from typing import Any, Callable, Sequence, Union, get_type_hints

__all__ = [
    "tool",
    "FunctionTool",
    "normalize_tools",
    "ToolTypes",
]

logger = logging.getLogger("agent_framework")

# ─── Type alias ──────────────────────────────────────────────

ToolTypes = Union["FunctionTool", Callable[..., Any]]

# ─── Type → JSON Schema 매핑 ────────────────────────────────

_TYPE_MAP: dict[type, str] = {
    str: "string",
    int: "integer",
    float: "number",
    bool: "boolean",
    list: "array",
    dict: "object",
}


def _python_type_to_json_schema(tp: Any) -> dict[str, Any]:
    """
    Python 타입 어노테이션을 JSON Schema로 변환.

    지원 타입:
        - 기본: str, int, float, bool, list, dict
        - 제네릭: list[X], dict[str, X]
        - Optional: Optional[X], X | None (Python 3.10+)
        - UnionType: X | Y (Python 3.10+ 파이프 문법, types.UnionType 지원)
    """
    if tp is None or tp is type(None):
        return {"type": "null"}

    origin = getattr(tp, "__origin__", None)

    # list[X] → {"type": "array", "items": {...}}
    if origin is list:
        args = getattr(tp, "__args__", ())
        if args:
            return {"type": "array", "items": _python_type_to_json_schema(args[0])}
        return {"type": "array"}

    # dict[str, X] → {"type": "object"}
    if origin is dict:
        return {"type": "object"}

    # 성능 최적화: Python 3.10+ 파이프 문법 (X | Y) 지원
    # types.UnionType는 3.10+에서만 존재하므로 getattr로 안전하게 접근
    if origin is Union or isinstance(tp, getattr(_types, "UnionType", type(None))):
        args = [a for a in getattr(tp, "__args__", ()) if a is not type(None)]
        if len(args) == 1:
            return _python_type_to_json_schema(args[0])
        return {}

    # 기본 타입 매핑
    json_type = _TYPE_MAP.get(tp)
    if json_type:
        return {"type": json_type}

    return {"type": "string"}


# ─── FunctionTool ────────────────────────────────────────────

class FunctionTool:
    """
    함수 도구 — agent_framework 1.0.0-rc1 호환

    함수를 AI 에이전트의 도구로 등록하기 위한 클래스입니다.
    함수 시그니처와 타입 힌트에서 OpenAI function calling
    JSON Schema를 자동 생성합니다.

    v5 대응: Tool dataclass (name, description, func)
    v6 변경: 자동 스키마 생성 + 비동기 호출 + OpenAI 스키마 변환.

    성능 최적화:
        - invoke()에서 asyncio.to_thread() 사용 (deprecated get_event_loop 대체)
        - docstring에서 파라미터 설명 자동 추출

    사용법:
        >>> @tool
        ... def get_weather(city: str) -> str:
        ...     \"\"\"도시의 날씨를 반환합니다.\"\"\"
        ...     return f"{city}: 맑음"
        >>>
        >>> # 수동 생성
        >>> ft = FunctionTool(
        ...     name="get_weather",
        ...     description="도시의 날씨를 반환",
        ...     func=some_function,
        ...     parameters={"city": {"type": "string", "description": "도시 이름"}},
        ... )
    """

    def __init__(
        self,
        name: str,
        description: str = "",
        func: Callable[..., Any] | None = None,
        *,
        parameters: dict[str, Any] | None = None,
        required: list[str] | None = None,
        strict: bool = False,
    ) -> None:
        self.name = name
        self.description = description
        self.func = func
        self._parameters = parameters
        self._required = required
        self.strict = strict

        # 함수에서 자동 스키마 생성
        if func and parameters is None:
            self._auto_generate_schema()

    def _auto_generate_schema(self) -> None:
        """함수 시그니처에서 자동으로 파라미터 스키마 생성."""
        if not self.func:
            return

        sig = inspect.signature(self.func)
        hints = get_type_hints(self.func) if self.func else {}

        params: dict[str, Any] = {}
        required: list[str] = []

        for param_name, param in sig.parameters.items():
            if param_name in ("self", "cls"):
                continue

            schema = _python_type_to_json_schema(hints.get(param_name, str))

            # docstring에서 파라미터 설명 추출
            doc = inspect.getdoc(self.func) or ""
            param_desc = _extract_param_doc(doc, param_name)
            if param_desc:
                schema["description"] = param_desc

            params[param_name] = schema

            if param.default is inspect.Parameter.empty:
                required.append(param_name)

        self._parameters = params
        self._required = required

    @property
    def parameters_schema(self) -> dict[str, Any]:
        """JSON Schema 형식의 파라미터 정의."""
        properties = {}
        for name, schema in (self._parameters or {}).items():
            if isinstance(schema, dict):
                properties[name] = schema
            else:
                properties[name] = {"type": "string"}

        result: dict[str, Any] = {
            "type": "object",
            "properties": properties,
        }
        if self._required:
            result["required"] = self._required
        if self.strict:
            result["additionalProperties"] = False
        return result

    def to_openai_schema(self) -> dict[str, Any]:
        """OpenAI function calling 스키마 형식으로 변환."""
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": self.parameters_schema,
            },
        }

    async def invoke(self, *, arguments: dict[str, Any] | None = None) -> Any:
        """
        도구 실행 (동기/비동기 함수 모두 지원).

        비동기 함수는 직접 await, 동기 함수는 asyncio.to_thread()로
        스레드 풀에서 실행하여 이벤트 루프를 차단하지 않습니다.

        Args:
            arguments: 도구에 전달할 인수 dict

        Returns:
            도구 함수의 반환값

        Raises:
            RuntimeError: func가 None인 경우
        """
        if self.func is None:
            raise RuntimeError(f"Tool '{self.name}' has no callable function.")

        args = arguments or {}

        try:
            if asyncio.iscoroutinefunction(self.func):
                result = await self.func(**args)
            else:
                # 성능 최적화: asyncio.to_thread()로 동기 함수 실행
                # deprecated get_event_loop().run_in_executor() 대체
                result = await asyncio.to_thread(self.func, **args)
            return result
        except Exception as e:
            logger.error("Tool %s failed: %s", self.name, e)
            raise

    def __repr__(self) -> str:
        return f"FunctionTool(name={self.name!r})"


# ─── @tool 데코레이터 ───────────────────────────────────────

def tool(
    func: Callable[..., Any] | None = None,
    *,
    name: str | None = None,
    description: str | None = None,
    strict: bool = False,
) -> FunctionTool | Callable[..., FunctionTool]:
    """
    @tool 데코레이터 — agent_framework 1.0.0-rc1 호환

    함수를 FunctionTool로 자동 변환합니다.
    함수 시그니처 + 타입 힌트에서 OpenAI function calling
    JSON Schema를 자동 생성하며, functools.update_wrapper로
    원본 함수의 메타데이터(__name__, __doc__ 등)를 보존합니다.

    v5 대응: @mcp_tool 데코레이터
    v6 변경: 간소화된 인터페이스 + 자동 스키마 생성.

    Args:
        func: 변환할 함수 (@tool 형태로 사용 시 자동 전달)
        name: 도구 이름 (기본: 함수명)
        description: 도구 설명 (기본: docstring 첫 줄)
        strict: 엄격 모드 (additionalProperties: false)

    Returns:
        FunctionTool 인스턴스

    사용법:
        >>> @tool
        ... def get_weather(city: str) -> str:
        ...     \"\"\"도시의 날씨를 반환합니다.\"\"\"
        ...     return f"{city}: 맑음 22°C"
        >>>
        >>> @tool(name="search_web", description="웹 검색")
        ... async def search(query: str, max_results: int = 5) -> str:
        ...     return f"Results for: {query}"
    """
    def _wrap(fn: Callable[..., Any]) -> FunctionTool:
        tool_name = name or fn.__name__
        tool_desc = description or (inspect.getdoc(fn) or "").split("\n")[0]

        ft = FunctionTool(
            name=tool_name,
            description=tool_desc,
            func=fn,
            strict=strict,
        )

        # 원본 함수의 메타데이터 보존
        functools.update_wrapper(ft, fn)
        return ft

    if func is not None:
        # @tool 형태 (괄호 없이 사용)
        return _wrap(func)

    # @tool(...) 형태 (괄호와 인수 사용)
    return _wrap


# ─── 도구 정규화 ─────────────────────────────────────────────

def normalize_tools(
    tools: ToolTypes | Sequence[ToolTypes] | None = None,
) -> list[FunctionTool]:
    """
    다양한 형식의 도구를 FunctionTool 리스트로 정규화.

    Agent 생성자와 ChatClient에서 도구 형식을 통일하는 데 사용됩니다.

    지원 형식:
        - FunctionTool 인스턴스 (그대로 사용)
        - callable (함수/메서드 → FunctionTool 자동 변환)
        - 리스트 / 튜플 (각 요소를 재귀적 정규화)
        - None (빈 리스트 반환)
    """
    if tools is None:
        return []

    if isinstance(tools, FunctionTool):
        return [tools]

    if callable(tools) and not isinstance(tools, (list, tuple)):
        return [_ensure_function_tool(tools)]

    result: list[FunctionTool] = []
    for t in tools:
        if isinstance(t, FunctionTool):
            result.append(t)
        elif callable(t):
            result.append(_ensure_function_tool(t))
        else:
            logger.warning("Unknown tool type: %s — skipping", type(t))
    return result


def _ensure_function_tool(fn: Any) -> FunctionTool:
    """Callable을 FunctionTool으로 변환."""
    if isinstance(fn, FunctionTool):
        return fn
    name = getattr(fn, "__name__", "unknown")
    desc = (inspect.getdoc(fn) or "").split("\n")[0]
    return FunctionTool(name=name, description=desc, func=fn)


# ─── 유틸리티 ────────────────────────────────────────────────

def _extract_param_doc(docstring: str, param_name: str) -> str:
    """docstring에서 파라미터 설명 추출 (Args 섹션)."""
    if not docstring:
        return ""

    lines = docstring.split("\n")
    in_args = False
    for line in lines:
        stripped = line.strip()
        if stripped.lower() in ("args:", "arguments:", "parameters:", "params:"):
            in_args = True
            continue
        if in_args:
            if stripped.startswith(f"{param_name}") and (":" in stripped or "(" in stripped):
                # param_name: description  또는  param_name (type): description
                parts = stripped.split(":", 1)
                if len(parts) > 1:
                    return parts[1].strip()
            elif stripped and not stripped[0].isspace() and ":" in stripped and not stripped.startswith(param_name):
                continue
            elif not stripped:
                in_args = False
    return ""
