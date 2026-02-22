#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Unified Agent Framework v5 — Tools

================================================================================
v4.1 대응: tools.py(316줄) + mcp_workbench.py 통합
축소 이유: v4.1에서는 AIFunction, MCPTool, ApprovalRequiredAIFunction,
          StructuredTool, LangChainTool 등 프레임워크별로 다른
          도구 정의 방식이 혼재했음.
          MCP(Model Context Protocol) 표준으로 일원화하고,
          OpenAI Function Calling 스키마를 자동 생성하여
          엔진 간 도구 공유를 단순화.
================================================================================

설계 원칙:
    - Tool = (이름, 설명, 파라미터 스키마, 실행 함수)
    - OpenAI Function Calling 스키마로 자동 변환 (to_openai_schema)
    - MCP 서버 도구를 동일한 인터페이스로 사용
    - @mcp_tool 데코레이터로 함수 시그니처에서 자동 스키마 추출

사용법:
    >>> from unified_agent_v5 import Tool, mcp_tool
    >>>
    >>> # 방법 1: Tool 직접 생성
    >>> tool = Tool(
    ...     name="web_search",
    ...     description="웹 검색",
    ...     parameters={"query": {"type": "string", "description": "검색어"}},
    ...     fn=my_search_function
    ... )
    >>>
    >>> # 방법 2: @mcp_tool 데코레이터 (권장)
    >>> @mcp_tool(description="날씨 조회")
    ... async def get_weather(city: str) -> str:
    ...     return f"{city}의 날씨: 맑음 22°C"
"""

from __future__ import annotations

import inspect
import logging
import re
from dataclasses import dataclass, field
from typing import Any, Callable, get_origin, get_type_hints

__all__ = ["Tool", "ToolRegistry", "mcp_tool"]

logger = logging.getLogger(__name__)


# Docstring 파라미터 설명 추출용 정규식 (O(L) 한 번 파싱)
_PARAM_DOC_RE = re.compile(r"^\s*(\w+)\s*:\s*(.+)$")


@dataclass(slots=True)
class Tool:
    """
    통합 도구 정의 — MCP 표준 + OpenAI Function Calling 호환

    v4.1 대응: AIFunction + MCPTool + ApprovalRequiredAIFunction 통합
    축소 이유: 프레임워크별 다른 도구 형식 → MCP 표준 하나로 단일화.
              to_openai_schema()로 모든 엔진이 자동 변환.

    모든 엔진(Direct, LangChain, CrewAI)에서 동일하게 사용되며,
    각 엔진은 이를 자신의 도구 형식으로 자동 변환합니다.
    """
    name: str
    description: str = ""
    parameters: dict[str, Any] = field(default_factory=dict)
    fn: Callable[..., Any] | None = None

    # MCP 메타데이터
    mcp_server: str | None = None
    requires_approval: bool = False

    # 스키마 캐시 (생성 후 변경되지 않으므로 한 번만 계산)
    _cached_schema: dict[str, Any] | None = field(
        default=None, repr=False, init=False, compare=False
    )

    def to_openai_schema(self) -> dict[str, Any]:
        """OpenAI Function Calling 스키마로 변환 (캐시됨)"""
        if self._cached_schema is not None:
            return self._cached_schema
        properties = {}
        required = []

        for param_name, param_info in self.parameters.items():
            if isinstance(param_info, dict):
                # 비표준 필드('optional', 'default') 제거 후 스키마에 추가
                clean = {k: v for k, v in param_info.items() if k not in ("optional", "default")}
                properties[param_name] = clean
            else:
                properties[param_name] = {"type": "string", "description": str(param_info)}
            # optional 파라미터가 아니면 required에 추가
            if isinstance(param_info, dict) and not param_info.get("optional", False):
                required.append(param_name)

        schema = {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": {
                    "type": "object",
                    "properties": properties,
                    "required": required,
                },
            },
        }
        self._cached_schema = schema
        return schema

    async def execute(self, **kwargs) -> str:
        """도구 실행"""
        if self.fn is None:
            raise RuntimeError(f"Tool '{self.name}' has no execution function")

        try:
            result = self.fn(**kwargs)
            if inspect.isawaitable(result):
                result = await result
            return str(result)
        except Exception as e:
            logger.error(f"Tool '{self.name}' execution error with args {kwargs}: {e}")
            raise

    def __repr__(self) -> str:
        return f"Tool(name={self.name!r}, description={self.description[:40]!r})"


class ToolRegistry:
    """
    도구 레지스트리 — 이름으로 도구를 관리

    v4.1 대응: SkillRegistry + MCPToolInventory 통합
    축소 이유: v4.1의 복잡한 스킬/도구 검색/승인 흐름 → 단순 dict 기반 레지스트리.

    사용법:
        >>> registry = ToolRegistry()
        >>> registry.register(search_tool)
        >>> registry.register(weather_tool)
        >>> tools = registry.get_all()
    """

    __slots__ = ('_tools',)

    def __init__(self):
        self._tools: dict[str, Tool] = {}

    def register(self, tool: Tool) -> None:
        """도구 등록"""
        self._tools[tool.name] = tool

    def get(self, name: str) -> Tool | None:
        """이름으로 도구 조회"""
        return self._tools.get(name)

    def get_all(self) -> list[Tool]:
        """모든 도구 리스트"""
        return list(self._tools.values())

    def get_openai_schemas(self) -> list[dict[str, Any]]:
        """모든 도구의 OpenAI 스키마"""
        return [t.to_openai_schema() for t in self._tools.values()]

    def unregister(self, name: str) -> None:
        """도구 해제"""
        self._tools.pop(name, None)

    def __len__(self) -> int:
        return len(self._tools)

    def __contains__(self, name: str) -> bool:
        return name in self._tools

    def __repr__(self) -> str:
        return f"ToolRegistry(tools={list(self._tools.keys())})"


def mcp_tool(
    name: str | None = None,
    description: str = "",
    mcp_server: str | None = None,
    requires_approval: bool = False,
) -> Callable:
    """
    @mcp_tool 데코레이터 — 함수를 MCP 표준 Tool로 변환

    v4.1 대응: @ai_function + @mcp_registered + @structured_tool 통합
    축소 이유: 프레임워크별 다른 데코레이터 → @mcp_tool 하나로 단일화.
              함수 시그니처에서 파라미터 스키마를 자동 추출하여
              수동 스키마 정의 불필요.

    사용법:
        >>> @mcp_tool(description="코드 검색")
        ... async def search_code(query: str, language: str = "python") -> str:
        ...     return f"Found code for: {query}"
        >>>
        >>> # search_code는 이제 Tool 인스턴스
        >>> schema = search_code.to_openai_schema()
    """

    def decorator(fn: Callable) -> Tool:
        tool_name = name or fn.__name__

        # 함수 시그니처에서 파라미터 스키마 자동 추출
        sig = inspect.signature(fn)
        hints = get_type_hints(fn)
        parameters: dict[str, Any] = {}

        type_map = {
            str: "string",
            int: "integer",
            float: "number",
            bool: "boolean",
            list: "array",
            dict: "object",
        }

        # docstring에서 파라미터 설명 한 번에 추출 (O(L))
        param_docs: dict[str, str] = {}
        if fn.__doc__:
            for line in fn.__doc__.split("\n"):
                m = _PARAM_DOC_RE.match(line)
                if m:
                    param_docs[m.group(1)] = m.group(2).strip()

        for param_name, param in sig.parameters.items():
            if param_name in ("self", "cls"):
                continue

            param_type = hints.get(param_name, str)
            # Generic 타입 (list[str], dict[str, Any], int | None 등) 처리
            origin = get_origin(param_type) or param_type
            json_type = type_map.get(origin, "string")

            param_schema: dict[str, Any] = {"type": json_type}

            # docstring에서 추출한 설명 적용 (O(1) 룩업)
            if param_name in param_docs:
                param_schema["description"] = param_docs[param_name]

            if param.default is not inspect.Parameter.empty:
                param_schema["optional"] = True
                param_schema["default"] = param.default

            parameters[param_name] = param_schema

        tool_description = description or (fn.__doc__ or "").strip().split("\n")[0]

        return Tool(
            name=tool_name,
            description=tool_description,
            parameters=parameters,
            fn=fn,
            mcp_server=mcp_server,
            requires_approval=requires_approval,
        )

    return decorator
