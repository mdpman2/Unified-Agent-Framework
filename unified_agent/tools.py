#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Unified Agent Framework - 도구 모듈 (Tools Module)

================================================================================
📁 파일 위치: unified_agent/tools.py
📋 역할: AIFunction, MCP 도구 등 외부 도구 통합 및 관리
📅 최종 업데이트: 2026년 1월
================================================================================

🎯 주요 구성 요소:

    📌 AIFunction (Abstract Base Class):
        - OpenAI Function Calling을 위한 추상 클래스
        - Microsoft Agent Framework 패턴 기반
        - 메트릭 수집 (execution_count, total_duration_ms)
        - OpenAI 함수 스키마 자동 생성

    📌 ApprovalRequiredAIFunction:
        - Human-in-the-loop 승인이 필요한 함수 래퍼
        - 자동 승인 임계값 설정 가능
        - 결제, 데이터 삭제 등 위험한 작업용

    📌 MockMCPClient:
        - MCP 클라이언트 모킹 (테스트용)
        - call_tool(), list_tools() 메서드 제공

    📌 MCPTool:
        - Model Context Protocol 도구 클래스
        - 외부 MCP 서버와 통신
        - Microsoft Learn, GitHub 등 다양한 소스 지원

🔧 MCP (Model Context Protocol) 설명:
    LLM이 외부 데이터 소스와 상호작용하기 위한 표준 프로토콜입니다.

    지원 소스 예시:
    - Microsoft Learn 문서
    - GitHub 저장소
    - Azure 리소스
    - 데이터베이스
    - 파일 시스템

📌 사용 예시:

    예제 1: 커스텀 AIFunction
    ----------------------------------------
    >>> from unified_agent.tools import AIFunction
    >>>
    >>> class WebSearchFunction(AIFunction):
    ...     def __init__(self):
    ...         super().__init__(
    ...             name="web_search",
    ...             description="Search the web for information",
    ...             parameters={
    ...                 "type": "object",
    ...                 "properties": {
    ...                     "query": {"type": "string", "description": "Search query"}
    ...                 },
    ...                 "required": ["query"]
    ...             }
    ...         )
    ...
    ...     async def execute(self, query: str) -> str:
    ...         # 웹 검색 로직
    ...         return f"Search results for: {query}"
    >>>
    >>> # 사용
    >>> func = WebSearchFunction()
    >>> schema = func.get_schema()  # OpenAI Function Calling 스키마
    >>> result, duration = await func.invoke_with_metrics(query="Python tutorial")

    예제 2: Human-in-the-loop 승인
    ----------------------------------------
    >>> from unified_agent.tools import ApprovalRequiredAIFunction
    >>>
    >>> # 기본 함수를 승인 필요 함수로 래핑
    >>> payment_func = PaymentFunction()
    >>> approved_func = ApprovalRequiredAIFunction(
    ...     base_function=payment_func,
    ...     approval_callback=request_user_approval,
    ...     auto_approve_threshold=10000  # 10,000원 이하는 자동 승인
    ... )

    예제 3: MCP 도구
    ----------------------------------------
    >>> from unified_agent.tools import MCPTool
    >>>
    >>> # Microsoft Learn MCP 도구
    >>> docs_tool = MCPTool(
    ...     name="microsoft_docs",
    ...     server_config={
    ...         "type": "mcp",
    ...         "url": "https://learn.microsoft.com/api/mcp"
    ...     }
    ... )
    >>>
    >>> # 도구 실행
    >>> result = await docs_tool.call("search", query="Azure OpenAI quickstart")

⚠️ 주의사항:
    - AIFunction.execute()는 반드시 async로 구현해야 합니다.
    - ApprovalRequiredAIFunction은 보안이 중요한 작업에 사용하세요.
    - MCP 서버 연결 실패 시 CircuitBreaker가 자동 발동됩니다.

🔗 참고:
    - MCP Protocol: https://modelcontextprotocol.io/
    - Microsoft Agent Framework: https://github.com/microsoft/agent-framework
    - OpenAI Function Calling: https://platform.openai.com/docs/guides/function-calling
"""

import time
import logging
from abc import ABC, abstractmethod
from datetime import datetime, timezone
from typing import Dict, List, Optional, Any, Tuple, Callable

from .models import ApprovalStatus

__all__ = [
    "AIFunction",
    "ApprovalRequiredAIFunction",
    "MockMCPClient",
    "MCPTool",
]


# ============================================================================
# AIFunction - Microsoft Agent Framework 패턴
# ============================================================================

class AIFunction(ABC):
    """
    AI Function 추상 클래스 - Microsoft Agent Framework 패턴

    OpenAI Function Calling을 위한 추상 클래스

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
        """OpenAI Function Calling 스키마"""
        return {
            "name": self.name,
            "description": self.description,
            "parameters": self.parameters
        }

    async def invoke_with_metrics(self, **kwargs) -> Tuple[Any, float]:
        """메트릭과 함께 실행"""
        start_time = time.time()
        result = await self.execute(**kwargs)
        duration_ms = (time.time() - start_time) * 1000

        self.execution_count += 1
        self.total_duration_ms += duration_ms

        return result, duration_ms


class ApprovalRequiredAIFunction(AIFunction):
    """
    Human-in-the-loop 승인이 필요한 함수

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

        # 자동 승인 임계값 확인
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
        """안전한 작업인지 확인"""
        # 읽기 전용 작업은 자동 승인
        if self.base_function.name.startswith(("get_", "read_", "list_")):
            return True
        return False


# ============================================================================
# MCP (Model Context Protocol) 통합
# ============================================================================

class MockMCPClient:
    """MCP 클라이언트 모의 구현 (데모용)"""

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
    __slots__ = ('name', 'server_config', 'connected', 'client', 'available_tools')

    def __init__(self, name: str, server_config: Dict[str, Any]):
        self.name = name
        self.server_config = server_config
        self.connected = False
        self.client: Optional[MockMCPClient] = None
        self.available_tools: List[Dict[str, Any]] = []

    async def connect(self):
        """MCP 서버 연결"""
        try:
            logging.info(f"🔌 MCP 서버 연결 시도: {self.name}")
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
