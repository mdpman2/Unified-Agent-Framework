"""
Unified Agent Framework - Enterprise Edition
Microsoft Agent Framework 패턴 통합 (MCP, Approval, Streaming 지원)

🔥 주요 고도화 내용:
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

기존 코드 대비 개선사항:
- 코드 라인: 500줄 → 1,100줄 (2.2배 증가)
- Agent 타입: 3개 → 5개
- 데모 워크플로우: 2개 → 4개
- CLI 명령어: 5개 → 12개
- 디자인 패턴: 4개 추가 (Circuit Breaker, Pub-Sub, LRU Cache, Supervisor)

pip install semantic-kernel python-dotenv redis opentelemetry-api opentelemetry-sdk pydantic
"""

import os
import asyncio
import json
import logging
from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Any, Callable, Set, AsyncIterator
from datetime import datetime, timezone
from enum import Enum
from collections import defaultdict
from dataclasses import dataclass, field
import time

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
# 설정 (Configuration)
# ============================================================================

# 🆕 LLM 모델 중앙 설정
DEFAULT_LLM_MODEL = "gpt-4.1"  # 또는 "gpt-4o-mini" 등 원하는 모델명
DEFAULT_API_VERSION = "2024-08-01-preview"


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
# MCP (Model Context Protocol) 통합
# ============================================================================

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
        circuit_breaker: Optional[CircuitBreaker] = None  # 🆕 회로 차단기
    ):
        self.name = name
        self.role = role
        self.system_prompt = system_prompt
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.enable_streaming = enable_streaming
        self.event_bus = event_bus
        self.circuit_breaker = circuit_breaker or CircuitBreaker()

        self.execution_settings = AzureChatPromptExecutionSettings(
            temperature=temperature,
            max_tokens=max_tokens,
            service_id=model
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
            service_id=self.model,
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


class SupervisorAgent(Agent):
    """
    Supervisor Agent - 여러 Agent를 감독하고 조율

    [신규] Microsoft AutoGen의 Supervisor 패턴

    기존 OrchestratorAgent vs SupervisorAgent:
    - Orchestrator: 순차 실행, 간단한 협업
    - Supervisor: 라운드 기반 협업, 조기 종료 조건, 실행 로그

    주요 기능:
    1. 라운드 기반 협업 (max_rounds)
    2. 조기 종료 조건 ("TERMINATE" 키워드)
    3. 상세한 실행 로그 (execution_log)
    4. 서브 에이전트 성능 추적

    사용 시나리오:
    - Research Agent + Writer Agent 협업
    - Coder + Reviewer 협업
    - 복잡한 multi-step 작업
    """

    def __init__(self, *args, sub_agents: List[Agent],
                 max_rounds: int = 3, **kwargs):
        super().__init__(*args, role=AgentRole.SUPERVISOR, **kwargs)
        self.sub_agents = {agent.name: agent for agent in sub_agents}
        self.max_rounds = max_rounds
        self.execution_log: List[Dict[str, Any]] = []  # 🆕 실행 로그

    async def execute(self, state: AgentState, kernel: Kernel) -> NodeResult:
        start_time = time.time()

        try:
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

            final_output = "\n\n".join(responses)
            duration_ms = (time.time() - start_time) * 1000

            # 최종 요약
            summary = f"Supervisor 실행 완료: {current_round}라운드"
            state.add_message(AgentRole.SUPERVISOR, summary, self.name)

            return NodeResult(
                node_name=self.name,
                output=final_output,
                success=True,
                duration_ms=duration_ms,
                metadata={
                    "rounds": current_round,
                    "agents": len(self.sub_agents),
                    "execution_log": self.execution_log
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

    [수정] 여러 기능 추가:
    1. mcp_tools: MCP 도구 관리
    2. event_bus: 이벤트 시스템
    3. global_metrics: 전역 메트릭
    4. register_mcp_tool(): MCP 도구 등록
    5. approve_pending_request(): 승인 처리
    6. get_workflow_stats(): 워크플로우 통계
    7. get_global_metrics(): 전역 메트릭
    8. cleanup(): 리소스 정리
    """

    def __init__(
        self,
        kernel: Kernel,
        memory_store: Optional[MemoryStore] = None,
        checkpoint_dir: str = "./checkpoints",
        enable_telemetry: bool = True,
        enable_events: bool = True  # 🆕 이벤트 시스템 옵션
    ):
        self.kernel = kernel
        self.memory_store = memory_store or CachedMemoryStore(max_cache_size=100)
        self.state_manager = StateManager(self.memory_store, checkpoint_dir)
        self.graphs: Dict[str, Graph] = {}
        self.mcp_tools: Dict[str, MCPTool] = {}  # 🆕 MCP 도구
        self.event_bus = EventBus() if enable_events else None  # 🆕 이벤트 버스

        if enable_telemetry:
            self.tracer = trace.get_tracer(__name__)
        else:
            self.tracer = None

        # 🆕 전역 메트릭
        self.global_metrics = {
            "total_workflows": 0,
            "total_executions": 0,
            "total_failures": 0,
            "start_time": datetime.now(timezone.utc).isoformat()
        }

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
    데모 1: 단순 대화

    [신규] 가장 기본적인 대화형 Agent

    학습 포인트:
    - SimpleAgent의 기본 사용법
    - 단순한 시작->종료 플로우
    """
    print("\n" + "="*60)
    print("📚 데모 1: 단순 대화 Agent")
    print("="*60)

    graph = framework.create_graph("simple_chat")

    assistant = SimpleAgent(
        name="assistant",
        system_prompt="You are a helpful AI assistant. Answer questions clearly and concisely.",
        model=DEFAULT_LLM_MODEL,
        enable_streaming=False,
        event_bus=framework.event_bus
    )

    graph.add_node(Node("assistant", assistant))
    graph.set_start("assistant")
    graph.set_end("assistant")

    print("\n워크플로우 구조:")
    print(framework.visualize_workflow("simple_chat"))


async def demo_routing_workflow(framework: UnifiedAgentFramework):
    """
    데모 2: 라우팅 워크플로우

    [신규] 인텐트 기반 라우팅

    학습 포인트:
    - RouterAgent로 동적 라우팅
    - 전문화된 Agent 활용
    - 다중 종료 노드
    """
    print("\n" + "="*60)
    print("📚 데모 2: 인텐트 기반 라우팅")
    print("="*60)

    graph = framework.create_graph("routing_workflow")

    # Router
    router = RouterAgent(
        name="router",
        system_prompt="Classify user intent accurately.",
        model=DEFAULT_LLM_MODEL,
        routes={
            "order": "order_agent",
            "support": "support_agent",
            "general": "general_agent"
        },
        event_bus=framework.event_bus
    )

    # Specialized Agents
    order_agent = SimpleAgent(
        name="order_agent",
        system_prompt="You are an order specialist. Help with ordering and purchases.",
        model=DEFAULT_LLM_MODEL,
        event_bus=framework.event_bus
    )

    support_agent = SimpleAgent(
        name="support_agent",
        system_prompt="You are a support specialist. Help troubleshoot and resolve issues.",
        model=DEFAULT_LLM_MODEL,
        event_bus=framework.event_bus
    )

    general_agent = SimpleAgent(
        name="general_agent",
        system_prompt="You are a general assistant. Answer various questions.",
        model=DEFAULT_LLM_MODEL,
        event_bus=framework.event_bus
    )

    # Build Graph
    graph.add_node(Node("router", router))
    graph.add_node(Node("order_agent", order_agent))
    graph.add_node(Node("support_agent", support_agent))
    graph.add_node(Node("general_agent", general_agent))

    graph.set_start("router")
    graph.set_end("order_agent")
    graph.set_end("support_agent")
    graph.set_end("general_agent")

    print("\n워크플로우 구조:")
    print(framework.visualize_workflow("routing_workflow"))


async def demo_supervisor_workflow(framework: UnifiedAgentFramework):
    """
    데모 3: Supervisor 패턴

    [신규] Microsoft AutoGen의 Supervisor 패턴

    학습 포인트:
    - SupervisorAgent로 멀티 에이전트 조율
    - 라운드 기반 협업
    - 조기 종료 조건
    """
    print("\n" + "="*60)
    print("📚 데모 3: Supervisor Multi-Agent 협업")
    print("="*60)

    graph = framework.create_graph("supervisor_workflow")

    # Sub-agents
    research_agent = SimpleAgent(
        name="researcher",
        system_prompt="You are a research specialist. Gather and analyze information.",
        model=DEFAULT_LLM_MODEL,
        event_bus=framework.event_bus
    )

    writer_agent = SimpleAgent(
        name="writer",
        system_prompt="You are a content writer. Create clear, engaging content.",
        model=DEFAULT_LLM_MODEL,
        event_bus=framework.event_bus
    )

    # Supervisor
    supervisor = SupervisorAgent(
        name="supervisor",
        system_prompt="Coordinate research and writing tasks.",
        model=DEFAULT_LLM_MODEL,
        sub_agents=[research_agent, writer_agent],
        max_rounds=2,
        event_bus=framework.event_bus
    )

    graph.add_node(Node("supervisor", supervisor))
    graph.set_start("supervisor")
    graph.set_end("supervisor")

    print("\n워크플로우 구조:")
    print(framework.visualize_workflow("supervisor_workflow"))


async def demo_conditional_workflow(framework: UnifiedAgentFramework):
    """
    데모 4: 조건부 라우팅

    [신규] LangGraph의 조건부 엣지 패턴

    학습 포인트:
    - 조건 함수 (condition_func)로 동적 라우팅
    - 복잡도 기반 처리 경로 분기
    - 조건부 엣지 사용법
    """
    print("\n" + "="*60)
    print("📚 데모 4: 조건부 라우팅 및 루프")
    print("="*60)

    graph = framework.create_graph("conditional_workflow")

    # Agents
    analyzer = SimpleAgent(
        name="analyzer",
        system_prompt="Analyze the complexity of the user's question. Respond with SIMPLE or COMPLEX.",
        model=DEFAULT_LLM_MODEL,
        event_bus=framework.event_bus
    )

    simple_handler = SimpleAgent(
        name="simple_handler",
        system_prompt="Answer simple questions directly and briefly.",
        model=DEFAULT_LLM_MODEL,
        event_bus=framework.event_bus
    )

    complex_handler = SimpleAgent(
        name="complex_handler",
        system_prompt="Provide detailed, comprehensive answers to complex questions.",
        model=DEFAULT_LLM_MODEL,
        max_tokens=2000,
        event_bus=framework.event_bus
    )

    # Build Graph
    analyzer_node = Node("analyzer", analyzer, edges={"simple": "simple_handler", "complex": "complex_handler"})

    # 🆕 조건부 라우팅 함수
    async def route_by_complexity(state: AgentState, result: NodeResult) -> str:
        """복잡도에 따라 라우팅"""
        output_lower = result.output.lower()
        if "simple" in output_lower:
            return "simple"
        else:
            return "complex"

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
    메인 실행 함수

    [수정] CLI 명령어 확장: 5개 → 12개

    기존 명령어:
    - exit, checkpoint, restore, visualize, switch

    새 명령어:
    - rollback, stats, metrics, events, list
    - checkpoint [tag], restore [tag], rollback [steps]
    """
    # 로깅 설정
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler("agent_framework.log", encoding='utf-8'),  # 🆕 파일 로깅
            logging.StreamHandler()
        ]
    )
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("semantic_kernel").setLevel(logging.WARNING)

    # OpenTelemetry 설정
    setup_telemetry("UnifiedAgentFramework-Enterprise", enable_console=False)

    # 환경 변수 로드
    load_dotenv()
    api_key = os.getenv("OPEN_AI_KEY_5")
    endpoint = os.getenv("OPEN_AI_ENDPOINT_5")
    deployment_name = os.getenv("AZURE_OPENAI_DEPLOYMENT_NAME")

    if not all([api_key, endpoint, deployment_name]):
        raise ValueError("❌ 필수 환경 변수 미설정: AZURE_OPENAI_API_KEY, AZURE_OPENAI_ENDPOINT, AZURE_OPENAI_DEPLOYMENT_NAME")

    print("\n" + "="*60)
    print("🚀 Unified Agent Framework - Enterprise Edition")
    print("="*60)
    print(f"✅ 엔드포인트: {endpoint}")
    print(f"✅ 모델: {deployment_name}")
    print("="*60)

    # Kernel 초기화
    kernel = Kernel()
    chat_service = AzureChatCompletion(
        deployment_name=deployment_name,
        api_key=api_key,
        endpoint=endpoint,
        service_id=DEFAULT_LLM_MODEL,  # 🆕 중앙 설정 사용
        api_version=DEFAULT_API_VERSION
    )
    kernel.add_service(chat_service)

    # Framework 초기화
    framework = UnifiedAgentFramework(
        kernel=kernel,
        checkpoint_dir="./checkpoints",
        enable_telemetry=True,
        enable_events=True
    )

    # 🆕 이벤트 리스너 등록
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
    print("  - exit: 종료")
    print("  - checkpoint [tag]: 체크포인트 저장")
    print("  - restore [tag]: 체크포인트 복원")
    print("  - rollback [steps]: 이전 상태로 롤백")  # 🆕
    print("  - visualize: 현재 워크플로우 시각화")
    print("  - switch [workflow]: 워크플로우 전환")
    print("  - stats: 워크플로우 통계")  # 🆕
    print("  - metrics: 전역 메트릭")  # 🆕
    print("  - events [type]: 이벤트 히스토리")  # 🆕
    print("  - list: 사용 가능한 워크플로우 목록")  # 🆕
    print("="*60 + "\n")

    session_id = f"session-{int(time.time())}"
    current_workflow = "simple_chat"

    try:
        while True:
            user_input = input(f"\n[{current_workflow}] User > ").strip()

            if not user_input:
                continue

            # 명령어 처리
            if user_input.lower() == "exit":
                print("\n👋 종료합니다...")
                break

            elif user_input.lower().startswith("checkpoint"):
                parts = user_input.split()
                tag = parts[1] if len(parts) > 1 else None
                state = await framework.state_manager.load_state(session_id)
                if state:
                    checkpoint_file = await framework.state_manager.save_checkpoint(state, tag=tag)
                    print(f"✅ 체크포인트 저장: {checkpoint_file}")
                else:
                    print("❌ 저장할 상태가 없습니다")
                continue

            elif user_input.lower().startswith("restore"):
                parts = user_input.split()
                tag = parts[1] if len(parts) > 1 else None
                state = await framework.state_manager.restore_checkpoint(session_id, tag=tag)
                if state:
                    print(f"✅ 체크포인트 복원 완료")
                else:
                    print("❌ 복원할 체크포인트가 없습니다")
                continue

            elif user_input.lower().startswith("rollback"):  # 🆕 롤백 명령어
                parts = user_input.split()
                steps = int(parts[1]) if len(parts) > 1 else 1
                state = await framework.state_manager.rollback(session_id, steps=steps)
                if state:
                    print(f"✅ {steps}단계 롤백 완료")
                else:
                    print("❌ 롤백 실패")
                continue

            elif user_input.lower() == "visualize":
                print("\n" + framework.visualize_workflow(current_workflow))
                continue

            elif user_input.lower().startswith("switch"):
                parts = user_input.split()
                if len(parts) > 1:
                    workflow_name = parts[1]
                    if workflow_name in framework.graphs:
                        current_workflow = workflow_name
                        print(f"✅ 워크플로우 전환: {workflow_name}")
                    else:
                        print(f"❌ 워크플로우 '{workflow_name}'를 찾을 수 없습니다")
                continue

            elif user_input.lower() == "stats":  # 🆕 통계 명령어
                stats = framework.get_workflow_stats(current_workflow)
                print("\n📊 워크플로우 통계:")
                print(json.dumps(stats, indent=2, ensure_ascii=False))
                continue

            elif user_input.lower() == "metrics":  # 🆕 메트릭 명령어
                metrics = framework.get_global_metrics()
                print("\n📈 전역 메트릭:")
                print(json.dumps(metrics, indent=2, ensure_ascii=False))
                continue

            elif user_input.lower().startswith("events"):  # 🆕 이벤트 명령어
                parts = user_input.split()
                event_type = parts[1] if len(parts) > 1 else None

                if framework.event_bus:
                    if event_type:
                        try:
                            et = EventType(event_type)
                            events = framework.event_bus.get_event_history(event_type=et, limit=10)
                        except ValueError:
                            print(f"❌ 잘못된 이벤트 타입: {event_type}")
                            continue
                    else:
                        events = framework.event_bus.get_event_history(limit=10)

                    print(f"\n📜 최근 이벤트 ({len(events)}개):")
                    for event in events:
                        print(f"  - {event.timestamp}: {event.event_type.value} ({event.agent_name or 'System'})")
                else:
                    print("❌ 이벤트 시스템이 비활성화되어 있습니다")
                continue

            elif user_input.lower() == "list":  # 🆕 목록 명령어
                print("\n📋 사용 가능한 워크플로우:")
                for name in framework.graphs.keys():
                    marker = "👉" if name == current_workflow else "  "
                    print(f"{marker} {name}")
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

                # 🆕 승인 대기 처리
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


if __name__ == "__main__":
    asyncio.run(main())
