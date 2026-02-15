#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
MCP Workbench 시스템 - 다중 MCP 서버 관리

================================================================================
📋 역할: 여러 MCP(Model Context Protocol) 서버의 연결, 라우팅, 관리
📅 버전: 3.4.0 (2026년 2월)
📦 영감: Microsoft Agent Framework MCP, Anthropic MCP
================================================================================

🎯 주요 기능:
    - 다중 MCP 서버 연결 관리
    - 커넥션 풀링
    - 자동 라우팅 (능력 기반)
    - 헬스체크 및 장애 복구
    - 로드 밸런싱
    - 도구 통합 뷰

📌 사용 시나리오:
    - 여러 MCP 서버 통합 관리
    - 분산 도구 환경
    - 고가용성 MCP 시스템
    - 마이크로서비스 아키텍처

📌 사용 예시:
    >>> from unified_agent import McpWorkbench, McpServerConfig
    >>>
    >>> workbench = McpWorkbench()
    >>>
    >>> # MCP 서버 등록
    >>> workbench.register_server(McpServerConfig(
    ...     name="filesystem",
    ...     uri="stdio://mcp-server-filesystem",
    ...     capabilities=["read_file", "write_file"]
    ... ))
    >>>
    >>> workbench.register_server(McpServerConfig(
    ...     name="database",
    ...     uri="http://localhost:3000/mcp",
    ...     capabilities=["query", "insert"]
    ... ))
    >>>
    >>> # 연결
    >>> await workbench.connect_all()
    >>>
    >>> # 도구 호출 (자동 라우팅)
    >>> result = await workbench.call_tool("read_file", path="/etc/hosts")
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import random
import time
import uuid
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime, timezone, timedelta
from enum import Enum
from typing import (
    Any,
    Callable,
    Coroutine,
    Generic,
    Protocol,
    TypeVar,
)

from .utils import StructuredLogger, CircuitBreaker

__all__ = [
    # 설정
    "McpServerConfig",
    "McpWorkbenchConfig",
    "ConnectionState",
    "LoadBalanceStrategy",
    # 서버
    "McpServerConnection",
    "McpServerInfo",
    # 워크벤치
    "McpWorkbench",
    "McpToolRegistry",
    # 라우터
    "McpRouter",
    "CapabilityRouter",
    "RoundRobinRouter",
    # 헬스체크
    "HealthChecker",
    "HealthStatus",
]

# ============================================================================
# 설정 및 상태
# ============================================================================

class ConnectionState(str, Enum):
    """연결 상태"""
    DISCONNECTED = "disconnected"
    CONNECTING = "connecting"
    CONNECTED = "connected"
    ERROR = "error"
    RECONNECTING = "reconnecting"

class LoadBalanceStrategy(str, Enum):
    """로드 밸런싱 전략"""
    ROUND_ROBIN = "round_robin"     # 순환
    RANDOM = "random"               # 랜덤
    LEAST_CONN = "least_conn"       # 최소 연결
    CAPABILITY = "capability"       # 능력 기반 (기본)

class HealthStatus(str, Enum):
    """헬스 상태"""
    HEALTHY = "healthy"
    DEGRADED = "degraded"
    UNHEALTHY = "unhealthy"
    UNKNOWN = "unknown"

@dataclass(frozen=True, slots=True)
class McpServerConfig:
    """
    MCP 서버 설정
    
    Args:
        name: 서버 이름 (고유)
        uri: 서버 URI (stdio://, http://, ws://)
        capabilities: 제공하는 도구 목록
        priority: 우선순위 (높을수록 선호)
        max_connections: 최대 연결 수
        timeout_seconds: 타임아웃
        retry_count: 재시도 횟수
        healthcheck_interval: 헬스체크 간격 (초)
        metadata: 추가 메타데이터
    """
    name: str
    uri: str
    capabilities: list[str] = field(default_factory=list)
    priority: int = 1
    max_connections: int = 5
    timeout_seconds: float = 30.0
    retry_count: int = 3
    healthcheck_interval: float = 30.0
    metadata: dict[str, Any] = field(default_factory=dict)
    
    # 인증 (선택적)
    auth_token: str | None = field(default=None, repr=False)
    auth_type: str = "bearer"  # bearer, basic, api_key

@dataclass(frozen=True, slots=True)
class McpWorkbenchConfig:
    """
    MCP Workbench 설정
    
    Args:
        load_balance_strategy: 로드 밸런싱 전략
        enable_healthcheck: 헬스체크 활성화
        enable_auto_reconnect: 자동 재연결
        max_total_connections: 전체 최대 연결 수
        default_timeout: 기본 타임아웃
    """
    load_balance_strategy: LoadBalanceStrategy = LoadBalanceStrategy.CAPABILITY
    enable_healthcheck: bool = True
    enable_auto_reconnect: bool = True
    max_total_connections: int = 50
    default_timeout: float = 30.0
    healthcheck_interval: float = 30.0

@dataclass(frozen=True, slots=True)
class McpServerInfo:
    """MCP 서버 정보"""
    name: str
    uri: str
    state: ConnectionState
    health: HealthStatus
    capabilities: list[str]
    active_connections: int
    total_calls: int = 0
    failed_calls: int = 0
    avg_latency_ms: float = 0.0
    last_healthcheck: datetime | None = None
    last_error: str | None = None
    
    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "uri": self.uri,
            "state": self.state.value,
            "health": self.health.value,
            "capabilities": self.capabilities,
            "active_connections": self.active_connections,
            "total_calls": self.total_calls,
            "failed_calls": self.failed_calls,
            "avg_latency_ms": round(self.avg_latency_ms, 2),
            "success_rate": f"{(1 - self.failed_calls / max(1, self.total_calls)) * 100:.1f}%",
        }

# ============================================================================
# MCP 서버 연결
# ============================================================================

class McpServerConnection:
    """
    MCP 서버 연결 관리
    
    개별 MCP 서버와의 연결을 관리
    """
    
    def __init__(self, config: McpServerConfig):
        self.config = config
        self.name = config.name
        self.uri = config.uri
        
        self._state = ConnectionState.DISCONNECTED
        self._health = HealthStatus.UNKNOWN
        self._active_connections = 0
        self._semaphore = asyncio.Semaphore(config.max_connections)
        
        # 통계
        self._total_calls = 0
        self._failed_calls = 0
        self._latencies: list[float] = []
        self._last_error: str | None = None
        self._last_healthcheck: datetime | None = None
        
        # 회로 차단기
        self._circuit_breaker = CircuitBreaker(
            failure_threshold=config.retry_count,
            timeout=30.0,
        )
        
        self._logger = StructuredLogger(f"mcp_conn.{config.name}")
        self._lock = asyncio.Lock()
        
        # 실제 연결 객체 (구현에 따라 다름)
        self._connection: Any | None = None
        
        # 제공하는 도구 목록 (연결 후 조회)
        self._tools: dict[str, dict[str, Any]] = {}
    
    @property
    def state(self) -> ConnectionState:
        return self._state
    
    @property
    def health(self) -> HealthStatus:
        return self._health
    
    @property
    def active_connections(self) -> int:
        return self._active_connections
    
    async def connect(self) -> bool:
        """서버 연결"""
        if self._state == ConnectionState.CONNECTED:
            return True
        
        self._state = ConnectionState.CONNECTING
        
        try:
            # URI 프로토콜에 따른 연결
            if self.uri.startswith("stdio://"):
                await self._connect_stdio()
            elif self.uri.startswith("http://") or self.uri.startswith("https://"):
                await self._connect_http()
            elif self.uri.startswith("ws://") or self.uri.startswith("wss://"):
                await self._connect_websocket()
            else:
                raise ValueError(f"Unsupported protocol: {self.uri}")
            
            self._state = ConnectionState.CONNECTED
            self._health = HealthStatus.HEALTHY
            
            # 도구 목록 조회
            await self._fetch_tools()
            
            self._logger.info("Connected", uri=self.uri)
            return True
            
        except Exception as e:
            self._state = ConnectionState.ERROR
            self._health = HealthStatus.UNHEALTHY
            self._last_error = str(e)
            self._logger.error("Connection failed", uri=self.uri, error=str(e))
            return False
    
    async def _connect_stdio(self):
        """STDIO 연결 (로컬 프로세스)"""
        # 실제 구현에서는 subprocess로 MCP 서버 실행
        self._connection = {"type": "stdio", "uri": self.uri}
    
    async def _connect_http(self):
        """HTTP 연결 (REST API)"""
        self._connection = {"type": "http", "uri": self.uri}
    
    async def _connect_websocket(self):
        """WebSocket 연결"""
        self._connection = {"type": "websocket", "uri": self.uri}
    
    async def _fetch_tools(self):
        """서버에서 도구 목록 조회"""
        # 실제 구현에서는 MCP 프로토콜로 tools/list 호출
        # 여기서는 config의 capabilities 사용
        for cap in self.config.capabilities:
            self._tools[cap] = {
                "name": cap,
                "description": f"Tool: {cap}",
                "server": self.name,
            }
    
    async def disconnect(self):
        """연결 해제"""
        self._state = ConnectionState.DISCONNECTED
        self._connection = None
        self._logger.info("Disconnected", uri=self.uri)
    
    async def call_tool(
        self,
        tool_name: str,
        arguments: dict[str, Any],
        timeout: float | None = None,
    ) -> Any:
        """
        도구 호출
        
        Args:
            tool_name: 도구 이름
            arguments: 도구 인자
            timeout: 타임아웃
            
        Returns:
            도구 실행 결과
        """
        if self._state != ConnectionState.CONNECTED:
            raise ConnectionError(f"Server {self.name} is not connected")
        
        timeout = timeout or self.config.timeout_seconds
        
        async with self._semaphore:
            self._active_connections += 1
            start_time = time.time()
            
            try:
                # 회로 차단기 체크
                result = await self._circuit_breaker.call(
                    self._execute_tool,
                    tool_name,
                    arguments,
                    timeout
                )
                
                latency = (time.time() - start_time) * 1000
                self._record_success(latency)
                
                return result
                
            except Exception as e:
                self._record_failure(str(e))
                raise
                
            finally:
                self._active_connections -= 1
    
    async def _execute_tool(
        self,
        tool_name: str,
        arguments: dict[str, Any],
        timeout: float,
    ) -> Any:
        """실제 도구 실행 (프로토콜별 구현)"""
        # 실제 구현에서는 MCP 프로토콜로 tools/call 호출
        # 여기서는 시뮬레이션
        await asyncio.sleep(0.01)  # 시뮬레이션
        
        return {
            "tool": tool_name,
            "server": self.name,
            "arguments": arguments,
            "result": f"Executed {tool_name} on {self.name}",
        }
    
    def _record_success(self, latency_ms: float):
        """성공 기록"""
        self._total_calls += 1
        self._latencies.append(latency_ms)
        if len(self._latencies) > 100:
            self._latencies = self._latencies[-100:]
    
    def _record_failure(self, error: str):
        """실패 기록"""
        self._total_calls += 1
        self._failed_calls += 1
        self._last_error = error
    
    def get_info(self) -> McpServerInfo:
        """서버 정보 조회"""
        avg_latency = sum(self._latencies) / len(self._latencies) if self._latencies else 0.0
        
        return McpServerInfo(
            name=self.name,
            uri=self.uri,
            state=self._state,
            health=self._health,
            capabilities=list(self._tools.keys()),
            active_connections=self._active_connections,
            total_calls=self._total_calls,
            failed_calls=self._failed_calls,
            avg_latency_ms=avg_latency,
            last_healthcheck=self._last_healthcheck,
            last_error=self._last_error,
        )
    
    def has_capability(self, capability: str) -> bool:
        """능력 보유 여부"""
        return capability in self._tools or capability in self.config.capabilities
    
    async def healthcheck(self) -> HealthStatus:
        """헬스체크 수행"""
        try:
            # 간단한 연결 테스트
            if self._state != ConnectionState.CONNECTED:
                self._health = HealthStatus.UNHEALTHY
            elif self._failed_calls > self._total_calls * 0.5 and self._total_calls > 10:
                self._health = HealthStatus.DEGRADED
            else:
                self._health = HealthStatus.HEALTHY
            
            self._last_healthcheck = datetime.now(timezone.utc)
            return self._health
            
        except Exception as e:
            self._health = HealthStatus.UNHEALTHY
            self._last_error = str(e)
            return self._health

# ============================================================================
# MCP Router - 라우팅
# ============================================================================

class McpRouter(ABC):
    """MCP 라우터 추상 클래스"""
    
    @abstractmethod
    def select_server(
        self,
        tool_name: str,
        servers: list[McpServerConnection],
    ) -> McpServerConnection | None:
        """서버 선택"""
        pass

class CapabilityRouter(McpRouter):
    """능력 기반 라우터"""
    
    def select_server(
        self,
        tool_name: str,
        servers: list[McpServerConnection],
    ) -> McpServerConnection | None:
        # 능력이 있는 서버 필터링
        capable = [s for s in servers if s.has_capability(tool_name)]
        
        if not capable:
            return None
        
        # 연결된 서버 중 선택
        connected = [s for s in capable if s.state == ConnectionState.CONNECTED]
        
        if not connected:
            return capable[0]  # 연결 시도할 서버
        
        # 우선순위 + 활성 연결 수로 선택
        return min(
            connected,
            key=lambda s: (-s.config.priority, s.active_connections)
        )

class RoundRobinRouter(McpRouter):
    """라운드 로빈 라우터"""
    
    def __init__(self):
        self._index = 0
    
    def select_server(
        self,
        tool_name: str,
        servers: list[McpServerConnection],
    ) -> McpServerConnection | None:
        capable = [
            s for s in servers 
            if s.has_capability(tool_name) and s.state == ConnectionState.CONNECTED
        ]
        
        if not capable:
            return None
        
        self._index = (self._index + 1) % len(capable)
        return capable[self._index]

# ============================================================================
# Health Checker - 헬스체크
# ============================================================================

class HealthChecker:
    """
    헬스체크 관리자
    
    모든 MCP 서버의 헬스를 주기적으로 체크
    """
    
    def __init__(
        self,
        servers: dict[str, McpServerConnection],
        interval: float = 30.0,
    ):
        self._servers = servers
        self._interval = interval
        self._running = False
        self._task: asyncio.Task | None = None
        self._logger = StructuredLogger("mcp_healthcheck")
    
    async def start(self):
        """헬스체크 시작"""
        self._running = True
        self._task = asyncio.create_task(self._healthcheck_loop())
        self._logger.info("Health checker started", interval=self._interval)
    
    async def stop(self):
        """헬스체크 중지"""
        self._running = False
        if self._task:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass
    
    async def _healthcheck_loop(self):
        """헬스체크 루프"""
        while self._running:
            try:
                await asyncio.sleep(self._interval)
                await self.check_all()
            except asyncio.CancelledError:
                break
            except Exception as e:
                self._logger.error("Healthcheck error", error=str(e))
    
    async def check_all(self) -> dict[str, HealthStatus]:
        """모든 서버 헬스체크"""
        results = {}
        
        for name, server in self._servers.items():
            try:
                status = await server.healthcheck()
                results[name] = status
            except Exception as e:
                results[name] = HealthStatus.UNHEALTHY
                self._logger.warning(
                    "Healthcheck failed",
                    server=name,
                    error=str(e)
                )
        
        return results
    
    def get_healthy_servers(self) -> list[str]:
        """건강한 서버 목록"""
        return [
            name for name, server in self._servers.items()
            if server.health == HealthStatus.HEALTHY
        ]

# ============================================================================
# MCP Tool Registry - 도구 레지스트리
# ============================================================================

class McpToolRegistry:
    """
    MCP 도구 통합 레지스트리
    
    여러 서버의 도구를 통합 관리
    """
    
    def __init__(self):
        self._tools: dict[str, list[str]] = {}  # tool_name -> [server_names]
        self._schemas: dict[str, dict[str, Any]] = {}
        self._logger = StructuredLogger("mcp_tool_registry")
    
    def register_tool(
        self,
        tool_name: str,
        server_name: str,
        schema: dict[str, Any] | None = None,
    ):
        """도구 등록"""
        if tool_name not in self._tools:
            self._tools[tool_name] = []
        
        if server_name not in self._tools[tool_name]:
            self._tools[tool_name].append(server_name)
        
        if schema:
            self._schemas[f"{server_name}.{tool_name}"] = schema
    
    def unregister_tool(self, tool_name: str, server_name: str):
        """도구 등록 해제"""
        if tool_name in self._tools:
            self._tools[tool_name] = [
                s for s in self._tools[tool_name] if s != server_name
            ]
    
    def get_servers_for_tool(self, tool_name: str) -> list[str]:
        """도구를 제공하는 서버 목록"""
        return self._tools.get(tool_name, [])
    
    def get_all_tools(self) -> list[str]:
        """모든 도구 목록"""
        return list(self._tools.keys())
    
    def get_tool_schema(
        self,
        tool_name: str,
        server_name: str | None = None,
    ) -> dict[str, Any] | None:
        """도구 스키마 조회"""
        if server_name:
            return self._schemas.get(f"{server_name}.{tool_name}")
        
        # 첫 번째 서버의 스키마 반환
        servers = self._tools.get(tool_name, [])
        if servers:
            return self._schemas.get(f"{servers[0]}.{tool_name}")
        
        return None

# ============================================================================
# MCP Workbench - 메인 클래스
# ============================================================================

class McpWorkbench:
    """
    MCP Workbench - 다중 MCP 서버 관리
    
    여러 MCP 서버를 통합 관리하고 도구 호출을 라우팅
    
    사용 예시:
        >>> workbench = McpWorkbench()
        >>>
        >>> # 서버 등록
        >>> workbench.register_server(McpServerConfig(
        ...     name="files",
        ...     uri="stdio://mcp-server-filesystem",
        ...     capabilities=["read_file", "write_file", "list_dir"]
        ... ))
        >>>
        >>> # 연결
        >>> await workbench.connect_all()
        >>>
        >>> # 도구 호출
        >>> result = await workbench.call_tool("read_file", path="/etc/hosts")
        >>>
        >>> # 상태 조회
        >>> status = workbench.get_status()
    """
    
    def __init__(self, config: McpWorkbenchConfig | None = None):
        self.config = config or McpWorkbenchConfig()
        
        self._servers: dict[str, McpServerConnection] = {}
        self._tool_registry = McpToolRegistry()
        self._logger = StructuredLogger("mcp_workbench")
        
        # 라우터 선택
        if self.config.load_balance_strategy == LoadBalanceStrategy.ROUND_ROBIN:
            self._router: McpRouter = RoundRobinRouter()
        else:
            self._router = CapabilityRouter()
        
        # 헬스체커
        self._health_checker: HealthChecker | None = None
    
    def register_server(self, server_config: McpServerConfig) -> McpServerConnection:
        """
        MCP 서버 등록
        
        Args:
            server_config: 서버 설정
            
        Returns:
            생성된 연결 객체
        """
        connection = McpServerConnection(server_config)
        self._servers[server_config.name] = connection
        
        # 도구 등록
        for cap in server_config.capabilities:
            self._tool_registry.register_tool(cap, server_config.name)
        
        self._logger.info(
            "Server registered",
            name=server_config.name,
            uri=server_config.uri,
            capabilities=server_config.capabilities
        )
        
        return connection
    
    def unregister_server(self, name: str) -> bool:
        """서버 등록 해제"""
        if name in self._servers:
            server = self._servers[name]
            
            # 도구 등록 해제
            for cap in server.config.capabilities:
                self._tool_registry.unregister_tool(cap, name)
            
            del self._servers[name]
            return True
        
        return False
    
    async def connect_all(self) -> dict[str, bool]:
        """모든 서버 연결"""
        results = {}
        
        tasks = [
            self._connect_server(name)
            for name in self._servers
        ]
        
        for name, success in zip(self._servers.keys(), await asyncio.gather(*tasks)):
            results[name] = success
        
        # 헬스체커 시작
        if self.config.enable_healthcheck:
            self._health_checker = HealthChecker(
                self._servers,
                self.config.healthcheck_interval
            )
            await self._health_checker.start()
        
        self._logger.info(
            "Connected all servers",
            total=len(results),
            success=sum(results.values())
        )
        
        return results
    
    async def _connect_server(self, name: str) -> bool:
        """개별 서버 연결"""
        server = self._servers.get(name)
        if server:
            return await server.connect()
        return False
    
    async def disconnect_all(self):
        """모든 서버 연결 해제"""
        if self._health_checker:
            await self._health_checker.stop()
        
        for server in self._servers.values():
            await server.disconnect()
        
        self._logger.info("Disconnected all servers")
    
    async def call_tool(
        self,
        tool_name: str,
        server_name: str | None = None,
        timeout: float | None = None,
        **arguments,
    ) -> Any:
        """
        도구 호출
        
        Args:
            tool_name: 도구 이름
            server_name: 특정 서버 지정 (선택적)
            timeout: 타임아웃
            **arguments: 도구 인자
            
        Returns:
            도구 실행 결과
        """
        # 서버 선택
        if server_name:
            server = self._servers.get(server_name)
            if not server:
                raise ValueError(f"Server {server_name} not found")
        else:
            servers = list(self._servers.values())
            server = self._router.select_server(tool_name, servers)
            
            if not server:
                raise ValueError(f"No server available for tool: {tool_name}")
        
        # 연결 확인
        if server.state != ConnectionState.CONNECTED:
            if self.config.enable_auto_reconnect:
                await server.connect()
            else:
                raise ConnectionError(f"Server {server.name} is not connected")
        
        self._logger.debug(
            "Calling tool",
            tool=tool_name,
            server=server.name,
            arguments=list(arguments.keys())
        )
        
        # 도구 호출
        return await server.call_tool(
            tool_name,
            arguments,
            timeout or self.config.default_timeout
        )
    
    def get_all_tools(self) -> list[dict[str, Any]]:
        """모든 사용 가능한 도구 목록"""
        tools = []
        
        for tool_name in self._tool_registry.get_all_tools():
            servers = self._tool_registry.get_servers_for_tool(tool_name)
            schema = self._tool_registry.get_tool_schema(tool_name)
            
            tools.append({
                "name": tool_name,
                "servers": servers,
                "schema": schema,
            })
        
        return tools
    
    def get_server_info(self, name: str) -> McpServerInfo | None:
        """서버 정보 조회"""
        server = self._servers.get(name)
        return server.get_info() if server else None
    
    def get_status(self) -> dict[str, Any]:
        """전체 상태 조회"""
        servers = {
            name: server.get_info().to_dict()
            for name, server in self._servers.items()
        }
        
        healthy = sum(
            1 for s in self._servers.values()
            if s.health == HealthStatus.HEALTHY
        )
        
        return {
            "total_servers": len(self._servers),
            "healthy_servers": healthy,
            "total_tools": len(self._tool_registry.get_all_tools()),
            "load_balance_strategy": self.config.load_balance_strategy.value,
            "servers": servers,
        }
    
    def get_tool_schema_for_llm(self) -> list[dict[str, Any]]:
        """LLM Function Calling용 스키마 생성"""
        schemas = []
        
        for tool_name in self._tool_registry.get_all_tools():
            schema = {
                "type": "function",
                "function": {
                    "name": tool_name,
                    "description": f"MCP Tool: {tool_name}",
                    "parameters": {
                        "type": "object",
                        "properties": {},
                        "required": []
                    }
                }
            }
            
            # 서버별 스키마가 있으면 병합
            tool_schema = self._tool_registry.get_tool_schema(tool_name)
            if tool_schema:
                schema["function"]["parameters"] = tool_schema.get(
                    "inputSchema",
                    schema["function"]["parameters"]
                )
                schema["function"]["description"] = tool_schema.get(
                    "description",
                    schema["function"]["description"]
                )
            
            schemas.append(schema)
        
        return schemas
    
    def __len__(self) -> int:
        return len(self._servers)
    
    def __contains__(self, name: str) -> bool:
        return name in self._servers
