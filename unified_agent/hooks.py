#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Hook 시스템 - 라이프사이클 이벤트 처리

================================================================================
📋 역할: 에이전트 실행의 핵심 지점에서 커스텀 로직 실행
📅 버전: 3.3.0 (2026년 2월)
📦 영감: Microsoft Agent Lightning의 Hook 시스템
================================================================================

🎯 주요 기능:
    - 트레이스/스팬 라이프사이클 훅
    - 롤아웃/어템프트 라이프사이클 훅
    - LLM/도구 호출 훅
    - 훅 우선순위 및 필터링

📌 사용 예시:
    >>> from unified_agent import HookManager, HookPriority
    >>>
    >>> hooks = HookManager()
    >>>
    >>> @hooks.on_trace_start
    >>> async def log_trace_start(trace_id: str, metadata: dict):
    ...     print(f"Trace started: {trace_id}")
    >>>
    >>> @hooks.on_llm_call(priority=HookPriority.HIGH)
    >>> async def rate_limit_check(span, request):
    ...     await check_rate_limit()
"""

from __future__ import annotations

import asyncio
import bisect
import functools
import inspect
import re
import traceback
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from enum import IntEnum
from typing import (
    Any,
    Callable,
    Coroutine,
    Dict,
    List,
    Optional,
    TypeVar,
    Union,
)

from .tracer import Span, SpanKind
from .agent_store import Rollout, Attempt, AttemptStatus
from .utils import StructuredLogger


# ============================================================================
# 타입 정의
# ============================================================================

T = TypeVar("T")

# 훅 함수 타입
SyncHookFunc = Callable[..., Any]
AsyncHookFunc = Callable[..., Coroutine[Any, Any, Any]]
HookFunc = Union[SyncHookFunc, AsyncHookFunc]


# ============================================================================
# 훅 우선순위
# ============================================================================

class HookPriority(IntEnum):
    """
    훅 실행 우선순위
    
    낮은 숫자가 먼저 실행됨.
    """
    HIGHEST = 0
    HIGH = 10
    NORMAL = 50
    LOW = 90
    LOWEST = 100


# ============================================================================
# 훅 이벤트 타입
# ============================================================================

class HookEvent:
    """훅 이벤트 타입 정의"""
    
    # 트레이스 라이프사이클
    TRACE_START = "trace.start"
    TRACE_END = "trace.end"
    
    # 스팬 라이프사이클
    SPAN_START = "span.start"
    SPAN_END = "span.end"
    
    # 롤아웃 라이프사이클
    ROLLOUT_START = "rollout.start"
    ROLLOUT_END = "rollout.end"
    ROLLOUT_QUEUED = "rollout.queued"
    ROLLOUT_DEQUEUED = "rollout.dequeued"
    
    # 어템프트 라이프사이클
    ATTEMPT_START = "attempt.start"
    ATTEMPT_END = "attempt.end"
    ATTEMPT_FAILED = "attempt.failed"
    ATTEMPT_SUCCESS = "attempt.success"
    
    # LLM 라이프사이클
    LLM_CALL_START = "llm.call.start"
    LLM_CALL_END = "llm.call.end"
    LLM_CALL_ERROR = "llm.call.error"
    
    # 도구 라이프사이클
    TOOL_CALL_START = "tool.call.start"
    TOOL_CALL_END = "tool.call.end"
    TOOL_CALL_ERROR = "tool.call.error"
    
    # 리워드
    REWARD_EMITTED = "reward.emitted"
    
    # 메모리
    MEMORY_SAVE = "memory.save"
    MEMORY_LOAD = "memory.load"
    MEMORY_COMPACTION = "memory.compaction"


# ============================================================================
# 훅 등록 정보
# ============================================================================

@dataclass
class HookRegistration:
    """훅 등록 정보"""
    event: str                              # 이벤트 타입
    func: HookFunc                          # 훅 함수
    priority: HookPriority = HookPriority.NORMAL
    name: Optional[str] = None              # 훅 이름 (디버깅용)
    filter_pattern: Optional[str] = None    # 필터 패턴 (정규식)
    once: bool = False                      # 한 번만 실행
    enabled: bool = True                    # 활성화 여부
    
    # 실행 통계
    call_count: int = 0
    last_error: Optional[str] = None
    
    def __post_init__(self):
        if self.name is None:
            self.name = self.func.__name__
        
        if self.filter_pattern:
            self._pattern = re.compile(self.filter_pattern)
        else:
            self._pattern = None
    
    def matches_filter(self, value: str) -> bool:
        """필터 패턴 매칭"""
        if self._pattern is None:
            return True
        return bool(self._pattern.search(value))
    
    @property
    def is_async(self) -> bool:
        """비동기 함수 여부"""
        return asyncio.iscoroutinefunction(self.func)


# ============================================================================
# 훅 컨텍스트
# ============================================================================

@dataclass
class HookContext:
    """
    훅 실행 컨텍스트
    
    훅 함수에 전달되는 컨텍스트 정보.
    """
    event: str                          # 이벤트 타입
    timestamp: float                    # 발생 시각
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    # 관련 객체
    span: Optional[Span] = None
    rollout: Optional[Rollout] = None
    attempt: Optional[Attempt] = None
    
    # 추가 데이터
    data: Dict[str, Any] = field(default_factory=dict)
    
    # 결과/에러
    result: Any = None
    error: Optional[Exception] = None
    
    def __getitem__(self, key: str) -> Any:
        return self.data.get(key)
    
    def __setitem__(self, key: str, value: Any) -> None:
        self.data[key] = value


# ============================================================================
# 훅 결과
# ============================================================================

@dataclass
class HookResult:
    """훅 실행 결과"""
    event: str
    hooks_called: int = 0
    hooks_succeeded: int = 0
    hooks_failed: int = 0
    errors: List[str] = field(default_factory=list)
    results: List[Any] = field(default_factory=list)
    
    @property
    def success(self) -> bool:
        return self.hooks_failed == 0


# ============================================================================
# 훅 매니저
# ============================================================================

class HookManager:
    """
    훅 매니저
    
    훅 등록 및 실행을 관리.
    """
    
    def __init__(
        self,
        logger: Optional[StructuredLogger] = None,
        suppress_errors: bool = True,
    ):
        """
        Args:
            logger: 로거
            suppress_errors: 훅 에러 억제 여부
        """
        self._hooks: Dict[str, List[HookRegistration]] = {}
        self._logger = logger or StructuredLogger("hooks")
        self._suppress_errors = suppress_errors
        self._enabled = True
    
    # ==========================================================================
    # 훅 등록
    # ==========================================================================
    
    def register(
        self,
        event: str,
        func: HookFunc,
        priority: HookPriority = HookPriority.NORMAL,
        name: Optional[str] = None,
        filter_pattern: Optional[str] = None,
        once: bool = False,
    ) -> HookRegistration:
        """
        훅 등록
        
        Args:
            event: 이벤트 타입
            func: 훅 함수
            priority: 우선순위
            name: 훅 이름
            filter_pattern: 필터 패턴
            once: 한 번만 실행
            
        Returns:
            HookRegistration
        """
        registration = HookRegistration(
            event=event,
            func=func,
            priority=priority,
            name=name,
            filter_pattern=filter_pattern,
            once=once,
        )
        
        if event not in self._hooks:
            self._hooks[event] = []
        
        # bisect를 사용한 O(log n) 삽입 (정렬 유지)
        hooks_list = self._hooks[event]
        # priority 기준으로 삽입 위치 찾기
        insert_pos = bisect.bisect_left(
            [h.priority for h in hooks_list], 
            registration.priority
        )
        hooks_list.insert(insert_pos, registration)
        
        self._logger.debug(
            "Hook registered",
            event=event,
            name=registration.name,
            priority=priority.name,
        )
        
        return registration
    
    def unregister(self, registration: HookRegistration) -> bool:
        """훅 등록 해제"""
        if registration.event in self._hooks:
            try:
                self._hooks[registration.event].remove(registration)
                return True
            except ValueError:
                pass
        return False
    
    def unregister_all(self, event: Optional[str] = None) -> int:
        """모든 훅 등록 해제"""
        if event:
            count = len(self._hooks.get(event, []))
            self._hooks[event] = []
            return count
        else:
            count = sum(len(hooks) for hooks in self._hooks.values())
            self._hooks.clear()
            return count
    
    # ==========================================================================
    # 데코레이터
    # ==========================================================================
    
    def hook(
        self,
        event: str,
        priority: HookPriority = HookPriority.NORMAL,
        name: Optional[str] = None,
        filter_pattern: Optional[str] = None,
        once: bool = False,
    ) -> Callable[[HookFunc], HookFunc]:
        """훅 등록 데코레이터"""
        def decorator(func: HookFunc) -> HookFunc:
            self.register(
                event=event,
                func=func,
                priority=priority,
                name=name,
                filter_pattern=filter_pattern,
                once=once,
            )
            return func
        return decorator
    
    # 편의 데코레이터
    def on_trace_start(
        self,
        priority: HookPriority = HookPriority.NORMAL,
    ) -> Callable[[HookFunc], HookFunc]:
        return self.hook(HookEvent.TRACE_START, priority)
    
    def on_trace_end(
        self,
        priority: HookPriority = HookPriority.NORMAL,
    ) -> Callable[[HookFunc], HookFunc]:
        return self.hook(HookEvent.TRACE_END, priority)
    
    def on_span_start(
        self,
        priority: HookPriority = HookPriority.NORMAL,
        filter_pattern: Optional[str] = None,
    ) -> Callable[[HookFunc], HookFunc]:
        return self.hook(HookEvent.SPAN_START, priority, filter_pattern=filter_pattern)
    
    def on_span_end(
        self,
        priority: HookPriority = HookPriority.NORMAL,
        filter_pattern: Optional[str] = None,
    ) -> Callable[[HookFunc], HookFunc]:
        return self.hook(HookEvent.SPAN_END, priority, filter_pattern=filter_pattern)
    
    def on_llm_call(
        self,
        priority: HookPriority = HookPriority.NORMAL,
    ) -> Callable[[HookFunc], HookFunc]:
        return self.hook(HookEvent.LLM_CALL_START, priority)
    
    def on_llm_call_end(
        self,
        priority: HookPriority = HookPriority.NORMAL,
    ) -> Callable[[HookFunc], HookFunc]:
        return self.hook(HookEvent.LLM_CALL_END, priority)
    
    def on_tool_call(
        self,
        priority: HookPriority = HookPriority.NORMAL,
    ) -> Callable[[HookFunc], HookFunc]:
        return self.hook(HookEvent.TOOL_CALL_START, priority)
    
    def on_tool_call_end(
        self,
        priority: HookPriority = HookPriority.NORMAL,
    ) -> Callable[[HookFunc], HookFunc]:
        return self.hook(HookEvent.TOOL_CALL_END, priority)
    
    def on_reward(
        self,
        priority: HookPriority = HookPriority.NORMAL,
    ) -> Callable[[HookFunc], HookFunc]:
        return self.hook(HookEvent.REWARD_EMITTED, priority)
    
    def on_rollout_start(
        self,
        priority: HookPriority = HookPriority.NORMAL,
    ) -> Callable[[HookFunc], HookFunc]:
        return self.hook(HookEvent.ROLLOUT_START, priority)
    
    def on_rollout_end(
        self,
        priority: HookPriority = HookPriority.NORMAL,
    ) -> Callable[[HookFunc], HookFunc]:
        return self.hook(HookEvent.ROLLOUT_END, priority)
    
    def on_attempt_start(
        self,
        priority: HookPriority = HookPriority.NORMAL,
    ) -> Callable[[HookFunc], HookFunc]:
        return self.hook(HookEvent.ATTEMPT_START, priority)
    
    def on_attempt_end(
        self,
        priority: HookPriority = HookPriority.NORMAL,
    ) -> Callable[[HookFunc], HookFunc]:
        return self.hook(HookEvent.ATTEMPT_END, priority)
    
    # ==========================================================================
    # 훅 실행
    # ==========================================================================
    
    async def emit(
        self,
        event: str,
        context: Optional[HookContext] = None,
        **kwargs,
    ) -> HookResult:
        """
        훅 이벤트 발행 (비동기)
        
        Args:
            event: 이벤트 타입
            context: 훅 컨텍스트
            **kwargs: 컨텍스트에 추가할 데이터
            
        Returns:
            HookResult
        """
        import time
        
        result = HookResult(event=event)
        
        if not self._enabled:
            return result
        
        hooks = self._hooks.get(event, [])
        if not hooks:
            return result
        
        # 컨텍스트 생성
        if context is None:
            context = HookContext(
                event=event,
                timestamp=time.time(),
            )
        
        context.data.update(kwargs)
        
        # 삭제할 훅 (once=True)
        to_remove: List[HookRegistration] = []
        
        for hook in hooks:
            if not hook.enabled:
                continue
            
            # 필터 체크
            filter_value = kwargs.get("name", kwargs.get("span_name", ""))
            if not hook.matches_filter(str(filter_value)):
                continue
            
            result.hooks_called += 1
            
            try:
                if hook.is_async:
                    hook_result = await hook.func(context)
                else:
                    hook_result = hook.func(context)
                
                result.results.append(hook_result)
                result.hooks_succeeded += 1
                hook.call_count += 1
                
            except Exception as e:
                result.hooks_failed += 1
                error_msg = f"{hook.name}: {str(e)}"
                result.errors.append(error_msg)
                hook.last_error = error_msg
                
                if not self._suppress_errors:
                    raise
                
                self._logger.error(
                    "Hook execution failed",
                    hook=hook.name,
                    event=event,
                    error=str(e),
                )
            
            if hook.once:
                to_remove.append(hook)
        
        # 한 번만 실행할 훅 제거
        for hook in to_remove:
            self.unregister(hook)
        
        return result
    
    def emit_sync(
        self,
        event: str,
        context: Optional[HookContext] = None,
        **kwargs,
    ) -> HookResult:
        """훅 이벤트 발행 (동기)"""
        try:
            loop = asyncio.get_event_loop()
        except RuntimeError:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
        
        return loop.run_until_complete(
            self.emit(event, context, **kwargs)
        )
    
    # ==========================================================================
    # 상태 관리
    # ==========================================================================
    
    def enable(self) -> None:
        """훅 시스템 활성화"""
        self._enabled = True
    
    def disable(self) -> None:
        """훅 시스템 비활성화"""
        self._enabled = False
    
    def get_hooks(self, event: Optional[str] = None) -> List[HookRegistration]:
        """등록된 훅 목록"""
        if event:
            return list(self._hooks.get(event, []))
        else:
            result: List[HookRegistration] = []
            for hooks in self._hooks.values():
                result.extend(hooks)
            return result
    
    def get_stats(self) -> Dict[str, Any]:
        """훅 통계"""
        stats = {
            "enabled": self._enabled,
            "total_hooks": sum(len(h) for h in self._hooks.values()),
            "events": {},
        }
        
        for event, hooks in self._hooks.items():
            stats["events"][event] = {
                "count": len(hooks),
                "hooks": [
                    {
                        "name": h.name,
                        "priority": h.priority.name,
                        "enabled": h.enabled,
                        "call_count": h.call_count,
                        "last_error": h.last_error,
                    }
                    for h in hooks
                ]
            }
        
        return stats


# ============================================================================
# 전역 훅 매니저
# ============================================================================

_global_hook_manager: Optional[HookManager] = None


def get_hook_manager() -> HookManager:
    """전역 훅 매니저 반환"""
    global _global_hook_manager
    if _global_hook_manager is None:
        _global_hook_manager = HookManager()
    return _global_hook_manager


def set_hook_manager(manager: HookManager) -> None:
    """전역 훅 매니저 설정"""
    global _global_hook_manager
    _global_hook_manager = manager


# ============================================================================
# 편의 함수
# ============================================================================

def on_trace_start(
    priority: HookPriority = HookPriority.NORMAL,
) -> Callable[[HookFunc], HookFunc]:
    """전역 트레이스 시작 훅"""
    return get_hook_manager().on_trace_start(priority)


def on_trace_end(
    priority: HookPriority = HookPriority.NORMAL,
) -> Callable[[HookFunc], HookFunc]:
    """전역 트레이스 종료 훅"""
    return get_hook_manager().on_trace_end(priority)


def on_span_start(
    priority: HookPriority = HookPriority.NORMAL,
    filter_pattern: Optional[str] = None,
) -> Callable[[HookFunc], HookFunc]:
    """전역 스팬 시작 훅"""
    return get_hook_manager().on_span_start(priority, filter_pattern)


def on_span_end(
    priority: HookPriority = HookPriority.NORMAL,
    filter_pattern: Optional[str] = None,
) -> Callable[[HookFunc], HookFunc]:
    """전역 스팬 종료 훅"""
    return get_hook_manager().on_span_end(priority, filter_pattern)


def on_llm_call(
    priority: HookPriority = HookPriority.NORMAL,
) -> Callable[[HookFunc], HookFunc]:
    """전역 LLM 호출 훅"""
    return get_hook_manager().on_llm_call(priority)


def on_tool_call(
    priority: HookPriority = HookPriority.NORMAL,
) -> Callable[[HookFunc], HookFunc]:
    """전역 도구 호출 훅"""
    return get_hook_manager().on_tool_call(priority)


def on_reward(
    priority: HookPriority = HookPriority.NORMAL,
) -> Callable[[HookFunc], HookFunc]:
    """전역 리워드 발행 훅"""
    return get_hook_manager().on_reward(priority)


async def emit_hook(
    event: str,
    **kwargs,
) -> HookResult:
    """전역 훅 이벤트 발행"""
    return await get_hook_manager().emit(event, **kwargs)


# ============================================================================
# 훅 인터셉터
# ============================================================================

class HookInterceptor:
    """
    훅 인터셉터
    
    함수 실행 전후에 훅을 자동으로 실행하는 래퍼.
    """
    
    def __init__(
        self,
        hook_manager: Optional[HookManager] = None,
    ):
        self._manager = hook_manager or get_hook_manager()
    
    def intercept(
        self,
        start_event: str,
        end_event: str,
        error_event: Optional[str] = None,
    ) -> Callable:
        """
        함수 인터셉트 데코레이터
        
        Args:
            start_event: 시작 이벤트
            end_event: 종료 이벤트
            error_event: 에러 이벤트
        """
        def decorator(func: Callable) -> Callable:
            if asyncio.iscoroutinefunction(func):
                @functools.wraps(func)
                async def async_wrapper(*args, **kwargs):
                    import time
                    
                    context = HookContext(
                        event=start_event,
                        timestamp=time.time(),
                        data={
                            "function": func.__name__,
                            "args": args,
                            "kwargs": kwargs,
                        }
                    )
                    
                    # 시작 훅
                    await self._manager.emit(start_event, context)
                    
                    try:
                        result = await func(*args, **kwargs)
                        context.result = result
                        
                        # 종료 훅
                        await self._manager.emit(end_event, context)
                        
                        return result
                        
                    except Exception as e:
                        context.error = e
                        
                        if error_event:
                            await self._manager.emit(error_event, context)
                        
                        raise
                
                return async_wrapper
            else:
                @functools.wraps(func)
                def sync_wrapper(*args, **kwargs):
                    import time
                    
                    context = HookContext(
                        event=start_event,
                        timestamp=time.time(),
                        data={
                            "function": func.__name__,
                            "args": args,
                            "kwargs": kwargs,
                        }
                    )
                    
                    # 시작 훅
                    self._manager.emit_sync(start_event, context)
                    
                    try:
                        result = func(*args, **kwargs)
                        context.result = result
                        
                        # 종료 훅
                        self._manager.emit_sync(end_event, context)
                        
                        return result
                        
                    except Exception as e:
                        context.error = e
                        
                        if error_event:
                            self._manager.emit_sync(error_event, context)
                        
                        raise
                
                return sync_wrapper
        
        return decorator


# ============================================================================
# 내장 훅
# ============================================================================

class BuiltinHooks:
    """기본 제공 훅"""
    
    @staticmethod
    def logging_hook(logger: Optional[StructuredLogger] = None) -> HookFunc:
        """로깅 훅"""
        _logger = logger or StructuredLogger("hooks.logging")
        
        async def hook(context: HookContext):
            _logger.info(
                f"Hook event: {context.event}",
                timestamp=context.timestamp,
                **context.data,
            )
        
        return hook
    
    @staticmethod
    def metrics_hook(
        metrics_collector: Optional[Any] = None,
    ) -> HookFunc:
        """메트릭 수집 훅"""
        async def hook(context: HookContext):
            if metrics_collector:
                metrics_collector.record(context.event, context.data)
        
        return hook
    
    @staticmethod
    def timing_hook() -> HookFunc:
        """타이밍 훅"""
        import time
        
        _start_times: Dict[str, float] = {}
        
        async def hook(context: HookContext):
            event = context.event
            
            if event.endswith(".start"):
                key = event.replace(".start", "")
                _start_times[key] = time.time()
            
            elif event.endswith(".end"):
                key = event.replace(".end", "")
                if key in _start_times:
                    duration = time.time() - _start_times[key]
                    context.data["duration_ms"] = duration * 1000
                    del _start_times[key]
        
        return hook


# ============================================================================
# 훅 컨텍스트 매니저
# ============================================================================

@asynccontextmanager
async def hooked_context(
    manager: HookManager,
    start_event: str,
    end_event: str,
    error_event: Optional[str] = None,
    **initial_data,
):
    """
    훅을 자동으로 발행하는 컨텍스트 매니저
    
    Usage:
        async with hooked_context(hooks, "trace.start", "trace.end") as ctx:
            # 작업 수행
            ctx["result"] = result
    """
    import time
    
    context = HookContext(
        event=start_event,
        timestamp=time.time(),
        data=initial_data,
    )
    
    await manager.emit(start_event, context)
    
    try:
        yield context
        await manager.emit(end_event, context)
        
    except Exception as e:
        context.error = e
        if error_event:
            await manager.emit(error_event, context)
        raise
