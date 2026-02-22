#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Unified Agent Framework - Compaction 시스템 (Context Compaction Module)

================================================================================
📁 파일 위치: unified_agent/compaction.py
📋 역할: 컨텍스트 압축, Memory Flush, Cache-TTL Pruning
📅 최종 업데이트: 2026년 2월
================================================================================

🎯 주요 구성 요소:

    📌 Compaction (컨텍스트 압축):
        - 긴 대화를 요약하여 컨텍스트 절약
        - 자동/수동 트리거 지원
        - 요약 후 디스크에 영속

    📌 Memory Flush (메모리 플러시):
        - Compaction 전 중요 정보를 디스크에 저장
        - 정보 손실 방지
        - 소프트 임계값 기반 자동 트리거

    📌 Cache-TTL Pruning (캐시 정리):
        - 오래된 도구 결과 정리
        - API 비용 최적화
        - Anthropic 캐시 TTL 활용

🔧 핵심 기능:
    - 자동 Compaction 트리거 (컨텍스트 리밋 75%)
    - Pre-compaction Memory Flush
    - 소프트/하드 트리밍
    - JSONL 세션 트랜스크립트 저장

📌 참고:
    - Clawdbot Compaction: https://manthanguptaa.in/posts/clawdbot_memory/
"""

from __future__ import annotations

import os
import json
import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime, timezone, timedelta
from pathlib import Path
from typing import Any, Callable
from enum import Enum

from .utils import StructuredLogger

__all__ = [
    # 설정
    "CompactionConfig",
    "PruningConfig",
    "MemoryFlushConfig",
    # 핵심 클래스
    "ContextCompactor",
    "MemoryFlusher",
    "CacheTTLPruner",
    # 매니저
    "CompactionManager",
    # 모델
    "CompactionSummary",
    "PruningResult",
]

# ============================================================================
# Configuration
# ============================================================================

@dataclass(frozen=True, slots=True)
class CompactionConfig:
    """
    Compaction 설정
    
    Args:
        context_window: 모델의 컨텍스트 윈도우 크기 (tokens)
        reserve_tokens: 출력용 예약 토큰
        trigger_threshold: Compaction 트리거 임계값 (0.0 ~ 1.0)
        keep_recent_turns: 최근 유지할 턴 수
        summary_max_tokens: 요약 최대 토큰 수
    """
    context_window: int = 200_000  # Claude: 200K, GPT-5: 1M
    reserve_tokens: int = 20_000   # 출력용 예약
    trigger_threshold: float = 0.75  # 75%에서 트리거
    keep_recent_turns: int = 10
    summary_max_tokens: int = 2000
    
    @property
    def trigger_tokens(self) -> int:
        """Compaction 트리거 토큰 수"""
        return int((self.context_window - self.reserve_tokens) * self.trigger_threshold)

@dataclass(frozen=True, slots=True)
class MemoryFlushConfig:
    """
    Memory Flush 설정
    
    Args:
        enabled: Memory Flush 활성화 여부
        soft_threshold_tokens: 소프트 임계값 (이 전에 플러시)
        system_prompt: 플러시 시스템 프롬프트
        user_prompt: 플러시 사용자 프롬프트
    """
    enabled: bool = True
    soft_threshold_tokens: int = 4000  # trigger 4000 tokens 전에 플러시
    system_prompt: str = "Session nearing compaction. Store durable memories now."
    user_prompt: str = "Write lasting notes to memory/YYYY-MM-DD.md; reply NO_REPLY if nothing to store."

@dataclass(frozen=True, slots=True)
class PruningConfig:
    """
    Cache-TTL Pruning 설정
    
    Args:
        mode: 프루닝 모드 ('always', 'cache-ttl', 'never')
        ttl_seconds: 캐시 TTL (초)
        keep_last_assistants: 최근 유지할 어시스턴트 메시지 수
        soft_trim_max_chars: 소프트 트리밍 최대 문자 수
        soft_trim_head_chars: 소프트 트리밍 헤드 문자 수
        soft_trim_tail_chars: 소프트 트리밍 테일 문자 수
        hard_clear_enabled: 하드 클리어 활성화
        hard_clear_placeholder: 하드 클리어 플레이스홀더
    """
    mode: str = "cache-ttl"  # 'always', 'cache-ttl', 'never'
    ttl_seconds: int = 300   # 5분 (Anthropic 캐시 기본)
    keep_last_assistants: int = 3
    soft_trim_max_chars: int = 4000
    soft_trim_head_chars: int = 1500
    soft_trim_tail_chars: int = 1500
    hard_clear_enabled: bool = True
    hard_clear_placeholder: str = "[Old tool result content cleared]"

# ============================================================================
# Data Models
# ============================================================================

@dataclass(frozen=True, slots=True)
class CompactionSummary:
    """Compaction 요약 결과"""
    original_turns: int
    compacted_turns: int
    original_tokens: int
    summary_tokens: int
    summary_text: str
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    metadata: dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> dict[str, Any]:
        return {
            "original_turns": self.original_turns,
            "compacted_turns": self.compacted_turns,
            "original_tokens": self.original_tokens,
            "summary_tokens": self.summary_tokens,
            "summary_text": self.summary_text,
            "timestamp": self.timestamp.isoformat(),
            "metadata": self.metadata
        }

@dataclass(frozen=True, slots=True)
class PruningResult:
    """Pruning 결과"""
    pruned_count: int
    original_chars: int
    pruned_chars: int
    mode: str
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))

# ============================================================================
# Message 인터페이스 (프레임워크 호환용)
# ============================================================================

@dataclass(frozen=True, slots=True)
class ConversationTurn:
    """대화 턴 (메시지 + 메타데이터)"""
    role: str  # 'user', 'assistant', 'tool', 'system'
    content: str
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    token_count: int = 0
    tool_results: list[dict[str, Any]] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)
    
    def estimate_tokens(self) -> int:
        """토큰 수 추정 (간단한 휴리스틱: 4 chars ≈ 1 token)"""
        if self.token_count > 0:
            return self.token_count
        return len(self.content) // 4 + sum(len(str(r)) // 4 for r in self.tool_results)

# ============================================================================
# Context Compactor
# ============================================================================

class ContextCompactor:
    """
    컨텍스트 압축기
    
    긴 대화를 요약하여 컨텍스트 절약:
    
    Before Compaction (180K / 200K tokens):
        [Turn 1-140] ... 많은 대화 ...
        [Turn 141-150] 최근 대화
    
    After Compaction (45K / 200K tokens):
        [SUMMARY] "Built REST API with /users, /auth endpoints..."
        [Turn 141-150] 최근 대화 유지
    
    사용 예시:
        >>> compactor = ContextCompactor(config, summarizer_func)
        >>> result = await compactor.compact(turns)
    """
    
    def __init__(
        self,
        config: CompactionConfig | None = None,
        summarizer: Callable[[list[ConversationTurn]], str] | None = None
    ):
        self.config = config or CompactionConfig()
        self._summarizer = summarizer
        self._logger = StructuredLogger("compactor")
    
    def set_summarizer(self, func: Callable[[list[ConversationTurn]], str]):
        """요약 함수 설정 (LLM 호출)"""
        self._summarizer = func
    
    def should_compact(self, turns: list[ConversationTurn]) -> bool:
        """Compaction 필요 여부 확인"""
        total_tokens = sum(t.estimate_tokens() for t in turns)
        return total_tokens >= self.config.trigger_tokens
    
    def get_compaction_point(self, turns: list[ConversationTurn]) -> int:
        """
        Compaction 분기점 계산
        
        최근 keep_recent_turns개는 유지하고 나머지를 요약
        """
        return max(0, len(turns) - self.config.keep_recent_turns)
    
    async def compact(
        self,
        turns: list[ConversationTurn],
        focus_hint: str | None = None
    ) -> tuple[list[ConversationTurn], CompactionSummary]:
        """
        대화 압축 수행
        
        Args:
            turns: 전체 대화 턴 리스트
            focus_hint: 요약 시 집중할 내용 힌트 (예: "decisions and open questions")
        
        Returns:
            (압축된 턴 리스트, 요약 정보)
        """
        if not self._summarizer:
            raise ValueError("Summarizer function not set. Call set_summarizer() first.")
        
        compaction_point = self.get_compaction_point(turns)
        
        if compaction_point == 0:
            self._logger.info("Nothing to compact")
            return turns, CompactionSummary(
                original_turns=len(turns),
                compacted_turns=len(turns),
                original_tokens=sum(t.estimate_tokens() for t in turns),
                summary_tokens=0,
                summary_text=""
            )
        
        # 요약할 턴과 유지할 턴 분리
        turns_to_summarize = turns[:compaction_point]
        turns_to_keep = turns[compaction_point:]
        
        original_tokens = sum(t.estimate_tokens() for t in turns_to_summarize)
        
        # LLM으로 요약 생성
        self._logger.info(
            f"Compacting {len(turns_to_summarize)} turns",
            original_tokens=original_tokens
        )
        
        summary_text = await self._summarizer(turns_to_summarize)
        
        # 요약 턴 생성
        summary_turn = ConversationTurn(
            role="system",
            content=f"[COMPACTION SUMMARY]\n{summary_text}",
            metadata={"type": "compaction_summary", "original_turns": len(turns_to_summarize)}
        )
        
        # 결과 조합
        compacted_turns = [summary_turn] + turns_to_keep
        
        summary = CompactionSummary(
            original_turns=len(turns),
            compacted_turns=len(compacted_turns),
            original_tokens=original_tokens,
            summary_tokens=summary_turn.estimate_tokens(),
            summary_text=summary_text,
            metadata={"focus_hint": focus_hint} if focus_hint else {}
        )
        
        self._logger.info(
            f"Compaction complete",
            original_turns=summary.original_turns,
            compacted_turns=summary.compacted_turns,
            saved_tokens=original_tokens - summary.summary_tokens
        )
        
        return compacted_turns, summary

# ============================================================================
# Memory Flusher
# ============================================================================

class MemoryFlusher:
    """
    Pre-Compaction Memory Flush
    
    Compaction 전에 중요 정보를 디스크에 저장하여 정보 손실 방지:
    
    1. 컨텍스트가 소프트 임계값에 도달
    2. 에이전트에게 중요 정보 저장 요청 (silent turn)
    3. 에이전트가 memory/YYYY-MM-DD.md에 기록
    4. Compaction 진행 (정보 안전)
    
    사용 예시:
        >>> flusher = MemoryFlusher(config, memory_system)
        >>> if flusher.should_flush(current_tokens):
        ...     await flusher.flush(agent, turns)
    """
    
    def __init__(
        self,
        config: MemoryFlushConfig | None = None,
        compaction_config: CompactionConfig | None = None,
        memory_write_func: Callable[[str], None] | None = None
    ):
        self.config = config or MemoryFlushConfig()
        self.compaction_config = compaction_config or CompactionConfig()
        self._memory_write = memory_write_func
        self._logger = StructuredLogger("memory_flusher")
        self._last_flush_tokens = 0
    
    def set_memory_writer(self, func: Callable[[str], None]):
        """메모리 쓰기 함수 설정"""
        self._memory_write = func
    
    def should_flush(self, current_tokens: int) -> bool:
        """
        Memory Flush 필요 여부 확인
        
        Compaction 트리거 전 soft_threshold_tokens 지점에서 플러시
        """
        if not self.config.enabled:
            return False
        
        flush_threshold = (
            self.compaction_config.trigger_tokens - 
            self.config.soft_threshold_tokens
        )
        
        # 이미 이 토큰 수에서 플러시했으면 스킵
        if current_tokens <= self._last_flush_tokens:
            return False
        
        return current_tokens >= flush_threshold
    
    def get_flush_prompt(self) -> tuple[str, str]:
        """플러시 프롬프트 반환 (system, user)"""
        return self.config.system_prompt, self.config.user_prompt
    
    async def flush(
        self,
        agent_response_func: Callable[[str, str], str],
        turns: list[ConversationTurn]
    ) -> str | None:
        """
        Memory Flush 수행
        
        Args:
            agent_response_func: 에이전트 응답 함수 (system_prompt, user_prompt) -> response
            turns: 현재 대화 턴
        
        Returns:
            에이전트 응답 또는 None (NO_REPLY인 경우)
        """
        self._logger.info("Initiating pre-compaction memory flush")
        
        system_prompt, user_prompt = self.get_flush_prompt()
        
        # 에이전트에게 플러시 요청 (silent turn - 사용자에게 보이지 않음)
        response = await agent_response_func(system_prompt, user_prompt)
        
        current_tokens = sum(t.estimate_tokens() for t in turns)
        self._last_flush_tokens = current_tokens
        
        if response.strip().upper() == "NO_REPLY":
            self._logger.info("Agent had nothing to flush")
            return None
        
        self._logger.info("Memory flush complete", response_length=len(response))
        return response

# ============================================================================
# Cache-TTL Pruner
# ============================================================================

class CacheTTLPruner:
    """
    Cache-TTL 기반 도구 결과 정리
    
    Anthropic은 프롬프트 프리픽스를 5분간 캐싱:
    - TTL 내: 캐시된 토큰 90% 할인
    - TTL 후: 전체 재캐싱 필요
    
    TTL 만료 후 오래된 도구 결과를 정리하여 비용 절감:
    
    Before Pruning:
        [Tool Result (exec): 50,000 chars of npm output]
        [Tool Result (read): Large config, 10,000 chars]
        [User: "What happened?"]
    
    After Pruning:
        [Tool Result (exec): "npm WARN...[truncated]...installed."]
        [Tool Result (read): "[Old tool result content cleared]"]
        [User: "What happened?"]
    
    JSONL 원본은 보존됨
    """
    
    def __init__(self, config: PruningConfig | None = None):
        self.config = config or PruningConfig()
        self._logger = StructuredLogger("pruner")
        self._last_cache_time: datetime | None = None
    
    def record_cache_time(self):
        """캐시 시간 기록"""
        self._last_cache_time = datetime.now(timezone.utc)
    
    def is_cache_expired(self) -> bool:
        """캐시 만료 여부 확인"""
        if self._last_cache_time is None:
            return True
        
        elapsed = (datetime.now(timezone.utc) - self._last_cache_time).total_seconds()
        return elapsed > self.config.ttl_seconds
    
    def should_prune(self) -> bool:
        """Pruning 필요 여부 확인"""
        if self.config.mode == "never":
            return False
        if self.config.mode == "always":
            return True
        # cache-ttl 모드
        return self.is_cache_expired()
    
    def _soft_trim(self, content: str) -> str:
        """소프트 트리밍: 앞뒤만 유지하고 중간 생략"""
        if len(content) <= self.config.soft_trim_max_chars:
            return content
        
        head = content[:self.config.soft_trim_head_chars]
        tail = content[-self.config.soft_trim_tail_chars:]
        
        return f"{head}\n...[truncated]...\n{tail}"
    
    def _hard_clear(self, content: str) -> str:
        """하드 클리어: 플레이스홀더로 대체"""
        return self.config.hard_clear_placeholder
    
    def prune_turns(
        self,
        turns: list[ConversationTurn],
        in_place: bool = False
    ) -> tuple[list[ConversationTurn], PruningResult]:
        """
        도구 결과 정리
        
        Args:
            turns: 대화 턴 리스트
            in_place: True면 원본 수정, False면 복사본 반환
        
        Returns:
            (정리된 턴 리스트, 정리 결과)
        """
        if not self.should_prune():
            return turns, PruningResult(
                pruned_count=0,
                original_chars=0,
                pruned_chars=0,
                mode=self.config.mode
            )
        
        result_turns = turns if in_place else [
            ConversationTurn(
                role=t.role,
                content=t.content,
                timestamp=t.timestamp,
                token_count=t.token_count,
                tool_results=t.tool_results.copy(),
                metadata=t.metadata.copy()
            ) for t in turns
        ]
        
        # 최근 N개 어시스턴트 메시지 인덱스 찾기
        assistant_indices = [
            i for i, t in enumerate(result_turns)
            if t.role == "assistant"
        ]
        protected_from = (
            assistant_indices[-self.config.keep_last_assistants]
            if len(assistant_indices) >= self.config.keep_last_assistants
            else 0
        )
        
        pruned_count = 0
        original_chars = 0
        pruned_chars = 0
        
        for i, turn in enumerate(result_turns):
            if i >= protected_from:
                continue  # 최근 턴은 보호
            
            # 도구 결과 정리
            if turn.tool_results:
                for j, tool_result in enumerate(turn.tool_results):
                    if 'content' in tool_result:
                        original = str(tool_result['content'])
                        original_chars += len(original)
                        
                        if self.config.hard_clear_enabled and len(original) > self.config.soft_trim_max_chars * 2:
                            turn.tool_results[j]['content'] = self._hard_clear(original)
                        else:
                            turn.tool_results[j]['content'] = self._soft_trim(original)
                        
                        pruned_chars += len(str(turn.tool_results[j]['content']))
                        pruned_count += 1
        
        result = PruningResult(
            pruned_count=pruned_count,
            original_chars=original_chars,
            pruned_chars=pruned_chars,
            mode=self.config.mode
        )
        
        if pruned_count > 0:
            self._logger.info(
                f"Pruned {pruned_count} tool results",
                saved_chars=original_chars - pruned_chars
            )
        
        return result_turns, result

# ============================================================================
# Compaction Manager (통합 관리)
# ============================================================================

class CompactionManager:
    """
    Compaction 통합 관리자
    
    Memory Flush + Compaction + Pruning을 조율:
    
    1. 컨텍스트 모니터링
    2. 소프트 임계값 → Memory Flush
    3. 하드 임계값 → Compaction
    4. 캐시 만료 → Pruning
    
    사용 예시:
        >>> manager = CompactionManager(
        ...     compaction_config=CompactionConfig(context_window=200000),
        ...     flush_config=MemoryFlushConfig(enabled=True),
        ...     pruning_config=PruningConfig(mode="cache-ttl")
        ... )
        >>> manager.set_summarizer(llm_summarize)
        >>> manager.set_memory_writer(memory.add_daily_note)
        >>> 
        >>> # 매 턴마다 호출
        >>> turns = await manager.process_turns(turns, agent_respond)
    """
    
    def __init__(
        self,
        compaction_config: CompactionConfig | None = None,
        flush_config: MemoryFlushConfig | None = None,
        pruning_config: PruningConfig | None = None,
        transcript_dir: str | None = None
    ):
        self.compaction_config = compaction_config or CompactionConfig()
        self.flush_config = flush_config or MemoryFlushConfig()
        self.pruning_config = pruning_config or PruningConfig()
        
        self.compactor = ContextCompactor(self.compaction_config)
        self.flusher = MemoryFlusher(self.flush_config, self.compaction_config)
        self.pruner = CacheTTLPruner(self.pruning_config)
        
        self.transcript_dir = Path(transcript_dir) if transcript_dir else None
        self._logger = StructuredLogger("compaction_manager")
        
        # 통계
        self.stats = {
            "compactions": 0,
            "flushes": 0,
            "prunes": 0,
            "total_tokens_saved": 0
        }
    
    def set_summarizer(self, func: Callable[[list[ConversationTurn]], str]):
        """요약 함수 설정"""
        self.compactor.set_summarizer(func)
    
    def set_memory_writer(self, func: Callable[[str], None]):
        """메모리 쓰기 함수 설정"""
        self.flusher.set_memory_writer(func)
    
    def get_current_tokens(self, turns: list[ConversationTurn]) -> int:
        """현재 토큰 수 계산"""
        return sum(t.estimate_tokens() for t in turns)
    
    async def process_turns(
        self,
        turns: list[ConversationTurn],
        agent_respond_func: Callable[[str, str], str] | None = None
    ) -> list[ConversationTurn]:
        """
        턴 처리 (필요시 Flush/Compaction/Pruning 수행)
        
        Args:
            turns: 현재 대화 턴 리스트
            agent_respond_func: 에이전트 응답 함수 (Memory Flush용)
        
        Returns:
            처리된 턴 리스트
        """
        current_tokens = self.get_current_tokens(turns)
        
        # 1. Pruning 체크
        if self.pruner.should_prune():
            turns, prune_result = self.pruner.prune_turns(turns)
            if prune_result.pruned_count > 0:
                self.stats["prunes"] += 1
        
        # 2. Memory Flush 체크 (Compaction 전)
        if agent_respond_func and self.flusher.should_flush(current_tokens):
            await self.flusher.flush(agent_respond_func, turns)
            self.stats["flushes"] += 1
        
        # 3. Compaction 체크
        if self.compactor.should_compact(turns):
            turns, summary = await self.compactor.compact(turns)
            self.stats["compactions"] += 1
            self.stats["total_tokens_saved"] += (summary.original_tokens - summary.summary_tokens)
            
            # 트랜스크립트 저장
            if self.transcript_dir:
                await self._save_transcript(summary)
        
        return turns
    
    async def force_compact(
        self,
        turns: list[ConversationTurn],
        focus_hint: str | None = None
    ) -> tuple[list[ConversationTurn], CompactionSummary]:
        """
        강제 Compaction (수동 /compact 명령)
        
        Args:
            turns: 대화 턴 리스트
            focus_hint: 집중할 내용 힌트
        
        Returns:
            (압축된 턴 리스트, 요약)
        """
        turns, summary = await self.compactor.compact(turns, focus_hint)
        self.stats["compactions"] += 1
        
        if self.transcript_dir:
            await self._save_transcript(summary)
        
        return turns, summary
    
    async def _save_transcript(self, summary: CompactionSummary):
        """Compaction 요약을 JSONL 트랜스크립트로 저장"""
        if not self.transcript_dir:
            return
        
        self.transcript_dir.mkdir(parents=True, exist_ok=True)
        
        transcript_file = self.transcript_dir / f"compaction_{datetime.now().strftime('%Y%m%d_%H%M%S')}.jsonl"
        
        with open(transcript_file, 'a', encoding='utf-8') as f:
            f.write(json.dumps(summary.to_dict(), ensure_ascii=False) + "\n")
        
        self._logger.info("Transcript saved", file=str(transcript_file))
    
    def get_stats(self) -> dict[str, Any]:
        """통계 반환"""
        return {
            **self.stats,
            "cache_expired": self.pruner.is_cache_expired()
        }
    
    def record_api_call(self):
        """API 호출 기록 (캐시 타이머 리셋)"""
        self.pruner.record_cache_time()
