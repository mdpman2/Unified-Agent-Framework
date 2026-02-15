#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Prompt Caching 시스템 - LLM 비용 절감

================================================================================
📋 역할: 프롬프트 캐싱을 통한 LLM API 비용 절감
📅 버전: 3.4.0 (2026년 2월)
📦 영감: Anthropic Prompt Caching, OpenAI Predicted Outputs
================================================================================

🎯 주요 기능:
    - 해시 기반 프롬프트 캐싱
    - TTL(Time-To-Live) 만료 관리
    - LRU(Least Recently Used) 퇴거 정책
    - 캐시 히트율 통계
    - 시맨틱 유사도 기반 근접 매칭 (선택적)
    - 메모리/디스크 2계층 캐시

📌 비용 절감 효과:
    - 반복 프롬프트 캐싱으로 90% 비용 절감 가능
    - Anthropic: 캐시된 토큰 90% 할인
    - OpenAI: Predicted Outputs 50% 절감

📌 사용 예시:
    >>> from unified_agent import PromptCache, CacheConfig
    >>>
    >>> cache = PromptCache(CacheConfig(
    ...     max_size_mb=100,
    ...     ttl_seconds=3600,
    ...     enable_semantic_match=True
    ... ))
    >>>
    >>> # 캐시 조회
    >>> cached = await cache.get(prompt, model="gpt-5.2")
    >>> if cached:
    ...     return cached.response
    >>>
    >>> # 응답 저장
    >>> await cache.set(prompt, response, model="gpt-5.2", tokens=1000)
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import pickle
import threading
import time
from abc import ABC, abstractmethod
from collections import OrderedDict
from dataclasses import dataclass, field
from datetime import datetime, timezone, timedelta
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Generic, TypeVar

from .utils import StructuredLogger

__all__ = [
    # 설정
    "CacheConfig",
    "CacheEntry",
    "CacheStats",
    # 캐시 백엔드
    "CacheBackend",
    "MemoryCacheBackend",
    "DiskCacheBackend",
    "TwoLevelCacheBackend",
    # 메인 캐시
    "PromptCache",
    # 유틸리티
    "compute_prompt_hash",
    "estimate_tokens",
]

# ============================================================================
# 설정 및 모델
# ============================================================================

class CacheEvictionPolicy(str, Enum):
    """캐시 퇴거 정책"""
    LRU = "lru"           # Least Recently Used
    LFU = "lfu"           # Least Frequently Used
    TTL = "ttl"           # Time-To-Live only
    FIFO = "fifo"         # First In First Out

@dataclass(frozen=True, slots=True)
class CacheConfig:
    """
    프롬프트 캐시 설정
    
    Args:
        max_size_mb: 최대 캐시 크기 (MB)
        max_entries: 최대 엔트리 수
        ttl_seconds: 기본 TTL (초)
        eviction_policy: 퇴거 정책
        enable_semantic_match: 시맨틱 유사도 매칭 활성화
        semantic_threshold: 시맨틱 유사도 임계값 (0.0 ~ 1.0)
        disk_cache_path: 디스크 캐시 경로 (None이면 메모리만)
        enable_compression: 압축 활성화
    """
    max_size_mb: int = 100
    max_entries: int = 10000
    ttl_seconds: int = 3600  # 1시간
    eviction_policy: CacheEvictionPolicy = CacheEvictionPolicy.LRU
    enable_semantic_match: bool = False
    semantic_threshold: float = 0.95
    disk_cache_path: str | None = None
    enable_compression: bool = True

@dataclass(slots=True)
class CacheEntry:
    """
    캐시 엔트리
    
    Args:
        key: 캐시 키 (해시)
        prompt: 원본 프롬프트
        response: 캐시된 응답
        model: 모델 이름
        created_at: 생성 시간
        expires_at: 만료 시간
        hit_count: 히트 횟수
        tokens_saved: 절감된 토큰 수
        metadata: 추가 메타데이터
    """
    key: str
    prompt: str
    response: str
    model: str
    created_at: datetime
    expires_at: datetime
    hit_count: int = 0
    tokens_saved: int = 0
    metadata: dict[str, Any] = field(default_factory=dict)
    
    @property
    def is_expired(self) -> bool:
        """만료 여부"""
        return datetime.now(timezone.utc) > self.expires_at
    
    @property
    def age_seconds(self) -> float:
        """캐시 경과 시간 (초)"""
        return (datetime.now(timezone.utc) - self.created_at).total_seconds()
    
    def to_dict(self) -> dict[str, Any]:
        return {
            "key": self.key,
            "prompt_preview": self.prompt[:100] + "..." if len(self.prompt) > 100 else self.prompt,
            "model": self.model,
            "created_at": self.created_at.isoformat(),
            "expires_at": self.expires_at.isoformat(),
            "hit_count": self.hit_count,
            "tokens_saved": self.tokens_saved,
        }

@dataclass(slots=True)
class CacheStats:
    """캐시 통계"""
    total_requests: int = 0
    cache_hits: int = 0
    cache_misses: int = 0
    total_tokens_saved: int = 0
    total_cost_saved_usd: float = 0.0
    current_entries: int = 0
    current_size_mb: float = 0.0
    
    @property
    def hit_rate(self) -> float:
        """캐시 히트율"""
        if self.total_requests == 0:
            return 0.0
        return self.cache_hits / self.total_requests
    
    def to_dict(self) -> dict[str, Any]:
        return {
            "total_requests": self.total_requests,
            "cache_hits": self.cache_hits,
            "cache_misses": self.cache_misses,
            "hit_rate": f"{self.hit_rate:.2%}",
            "total_tokens_saved": self.total_tokens_saved,
            "total_cost_saved_usd": f"${self.total_cost_saved_usd:.4f}",
            "current_entries": self.current_entries,
            "current_size_mb": f"{self.current_size_mb:.2f}MB",
        }

# ============================================================================
# 유틸리티 함수
# ============================================================================

def compute_prompt_hash(
    prompt: str,
    model: str,
    system_prompt: str | None = None,
    temperature: float = 0.0,
) -> str:
    """
    프롬프트 해시 계산
    
    동일한 프롬프트+모델+설정에 대해 동일한 해시 생성
    
    Args:
        prompt: 사용자 프롬프트
        model: 모델 이름
        system_prompt: 시스템 프롬프트
        temperature: 온도 설정
        
    Returns:
        SHA256 해시 (16자)
    """
    content = json.dumps({
        "prompt": prompt,
        "model": model,
        "system_prompt": system_prompt or "",
        "temperature": temperature,
    }, sort_keys=True, ensure_ascii=False)
    
    return hashlib.sha256(content.encode('utf-8')).hexdigest()[:16]

def estimate_tokens(text: str) -> int:
    """
    토큰 수 추정 (간단한 휴리스틱)
    
    실제로는 tiktoken 사용 권장
    
    Args:
        text: 텍스트
        
    Returns:
        추정 토큰 수
    """
    # 영어: ~4자 = 1토큰, 한국어: ~2자 = 1토큰
    # 평균적으로 3자 = 1토큰으로 계산
    return max(1, len(text) // 3)

# ============================================================================
# 캐시 백엔드 (추상 클래스)
# ============================================================================

class CacheBackend(ABC):
    """캐시 백엔드 추상 클래스"""
    
    @abstractmethod
    async def get(self, key: str) -> CacheEntry | None:
        """캐시 조회"""
        pass
    
    @abstractmethod
    async def set(self, entry: CacheEntry) -> None:
        """캐시 저장"""
        pass
    
    @abstractmethod
    async def delete(self, key: str) -> bool:
        """캐시 삭제"""
        pass
    
    @abstractmethod
    async def clear(self) -> None:
        """전체 삭제"""
        pass
    
    @abstractmethod
    async def size(self) -> int:
        """현재 엔트리 수"""
        pass
    
    @abstractmethod
    async def keys(self) -> list[str]:
        """모든 키 조회"""
        pass

# ============================================================================
# 메모리 캐시 백엔드 (LRU)
# ============================================================================

class MemoryCacheBackend(CacheBackend):
    """
    메모리 기반 캐시 백엔드 (LRU)
    
    OrderedDict를 사용한 LRU 구현
    """
    
    def __init__(self, config: CacheConfig):
        self.config = config
        self._cache: OrderedDict[str, CacheEntry] = OrderedDict()
        self._lock = asyncio.Lock()
        self._logger = StructuredLogger("memory_cache")
    
    async def get(self, key: str) -> CacheEntry | None:
        async with self._lock:
            if key not in self._cache:
                return None
            
            entry = self._cache[key]
            
            # 만료 체크
            if entry.is_expired:
                del self._cache[key]
                return None
            
            # LRU: 최근 사용으로 이동
            self._cache.move_to_end(key)
            entry.hit_count += 1
            
            return entry
    
    async def set(self, entry: CacheEntry) -> None:
        async with self._lock:
            # 최대 엔트리 수 체크
            while len(self._cache) >= self.config.max_entries:
                # LRU: 가장 오래된 항목 제거
                oldest_key = next(iter(self._cache))
                del self._cache[oldest_key]
                self._logger.debug("Evicted oldest entry", key=oldest_key)
            
            self._cache[entry.key] = entry
            self._cache.move_to_end(entry.key)
    
    async def delete(self, key: str) -> bool:
        async with self._lock:
            if key in self._cache:
                del self._cache[key]
                return True
            return False
    
    async def clear(self) -> None:
        async with self._lock:
            self._cache.clear()
    
    async def size(self) -> int:
        return len(self._cache)
    
    async def keys(self) -> list[str]:
        return list(self._cache.keys())
    
    async def cleanup_expired(self) -> int:
        """만료된 엔트리 정리"""
        async with self._lock:
            expired_keys = [
                key for key, entry in self._cache.items()
                if entry.is_expired
            ]
            for key in expired_keys:
                del self._cache[key]
            return len(expired_keys)

# ============================================================================
# 디스크 캐시 백엔드
# ============================================================================

class DiskCacheBackend(CacheBackend):
    """
    디스크 기반 캐시 백엔드
    
    파일 시스템을 사용한 영속 캐시
    """
    
    def __init__(self, config: CacheConfig):
        self.config = config
        self._cache_dir = Path(config.disk_cache_path or "~/.agent_cache").expanduser()
        self._cache_dir.mkdir(parents=True, exist_ok=True)
        self._lock = asyncio.Lock()
        self._logger = StructuredLogger("disk_cache")
    
    def _get_file_path(self, key: str) -> Path:
        return self._cache_dir / f"{key}.cache"
    
    async def get(self, key: str) -> CacheEntry | None:
        file_path = self._get_file_path(key)
        
        if not file_path.exists():
            return None
        
        try:
            async with self._lock:
                with open(file_path, 'rb') as f:
                    entry = pickle.load(f)
                
                if entry.is_expired:
                    file_path.unlink()
                    return None
                
                entry.hit_count += 1
                with open(file_path, 'wb') as f:
                    pickle.dump(entry, f)
                
                return entry
        except Exception as e:
            self._logger.error("Failed to read cache", key=key, error=str(e))
            return None
    
    async def set(self, entry: CacheEntry) -> None:
        file_path = self._get_file_path(entry.key)
        
        try:
            async with self._lock:
                with open(file_path, 'wb') as f:
                    pickle.dump(entry, f)
        except Exception as e:
            self._logger.error("Failed to write cache", key=entry.key, error=str(e))
    
    async def delete(self, key: str) -> bool:
        file_path = self._get_file_path(key)
        
        if file_path.exists():
            file_path.unlink()
            return True
        return False
    
    async def clear(self) -> None:
        async with self._lock:
            for file_path in self._cache_dir.glob("*.cache"):
                file_path.unlink()
    
    async def size(self) -> int:
        return len(list(self._cache_dir.glob("*.cache")))
    
    async def keys(self) -> list[str]:
        return [f.stem for f in self._cache_dir.glob("*.cache")]

# ============================================================================
# 2계층 캐시 백엔드
# ============================================================================

class TwoLevelCacheBackend(CacheBackend):
    """
    2계층 캐시 백엔드 (메모리 + 디스크)
    
    L1: 메모리 (빠름, 제한적)
    L2: 디스크 (느림, 대용량)
    """
    
    def __init__(self, config: CacheConfig):
        self.config = config
        self._l1 = MemoryCacheBackend(CacheConfig(
            max_entries=min(1000, config.max_entries // 10),
            ttl_seconds=config.ttl_seconds,
        ))
        self._l2 = DiskCacheBackend(config)
        self._logger = StructuredLogger("two_level_cache")
    
    async def get(self, key: str) -> CacheEntry | None:
        # L1에서 먼저 조회
        entry = await self._l1.get(key)
        if entry:
            return entry
        
        # L2에서 조회
        entry = await self._l2.get(key)
        if entry:
            # L1에 승격
            await self._l1.set(entry)
            return entry
        
        return None
    
    async def set(self, entry: CacheEntry) -> None:
        # 양쪽에 저장
        await self._l1.set(entry)
        await self._l2.set(entry)
    
    async def delete(self, key: str) -> bool:
        l1_result = await self._l1.delete(key)
        l2_result = await self._l2.delete(key)
        return l1_result or l2_result
    
    async def clear(self) -> None:
        await self._l1.clear()
        await self._l2.clear()
    
    async def size(self) -> int:
        return await self._l2.size()  # L2가 전체 크기
    
    async def keys(self) -> list[str]:
        return await self._l2.keys()

# ============================================================================
# 메인 프롬프트 캐시
# ============================================================================

# 모델별 토큰 가격 (USD per 1K tokens, 2026년 기준)
MODEL_PRICING = {
    "gpt-5.2": {"input": 0.005, "output": 0.015, "cached": 0.0005},
    "gpt-5.1": {"input": 0.004, "output": 0.012, "cached": 0.0004},
    "gpt-4o": {"input": 0.003, "output": 0.01, "cached": 0.0003},
    "claude-opus-4.5": {"input": 0.015, "output": 0.075, "cached": 0.0015},
    "claude-sonnet-4": {"input": 0.003, "output": 0.015, "cached": 0.0003},
    "o3": {"input": 0.015, "output": 0.060, "cached": 0.0015},
    "default": {"input": 0.003, "output": 0.01, "cached": 0.0003},
}

class PromptCache:
    """
    프롬프트 캐시 시스템
    
    LLM API 호출 비용을 절감하기 위한 캐싱 시스템
    
    주요 기능:
    1. 해시 기반 정확 매칭
    2. TTL 만료 관리
    3. LRU 퇴거 정책
    4. 캐시 히트율 통계
    5. 비용 절감 추적
    
    사용 예시:
        >>> cache = PromptCache(CacheConfig(ttl_seconds=3600))
        >>> await cache.initialize()
        >>>
        >>> # 캐시 조회
        >>> result = await cache.get(
        ...     prompt="Hello, world!",
        ...     model="gpt-5.2"
        ... )
        >>>
        >>> if result:
        ...     print(f"Cache hit! Response: {result.response}")
        ... else:
        ...     response = await llm.chat(prompt)
        ...     await cache.set(prompt, response, model="gpt-5.2")
    """
    
    def __init__(self, config: CacheConfig | None = None):
        self.config = config or CacheConfig()
        self._stats = CacheStats()
        self._logger = StructuredLogger("prompt_cache")
        
        # 백엔드 선택
        if self.config.disk_cache_path:
            self._backend = TwoLevelCacheBackend(self.config)
        else:
            self._backend = MemoryCacheBackend(self.config)
        
        # 시맨틱 매칭용 임베딩 캐시 (선택적)
        self._embedding_cache: dict[str, list[float]] = {}
        self._embedding_func: Callable[[str], list[float]] | None = None
        
        # 백그라운드 정리 태스크
        self._cleanup_task: asyncio.Task | None = None
        self._running = False
    
    async def initialize(self):
        """캐시 초기화 및 백그라운드 정리 시작"""
        self._running = True
        self._cleanup_task = asyncio.create_task(self._cleanup_loop())
        self._logger.info("PromptCache initialized", config=self.config)
    
    async def close(self):
        """캐시 종료"""
        self._running = False
        if self._cleanup_task:
            self._cleanup_task.cancel()
            try:
                await self._cleanup_task
            except asyncio.CancelledError:
                pass
    
    async def _cleanup_loop(self):
        """백그라운드 정리 루프"""
        while self._running:
            try:
                await asyncio.sleep(60)  # 1분마다
                if isinstance(self._backend, MemoryCacheBackend):
                    cleaned = await self._backend.cleanup_expired()
                    if cleaned > 0:
                        self._logger.debug("Cleaned expired entries", count=cleaned)
            except asyncio.CancelledError:
                break
            except Exception as e:
                self._logger.error("Cleanup error", error=str(e))
    
    def set_embedding_function(self, func: Callable[[str], list[float]]):
        """시맨틱 매칭용 임베딩 함수 설정"""
        self._embedding_func = func
    
    async def get(
        self,
        prompt: str,
        model: str,
        system_prompt: str | None = None,
        temperature: float = 0.0,
    ) -> CacheEntry | None:
        """
        캐시 조회
        
        Args:
            prompt: 사용자 프롬프트
            model: 모델 이름
            system_prompt: 시스템 프롬프트
            temperature: 온도 설정
            
        Returns:
            캐시된 엔트리 또는 None
        """
        self._stats.total_requests += 1
        
        # 해시 키 생성
        key = compute_prompt_hash(prompt, model, system_prompt, temperature)
        
        # 정확 매칭 조회
        entry = await self._backend.get(key)
        
        if entry:
            self._stats.cache_hits += 1
            tokens = estimate_tokens(prompt)
            self._stats.total_tokens_saved += tokens
            
            # 비용 절감 계산
            pricing = MODEL_PRICING.get(model, MODEL_PRICING["default"])
            saved = (pricing["input"] - pricing["cached"]) * tokens / 1000
            self._stats.total_cost_saved_usd += saved
            entry.tokens_saved += tokens
            
            self._logger.debug("Cache hit", key=key, model=model)
            return entry
        
        # 시맨틱 매칭 (선택적)
        if self.config.enable_semantic_match and self._embedding_func:
            entry = await self._semantic_search(prompt, model)
            if entry:
                self._stats.cache_hits += 1
                return entry
        
        self._stats.cache_misses += 1
        return None
    
    async def set(
        self,
        prompt: str,
        response: str,
        model: str,
        system_prompt: str | None = None,
        temperature: float = 0.0,
        ttl_seconds: int | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> CacheEntry:
        """
        캐시 저장
        
        Args:
            prompt: 사용자 프롬프트
            response: LLM 응답
            model: 모델 이름
            system_prompt: 시스템 프롬프트
            temperature: 온도 설정
            ttl_seconds: TTL (초)
            metadata: 추가 메타데이터
            
        Returns:
            생성된 캐시 엔트리
        """
        key = compute_prompt_hash(prompt, model, system_prompt, temperature)
        now = datetime.now(timezone.utc)
        ttl = ttl_seconds or self.config.ttl_seconds
        
        entry = CacheEntry(
            key=key,
            prompt=prompt,
            response=response,
            model=model,
            created_at=now,
            expires_at=now + timedelta(seconds=ttl),
            metadata=metadata or {},
        )
        
        await self._backend.set(entry)
        self._stats.current_entries = await self._backend.size()
        
        self._logger.debug("Cache set", key=key, model=model, ttl=ttl)
        return entry
    
    async def _semantic_search(
        self,
        prompt: str,
        model: str,
    ) -> CacheEntry | None:
        """시맨틱 유사도 검색 (근접 매칭)"""
        if not self._embedding_func:
            return None
        
        try:
            query_embedding = self._embedding_func(prompt)
            
            best_entry = None
            best_score = 0.0
            
            for key in await self._backend.keys():
                entry = await self._backend.get(key)
                if not entry or entry.model != model:
                    continue
                
                # 캐시된 임베딩 조회 또는 생성
                if entry.key not in self._embedding_cache:
                    self._embedding_cache[entry.key] = self._embedding_func(entry.prompt)
                
                cached_embedding = self._embedding_cache[entry.key]
                
                # 코사인 유사도 계산
                score = self._cosine_similarity(query_embedding, cached_embedding)
                
                if score > best_score and score >= self.config.semantic_threshold:
                    best_score = score
                    best_entry = entry
            
            if best_entry:
                self._logger.debug(
                    "Semantic match found",
                    score=f"{best_score:.3f}",
                    key=best_entry.key
                )
            
            return best_entry
        except Exception as e:
            self._logger.error("Semantic search failed", error=str(e))
            return None
    
    @staticmethod
    def _cosine_similarity(a: list[float], b: list[float]) -> float:
        """코사인 유사도 계산"""
        dot_product = sum(x * y for x, y in zip(a, b))
        norm_a = sum(x * x for x in a) ** 0.5
        norm_b = sum(x * x for x in b) ** 0.5
        
        if norm_a == 0 or norm_b == 0:
            return 0.0
        
        return dot_product / (norm_a * norm_b)
    
    async def invalidate(
        self,
        prompt: str | None = None,
        model: str | None = None,
        key: str | None = None,
    ) -> bool:
        """
        캐시 무효화
        
        Args:
            prompt: 프롬프트 (key 계산용)
            model: 모델 이름
            key: 직접 키 지정
            
        Returns:
            삭제 성공 여부
        """
        if key:
            return await self._backend.delete(key)
        elif prompt and model:
            key = compute_prompt_hash(prompt, model)
            return await self._backend.delete(key)
        return False
    
    async def clear(self):
        """전체 캐시 삭제"""
        await self._backend.clear()
        self._stats = CacheStats()
        self._embedding_cache.clear()
        self._logger.info("Cache cleared")
    
    def get_stats(self) -> CacheStats:
        """캐시 통계 조회"""
        return self._stats
    
    async def get_all_entries(self) -> list[CacheEntry]:
        """모든 캐시 엔트리 조회"""
        entries = []
        for key in await self._backend.keys():
            entry = await self._backend.get(key)
            if entry:
                entries.append(entry)
        return entries

# ============================================================================
# 데코레이터
# ============================================================================

def cached_prompt(
    cache: PromptCache,
    model: str,
    ttl_seconds: int | None = None,
):
    """
    프롬프트 캐싱 데코레이터
    
    사용 예시:
        >>> @cached_prompt(cache, model="gpt-5.2")
        >>> async def chat(prompt: str) -> str:
        ...     return await llm.chat(prompt)
    """
    def decorator(func):
        async def wrapper(prompt: str, *args, **kwargs):
            # 캐시 조회
            cached = await cache.get(prompt, model)
            if cached:
                return cached.response
            
            # 원본 함수 실행
            response = await func(prompt, *args, **kwargs)
            
            # 캐시 저장
            await cache.set(prompt, response, model, ttl_seconds=ttl_seconds)
            
            return response
        return wrapper
    return decorator
