#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Unified Agent Framework - 영속 메모리 시스템 (Persistent Memory Module)

================================================================================
📁 파일 위치: unified_agent/persistent_memory.py
📋 역할: Clawdbot 스타일 2계층 영속 메모리 + 하이브리드 검색
📅 최종 업데이트: 2026년 2월
================================================================================

🎯 주요 구성 요소:

    📌 2계층 메모리 시스템:
        - Layer 1: Daily Logs (memory/YYYY-MM-DD.md) - 일별 기록
        - Layer 2: Long-term Memory (MEMORY.md) - 장기 기억

    📌 메모리 도구:
        - memory_search: 시맨틱 + 키워드 하이브리드 검색
        - memory_get: 특정 라인 범위 읽기
        - memory_write: 메모리 파일에 기록

    📌 인덱싱 시스템:
        - 청킹 (400 tokens, 80 overlap)
        - 임베딩 (OpenAI text-embedding-3-small)
        - SQLite + FTS5 (전문 검색)

🔧 핵심 기능:
    - 하이브리드 검색: Vector (70%) + BM25 (30%)
    - 자동 인덱싱: 파일 변경 시 자동 재인덱싱
    - Multi-Agent 메모리 격리
    - Bootstrap Files 패턴 (AGENTS.md, SOUL.md, USER.md)

📌 참고:
    - Clawdbot Memory System: https://manthanguptaa.in/posts/clawdbot_memory/
    - sqlite-vec: https://github.com/asg017/sqlite-vec
"""

import os
import re
import json
import sqlite3
import hashlib
import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime, timezone, date
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple, Callable
from enum import Enum

from .utils import StructuredLogger

__all__ = [
    # 메모리 시스템
    "PersistentMemory",
    "MemoryConfig",
    "MemoryLayer",
    # 검색 결과
    "MemorySearchResult",
    "MemoryChunk",
    # 도구
    "MemorySearchTool",
    "MemoryGetTool",
    "MemoryWriteTool",
    # Bootstrap Files
    "BootstrapFileManager",
    "BootstrapFileType",
    # 인덱서
    "MemoryIndexer",
]


# ============================================================================
# Enums & Constants
# ============================================================================

class MemoryLayer(Enum):
    """메모리 계층"""
    DAILY_LOG = "daily_log"        # Layer 1: 일별 기록 (memory/YYYY-MM-DD.md)
    LONG_TERM = "long_term"        # Layer 2: 장기 기억 (MEMORY.md)
    BOOTSTRAP = "bootstrap"        # Bootstrap 파일 (AGENTS.md, SOUL.md 등)


class BootstrapFileType(Enum):
    """Bootstrap 파일 유형 (Clawdbot 패턴)"""
    AGENTS = "AGENTS.md"     # 에이전트 지시사항, 메모리 가이드라인
    SOUL = "SOUL.md"         # 성격과 톤
    USER = "USER.md"         # 사용자 정보
    TOOLS = "TOOLS.md"       # 외부 도구 사용 가이드
    MEMORY = "MEMORY.md"     # 장기 기억


# ============================================================================
# Configuration
# ============================================================================

@dataclass
class MemoryConfig:
    """
    영속 메모리 설정
    
    Args:
        workspace_dir: 메모리 워크스페이스 디렉토리
        state_dir: SQLite 인덱스 저장 디렉토리
        chunk_size: 청크 크기 (토큰)
        chunk_overlap: 청크 오버랩 (토큰)
        vector_weight: 하이브리드 검색에서 벡터 가중치 (0.0 ~ 1.0)
        min_search_score: 최소 검색 점수 임계값
        max_search_results: 최대 검색 결과 수
        embedding_model: 임베딩 모델명
    """
    workspace_dir: str = field(default_factory=lambda: os.path.expanduser("~/agent_memory"))
    state_dir: str = field(default_factory=lambda: os.path.expanduser("~/.agent_memory"))
    chunk_size: int = 400          # ~400 tokens per chunk
    chunk_overlap: int = 80        # 80 token overlap
    vector_weight: float = 0.7     # 70% vector, 30% BM25
    min_search_score: float = 0.35
    max_search_results: int = 10
    embedding_model: str = "text-embedding-3-small"
    embedding_dimensions: int = 1536


# ============================================================================
# Data Models
# ============================================================================

@dataclass
class MemoryChunk:
    """메모리 청크 (인덱싱 단위)"""
    id: str
    path: str
    start_line: int
    end_line: int
    text: str
    content_hash: str
    layer: MemoryLayer
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))


@dataclass
class MemorySearchResult:
    """메모리 검색 결과"""
    path: str
    start_line: int
    end_line: int
    score: float
    snippet: str
    layer: MemoryLayer
    vector_score: float = 0.0
    text_score: float = 0.0
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "path": self.path,
            "startLine": self.start_line,
            "endLine": self.end_line,
            "score": round(self.score, 3),
            "snippet": self.snippet,
            "source": self.layer.value,
            "vectorScore": round(self.vector_score, 3),
            "textScore": round(self.text_score, 3),
        }


# ============================================================================
# Memory Indexer (SQLite + FTS5)
# ============================================================================

class MemoryIndexer:
    """
    메모리 인덱서 - SQLite + FTS5 기반 하이브리드 검색
    
    SQLite 테이블:
        - chunks: 청크 메타데이터 (id, path, start_line, end_line, text, hash)
        - chunks_fts: FTS5 전문 검색 인덱스
        - embeddings: 임베딩 벡터 캐시 (hash -> vector)
    """
    
    def __init__(
        self,
        db_path: str,
        embedding_func: Optional[Callable[[str], List[float]]] = None,
        config: Optional[MemoryConfig] = None
    ):
        self.db_path = db_path
        self.config = config or MemoryConfig()
        self._embedding_func = embedding_func
        self._logger = StructuredLogger("memory_indexer")
        self._conn: Optional[sqlite3.Connection] = None
        self._init_database()
    
    def _init_database(self):
        """데이터베이스 초기화"""
        os.makedirs(os.path.dirname(self.db_path), exist_ok=True)
        self._conn = sqlite3.connect(self.db_path, check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        
        # 청크 테이블
        self._conn.execute("""
            CREATE TABLE IF NOT EXISTS chunks (
                id TEXT PRIMARY KEY,
                path TEXT NOT NULL,
                start_line INTEGER NOT NULL,
                end_line INTEGER NOT NULL,
                text TEXT NOT NULL,
                content_hash TEXT NOT NULL,
                layer TEXT NOT NULL,
                created_at TEXT NOT NULL
            )
        """)
        
        # FTS5 전문 검색 테이블
        self._conn.execute("""
            CREATE VIRTUAL TABLE IF NOT EXISTS chunks_fts USING fts5(
                text,
                content=chunks,
                content_rowid=rowid
            )
        """)
        
        # 임베딩 캐시 테이블
        self._conn.execute("""
            CREATE TABLE IF NOT EXISTS embeddings (
                content_hash TEXT PRIMARY KEY,
                vector BLOB NOT NULL,
                model TEXT NOT NULL,
                created_at TEXT NOT NULL
            )
        """)
        
        # 인덱스
        self._conn.execute("CREATE INDEX IF NOT EXISTS idx_chunks_path ON chunks(path)")
        self._conn.execute("CREATE INDEX IF NOT EXISTS idx_chunks_hash ON chunks(content_hash)")
        
        self._conn.commit()
        self._logger.info("Database initialized", db_path=self.db_path)
    
    def set_embedding_function(self, func: Callable[[str], List[float]]):
        """임베딩 함수 설정"""
        self._embedding_func = func
    
    def _compute_hash(self, text: str) -> str:
        """텍스트 해시 계산"""
        return hashlib.sha256(text.encode('utf-8')).hexdigest()[:16]
    
    def _chunk_text(self, text: str, path: str, layer: MemoryLayer) -> List[MemoryChunk]:
        """텍스트를 청크로 분할"""
        lines = text.split('\n')
        chunks = []
        
        # 간단한 라인 기반 청킹 (토큰 추정: ~4 chars/token)
        chars_per_chunk = self.config.chunk_size * 4
        overlap_chars = self.config.chunk_overlap * 4
        
        current_text = ""
        current_start = 1
        
        for i, line in enumerate(lines, start=1):
            current_text += line + "\n"
            
            if len(current_text) >= chars_per_chunk:
                chunk_id = f"{self._compute_hash(path)}_{current_start}_{i}"
                chunks.append(MemoryChunk(
                    id=chunk_id,
                    path=path,
                    start_line=current_start,
                    end_line=i,
                    text=current_text.strip(),
                    content_hash=self._compute_hash(current_text),
                    layer=layer
                ))
                
                # 오버랩을 위해 일부 보존
                overlap_text = current_text[-overlap_chars:] if len(current_text) > overlap_chars else ""
                current_text = overlap_text
                current_start = max(1, i - len(overlap_text.split('\n')) + 1)
        
        # 마지막 청크
        if current_text.strip():
            chunk_id = f"{self._compute_hash(path)}_{current_start}_{len(lines)}"
            chunks.append(MemoryChunk(
                id=chunk_id,
                path=path,
                start_line=current_start,
                end_line=len(lines),
                text=current_text.strip(),
                content_hash=self._compute_hash(current_text),
                layer=layer
            ))
        
        return chunks
    
    async def index_file(self, file_path: str, layer: MemoryLayer) -> int:
        """파일 인덱싱"""
        if not os.path.exists(file_path):
            self._logger.warning("File not found", path=file_path)
            return 0
        
        with open(file_path, 'r', encoding='utf-8') as f:
            text = f.read()
        
        # 기존 청크 삭제
        self._conn.execute("DELETE FROM chunks WHERE path = ?", (file_path,))
        
        # 새 청크 생성 및 저장
        chunks = self._chunk_text(text, file_path, layer)
        
        for chunk in chunks:
            self._conn.execute("""
                INSERT OR REPLACE INTO chunks (id, path, start_line, end_line, text, content_hash, layer, created_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                chunk.id, chunk.path, chunk.start_line, chunk.end_line,
                chunk.text, chunk.content_hash, chunk.layer.value,
                chunk.created_at.isoformat()
            ))
        
        # FTS 인덱스 재구축
        self._conn.execute("INSERT INTO chunks_fts(chunks_fts) VALUES('rebuild')")
        self._conn.commit()
        
        self._logger.info("File indexed", path=file_path, chunks=len(chunks))
        return len(chunks)
    
    async def search_bm25(self, query: str, limit: int = 10) -> List[Tuple[str, float]]:
        """BM25 키워드 검색"""
        cursor = self._conn.execute("""
            SELECT chunks.id, bm25(chunks_fts) as score
            FROM chunks_fts
            JOIN chunks ON chunks.rowid = chunks_fts.rowid
            WHERE chunks_fts MATCH ?
            ORDER BY score
            LIMIT ?
        """, (query, limit))
        
        results = []
        for row in cursor.fetchall():
            # BM25 점수 정규화 (음수 -> 양수)
            normalized_score = 1.0 / (1.0 + abs(row['score']))
            results.append((row['id'], normalized_score))
        
        return results
    
    async def search_vector(self, query: str, limit: int = 10) -> List[Tuple[str, float]]:
        """벡터 시맨틱 검색"""
        if not self._embedding_func:
            self._logger.debug("Embedding function not set, skipping vector search")
            return []
        
        try:
            query_embedding = self._embedding_func(query)
        except Exception as e:
            self._logger.error(f"Embedding failed: {e}")
            return []
        
        # 모든 청크의 임베딩과 코사인 유사도 계산
        cursor = self._conn.execute("""
            SELECT c.id, c.content_hash, e.vector
            FROM chunks c
            LEFT JOIN embeddings e ON c.content_hash = e.content_hash
        """)
        
        results = []
        for row in cursor.fetchall():
            if row['vector']:
                stored_embedding = json.loads(row['vector'])
                similarity = self._cosine_similarity(query_embedding, stored_embedding)
                results.append((row['id'], similarity))
        
        # 점수순 정렬
        results.sort(key=lambda x: x[1], reverse=True)
        return results[:limit]
    
    def _cosine_similarity(self, vec1: List[float], vec2: List[float]) -> float:
        """코사인 유사도 계산"""
        dot_product = sum(a * b for a, b in zip(vec1, vec2))
        norm1 = sum(a * a for a in vec1) ** 0.5
        norm2 = sum(b * b for b in vec2) ** 0.5
        
        if norm1 == 0 or norm2 == 0:
            return 0.0
        return dot_product / (norm1 * norm2)
    
    async def hybrid_search(
        self,
        query: str,
        limit: int = 10,
        min_score: float = 0.35
    ) -> List[MemorySearchResult]:
        """
        하이브리드 검색 (Vector 70% + BM25 30%)
        """
        vector_results = await self.search_vector(query, limit * 2)
        bm25_results = await self.search_bm25(query, limit * 2)
        
        # 점수 결합
        combined_scores: Dict[str, Dict[str, float]] = {}
        
        for chunk_id, score in vector_results:
            combined_scores[chunk_id] = {'vector': score, 'text': 0.0}
        
        for chunk_id, score in bm25_results:
            if chunk_id in combined_scores:
                combined_scores[chunk_id]['text'] = score
            else:
                combined_scores[chunk_id] = {'vector': 0.0, 'text': score}
        
        # 가중 평균 계산
        final_scores = []
        for chunk_id, scores in combined_scores.items():
            final_score = (
                self.config.vector_weight * scores['vector'] +
                (1 - self.config.vector_weight) * scores['text']
            )
            if final_score >= min_score:
                final_scores.append((chunk_id, final_score, scores['vector'], scores['text']))
        
        # 정렬 및 상위 결과 선택
        final_scores.sort(key=lambda x: x[1], reverse=True)
        
        # 결과 조회
        results = []
        for chunk_id, final_score, vector_score, text_score in final_scores[:limit]:
            cursor = self._conn.execute(
                "SELECT * FROM chunks WHERE id = ?", (chunk_id,)
            )
            row = cursor.fetchone()
            if row:
                results.append(MemorySearchResult(
                    path=row['path'],
                    start_line=row['start_line'],
                    end_line=row['end_line'],
                    score=final_score,
                    snippet=row['text'][:500] + "..." if len(row['text']) > 500 else row['text'],
                    layer=MemoryLayer(row['layer']),
                    vector_score=vector_score,
                    text_score=text_score
                ))
        
        return results
    
    def get_chunk_by_id(self, chunk_id: str) -> Optional[MemoryChunk]:
        """청크 조회"""
        cursor = self._conn.execute("SELECT * FROM chunks WHERE id = ?", (chunk_id,))
        row = cursor.fetchone()
        if row:
            return MemoryChunk(
                id=row['id'],
                path=row['path'],
                start_line=row['start_line'],
                end_line=row['end_line'],
                text=row['text'],
                content_hash=row['content_hash'],
                layer=MemoryLayer(row['layer']),
                created_at=datetime.fromisoformat(row['created_at'])
            )
        return None
    
    async def store_embedding(self, content_hash: str, embedding: List[float], model: str):
        """임베딩 저장"""
        self._conn.execute("""
            INSERT OR REPLACE INTO embeddings (content_hash, vector, model, created_at)
            VALUES (?, ?, ?, ?)
        """, (
            content_hash,
            json.dumps(embedding),
            model,
            datetime.now(timezone.utc).isoformat()
        ))
        self._conn.commit()
    
    def close(self):
        """연결 종료"""
        if self._conn:
            self._conn.close()


# ============================================================================
# Bootstrap File Manager
# ============================================================================

class BootstrapFileManager:
    """
    Bootstrap 파일 관리자 (Clawdbot 패턴)
    
    에이전트 설정을 투명하게 관리하는 Markdown 파일들:
        - AGENTS.md: 에이전트 지시사항
        - SOUL.md: 성격과 톤
        - USER.md: 사용자 정보
        - TOOLS.md: 도구 사용 가이드
        - MEMORY.md: 장기 기억
    """
    
    DEFAULT_AGENTS_MD = """# Agent Instructions

## Every Session

Before doing anything else:
1. Read SOUL.md - this is who you are
2. Read USER.md - this is who you are helping
3. Read memory/YYYY-MM-DD.md (today and yesterday) for recent context
4. Read MEMORY.md for long-term knowledge

Don't ask permission, just do it.

## Memory Guidelines

### Where to Write
| Type | Location |
|------|----------|
| Day-to-day notes, "remember this" | `memory/YYYY-MM-DD.md` |
| Durable facts, preferences, decisions | `MEMORY.md` |
| Lessons learned | `AGENTS.md` or `TOOLS.md` |

### When to Search Memory
Before answering questions about:
- Prior work or decisions
- Dates and timelines
- People and contacts
- User preferences
- Todos and tasks
"""

    DEFAULT_SOUL_MD = """# Agent Personality

## Core Traits
- Professional but friendly
- Concise and clear
- Proactive and helpful
- Honest about limitations

## Communication Style
- Use Korean by default (respond in user's language)
- Provide explanations when needed
- Ask clarifying questions when uncertain
"""

    DEFAULT_USER_MD = """# User Information

## Preferences
- Language: Korean (한국어)
- Response style: Detailed but concise

## Current Projects
(To be filled by the agent during conversations)

## Important Contacts
(To be filled by the agent during conversations)
"""

    DEFAULT_MEMORY_MD = """# Long-term Memory

## User Preferences
(Curated knowledge about user preferences)

## Important Decisions
(Key decisions and their rationale)

## Key Contacts
(Important people and their roles)

## Lessons Learned
(What worked and what didn't)
"""
    
    def __init__(self, workspace_dir: str):
        self.workspace_dir = Path(workspace_dir)
        self._logger = StructuredLogger("bootstrap_files")
    
    def ensure_bootstrap_files(self):
        """Bootstrap 파일들이 존재하는지 확인하고 없으면 생성"""
        self.workspace_dir.mkdir(parents=True, exist_ok=True)
        
        defaults = {
            BootstrapFileType.AGENTS: self.DEFAULT_AGENTS_MD,
            BootstrapFileType.SOUL: self.DEFAULT_SOUL_MD,
            BootstrapFileType.USER: self.DEFAULT_USER_MD,
            BootstrapFileType.MEMORY: self.DEFAULT_MEMORY_MD,
        }
        
        for file_type, default_content in defaults.items():
            file_path = self.workspace_dir / file_type.value
            if not file_path.exists():
                file_path.write_text(default_content, encoding='utf-8')
                self._logger.info(f"Created {file_type.value}")
    
    def get_file_path(self, file_type: BootstrapFileType) -> Path:
        """Bootstrap 파일 경로 반환"""
        return self.workspace_dir / file_type.value
    
    def read_file(self, file_type: BootstrapFileType) -> str:
        """Bootstrap 파일 읽기"""
        file_path = self.get_file_path(file_type)
        if file_path.exists():
            return file_path.read_text(encoding='utf-8')
        return ""
    
    def write_file(self, file_type: BootstrapFileType, content: str):
        """Bootstrap 파일 쓰기"""
        file_path = self.get_file_path(file_type)
        file_path.write_text(content, encoding='utf-8')
        self._logger.info(f"Updated {file_type.value}")
    
    def append_to_file(self, file_type: BootstrapFileType, content: str):
        """Bootstrap 파일에 추가"""
        existing = self.read_file(file_type)
        self.write_file(file_type, existing + "\n" + content)
    
    def get_project_context(self) -> str:
        """모든 Bootstrap 파일을 결합하여 프로젝트 컨텍스트 생성"""
        context_parts = []
        
        for file_type in [BootstrapFileType.AGENTS, BootstrapFileType.SOUL, 
                          BootstrapFileType.USER, BootstrapFileType.MEMORY]:
            content = self.read_file(file_type)
            if content:
                context_parts.append(f"=== {file_type.value} ===\n{content}")
        
        return "\n\n".join(context_parts)


# ============================================================================
# Persistent Memory System
# ============================================================================

class PersistentMemory:
    """
    2계층 영속 메모리 시스템 (Clawdbot 스타일)
    
    Layer 1: Daily Logs (memory/YYYY-MM-DD.md)
        - 일별 기록
        - append-only
        - "remember this" 류의 메모
    
    Layer 2: Long-term Memory (MEMORY.md)
        - 장기 기억
        - 중요한 결정, 선호도, 연락처 등
        - 에이전트가 큐레이션
    
    v3.3: Compaction 자동 연동
        - 컨텍스트 임계값 도달 시 자동 Compaction 트리거
        - Memory Flush → Compaction → Pruning 순서로 진행
    
    사용 예시:
        >>> memory = PersistentMemory(agent_id="main")
        >>> await memory.initialize()
        >>> 
        >>> # 오늘 기록에 추가
        >>> await memory.add_daily_note("오늘 API 설계 결정: REST over GraphQL")
        >>> 
        >>> # 장기 기억에 추가
        >>> await memory.add_long_term_memory("## 사용자 선호도\\n- TypeScript 선호")
        >>> 
        >>> # 검색
        >>> results = await memory.search("API 설계")
        >>> 
        >>> # v3.3: 컨텍스트 체크 및 자동 Compaction
        >>> turns = await memory.check_and_compact(turns, agent_func)
    """
    
    def __init__(
        self,
        agent_id: str = "main",
        config: Optional[MemoryConfig] = None,
        embedding_func: Optional[Callable[[str], List[float]]] = None,
        compaction_manager: Optional[Any] = None  # v3.3: CompactionManager 연동
    ):
        self.agent_id = agent_id
        self.config = config or MemoryConfig()
        
        # 워크스페이스 설정
        self.workspace_dir = Path(self.config.workspace_dir) / agent_id
        self.memory_dir = self.workspace_dir / "memory"
        
        # 인덱서 설정
        state_dir = Path(self.config.state_dir)
        db_path = state_dir / f"{agent_id}.sqlite"
        self.indexer = MemoryIndexer(str(db_path), embedding_func, config)
        
        # Bootstrap 파일 관리자
        self.bootstrap = BootstrapFileManager(str(self.workspace_dir))
        
        # v3.3: Compaction 연동
        self._compaction_manager = compaction_manager
        self._auto_compact_enabled = True
        self._context_threshold = 0.75  # 75%에서 자동 Compaction
        
        self._logger = StructuredLogger("persistent_memory")
    
    def set_compaction_manager(self, manager: Any):
        """v3.3: CompactionManager 설정"""
        self._compaction_manager = manager
        # Memory writer 연결
        if hasattr(manager, 'set_memory_writer'):
            manager.set_memory_writer(lambda content: self._sync_add_daily_note(content))
        self._logger.info("CompactionManager connected to PersistentMemory")
    
    def _sync_add_daily_note(self, content: str):
        """동기 방식 daily note 추가 (Compaction용)"""
        import asyncio
        loop = asyncio.get_event_loop()
        if loop.is_running():
            asyncio.create_task(self.add_daily_note(content))
        else:
            loop.run_until_complete(self.add_daily_note(content))
    
    async def check_and_compact(
        self,
        turns: List[Any],
        agent_respond_func: Optional[Callable] = None
    ) -> List[Any]:
        """
        v3.3: 컨텍스트 체크 및 자동 Compaction
        
        Args:
            turns: 현재 대화 턴 리스트
            agent_respond_func: 에이전트 응답 함수
        
        Returns:
            처리된 턴 리스트 (필요시 압축됨)
        """
        if not self._compaction_manager or not self._auto_compact_enabled:
            return turns
        
        # CompactionManager.process_turns() 호출
        return await self._compaction_manager.process_turns(turns, agent_respond_func)
    
    async def initialize(self):
        """메모리 시스템 초기화"""
        # 디렉토리 생성
        self.workspace_dir.mkdir(parents=True, exist_ok=True)
        self.memory_dir.mkdir(parents=True, exist_ok=True)
        
        # Bootstrap 파일 생성
        self.bootstrap.ensure_bootstrap_files()
        
        # 기존 파일 인덱싱
        await self._index_all_memory_files()
        
        self._logger.info(
            "Persistent memory initialized",
            agent_id=self.agent_id,
            workspace=str(self.workspace_dir)
        )
    
    async def _index_all_memory_files(self):
        """모든 메모리 파일 인덱싱"""
        # MEMORY.md 인덱싱
        memory_md = self.workspace_dir / "MEMORY.md"
        if memory_md.exists():
            await self.indexer.index_file(str(memory_md), MemoryLayer.LONG_TERM)
        
        # Daily logs 인덱싱
        for daily_file in self.memory_dir.glob("*.md"):
            await self.indexer.index_file(str(daily_file), MemoryLayer.DAILY_LOG)
    
    def _get_today_log_path(self) -> Path:
        """오늘 날짜의 로그 파일 경로"""
        today = date.today().isoformat()  # YYYY-MM-DD
        return self.memory_dir / f"{today}.md"
    
    async def add_daily_note(self, content: str, timestamp: Optional[datetime] = None):
        """
        오늘 기록에 메모 추가 (Layer 1)
        
        Args:
            content: 메모 내용
            timestamp: 타임스탬프 (기본: 현재 시간)
        """
        log_path = self._get_today_log_path()
        ts = timestamp or datetime.now(timezone.utc)
        time_str = ts.strftime("%H:%M")
        
        # 파일이 없으면 헤더 생성
        if not log_path.exists():
            header = f"# {date.today().isoformat()}\n\n"
            log_path.write_text(header, encoding='utf-8')
        
        # 메모 추가
        entry = f"## {time_str}\n{content}\n\n"
        with open(log_path, 'a', encoding='utf-8') as f:
            f.write(entry)
        
        # 재인덱싱
        await self.indexer.index_file(str(log_path), MemoryLayer.DAILY_LOG)
        
        self._logger.info("Added daily note", time=time_str)
    
    async def add_long_term_memory(self, content: str, section: Optional[str] = None):
        """
        장기 기억에 추가 (Layer 2)
        
        Args:
            content: 추가할 내용
            section: 섹션 이름 (예: "## User Preferences")
        """
        memory_md = self.workspace_dir / "MEMORY.md"
        existing = memory_md.read_text(encoding='utf-8') if memory_md.exists() else ""
        
        if section:
            # 특정 섹션에 추가
            pattern = rf"(## {re.escape(section)}.*?)(?=\n## |\Z)"
            match = re.search(pattern, existing, re.DOTALL)
            if match:
                section_content = match.group(1)
                new_section = section_content.rstrip() + f"\n{content}\n"
                existing = existing[:match.start()] + new_section + existing[match.end():]
            else:
                existing += f"\n## {section}\n{content}\n"
        else:
            existing += f"\n{content}\n"
        
        memory_md.write_text(existing, encoding='utf-8')
        
        # 재인덱싱
        await self.indexer.index_file(str(memory_md), MemoryLayer.LONG_TERM)
        
        self._logger.info("Added long-term memory", section=section)
    
    async def search(
        self,
        query: str,
        max_results: int = 6,
        min_score: float = 0.35,
        layer: Optional[MemoryLayer] = None
    ) -> List[MemorySearchResult]:
        """
        메모리 검색 (하이브리드: Vector 70% + BM25 30%)
        
        Args:
            query: 검색 쿼리
            max_results: 최대 결과 수
            min_score: 최소 점수 임계값
            layer: 특정 계층만 검색 (None이면 전체)
        
        Returns:
            검색 결과 리스트
        """
        results = await self.indexer.hybrid_search(query, max_results * 2, min_score)
        
        # 계층 필터링
        if layer:
            results = [r for r in results if r.layer == layer]
        
        return results[:max_results]
    
    async def get_memory_content(
        self,
        path: str,
        start_line: int = 1,
        lines: int = 15
    ) -> Optional[str]:
        """
        특정 메모리 파일의 내용 읽기
        
        Args:
            path: 파일 경로
            start_line: 시작 라인 (1-based)
            lines: 읽을 라인 수
        
        Returns:
            파일 내용 또는 None
        """
        file_path = Path(path)
        if not file_path.exists():
            return None
        
        with open(file_path, 'r', encoding='utf-8') as f:
            all_lines = f.readlines()
        
        end_line = min(start_line + lines - 1, len(all_lines))
        return "".join(all_lines[start_line - 1:end_line])
    
    async def get_recent_daily_logs(self, days: int = 2) -> List[Dict[str, Any]]:
        """
        최근 N일간의 일별 로그 조회
        
        Args:
            days: 조회할 일수
        
        Returns:
            일별 로그 리스트
        """
        logs = []
        today = date.today()
        
        for i in range(days):
            log_date = today - timedelta(days=i)
            log_path = self.memory_dir / f"{log_date.isoformat()}.md"
            
            if log_path.exists():
                content = log_path.read_text(encoding='utf-8')
                logs.append({
                    "date": log_date.isoformat(),
                    "path": str(log_path),
                    "content": content
                })
        
        return logs
    
    def get_project_context(self) -> str:
        """에이전트 초기화용 프로젝트 컨텍스트"""
        return self.bootstrap.get_project_context()
    
    def close(self):
        """리소스 정리"""
        self.indexer.close()


# ============================================================================
# Memory Tools (에이전트가 사용하는 도구)
# ============================================================================

@dataclass
class MemorySearchTool:
    """
    memory_search 도구 - 메모리에서 관련 정보 검색
    
    Clawdbot 패턴:
        사전 질문에 답하기 전 반드시 메모리 검색 권장
        - 이전 작업/결정
        - 날짜/일정
        - 사람/연락처
        - 선호도
        - 할 일
    """
    
    name: str = "memory_search"
    description: str = """Mandatory recall step: semantically search MEMORY.md + memory/*.md 
before answering questions about prior work, decisions, dates, people, preferences, or todos"""
    
    memory: Optional[PersistentMemory] = None
    
    def get_schema(self) -> Dict[str, Any]:
        """OpenAI Function Calling 스키마"""
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": {
                    "type": "object",
                    "properties": {
                        "query": {
                            "type": "string",
                            "description": "Search query for finding relevant memories"
                        },
                        "maxResults": {
                            "type": "integer",
                            "description": "Maximum number of results to return",
                            "default": 6
                        },
                        "minScore": {
                            "type": "number",
                            "description": "Minimum relevance score threshold (0.0-1.0)",
                            "default": 0.35
                        }
                    },
                    "required": ["query"]
                }
            }
        }
    
    async def execute(
        self,
        query: str,
        maxResults: int = 6,
        minScore: float = 0.35
    ) -> Dict[str, Any]:
        """도구 실행"""
        if not self.memory:
            return {"error": "Memory system not initialized"}
        
        results = await self.memory.search(query, maxResults, minScore)
        
        return {
            "results": [r.to_dict() for r in results],
            "provider": "hybrid",
            "model": self.memory.config.embedding_model
        }


@dataclass
class MemoryGetTool:
    """
    memory_get 도구 - 특정 메모리 파일 내용 읽기
    
    memory_search로 위치를 찾은 후 상세 내용 조회용
    """
    
    name: str = "memory_get"
    description: str = "Read specific lines from a memory file after memory_search"
    
    memory: Optional[PersistentMemory] = None
    
    def get_schema(self) -> Dict[str, Any]:
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": {
                    "type": "object",
                    "properties": {
                        "path": {
                            "type": "string",
                            "description": "Path to the memory file"
                        },
                        "from": {
                            "type": "integer",
                            "description": "Starting line number (1-based)",
                            "default": 1
                        },
                        "lines": {
                            "type": "integer",
                            "description": "Number of lines to read",
                            "default": 15
                        }
                    },
                    "required": ["path"]
                }
            }
        }
    
    async def execute(
        self,
        path: str,
        **kwargs
    ) -> Dict[str, Any]:
        """도구 실행"""
        if not self.memory:
            return {"error": "Memory system not initialized"}
        
        from_line = kwargs.get("from", 1)
        lines = kwargs.get("lines", 15)
        
        content = await self.memory.get_memory_content(path, from_line, lines)
        
        if content is None:
            return {"error": f"File not found: {path}"}
        
        return {
            "path": path,
            "text": content
        }


@dataclass
class MemoryWriteTool:
    """
    memory_write 도구 - 메모리에 기록
    
    일반적인 write/edit 도구로도 가능하지만
    편의를 위한 전용 도구
    """
    
    name: str = "memory_write"
    description: str = "Write to memory files (daily log or long-term memory)"
    
    memory: Optional[PersistentMemory] = None
    
    def get_schema(self) -> Dict[str, Any]:
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": {
                    "type": "object",
                    "properties": {
                        "content": {
                            "type": "string",
                            "description": "Content to write"
                        },
                        "layer": {
                            "type": "string",
                            "enum": ["daily", "long_term"],
                            "description": "Memory layer: 'daily' for daily log, 'long_term' for MEMORY.md",
                            "default": "daily"
                        },
                        "section": {
                            "type": "string",
                            "description": "Section name for long-term memory (optional)"
                        }
                    },
                    "required": ["content"]
                }
            }
        }
    
    async def execute(
        self,
        content: str,
        layer: str = "daily",
        section: Optional[str] = None
    ) -> Dict[str, Any]:
        """도구 실행"""
        if not self.memory:
            return {"error": "Memory system not initialized"}
        
        if layer == "daily":
            await self.memory.add_daily_note(content)
            return {"success": True, "layer": "daily_log"}
        else:
            await self.memory.add_long_term_memory(content, section)
            return {"success": True, "layer": "long_term", "section": section}


# 추가 import
from datetime import timedelta
