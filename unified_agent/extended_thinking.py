#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Extended Thinking 시스템 - Reasoning 추적

================================================================================
📋 역할: LLM의 사고 과정 추적 및 기록
📅 버전: 3.4.0 (2026년 2월)
📦 영감: OpenAI o1/o3 Extended Thinking, Anthropic Claude Thinking
================================================================================

🎯 주요 기능:
    - Chain-of-Thought 추적
    - 단계별 추론 기록
    - 사고 과정 시각화
    - 추론 품질 평가
    - 디버깅 지원

📌 사용 시나리오:
    - 복잡한 문제 해결 추적
    - 추론 과정 검토
    - 에러 디버깅
    - 모델 행동 분석

📌 사용 예시:
    >>> from unified_agent import ThinkingTracker, ThinkingStep
    >>>
    >>> tracker = ThinkingTracker()
    >>>
    >>> # 사고 과정 기록
    >>> with tracker.thinking_context("complex_problem"):
    ...     tracker.add_step("분석", "문제를 분석합니다")
    ...     tracker.add_step("추론", "해결책을 도출합니다")
    ...     tracker.add_step("검증", "답을 검증합니다")
    >>>
    >>> # 사고 과정 출력
    >>> tracker.visualize()
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import time
import uuid
from abc import ABC, abstractmethod
from contextlib import asynccontextmanager, contextmanager
from dataclasses import dataclass, field
from datetime import datetime, timezone, timedelta
from enum import Enum
from typing import (
    Any,
    Callable,
    Dict,
    Generator,
    Generic,
    List,
    Optional,
    Set,
    Tuple,
    TypeVar,
)

from .utils import StructuredLogger


__all__ = [
    # 설정
    "ThinkingConfig",
    "ThinkingMode",
    # 단계
    "ThinkingStep",
    "ThinkingStepType",
    "ThinkingChain",
    # 트래커
    "ThinkingTracker",
    "ThinkingContext",
    # 분석
    "ThinkingAnalyzer",
    "ThinkingMetrics",
    # 저장소
    "ThinkingStore",
]


# ============================================================================
# 설정 및 타입
# ============================================================================

class ThinkingMode(str, Enum):
    """사고 모드"""
    SEQUENTIAL = "sequential"     # 순차적 사고
    BRANCHING = "branching"       # 분기 사고
    ITERATIVE = "iterative"       # 반복 사고
    PARALLEL = "parallel"         # 병렬 사고


class ThinkingStepType(str, Enum):
    """사고 단계 유형"""
    OBSERVATION = "observation"   # 관찰
    ANALYSIS = "analysis"         # 분석
    HYPOTHESIS = "hypothesis"     # 가설
    REASONING = "reasoning"       # 추론
    VERIFICATION = "verification" # 검증
    CONCLUSION = "conclusion"     # 결론
    QUESTION = "question"         # 질문
    REFLECTION = "reflection"     # 반성
    CORRECTION = "correction"     # 수정


@dataclass
class ThinkingConfig:
    """
    Extended Thinking 설정
    
    Args:
        max_steps: 최대 사고 단계 수
        max_depth: 최대 사고 깊이 (분기용)
        timeout_seconds: 타임아웃
        enable_caching: 캐싱 활성화
        record_timestamps: 타임스탬프 기록
        record_token_usage: 토큰 사용량 기록
    """
    max_steps: int = 100
    max_depth: int = 10
    timeout_seconds: float = 300.0
    enable_caching: bool = True
    record_timestamps: bool = True
    record_token_usage: bool = True


# ============================================================================
# Thinking Step - 사고 단계
# ============================================================================

@dataclass
class ThinkingStep:
    """
    사고 단계
    
    개별 사고 과정의 단위
    """
    id: str
    step_type: ThinkingStepType
    title: str
    content: str
    parent_id: Optional[str] = None
    depth: int = 0
    
    # 메타데이터
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    duration_ms: float = 0.0
    tokens_used: int = 0
    confidence: float = 1.0
    
    # 연결
    children: List[str] = field(default_factory=list)
    references: List[str] = field(default_factory=list)
    
    # 추가 정보
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "step_type": self.step_type.value,
            "title": self.title,
            "content": self.content,
            "parent_id": self.parent_id,
            "depth": self.depth,
            "created_at": self.created_at.isoformat(),
            "duration_ms": self.duration_ms,
            "tokens_used": self.tokens_used,
            "confidence": self.confidence,
            "children": self.children,
        }
    
    def __repr__(self) -> str:
        return f"ThinkingStep({self.step_type.value}: {self.title[:30]}...)"


# ============================================================================
# Thinking Chain - 사고 체인
# ============================================================================

@dataclass
class ThinkingChain:
    """
    사고 체인 (연결된 사고 과정)
    
    관련된 사고 단계들의 집합
    """
    id: str
    name: str
    mode: ThinkingMode
    steps: List[ThinkingStep] = field(default_factory=list)
    
    # 상태
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    status: str = "pending"  # pending, running, completed, failed
    
    # 결과
    conclusion: Optional[str] = None
    final_answer: Optional[str] = None
    
    # 메트릭
    total_steps: int = 0
    total_tokens: int = 0
    total_duration_ms: float = 0.0
    
    # 메타데이터
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def add_step(self, step: ThinkingStep) -> None:
        """단계 추가"""
        self.steps.append(step)
        self.total_steps += 1
        self.total_tokens += step.tokens_used
        self.total_duration_ms += step.duration_ms
    
    def get_step(self, step_id: str) -> Optional[ThinkingStep]:
        """ID로 단계 조회"""
        for step in self.steps:
            if step.id == step_id:
                return step
        return None
    
    def get_root_steps(self) -> List[ThinkingStep]:
        """루트 단계 (부모 없음) 조회"""
        return [s for s in self.steps if s.parent_id is None]
    
    def get_children(self, parent_id: str) -> List[ThinkingStep]:
        """자식 단계 조회"""
        return [s for s in self.steps if s.parent_id == parent_id]
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "name": self.name,
            "mode": self.mode.value,
            "status": self.status,
            "total_steps": self.total_steps,
            "total_tokens": self.total_tokens,
            "total_duration_ms": self.total_duration_ms,
            "steps": [s.to_dict() for s in self.steps],
            "conclusion": self.conclusion,
        }
    
    def visualize(self, indent: str = "  ") -> str:
        """사고 과정 시각화"""
        lines = [f"🧠 Thinking Chain: {self.name}"]
        lines.append(f"   Mode: {self.mode.value} | Steps: {self.total_steps}")
        lines.append("")
        
        def render_step(step: ThinkingStep, level: int = 0):
            prefix = indent * level
            icon = self._get_step_icon(step.step_type)
            lines.append(f"{prefix}{icon} [{step.step_type.value}] {step.title}")
            if step.content:
                content_preview = step.content[:100] + "..." if len(step.content) > 100 else step.content
                lines.append(f"{prefix}   └─ {content_preview}")
            
            for child in self.get_children(step.id):
                render_step(child, level + 1)
        
        for root in self.get_root_steps():
            render_step(root)
        
        if self.conclusion:
            lines.append("")
            lines.append(f"📝 Conclusion: {self.conclusion}")
        
        return "\n".join(lines)
    
    @staticmethod
    def _get_step_icon(step_type: ThinkingStepType) -> str:
        icons = {
            ThinkingStepType.OBSERVATION: "👁️",
            ThinkingStepType.ANALYSIS: "🔍",
            ThinkingStepType.HYPOTHESIS: "💡",
            ThinkingStepType.REASONING: "🤔",
            ThinkingStepType.VERIFICATION: "✅",
            ThinkingStepType.CONCLUSION: "📌",
            ThinkingStepType.QUESTION: "❓",
            ThinkingStepType.REFLECTION: "🪞",
            ThinkingStepType.CORRECTION: "✏️",
        }
        return icons.get(step_type, "•")


# ============================================================================
# Thinking Context - 사고 컨텍스트
# ============================================================================

class ThinkingContext:
    """
    사고 컨텍스트 (Context Manager)
    
    사고 과정을 추적하는 컨텍스트
    """
    
    def __init__(
        self,
        chain: ThinkingChain,
        tracker: "ThinkingTracker",
        parent_step: Optional[ThinkingStep] = None,
    ):
        self._chain = chain
        self._tracker = tracker
        self._parent_step = parent_step
        self._current_step: Optional[ThinkingStep] = None
        self._start_time: Optional[float] = None
    
    def add_step(
        self,
        title: str,
        content: str = "",
        step_type: ThinkingStepType = ThinkingStepType.REASONING,
        confidence: float = 1.0,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> ThinkingStep:
        """
        사고 단계 추가
        
        Args:
            title: 단계 제목
            content: 단계 내용
            step_type: 단계 유형
            confidence: 신뢰도 (0.0 ~ 1.0)
            metadata: 추가 메타데이터
            
        Returns:
            생성된 ThinkingStep
        """
        parent_id = self._parent_step.id if self._parent_step else None
        depth = (self._parent_step.depth + 1) if self._parent_step else 0
        
        step = ThinkingStep(
            id=str(uuid.uuid4())[:8],
            step_type=step_type,
            title=title,
            content=content,
            parent_id=parent_id,
            depth=depth,
            confidence=confidence,
            metadata=metadata or {},
        )
        
        self._chain.add_step(step)
        
        if self._parent_step:
            self._parent_step.children.append(step.id)
        
        self._current_step = step
        return step
    
    def observe(self, title: str, content: str = "", **kwargs) -> ThinkingStep:
        """관찰 단계 추가"""
        return self.add_step(title, content, ThinkingStepType.OBSERVATION, **kwargs)
    
    def analyze(self, title: str, content: str = "", **kwargs) -> ThinkingStep:
        """분석 단계 추가"""
        return self.add_step(title, content, ThinkingStepType.ANALYSIS, **kwargs)
    
    def hypothesize(self, title: str, content: str = "", **kwargs) -> ThinkingStep:
        """가설 단계 추가"""
        return self.add_step(title, content, ThinkingStepType.HYPOTHESIS, **kwargs)
    
    def reason(self, title: str, content: str = "", **kwargs) -> ThinkingStep:
        """추론 단계 추가"""
        return self.add_step(title, content, ThinkingStepType.REASONING, **kwargs)
    
    def verify(self, title: str, content: str = "", **kwargs) -> ThinkingStep:
        """검증 단계 추가"""
        return self.add_step(title, content, ThinkingStepType.VERIFICATION, **kwargs)
    
    def conclude(self, title: str, content: str = "", **kwargs) -> ThinkingStep:
        """결론 단계 추가"""
        return self.add_step(title, content, ThinkingStepType.CONCLUSION, **kwargs)
    
    def question(self, title: str, content: str = "", **kwargs) -> ThinkingStep:
        """질문 단계 추가"""
        return self.add_step(title, content, ThinkingStepType.QUESTION, **kwargs)
    
    def reflect(self, title: str, content: str = "", **kwargs) -> ThinkingStep:
        """반성 단계 추가"""
        return self.add_step(title, content, ThinkingStepType.REFLECTION, **kwargs)
    
    def correct(self, title: str, content: str = "", **kwargs) -> ThinkingStep:
        """수정 단계 추가"""
        return self.add_step(title, content, ThinkingStepType.CORRECTION, **kwargs)
    
    @contextmanager
    def branch(self, title: str) -> Generator["ThinkingContext", None, None]:
        """
        분기 사고 컨텍스트 생성
        
        Args:
            title: 분기 제목
            
        Yields:
            새로운 ThinkingContext
        """
        branch_step = self.add_step(
            title=title,
            content="Branch point",
            step_type=ThinkingStepType.HYPOTHESIS,
        )
        
        branch_context = ThinkingContext(
            chain=self._chain,
            tracker=self._tracker,
            parent_step=branch_step,
        )
        
        try:
            yield branch_context
        finally:
            pass  # 분기 종료
    
    def set_conclusion(self, conclusion: str, answer: Optional[str] = None):
        """결론 설정"""
        self._chain.conclusion = conclusion
        self._chain.final_answer = answer
    
    @property
    def chain(self) -> ThinkingChain:
        return self._chain


# ============================================================================
# Thinking Tracker - 사고 추적기
# ============================================================================

class ThinkingTracker:
    """
    사고 추적기
    
    LLM의 사고 과정을 추적하고 기록
    
    사용 예시:
        >>> tracker = ThinkingTracker()
        >>>
        >>> with tracker.thinking_context("problem_solving") as ctx:
        ...     ctx.observe("문제 파악", "사용자의 질문을 분석합니다")
        ...     ctx.analyze("핵심 요소 분석", "주요 키워드: AI, 학습")
        ...     
        ...     with ctx.branch("접근법 1: 직접 해결"):
        ...         ctx.reason("단계별 해결", "1. 먼저...")
        ...         ctx.verify("검증", "결과가 맞는지 확인")
        ...     
        ...     ctx.conclude("최종 답변", "AI 학습은...")
        >>>
        >>> # 시각화
        >>> print(tracker.get_chain("problem_solving").visualize())
    """
    
    def __init__(self, config: Optional[ThinkingConfig] = None):
        self._config = config or ThinkingConfig()
        self._chains: Dict[str, ThinkingChain] = {}
        self._current_chain: Optional[ThinkingChain] = None
        self._logger = StructuredLogger("thinking_tracker")
    
    @contextmanager
    def thinking_context(
        self,
        name: str,
        mode: ThinkingMode = ThinkingMode.SEQUENTIAL,
    ) -> Generator[ThinkingContext, None, None]:
        """
        사고 컨텍스트 시작
        
        Args:
            name: 체인 이름
            mode: 사고 모드
            
        Yields:
            ThinkingContext
        """
        chain = ThinkingChain(
            id=str(uuid.uuid4())[:8],
            name=name,
            mode=mode,
            started_at=datetime.now(timezone.utc),
            status="running",
        )
        
        self._chains[name] = chain
        self._current_chain = chain
        
        context = ThinkingContext(chain, self)
        
        self._logger.info("Thinking started", name=name, mode=mode.value)
        
        try:
            yield context
            chain.status = "completed"
            chain.completed_at = datetime.now(timezone.utc)
            
        except Exception as e:
            chain.status = "failed"
            chain.metadata["error"] = str(e)
            raise
            
        finally:
            self._current_chain = None
            self._logger.info(
                "Thinking ended",
                name=name,
                steps=chain.total_steps,
                duration_ms=chain.total_duration_ms
            )
    
    @asynccontextmanager
    async def async_thinking_context(
        self,
        name: str,
        mode: ThinkingMode = ThinkingMode.SEQUENTIAL,
    ):
        """비동기 사고 컨텍스트"""
        with self.thinking_context(name, mode) as ctx:
            yield ctx
    
    def get_chain(self, name: str) -> Optional[ThinkingChain]:
        """이름으로 체인 조회"""
        return self._chains.get(name)
    
    def get_all_chains(self) -> List[ThinkingChain]:
        """모든 체인 조회"""
        return list(self._chains.values())
    
    def clear(self):
        """모든 체인 삭제"""
        self._chains.clear()
        self._current_chain = None


# ============================================================================
# Thinking Analyzer - 사고 분석기
# ============================================================================

@dataclass
class ThinkingMetrics:
    """사고 메트릭"""
    total_steps: int = 0
    total_tokens: int = 0
    total_duration_ms: float = 0.0
    
    # 단계 유형별 카운트
    step_type_counts: Dict[str, int] = field(default_factory=dict)
    
    # 깊이 분석
    max_depth: int = 0
    avg_depth: float = 0.0
    
    # 신뢰도 분석
    avg_confidence: float = 0.0
    min_confidence: float = 1.0
    
    # 분기 분석
    branch_count: int = 0
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "total_steps": self.total_steps,
            "total_tokens": self.total_tokens,
            "total_duration_ms": self.total_duration_ms,
            "step_type_counts": self.step_type_counts,
            "max_depth": self.max_depth,
            "avg_depth": round(self.avg_depth, 2),
            "avg_confidence": round(self.avg_confidence, 2),
            "min_confidence": round(self.min_confidence, 2),
            "branch_count": self.branch_count,
        }


class ThinkingAnalyzer:
    """
    사고 분석기
    
    사고 과정의 품질과 패턴 분석
    
    사용 예시:
        >>> analyzer = ThinkingAnalyzer()
        >>> metrics = analyzer.analyze_chain(chain)
        >>> quality = analyzer.assess_quality(chain)
    """
    
    def __init__(self):
        self._logger = StructuredLogger("thinking_analyzer")
    
    def analyze_chain(self, chain: ThinkingChain) -> ThinkingMetrics:
        """
        체인 분석
        
        Args:
            chain: 분석할 체인
            
        Returns:
            메트릭
        """
        metrics = ThinkingMetrics(
            total_steps=chain.total_steps,
            total_tokens=chain.total_tokens,
            total_duration_ms=chain.total_duration_ms,
        )
        
        if not chain.steps:
            return metrics
        
        # 단계 유형별 카운트
        type_counts: Dict[str, int] = {}
        depths: List[int] = []
        confidences: List[float] = []
        
        for step in chain.steps:
            type_key = step.step_type.value
            type_counts[type_key] = type_counts.get(type_key, 0) + 1
            depths.append(step.depth)
            confidences.append(step.confidence)
            
            # 분기 카운트
            if len(step.children) > 1:
                metrics.branch_count += 1
        
        metrics.step_type_counts = type_counts
        metrics.max_depth = max(depths) if depths else 0
        metrics.avg_depth = sum(depths) / len(depths) if depths else 0.0
        metrics.avg_confidence = sum(confidences) / len(confidences) if confidences else 0.0
        metrics.min_confidence = min(confidences) if confidences else 1.0
        
        return metrics
    
    def assess_quality(self, chain: ThinkingChain) -> Dict[str, Any]:
        """
        사고 품질 평가
        
        Args:
            chain: 평가할 체인
            
        Returns:
            품질 평가 결과
        """
        metrics = self.analyze_chain(chain)
        
        quality_score = 0.0
        issues = []
        suggestions = []
        
        # 1. 단계 다양성 체크
        type_count = len(metrics.step_type_counts)
        if type_count >= 4:
            quality_score += 0.2
        elif type_count >= 2:
            quality_score += 0.1
        else:
            issues.append("사고 단계 유형이 단조로움")
            suggestions.append("다양한 사고 단계 활용 (분석, 추론, 검증 등)")
        
        # 2. 검증 단계 체크
        verification_count = metrics.step_type_counts.get("verification", 0)
        if verification_count >= 1:
            quality_score += 0.2
        else:
            issues.append("검증 단계 누락")
            suggestions.append("추론 결과에 대한 검증 단계 추가")
        
        # 3. 결론 체크
        conclusion_count = metrics.step_type_counts.get("conclusion", 0)
        if conclusion_count >= 1:
            quality_score += 0.2
        else:
            issues.append("명확한 결론 누락")
            suggestions.append("최종 결론 단계 추가")
        
        # 4. 신뢰도 체크
        if metrics.avg_confidence >= 0.7:
            quality_score += 0.2
        elif metrics.avg_confidence >= 0.5:
            quality_score += 0.1
            suggestions.append("일부 단계의 신뢰도가 낮음")
        else:
            issues.append("전체적으로 낮은 신뢰도")
        
        # 5. 깊이 체크 (너무 얕거나 깊지 않은지)
        if 2 <= metrics.max_depth <= 5:
            quality_score += 0.2
        elif metrics.max_depth > 5:
            suggestions.append("사고 깊이가 깊음 - 단순화 고려")
        else:
            suggestions.append("좀 더 깊이 있는 분석 필요")
        
        return {
            "quality_score": round(quality_score, 2),
            "grade": self._score_to_grade(quality_score),
            "metrics": metrics.to_dict(),
            "issues": issues,
            "suggestions": suggestions,
        }
    
    @staticmethod
    def _score_to_grade(score: float) -> str:
        """점수를 등급으로 변환"""
        if score >= 0.9:
            return "A+"
        elif score >= 0.8:
            return "A"
        elif score >= 0.7:
            return "B+"
        elif score >= 0.6:
            return "B"
        elif score >= 0.5:
            return "C+"
        elif score >= 0.4:
            return "C"
        else:
            return "D"
    
    def compare_chains(
        self,
        chain1: ThinkingChain,
        chain2: ThinkingChain,
    ) -> Dict[str, Any]:
        """
        두 체인 비교
        
        Args:
            chain1: 첫 번째 체인
            chain2: 두 번째 체인
            
        Returns:
            비교 결과
        """
        metrics1 = self.analyze_chain(chain1)
        metrics2 = self.analyze_chain(chain2)
        
        return {
            "chain1": {
                "name": chain1.name,
                "metrics": metrics1.to_dict(),
            },
            "chain2": {
                "name": chain2.name,
                "metrics": metrics2.to_dict(),
            },
            "comparison": {
                "steps_diff": metrics2.total_steps - metrics1.total_steps,
                "tokens_diff": metrics2.total_tokens - metrics1.total_tokens,
                "duration_diff_ms": metrics2.total_duration_ms - metrics1.total_duration_ms,
                "confidence_diff": metrics2.avg_confidence - metrics1.avg_confidence,
            }
        }


# ============================================================================
# Thinking Store - 사고 저장소
# ============================================================================

class ThinkingStore:
    """
    사고 과정 저장소
    
    사고 체인을 저장하고 조회
    """
    
    def __init__(self, storage_path: Optional[str] = None):
        from pathlib import Path
        self._storage_path = Path(storage_path or "~/.thinking_store").expanduser()
        self._storage_path.mkdir(parents=True, exist_ok=True)
        self._logger = StructuredLogger("thinking_store")
    
    async def save(self, chain: ThinkingChain) -> None:
        """체인 저장"""
        import pickle
        file_path = self._storage_path / f"{chain.id}.thinking"
        
        with open(file_path, 'wb') as f:
            pickle.dump(chain, f)
        
        self._logger.debug("Chain saved", id=chain.id)
    
    async def load(self, chain_id: str) -> Optional[ThinkingChain]:
        """체인 로드"""
        import pickle
        file_path = self._storage_path / f"{chain_id}.thinking"
        
        if not file_path.exists():
            return None
        
        with open(file_path, 'rb') as f:
            return pickle.load(f)
    
    async def list_chains(self, limit: int = 100) -> List[str]:
        """체인 ID 목록"""
        files = list(self._storage_path.glob("*.thinking"))[:limit]
        return [f.stem for f in files]
    
    async def delete(self, chain_id: str) -> bool:
        """체인 삭제"""
        file_path = self._storage_path / f"{chain_id}.thinking"
        
        if file_path.exists():
            file_path.unlink()
            return True
        return False
