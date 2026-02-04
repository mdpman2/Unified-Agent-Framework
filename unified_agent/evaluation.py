#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Unified Agent Framework - 평가 모듈 (Evaluation Module)

================================================================================
📁 파일 위치: unified_agent/evaluation.py
📋 역할: 에이전트 품질 측정, PDCA 평가, LLM-as-Judge, Check-Act Iteration
📅 최종 업데이트: 2026년 2월 4일
📦 버전: v3.5.0
✅ 테스트: test_new_modules.py, test_v35_scenarios.py
🔗 참조: bkit-claude-code PDCA 방법론
================================================================================

🎯 주요 구성 요소:
    1. PDCAEvaluator - PDCA(Plan-Do-Check-Act) 사이클 평가
    2. LLMJudge - LLM 기반 품질 평가 (LLM-as-Judge)
    3. CheckActIterator - Check-Act 반복 최적화 (Evaluator-Optimizer 패턴)
    4. AgentBenchmark - 에이전트 벤치마크 테스트
    5. QualityMetrics - 품질 메트릭 수집 및 분석
    6. GapAnalyzer - 계획 vs 실제 갭 분석

🔧 2026년 2월 기능 (bkit 영감):
    - PDCA 방법론 기반 체계적 평가
    - Evaluator-Optimizer 패턴 (자동 개선 루프)
    - Check-Act Iteration (90% 임계값, 최대 5회)
    - LLM-as-Judge 다차원 평가
    - 갭 분석 및 자동 수정 제안

📌 사용 예시:
    >>> from unified_agent.evaluation import (
    ...     PDCAEvaluator, LLMJudge, CheckActIterator,
    ...     EvaluationConfig, QualityMetrics
    ... )
    >>>
    >>> # PDCA 평가
    >>> evaluator = PDCAEvaluator()
    >>> result = await evaluator.evaluate_cycle(
    ...     plan=plan_doc,
    ...     implementation=code,
    ...     expected_outcome=spec
    ... )
    >>>
    >>> # Check-Act 자동 개선 루프
    >>> iterator = CheckActIterator(
    ...     evaluator=LLMJudge(),
    ...     optimizer=optimizer_agent,
    ...     threshold=0.9,      # 90% 목표
    ...     max_iterations=5    # 최대 5회
    ... )
    >>> final_result = await iterator.iterate(initial_output)

⚠️ 주의사항:
    - LLM-as-Judge는 평가 모델이 대상 모델보다 강력해야 합니다.
    - Check-Act Iteration은 토큰 사용량이 증가할 수 있습니다.
    - 90% 임계값은 도메인에 따라 조정이 필요합니다.

🔗 관련 문서:
    - bkit PDCA: https://github.com/popup-studio-ai/bkit-claude-code
    - Anthropic Evaluator-Optimizer: https://www.anthropic.com/research
"""

import asyncio
import json
import logging
import statistics
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import (
    Any, Callable, Dict, Generic, List, Optional, 
    Protocol, Tuple, TypeVar, Union
)

__all__ = [
    # Enums
    "PDCAPhase",
    "EvaluationDimension",
    "QualityLevel",
    "GapSeverity",
    # Config
    "EvaluationConfig",
    "JudgeConfig",
    "IterationConfig",
    # Results
    "EvaluationResult",
    "JudgeVerdict",
    "GapAnalysisResult",
    "IterationResult",
    "BenchmarkResult",
    "QualityReport",
    # Core Components
    "PDCAEvaluator",
    "LLMJudge",
    "CheckActIterator",
    "GapAnalyzer",
    "AgentBenchmark",
    "QualityMetrics",
    # Protocols
    "Evaluator",
    "Optimizer",
]


T = TypeVar("T")


# ============================================================================
# Enums
# ============================================================================

class PDCAPhase(str, Enum):
    """PDCA 사이클 단계"""
    PLAN = "plan"         # 계획: 목표 및 프로세스 정의
    DO = "do"             # 실행: 계획 실행
    CHECK = "check"       # 점검: 결과 평가
    ACT = "act"           # 조치: 개선 적용


class EvaluationDimension(str, Enum):
    """평가 차원"""
    TASK_COMPLETION = "task_completion"       # 작업 완료도
    FACTUAL_ACCURACY = "factual_accuracy"     # 사실 정확도
    RESPONSE_QUALITY = "response_quality"     # 응답 품질
    CODE_QUALITY = "code_quality"             # 코드 품질
    TOOL_USAGE = "tool_usage"                 # 도구 사용 효율
    INSTRUCTION_FOLLOWING = "instruction_following"  # 지시 준수
    CREATIVITY = "creativity"                 # 창의성
    EFFICIENCY = "efficiency"                 # 효율성
    SAFETY = "safety"                         # 안전성


class QualityLevel(str, Enum):
    """품질 수준"""
    EXCELLENT = "excellent"    # 90%+
    GOOD = "good"              # 70-89%
    ACCEPTABLE = "acceptable"  # 50-69%
    POOR = "poor"              # 30-49%
    FAIL = "fail"              # 0-29%


class GapSeverity(str, Enum):
    """갭 심각도"""
    CRITICAL = "critical"      # 치명적: 즉시 수정 필요
    MAJOR = "major"            # 주요: 빠른 수정 필요
    MINOR = "minor"            # 경미: 개선 권장
    TRIVIAL = "trivial"        # 사소: 선택적 개선


# ============================================================================
# Protocols
# ============================================================================

class Evaluator(Protocol):
    """평가자 프로토콜"""
    async def evaluate(self, output: str, context: Dict[str, Any]) -> "EvaluationResult":
        ...


class Optimizer(Protocol):
    """최적화자 프로토콜"""
    async def optimize(self, output: str, feedback: str) -> str:
        ...


# ============================================================================
# Data Classes - Config
# ============================================================================

@dataclass
class EvaluationConfig:
    """
    평가 설정
    
    Attributes:
        dimensions: 평가 차원 목록
        weights: 차원별 가중치
        threshold: 통과 임계값 (0.0 ~ 1.0)
        model: 평가에 사용할 LLM 모델
        detailed_feedback: 상세 피드백 생성 여부
    """
    dimensions: List[EvaluationDimension] = field(default_factory=lambda: [
        EvaluationDimension.TASK_COMPLETION,
        EvaluationDimension.FACTUAL_ACCURACY,
        EvaluationDimension.RESPONSE_QUALITY,
    ])
    weights: Dict[EvaluationDimension, float] = field(default_factory=dict)
    threshold: float = 0.7  # 70% 기본 임계값
    model: str = "gpt-5.2"
    detailed_feedback: bool = True
    
    def __post_init__(self):
        # 가중치 기본값 설정
        if not self.weights:
            self.weights = {dim: 1.0 / len(self.dimensions) for dim in self.dimensions}


@dataclass
class JudgeConfig:
    """
    LLM Judge 설정
    
    Attributes:
        judge_model: 판단에 사용할 모델
        reference_model: 참조 모델 (비교 평가용)
        rubric: 평가 루브릭 (점수 기준)
        temperature: 일관성을 위해 낮은 값 권장
        multi_judge: 다중 판사 앙상블 사용
    """
    judge_model: str = "gpt-5.2"
    reference_model: Optional[str] = None
    rubric: Optional[str] = None
    temperature: float = 0.1
    multi_judge: bool = False
    num_judges: int = 3


@dataclass
class IterationConfig:
    """
    Check-Act Iteration 설정 (bkit 스타일)
    
    Attributes:
        threshold: 목표 품질 임계값 (기본: 90%)
        max_iterations: 최대 반복 횟수 (기본: 5회)
        improvement_threshold: 개선 없음 판단 임계값
        early_stop: 목표 달성 시 조기 종료
        verbose: 상세 로그 출력
    """
    threshold: float = 0.9           # 90% 목표 (bkit 기준)
    max_iterations: int = 5          # 최대 5회 (bkit 기준)
    improvement_threshold: float = 0.01  # 1% 미만 개선 시 종료
    early_stop: bool = True
    verbose: bool = True


# ============================================================================
# Data Classes - Results
# ============================================================================

@dataclass
class EvaluationResult:
    """
    평가 결과
    
    Attributes:
        overall_score: 종합 점수 (0.0 ~ 1.0)
        dimension_scores: 차원별 점수
        quality_level: 품질 수준
        passed: 임계값 통과 여부
        feedback: 피드백 메시지
        suggestions: 개선 제안
        metadata: 추가 메타데이터
    """
    overall_score: float
    dimension_scores: Dict[EvaluationDimension, float] = field(default_factory=dict)
    quality_level: QualityLevel = QualityLevel.ACCEPTABLE
    passed: bool = False
    feedback: str = ""
    suggestions: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    
    def __post_init__(self):
        # 품질 수준 자동 계산
        if self.overall_score >= 0.9:
            self.quality_level = QualityLevel.EXCELLENT
        elif self.overall_score >= 0.7:
            self.quality_level = QualityLevel.GOOD
        elif self.overall_score >= 0.5:
            self.quality_level = QualityLevel.ACCEPTABLE
        elif self.overall_score >= 0.3:
            self.quality_level = QualityLevel.POOR
        else:
            self.quality_level = QualityLevel.FAIL


@dataclass
class JudgeVerdict:
    """
    LLM Judge 판결
    
    Attributes:
        score: 점수 (0-10 또는 0-100)
        reasoning: 판단 근거
        strengths: 강점
        weaknesses: 약점
        comparison: 비교 결과 (A/B 테스트 시)
    """
    score: float
    reasoning: str = ""
    strengths: List[str] = field(default_factory=list)
    weaknesses: List[str] = field(default_factory=list)
    comparison: Optional[str] = None  # "A > B", "A < B", "A = B"
    confidence: float = 1.0


@dataclass
class GapAnalysisResult:
    """
    갭 분석 결과 (bkit 스타일)
    
    Attributes:
        match_rate: 일치율 (0.0 ~ 1.0)
        gaps: 발견된 갭 목록
        missing_features: 누락된 기능
        extra_features: 추가된 기능 (범위 초과)
        severity_summary: 심각도별 요약
    """
    match_rate: float
    gaps: List[Dict[str, Any]] = field(default_factory=list)
    missing_features: List[str] = field(default_factory=list)
    extra_features: List[str] = field(default_factory=list)
    severity_summary: Dict[GapSeverity, int] = field(default_factory=dict)
    recommendations: List[str] = field(default_factory=list)


@dataclass
class IterationResult:
    """
    Check-Act Iteration 결과
    
    Attributes:
        final_output: 최종 출력
        iterations: 반복 횟수
        score_history: 점수 이력
        converged: 수렴 여부
        improvement: 총 개선율
    """
    final_output: str
    iterations: int
    score_history: List[float] = field(default_factory=list)
    feedback_history: List[str] = field(default_factory=list)
    converged: bool = False
    improvement: float = 0.0
    final_score: float = 0.0


@dataclass
class BenchmarkResult:
    """
    벤치마크 결과
    
    Attributes:
        agent_name: 에이전트 이름
        test_suite: 테스트 스위트 이름
        total_tests: 총 테스트 수
        passed: 통과 수
        failed: 실패 수
        scores: 테스트별 점수
        avg_score: 평균 점수
        percentile: 백분위
    """
    agent_name: str
    test_suite: str
    total_tests: int
    passed: int = 0
    failed: int = 0
    scores: List[float] = field(default_factory=list)
    avg_score: float = 0.0
    percentile: float = 0.0
    details: List[Dict[str, Any]] = field(default_factory=list)
    
    def __post_init__(self):
        if self.scores:
            self.avg_score = statistics.mean(self.scores)


@dataclass
class QualityReport:
    """
    품질 리포트
    
    Attributes:
        summary: 요약
        overall_score: 종합 점수
        dimension_breakdown: 차원별 분석
        trends: 트렌드 (시계열)
        recommendations: 권장 사항
    """
    summary: str
    overall_score: float
    dimension_breakdown: Dict[str, float] = field(default_factory=dict)
    trends: List[Dict[str, Any]] = field(default_factory=list)
    recommendations: List[str] = field(default_factory=list)
    generated_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))


# ============================================================================
# PDCA Evaluator
# ============================================================================

class PDCAEvaluator:
    """
    PDCA 사이클 평가자
    
    ================================================================================
    📋 역할: PDCA (Plan-Do-Check-Act) 방법론 기반 체계적 평가
    📅 최종 업데이트: 2026년 2월 (bkit 영감)
    ================================================================================
    
    🎯 PDCA 사이클:
        Plan  → 목표 및 프로세스 정의 (설계 문서)
        Do    → 계획 실행 (구현)
        Check → 결과 평가 (갭 분석)
        Act   → 개선 적용 (수정 반복)
    
    📌 사용 예시:
        >>> evaluator = PDCAEvaluator()
        >>> 
        >>> # 전체 사이클 평가
        >>> result = await evaluator.evaluate_cycle(
        ...     plan="설계 문서 내용",
        ...     implementation="구현된 코드",
        ...     expected_outcome="예상 결과"
        ... )
        >>> print(f"일치율: {result.match_rate:.1%}")
        >>>
        >>> # 개별 단계 평가
        >>> plan_result = await evaluator.evaluate_plan(plan_doc)
        >>> do_result = await evaluator.evaluate_do(implementation, plan_doc)
    """
    
    def __init__(
        self,
        config: Optional[EvaluationConfig] = None,
        llm_client: Optional[Any] = None
    ):
        self.config = config or EvaluationConfig()
        self.llm_client = llm_client
        self.logger = logging.getLogger(__name__)
        self.gap_analyzer = GapAnalyzer()
    
    async def evaluate_cycle(
        self,
        plan: str,
        implementation: str,
        expected_outcome: Optional[str] = None,
        actual_outcome: Optional[str] = None
    ) -> GapAnalysisResult:
        """
        전체 PDCA 사이클 평가
        
        Args:
            plan: 계획/설계 문서
            implementation: 실제 구현
            expected_outcome: 예상 결과
            actual_outcome: 실제 결과
        
        Returns:
            GapAnalysisResult: 갭 분석 결과
        """
        self.logger.info("PDCA 사이클 평가 시작")
        
        # 갭 분석
        gap_result = await self.gap_analyzer.analyze(
            plan=plan,
            implementation=implementation,
            expected=expected_outcome,
            actual=actual_outcome
        )
        
        self.logger.info(f"PDCA 평가 완료: 일치율 {gap_result.match_rate:.1%}")
        return gap_result
    
    async def evaluate_plan(self, plan: str) -> EvaluationResult:
        """
        Plan 단계 평가 - 계획의 완전성 및 명확성 평가
        
        Args:
            plan: 계획/설계 문서
        
        Returns:
            EvaluationResult: 평가 결과
        """
        criteria = {
            "completeness": "모든 필수 요소가 포함되어 있는가?",
            "clarity": "요구사항이 명확하게 정의되어 있는가?",
            "feasibility": "실현 가능한 계획인가?",
            "measurability": "성공 기준이 측정 가능한가?",
        }
        
        scores = {}
        feedback_parts = []
        
        for criterion, question in criteria.items():
            # 간단한 휴리스틱 평가 (LLM 없이)
            score = self._evaluate_criterion(plan, criterion)
            scores[criterion] = score
            
            if score < 0.7:
                feedback_parts.append(f"- {criterion}: 개선 필요 ({question})")
        
        overall_score = sum(scores.values()) / len(scores)
        
        return EvaluationResult(
            overall_score=overall_score,
            passed=overall_score >= self.config.threshold,
            feedback="\n".join(feedback_parts) if feedback_parts else "계획이 양호합니다.",
            metadata={"phase": PDCAPhase.PLAN.value, "criteria_scores": scores}
        )
    
    async def evaluate_do(
        self,
        implementation: str,
        plan: str
    ) -> EvaluationResult:
        """
        Do 단계 평가 - 구현의 계획 준수도 평가
        
        Args:
            implementation: 구현 결과
            plan: 원본 계획
        
        Returns:
            EvaluationResult: 평가 결과
        """
        # 갭 분석
        gap_result = await self.gap_analyzer.analyze(
            plan=plan,
            implementation=implementation
        )
        
        suggestions = []
        if gap_result.missing_features:
            suggestions.append(f"누락된 기능: {', '.join(gap_result.missing_features[:5])}")
        if gap_result.gaps:
            suggestions.extend([g.get("recommendation", "") for g in gap_result.gaps[:3]])
        
        return EvaluationResult(
            overall_score=gap_result.match_rate,
            passed=gap_result.match_rate >= self.config.threshold,
            feedback=f"계획 대비 구현 일치율: {gap_result.match_rate:.1%}",
            suggestions=suggestions,
            metadata={"phase": PDCAPhase.DO.value, "gap_analysis": gap_result}
        )
    
    async def evaluate_check(
        self,
        actual_outcome: str,
        expected_outcome: str
    ) -> EvaluationResult:
        """
        Check 단계 평가 - 결과의 기대치 충족도 평가
        
        Args:
            actual_outcome: 실제 결과
            expected_outcome: 예상 결과
        
        Returns:
            EvaluationResult: 평가 결과
        """
        # 결과 비교
        gap_result = await self.gap_analyzer.analyze(
            plan=expected_outcome,
            implementation=actual_outcome
        )
        
        return EvaluationResult(
            overall_score=gap_result.match_rate,
            passed=gap_result.match_rate >= self.config.threshold,
            feedback=f"기대 결과 충족률: {gap_result.match_rate:.1%}",
            suggestions=gap_result.recommendations,
            metadata={"phase": PDCAPhase.CHECK.value}
        )
    
    def _evaluate_criterion(self, text: str, criterion: str) -> float:
        """기준별 휴리스틱 평가"""
        text_lower = text.lower()
        
        if criterion == "completeness":
            # 필수 섹션 존재 여부
            sections = ["목표", "요구사항", "범위", "일정", "goal", "requirement", "scope"]
            found = sum(1 for s in sections if s in text_lower)
            return min(1.0, found / 3)
        
        elif criterion == "clarity":
            # 문장 길이와 구조
            sentences = text.split(".")
            avg_length = sum(len(s.split()) for s in sentences) / max(len(sentences), 1)
            # 적절한 문장 길이 (10-25단어)
            if 10 <= avg_length <= 25:
                return 0.9
            elif 5 <= avg_length <= 35:
                return 0.7
            else:
                return 0.5
        
        elif criterion == "feasibility":
            # 구체적 수치/기한 언급
            import re
            numbers = len(re.findall(r'\d+', text))
            dates = len(re.findall(r'\d{4}[-/]\d{2}[-/]\d{2}', text))
            return min(1.0, (numbers + dates * 2) / 10)
        
        elif criterion == "measurability":
            # 측정 가능한 키워드
            keywords = ["kpi", "metric", "측정", "지표", "percent", "%", "rate", "score"]
            found = sum(1 for k in keywords if k in text_lower)
            return min(1.0, found / 3)
        
        return 0.5


# ============================================================================
# LLM Judge
# ============================================================================

class LLMJudge:
    """
    LLM 기반 품질 평가자 (LLM-as-Judge)
    
    ================================================================================
    📋 역할: LLM을 활용한 다차원 품질 평가
    📅 최종 업데이트: 2026년 2월
    ================================================================================
    
    🎯 평가 방식:
        - Single Rating: 단일 점수 평가
        - Pairwise Comparison: A/B 비교 평가
        - Multi-Dimension: 다차원 평가
        - Rubric-Based: 루브릭 기반 평가
    
    📌 사용 예시:
        >>> judge = LLMJudge(JudgeConfig(judge_model="gpt-5.2"))
        >>> 
        >>> # 단일 평가
        >>> verdict = await judge.evaluate(
        ...     output="AI 생성 응답",
        ...     criteria="정확성, 유용성, 명확성",
        ...     context={"task": "코드 리뷰"}
        ... )
        >>> print(f"점수: {verdict.score}/10")
        >>> 
        >>> # A/B 비교
        >>> comparison = await judge.compare(
        ...     output_a="응답 A",
        ...     output_b="응답 B",
        ...     criteria="어느 응답이 더 정확한가?"
        ... )
    """
    
    DEFAULT_RUBRIC = """
    평가 기준 (1-10점):
    - 10점: 완벽함, 개선 여지 없음
    - 8-9점: 우수함, 경미한 개선 가능
    - 6-7점: 양호함, 일부 개선 필요
    - 4-5점: 보통, 상당한 개선 필요
    - 2-3점: 미흡, 많은 개선 필요
    - 1점: 매우 부족, 전면 재작업 필요
    """
    
    def __init__(
        self,
        config: Optional[JudgeConfig] = None,
        llm_client: Optional[Any] = None
    ):
        self.config = config or JudgeConfig()
        self.llm_client = llm_client
        self.logger = logging.getLogger(__name__)
    
    async def evaluate(
        self,
        output: str,
        criteria: str,
        context: Optional[Dict[str, Any]] = None,
        reference: Optional[str] = None
    ) -> JudgeVerdict:
        """
        단일 출력 평가
        
        Args:
            output: 평가 대상 출력
            criteria: 평가 기준
            context: 추가 컨텍스트
            reference: 참조 답변 (있는 경우)
        
        Returns:
            JudgeVerdict: 판결
        """
        # LLM 호출이 없을 경우 휴리스틱 평가
        if not self.llm_client:
            return await self._heuristic_evaluate(output, criteria, context)
        
        prompt = self._build_evaluation_prompt(output, criteria, context, reference)
        
        try:
            # LLM API 호출 (구현 필요)
            response = await self._call_llm(prompt)
            return self._parse_verdict(response)
        except Exception as e:
            self.logger.error(f"LLM 평가 실패: {e}")
            return await self._heuristic_evaluate(output, criteria, context)
    
    async def compare(
        self,
        output_a: str,
        output_b: str,
        criteria: str,
        context: Optional[Dict[str, Any]] = None
    ) -> JudgeVerdict:
        """
        A/B 비교 평가
        
        Args:
            output_a: 출력 A
            output_b: 출력 B
            criteria: 비교 기준
            context: 추가 컨텍스트
        
        Returns:
            JudgeVerdict: 비교 판결
        """
        # 휴리스틱 비교
        score_a = await self._heuristic_evaluate(output_a, criteria, context)
        score_b = await self._heuristic_evaluate(output_b, criteria, context)
        
        if score_a.score > score_b.score + 0.5:
            comparison = "A > B"
        elif score_b.score > score_a.score + 0.5:
            comparison = "A < B"
        else:
            comparison = "A = B"
        
        return JudgeVerdict(
            score=(score_a.score + score_b.score) / 2,
            reasoning=f"A: {score_a.score:.1f}, B: {score_b.score:.1f}",
            comparison=comparison
        )
    
    async def multi_dimension_evaluate(
        self,
        output: str,
        dimensions: List[EvaluationDimension],
        context: Optional[Dict[str, Any]] = None
    ) -> Dict[EvaluationDimension, JudgeVerdict]:
        """
        다차원 평가
        
        Args:
            output: 평가 대상
            dimensions: 평가 차원 목록
            context: 컨텍스트
        
        Returns:
            Dict: 차원별 판결
        """
        results = {}
        
        for dim in dimensions:
            criteria = self._dimension_to_criteria(dim)
            verdict = await self.evaluate(output, criteria, context)
            results[dim] = verdict
        
        return results
    
    async def _heuristic_evaluate(
        self,
        output: str,
        criteria: str,
        context: Optional[Dict[str, Any]] = None
    ) -> JudgeVerdict:
        """휴리스틱 기반 평가 (LLM 없이)"""
        score = 5.0  # 기본 점수
        strengths = []
        weaknesses = []
        
        # 길이 평가
        word_count = len(output.split())
        if 50 <= word_count <= 500:
            score += 1
            strengths.append("적절한 응답 길이")
        elif word_count < 20:
            score -= 1
            weaknesses.append("응답이 너무 짧음")
        elif word_count > 1000:
            score -= 0.5
            weaknesses.append("응답이 다소 김")
        
        # 구조 평가
        if output.count("\n") >= 2:
            score += 0.5
            strengths.append("적절한 구조화")
        
        # 키워드 매칭 (기준에서)
        criteria_words = set(criteria.lower().split())
        output_words = set(output.lower().split())
        overlap = len(criteria_words & output_words) / max(len(criteria_words), 1)
        if overlap > 0.3:
            score += 1
            strengths.append("기준 관련 내용 포함")
        
        # 점수 정규화 (1-10)
        score = max(1.0, min(10.0, score))
        
        return JudgeVerdict(
            score=score,
            reasoning=f"휴리스틱 평가: {len(strengths)}개 강점, {len(weaknesses)}개 약점",
            strengths=strengths,
            weaknesses=weaknesses
        )
    
    def _dimension_to_criteria(self, dim: EvaluationDimension) -> str:
        """평가 차원을 기준 문자열로 변환"""
        mapping = {
            EvaluationDimension.TASK_COMPLETION: "작업이 완전히 수행되었는가?",
            EvaluationDimension.FACTUAL_ACCURACY: "정보가 정확하고 사실에 기반하는가?",
            EvaluationDimension.RESPONSE_QUALITY: "응답의 품질이 높고 유용한가?",
            EvaluationDimension.CODE_QUALITY: "코드가 깔끔하고 모범 사례를 따르는가?",
            EvaluationDimension.TOOL_USAGE: "도구를 효율적으로 사용했는가?",
            EvaluationDimension.INSTRUCTION_FOLLOWING: "지시사항을 정확히 따랐는가?",
            EvaluationDimension.CREATIVITY: "창의적이고 혁신적인 접근인가?",
            EvaluationDimension.EFFICIENCY: "효율적으로 작업을 수행했는가?",
            EvaluationDimension.SAFETY: "안전하고 책임감 있는 응답인가?",
        }
        return mapping.get(dim, "전반적인 품질을 평가하시오.")
    
    def _build_evaluation_prompt(
        self,
        output: str,
        criteria: str,
        context: Optional[Dict[str, Any]],
        reference: Optional[str]
    ) -> str:
        """평가 프롬프트 생성"""
        rubric = self.config.rubric or self.DEFAULT_RUBRIC
        
        prompt = f"""다음 출력물을 평가해주세요.

## 평가 기준
{criteria}

## 루브릭
{rubric}

## 평가 대상
{output}

"""
        if reference:
            prompt += f"""
## 참조 답변
{reference}
"""
        
        if context:
            prompt += f"""
## 컨텍스트
{json.dumps(context, ensure_ascii=False, indent=2)}
"""
        
        prompt += """
## 출력 형식
JSON으로 응답해주세요:
{
    "score": <1-10 점수>,
    "reasoning": "<판단 근거>",
    "strengths": ["<강점1>", "<강점2>"],
    "weaknesses": ["<약점1>", "<약점2>"]
}
"""
        return prompt
    
    async def _call_llm(self, prompt: str) -> str:
        """LLM API 호출 (구현 필요)"""
        # 실제 구현에서는 OpenAI/Azure API 호출
        raise NotImplementedError("LLM client not configured")
    
    def _parse_verdict(self, response: str) -> JudgeVerdict:
        """응답 파싱"""
        try:
            data = json.loads(response)
            return JudgeVerdict(
                score=float(data.get("score", 5)),
                reasoning=data.get("reasoning", ""),
                strengths=data.get("strengths", []),
                weaknesses=data.get("weaknesses", [])
            )
        except (json.JSONDecodeError, KeyError):
            return JudgeVerdict(score=5.0, reasoning="파싱 실패")


# ============================================================================
# Check-Act Iterator (Evaluator-Optimizer Pattern)
# ============================================================================

class CheckActIterator:
    """
    Check-Act 반복 최적화 (Evaluator-Optimizer 패턴)
    
    ================================================================================
    📋 역할: 자동 개선 루프를 통한 출력 품질 향상
    📅 최종 업데이트: 2026년 2월 (bkit Evaluator-Optimizer 영감)
    ================================================================================
    
    🎯 작동 방식 (bkit 스타일):
        1. 초기 출력 생성
        2. Check: 품질 평가 (목표: 90%)
        3. 미달 시 Act: 피드백 기반 개선
        4. 2-3 반복 (최대 5회)
        5. 목표 달성 또는 최대 반복 도달 시 종료
    
    📌 사용 예시:
        >>> iterator = CheckActIterator(
        ...     evaluator=LLMJudge(),
        ...     optimizer=optimizer_function,
        ...     threshold=0.9,      # 90% 목표
        ...     max_iterations=5    # 최대 5회
        ... )
        >>> 
        >>> result = await iterator.iterate(
        ...     initial_output="초기 생성 결과",
        ...     criteria="코드 품질, 문서화, 테스트 커버리지"
        ... )
        >>> 
        >>> print(f"반복 횟수: {result.iterations}")
        >>> print(f"최종 점수: {result.final_score:.1%}")
        >>> print(f"개선율: {result.improvement:.1%}")
    """
    
    def __init__(
        self,
        evaluator: Optional[LLMJudge] = None,
        optimizer: Optional[Callable] = None,
        config: Optional[IterationConfig] = None
    ):
        self.evaluator = evaluator or LLMJudge()
        self.optimizer = optimizer
        self.config = config or IterationConfig()
        self.logger = logging.getLogger(__name__)
    
    async def iterate(
        self,
        initial_output: str,
        criteria: str = "전반적인 품질",
        context: Optional[Dict[str, Any]] = None
    ) -> IterationResult:
        """
        Check-Act 반복 실행
        
        Args:
            initial_output: 초기 출력
            criteria: 평가 기준
            context: 추가 컨텍스트
        
        Returns:
            IterationResult: 반복 결과
        """
        current_output = initial_output
        score_history = []
        feedback_history = []
        
        if self.config.verbose:
            self.logger.info(f"Check-Act Iteration 시작 (목표: {self.config.threshold:.0%})")
        
        for iteration in range(1, self.config.max_iterations + 1):
            # Check: 평가
            verdict = await self.evaluator.evaluate(
                current_output, 
                criteria, 
                context
            )
            current_score = verdict.score / 10.0  # 0-1 정규화
            score_history.append(current_score)
            
            if self.config.verbose:
                self.logger.info(f"  [{iteration}] 점수: {current_score:.1%}")
            
            # 목표 달성 확인
            if current_score >= self.config.threshold:
                if self.config.verbose:
                    self.logger.info(f"✅ 목표 달성! ({current_score:.1%} >= {self.config.threshold:.0%})")
                
                return IterationResult(
                    final_output=current_output,
                    iterations=iteration,
                    score_history=score_history,
                    feedback_history=feedback_history,
                    converged=True,
                    improvement=current_score - score_history[0],
                    final_score=current_score
                )
            
            # 개선 여지 확인
            if len(score_history) >= 2:
                improvement = current_score - score_history[-2]
                if improvement < self.config.improvement_threshold:
                    if self.config.verbose:
                        self.logger.info(f"⚠️ 개선 정체 ({improvement:.1%} < {self.config.improvement_threshold:.1%})")
                    
                    if self.config.early_stop:
                        break
            
            # Act: 개선
            feedback = self._generate_feedback(verdict)
            feedback_history.append(feedback)
            
            if self.optimizer:
                current_output = await self._optimize(current_output, feedback)
            else:
                # 기본 최적화 (피드백 추가 요청)
                current_output = await self._default_optimize(current_output, feedback)
        
        # 최대 반복 도달
        if self.config.verbose:
            self.logger.info(f"🔄 최대 반복 도달 (최종: {score_history[-1]:.1%})")
        
        return IterationResult(
            final_output=current_output,
            iterations=len(score_history),
            score_history=score_history,
            feedback_history=feedback_history,
            converged=score_history[-1] >= self.config.threshold,
            improvement=score_history[-1] - score_history[0],
            final_score=score_history[-1]
        )
    
    def _generate_feedback(self, verdict: JudgeVerdict) -> str:
        """평가 결과에서 피드백 생성"""
        feedback_parts = []
        
        if verdict.weaknesses:
            feedback_parts.append("개선 필요 사항:")
            for w in verdict.weaknesses:
                feedback_parts.append(f"  - {w}")
        
        if verdict.reasoning:
            feedback_parts.append(f"\n평가 의견: {verdict.reasoning}")
        
        return "\n".join(feedback_parts)
    
    async def _optimize(self, output: str, feedback: str) -> str:
        """최적화 함수 호출"""
        if asyncio.iscoroutinefunction(self.optimizer):
            return await self.optimizer(output, feedback)
        else:
            return self.optimizer(output, feedback)
    
    async def _default_optimize(self, output: str, feedback: str) -> str:
        """기본 최적화 (변경 없음)"""
        # 실제 구현에서는 LLM을 통한 수정 요청
        return output


# ============================================================================
# Gap Analyzer
# ============================================================================

class GapAnalyzer:
    """
    갭 분석기 (bkit 스타일)
    
    ================================================================================
    📋 역할: 계획과 구현 간의 갭 분석
    📅 최종 업데이트: 2026년 2월
    ================================================================================
    
    📌 사용 예시:
        >>> analyzer = GapAnalyzer()
        >>> result = await analyzer.analyze(
        ...     plan="설계 문서",
        ...     implementation="구현 코드"
        ... )
        >>> print(f"일치율: {result.match_rate:.1%}")
    """
    
    def __init__(self, llm_client: Optional[Any] = None):
        self.llm_client = llm_client
        self.logger = logging.getLogger(__name__)
    
    async def analyze(
        self,
        plan: str,
        implementation: str,
        expected: Optional[str] = None,
        actual: Optional[str] = None
    ) -> GapAnalysisResult:
        """
        갭 분석 수행
        
        Args:
            plan: 계획/설계 문서
            implementation: 실제 구현
            expected: 예상 결과
            actual: 실제 결과
        
        Returns:
            GapAnalysisResult: 분석 결과
        """
        # 휴리스틱 분석
        gaps = []
        missing = []
        extra = []
        
        # 키워드 추출 및 비교
        plan_keywords = self._extract_keywords(plan)
        impl_keywords = self._extract_keywords(implementation)
        
        # 누락된 항목
        missing_keywords = plan_keywords - impl_keywords
        for kw in missing_keywords:
            missing.append(kw)
            gaps.append({
                "type": "missing",
                "item": kw,
                "severity": GapSeverity.MAJOR.value,
                "recommendation": f"'{kw}' 구현 필요"
            })
        
        # 추가된 항목 (범위 초과)
        extra_keywords = impl_keywords - plan_keywords
        for kw in list(extra_keywords)[:5]:  # 상위 5개만
            extra.append(kw)
        
        # 일치율 계산
        if plan_keywords:
            match_rate = len(plan_keywords & impl_keywords) / len(plan_keywords)
        else:
            match_rate = 1.0 if not impl_keywords else 0.5
        
        # 심각도 집계
        severity_summary = {s: 0 for s in GapSeverity}
        for gap in gaps:
            sev = GapSeverity(gap["severity"])
            severity_summary[sev] += 1
        
        # 권장 사항
        recommendations = []
        if match_rate < 0.5:
            recommendations.append("⚠️ 계획 대비 구현 일치율이 낮습니다. 전체 검토 필요.")
        elif match_rate < 0.7:
            recommendations.append("일부 기능이 누락되었습니다. 우선순위별 구현 필요.")
        elif match_rate < 0.9:
            recommendations.append("대부분 구현되었습니다. 세부 사항 점검 권장.")
        else:
            recommendations.append("✅ 계획 대비 구현이 잘 완료되었습니다.")
        
        return GapAnalysisResult(
            match_rate=match_rate,
            gaps=gaps,
            missing_features=missing,
            extra_features=extra,
            severity_summary=severity_summary,
            recommendations=recommendations
        )
    
    def _extract_keywords(self, text: str) -> set:
        """텍스트에서 키워드 추출"""
        import re
        
        # 소문자 변환 및 특수문자 제거
        text = re.sub(r'[^\w\s가-힣]', ' ', text.lower())
        
        # 불용어
        stopwords = {
            'the', 'a', 'an', 'is', 'are', 'was', 'were', 'be', 'been',
            'and', 'or', 'but', 'if', 'then', 'else', 'when', 'where',
            '이', '그', '저', '것', '수', '등', '및', '또는', '그리고',
            'def', 'class', 'return', 'import', 'from', 'async', 'await'
        }
        
        words = text.split()
        keywords = {w for w in words if len(w) > 2 and w not in stopwords}
        
        return keywords


# ============================================================================
# Agent Benchmark
# ============================================================================

class AgentBenchmark:
    """
    에이전트 벤치마크 테스트
    
    ================================================================================
    📋 역할: 에이전트 성능 측정 및 비교
    📅 최종 업데이트: 2026년 2월
    ================================================================================
    
    📌 사용 예시:
        >>> benchmark = AgentBenchmark()
        >>> 
        >>> # 테스트 케이스 추가
        >>> benchmark.add_test_case(
        ...     name="simple_qa",
        ...     input="서울의 수도는?",
        ...     expected="서울은 대한민국의 수도입니다.",
        ...     criteria="factual_accuracy"
        ... )
        >>> 
        >>> # 벤치마크 실행
        >>> result = await benchmark.run(agent)
        >>> print(f"평균 점수: {result.avg_score:.1%}")
    """
    
    def __init__(
        self,
        suite_name: str = "default",
        evaluator: Optional[LLMJudge] = None
    ):
        self.suite_name = suite_name
        self.evaluator = evaluator or LLMJudge()
        self.test_cases: List[Dict[str, Any]] = []
        self.logger = logging.getLogger(__name__)
    
    def add_test_case(
        self,
        name: str,
        input_text: str,
        expected: Optional[str] = None,
        criteria: str = "quality",
        weight: float = 1.0
    ):
        """
        테스트 케이스 추가
        
        Args:
            name: 테스트 이름
            input_text: 입력
            expected: 예상 출력
            criteria: 평가 기준
            weight: 가중치
        """
        self.test_cases.append({
            "name": name,
            "input": input_text,
            "expected": expected,
            "criteria": criteria,
            "weight": weight
        })
    
    async def run(
        self,
        agent_fn: Callable,
        agent_name: str = "test_agent"
    ) -> BenchmarkResult:
        """
        벤치마크 실행
        
        Args:
            agent_fn: 에이전트 함수 (async)
            agent_name: 에이전트 이름
        
        Returns:
            BenchmarkResult: 벤치마크 결과
        """
        scores = []
        details = []
        passed = 0
        failed = 0
        
        for tc in self.test_cases:
            try:
                # 에이전트 호출
                if asyncio.iscoroutinefunction(agent_fn):
                    output = await agent_fn(tc["input"])
                else:
                    output = agent_fn(tc["input"])
                
                # 평가
                verdict = await self.evaluator.evaluate(
                    output=output,
                    criteria=tc["criteria"],
                    reference=tc.get("expected")
                )
                
                score = verdict.score / 10.0
                scores.append(score * tc["weight"])
                
                if score >= 0.7:
                    passed += 1
                else:
                    failed += 1
                
                details.append({
                    "name": tc["name"],
                    "score": score,
                    "passed": score >= 0.7,
                    "verdict": verdict
                })
                
            except Exception as e:
                self.logger.error(f"테스트 실패 ({tc['name']}): {e}")
                failed += 1
                scores.append(0.0)
                details.append({
                    "name": tc["name"],
                    "score": 0.0,
                    "passed": False,
                    "error": str(e)
                })
        
        return BenchmarkResult(
            agent_name=agent_name,
            test_suite=self.suite_name,
            total_tests=len(self.test_cases),
            passed=passed,
            failed=failed,
            scores=scores,
            details=details
        )


# ============================================================================
# Quality Metrics
# ============================================================================

class QualityMetrics:
    """
    품질 메트릭 수집기
    
    ================================================================================
    📋 역할: 에이전트 품질 메트릭 수집 및 분석
    📅 최종 업데이트: 2026년 2월
    ================================================================================
    
    📌 사용 예시:
        >>> metrics = QualityMetrics()
        >>> 
        >>> # 메트릭 기록
        >>> metrics.record("task_completion", 0.95)
        >>> metrics.record("response_time_ms", 250)
        >>> 
        >>> # 리포트 생성
        >>> report = metrics.generate_report()
    """
    
    def __init__(self):
        self.metrics: Dict[str, List[float]] = {}
        self.timestamps: Dict[str, List[datetime]] = {}
        self.logger = logging.getLogger(__name__)
    
    def record(self, name: str, value: float):
        """
        메트릭 기록
        
        Args:
            name: 메트릭 이름
            value: 값
        """
        if name not in self.metrics:
            self.metrics[name] = []
            self.timestamps[name] = []
        
        self.metrics[name].append(value)
        self.timestamps[name].append(datetime.now(timezone.utc))
    
    def get_stats(self, name: str) -> Dict[str, float]:
        """
        메트릭 통계 조회
        
        Args:
            name: 메트릭 이름
        
        Returns:
            Dict: 통계 (mean, std, min, max, count)
        """
        if name not in self.metrics or not self.metrics[name]:
            return {}
        
        values = self.metrics[name]
        return {
            "mean": statistics.mean(values),
            "std": statistics.stdev(values) if len(values) > 1 else 0,
            "min": min(values),
            "max": max(values),
            "count": len(values),
            "latest": values[-1]
        }
    
    def generate_report(self) -> QualityReport:
        """
        품질 리포트 생성
        
        Returns:
            QualityReport: 리포트
        """
        dimension_breakdown = {}
        overall_scores = []
        
        for name, values in self.metrics.items():
            if values:
                avg = statistics.mean(values)
                dimension_breakdown[name] = avg
                
                # 0-1 범위 메트릭만 전체 점수에 포함
                if 0 <= avg <= 1:
                    overall_scores.append(avg)
        
        overall_score = statistics.mean(overall_scores) if overall_scores else 0.0
        
        # 트렌드 계산
        trends = []
        for name, values in self.metrics.items():
            if len(values) >= 2:
                recent = statistics.mean(values[-5:])
                older = statistics.mean(values[:-5]) if len(values) > 5 else values[0]
                trend = "improving" if recent > older else "declining" if recent < older else "stable"
                trends.append({
                    "metric": name,
                    "trend": trend,
                    "recent_avg": recent,
                    "change": recent - older
                })
        
        # 권장 사항
        recommendations = []
        for name, stats in [(n, self.get_stats(n)) for n in self.metrics]:
            if stats and stats.get("mean", 1) < 0.7:
                recommendations.append(f"'{name}' 개선 필요 (현재: {stats['mean']:.1%})")
        
        # 요약
        if overall_score >= 0.9:
            summary = "전반적으로 우수한 품질입니다."
        elif overall_score >= 0.7:
            summary = "양호한 품질이나 일부 개선 필요합니다."
        elif overall_score >= 0.5:
            summary = "개선이 필요한 영역이 있습니다."
        else:
            summary = "전반적인 품질 개선이 필요합니다."
        
        return QualityReport(
            summary=summary,
            overall_score=overall_score,
            dimension_breakdown=dimension_breakdown,
            trends=trends,
            recommendations=recommendations
        )
    
    def reset(self):
        """메트릭 초기화"""
        self.metrics.clear()
        self.timestamps.clear()
