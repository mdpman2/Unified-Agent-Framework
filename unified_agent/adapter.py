#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Adapter 시스템 - Span → 학습 데이터 변환

================================================================================
📋 역할: 트레이스 스팬을 RL/SFT 학습용 데이터로 변환
📅 버전: 3.3.0 (2026년 2월)
📦 영감: Microsoft Agent Lightning의 TraceAdapter 시스템
================================================================================

🎯 주요 기능:
    - Span → Triplet (prompt, response, reward) 변환
    - LLM 호출-리워드 매칭
    - 학습 데이터셋 생성
    - OpenAI 메시지 형식 변환

📌 사용 예시:
    >>> from unified_agent import TracerTraceToTriplet, Triplet
    >>>
    >>> adapter = TracerTraceToTriplet()
    >>> triplets = adapter.adapt(spans)
    >>>
    >>> for t in triplets:
    ...     print(f"Prompt: {t.prompt}")
    ...     print(f"Response: {t.response}")
    ...     print(f"Reward: {t.reward}")
"""

from __future__ import annotations

import json
import re
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import (
    Any,
    Dict,
    Generic,
    List,
    Optional,
    Sequence,
    Set,
    Tuple,
    TypeVar,
)

from .tracer import Span, SpanKind, SpanStatus
from .reward import is_reward_span, get_reward_value, find_reward_spans
from .utils import StructuredLogger


# ============================================================================
# 타입 변수
# ============================================================================

T_from = TypeVar("T_from")
T_to = TypeVar("T_to")


# ============================================================================
# Triplet 모델
# ============================================================================

@dataclass
class Triplet:
    """
    (Prompt, Response, Reward) 트리플렛
    
    강화학습 및 SFT에서 사용하는 기본 학습 단위.
    """
    prompt: Dict[str, Any]       # 프롬프트 정보
    response: Dict[str, Any]     # 응답 정보
    reward: Optional[float]      # 리워드 (없을 수 있음)
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "prompt": self.prompt,
            "response": self.response,
            "reward": self.reward,
            "metadata": self.metadata,
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Triplet":
        return cls(
            prompt=data.get("prompt", {}),
            response=data.get("response", {}),
            reward=data.get("reward"),
            metadata=data.get("metadata", {}),
        )


@dataclass
class Transition:
    """
    상태 전이 (RL용)
    
    State → Action → Reward → Next State
    """
    state: Dict[str, Any]
    action: Dict[str, Any]
    reward: float
    next_state: Optional[Dict[str, Any]] = None
    done: bool = False
    info: Dict[str, Any] = field(default_factory=dict)


@dataclass
class Trajectory:
    """
    전체 궤적 (트랜지션 시퀀스)
    """
    rollout_id: str
    attempt_id: Optional[str] = None
    transitions: List[Triplet] = field(default_factory=list)
    total_reward: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def add_triplet(self, triplet: Triplet) -> None:
        """트리플렛 추가"""
        self.transitions.append(triplet)
        if triplet.reward is not None:
            self.total_reward += triplet.reward
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "rollout_id": self.rollout_id,
            "attempt_id": self.attempt_id,
            "transitions": [t.to_dict() for t in self.transitions],
            "total_reward": self.total_reward,
            "metadata": self.metadata,
        }


# ============================================================================
# Reward 매칭 정책
# ============================================================================

class RewardMatchPolicy(str, Enum):
    """리워드-LLM 호출 매칭 정책"""
    FIRST_OCCURRENCE = "first"      # 첫 번째 발견된 리워드
    LAST_OCCURRENCE = "last"        # 마지막 리워드
    CLOSEST_BEFORE = "closest"      # 가장 가까운 이전 리워드
    FINAL_ONLY = "final"           # 마지막 LLM에만 리워드


# ============================================================================
# Adapter 베이스
# ============================================================================

class Adapter(ABC, Generic[T_from, T_to]):
    """어댑터 추상 베이스"""
    
    @abstractmethod
    def adapt(self, source: T_from) -> T_to:
        """소스 데이터를 타겟 형식으로 변환"""
        pass


class TraceAdapter(Adapter[Sequence[Span], T_to], Generic[T_to]):
    """
    트레이스 어댑터 베이스
    
    Span 시퀀스를 특정 형식으로 변환.
    Agent Lightning의 TraceAdapter 참고.
    """
    pass


# ============================================================================
# OpenAI Messages Adapter
# ============================================================================

@dataclass
class OpenAIMessage:
    """OpenAI 메시지 형식"""
    role: str
    content: str
    name: Optional[str] = None
    function_call: Optional[Dict[str, Any]] = None
    tool_calls: Optional[List[Dict[str, Any]]] = None
    
    def to_dict(self) -> Dict[str, Any]:
        result = {"role": self.role, "content": self.content}
        if self.name:
            result["name"] = self.name
        if self.function_call:
            result["function_call"] = self.function_call
        if self.tool_calls:
            result["tool_calls"] = self.tool_calls
        return result


class OpenAIMessagesAdapter(TraceAdapter[List[OpenAIMessage]]):
    """
    Span → OpenAI 메시지 형식 변환
    """
    
    def __init__(
        self,
        llm_call_pattern: str = r"openai\.chat\.completion|llm_call",
        include_system: bool = True,
    ):
        """
        Args:
            llm_call_pattern: LLM 호출 스팬 이름 패턴
            include_system: 시스템 메시지 포함 여부
        """
        self._llm_pattern = re.compile(llm_call_pattern, re.IGNORECASE)
        self._include_system = include_system
        self._logger = StructuredLogger("adapter.openai_messages")
    
    def adapt(self, source: Sequence[Span]) -> List[OpenAIMessage]:
        """Span → OpenAI Messages"""
        messages: List[OpenAIMessage] = []
        
        for span in source:
            if not self._llm_pattern.search(span.name):
                continue
            
            attrs = span.attributes or {}
            
            # 요청 메시지 추출
            request_messages = attrs.get("llm.request.messages", [])
            if isinstance(request_messages, str):
                try:
                    request_messages = json.loads(request_messages)
                except json.JSONDecodeError:
                    request_messages = []
            
            for msg in request_messages:
                if not self._include_system and msg.get("role") == "system":
                    continue
                
                messages.append(OpenAIMessage(
                    role=msg.get("role", "user"),
                    content=msg.get("content", ""),
                    name=msg.get("name"),
                    function_call=msg.get("function_call"),
                    tool_calls=msg.get("tool_calls"),
                ))
            
            # 응답 추출
            response_content = attrs.get("llm.response.content", "")
            if response_content:
                messages.append(OpenAIMessage(
                    role="assistant",
                    content=response_content,
                ))
        
        return messages


# ============================================================================
# Trace Tree (스팬 계층 구조)
# ============================================================================

class TraceTree:
    """
    스팬 트리 구조
    
    스팬들의 부모-자식 관계를 트리로 구성.
    """
    
    def __init__(self, spans: Sequence[Span]):
        """
        Args:
            spans: 스팬 시퀀스
        """
        self._spans = list(spans)
        self._by_id: Dict[str, Span] = {s.span_id: s for s in spans}
        self._children: Dict[str, List[Span]] = {}
        self._root_spans: List[Span] = []
        
        self._build_tree()
    
    def _build_tree(self) -> None:
        """트리 구축"""
        for span in self._spans:
            parent_id = span.parent_span_id
            
            if parent_id and parent_id in self._by_id:
                if parent_id not in self._children:
                    self._children[parent_id] = []
                self._children[parent_id].append(span)
            else:
                self._root_spans.append(span)
    
    def get_children(self, span_id: str) -> List[Span]:
        """자식 스팬들 반환"""
        return self._children.get(span_id, [])
    
    def get_descendants(self, span_id: str) -> List[Span]:
        """모든 후손 스팬들 반환"""
        result: List[Span] = []
        children = self.get_children(span_id)
        
        for child in children:
            result.append(child)
            result.extend(self.get_descendants(child.span_id))
        
        return result
    
    def get_path_to_root(self, span_id: str) -> List[Span]:
        """루트까지의 경로"""
        path: List[Span] = []
        current = self._by_id.get(span_id)
        
        while current:
            path.append(current)
            parent_id = current.parent_span_id
            current = self._by_id.get(parent_id) if parent_id else None
        
        return list(reversed(path))
    
    @property
    def roots(self) -> List[Span]:
        """루트 스팬들"""
        return self._root_spans
    
    def find_spans_by_kind(self, kind: SpanKind) -> List[Span]:
        """종류로 스팬 찾기"""
        return [s for s in self._spans if s.kind == kind]
    
    def find_spans_by_name(self, pattern: str) -> List[Span]:
        """이름 패턴으로 스팬 찾기"""
        regex = re.compile(pattern, re.IGNORECASE)
        return [s for s in self._spans if regex.search(s.name)]


# ============================================================================
# Tracer Trace to Triplet Adapter
# ============================================================================

class TracerTraceToTriplet(TraceAdapter[List[Triplet]]):
    """
    트레이서 스팬 → 트리플렛 변환
    
    Agent Lightning의 TracerTraceToTriplet 참고.
    
    전략:
        1. LLM 호출 스팬 추출
        2. 리워드 스팬 추출
        3. 리워드-LLM 호출 매칭
        4. Triplet 생성
    """
    
    def __init__(
        self,
        llm_call_pattern: str = r"openai\.chat\.completion|llm_call|llm:",
        agent_pattern: Optional[str] = None,
        reward_match_policy: RewardMatchPolicy = RewardMatchPolicy.FIRST_OCCURRENCE,
        exclude_llm_in_reward: bool = True,
        final_reward: Optional[float] = None,
    ):
        """
        Args:
            llm_call_pattern: LLM 호출 스팬 이름 패턴
            agent_pattern: 에이전트 스팬 이름 패턴 (필터용)
            reward_match_policy: 리워드 매칭 정책
            exclude_llm_in_reward: 리워드 스팬 내 LLM 호출 제외
            final_reward: 최종 리워드 (지정 시 마지막 트리플렛에만)
        """
        self._llm_pattern = re.compile(llm_call_pattern, re.IGNORECASE)
        self._agent_pattern = re.compile(agent_pattern) if agent_pattern else None
        self._reward_policy = reward_match_policy
        self._exclude_llm_in_reward = exclude_llm_in_reward
        self._final_reward = final_reward
        self._logger = StructuredLogger("adapter.triplet")
    
    def adapt(self, source: Sequence[Span]) -> List[Triplet]:
        """
        Span 시퀀스 → Triplet 리스트
        
        Args:
            source: 스팬 시퀀스
            
        Returns:
            Triplet 리스트
        """
        # 시퀀스 ID로 정렬
        spans = sorted(source, key=lambda s: s.sequence_id)
        
        # 트리 구축
        tree = TraceTree(spans)
        
        # LLM 호출 스팬 추출
        llm_spans = self._extract_llm_calls(spans, tree)
        
        # 리워드 스팬 추출
        reward_spans = find_reward_spans(spans)
        
        # 리워드 매칭
        matched_rewards = self._match_rewards(llm_spans, reward_spans)
        
        # 트리플렛 생성
        triplets: List[Triplet] = []
        
        for i, llm_span in enumerate(llm_spans):
            triplet = self._span_to_triplet(
                llm_span,
                matched_rewards.get(llm_span.span_id),
            )
            
            # 최종 리워드 적용
            if self._final_reward is not None and i == len(llm_spans) - 1:
                triplet.reward = self._final_reward
            
            triplets.append(triplet)
        
        self._logger.debug(
            "Adapted spans to triplets",
            span_count=len(spans),
            llm_count=len(llm_spans),
            triplet_count=len(triplets),
        )
        
        return triplets
    
    def _extract_llm_calls(
        self,
        spans: Sequence[Span],
        tree: TraceTree,
    ) -> List[Span]:
        """LLM 호출 스팬 추출"""
        llm_spans: List[Span] = []
        
        # 리워드 스팬 ID 집합 (제외용)
        reward_span_ids: Set[str] = set()
        if self._exclude_llm_in_reward:
            for span in spans:
                if is_reward_span(span):
                    reward_span_ids.add(span.span_id)
                    # 리워드 스팬의 모든 후손도 제외
                    for desc in tree.get_descendants(span.span_id):
                        reward_span_ids.add(desc.span_id)
        
        for span in spans:
            # 리워드 내 LLM 제외
            if span.span_id in reward_span_ids:
                continue
            
            # LLM 패턴 매칭
            if not self._llm_pattern.search(span.name):
                continue
            
            # 에이전트 패턴 필터
            if self._agent_pattern:
                agent_name = span.agent_name or ""
                if not self._agent_pattern.search(agent_name):
                    continue
            
            llm_spans.append(span)
        
        return llm_spans
    
    def _match_rewards(
        self,
        llm_spans: List[Span],
        reward_spans: List[Span],
    ) -> Dict[str, Optional[float]]:
        """
        리워드-LLM 호출 매칭
        
        Returns:
            LLM span_id → reward 매핑
        """
        matched: Dict[str, Optional[float]] = {}
        
        if not reward_spans:
            return matched
        
        # 시퀀스로 정렬
        reward_spans = sorted(reward_spans, key=lambda s: s.sequence_id)
        
        if self._reward_policy == RewardMatchPolicy.FINAL_ONLY:
            # 마지막 LLM에만 마지막 리워드
            if llm_spans and reward_spans:
                last_reward = get_reward_value(reward_spans[-1])
                matched[llm_spans[-1].span_id] = last_reward
        
        elif self._reward_policy == RewardMatchPolicy.FIRST_OCCURRENCE:
            # 각 LLM에 대해 그 다음에 오는 첫 리워드 매칭
            reward_idx = 0
            for llm_span in llm_spans:
                # LLM 시퀀스 이후의 첫 리워드 찾기
                while reward_idx < len(reward_spans):
                    if reward_spans[reward_idx].sequence_id > llm_span.sequence_id:
                        matched[llm_span.span_id] = get_reward_value(
                            reward_spans[reward_idx]
                        )
                        reward_idx += 1
                        break
                    reward_idx += 1
        
        elif self._reward_policy == RewardMatchPolicy.LAST_OCCURRENCE:
            # 각 LLM에 대해 그 다음 LLM 전까지의 마지막 리워드
            for i, llm_span in enumerate(llm_spans):
                next_llm_seq = (
                    llm_spans[i + 1].sequence_id
                    if i + 1 < len(llm_spans)
                    else float('inf')
                )
                
                last_reward = None
                for r_span in reward_spans:
                    if llm_span.sequence_id < r_span.sequence_id < next_llm_seq:
                        last_reward = get_reward_value(r_span)
                
                if last_reward is not None:
                    matched[llm_span.span_id] = last_reward
        
        elif self._reward_policy == RewardMatchPolicy.CLOSEST_BEFORE:
            # 각 LLM에 대해 직전의 가장 가까운 리워드
            for llm_span in llm_spans:
                closest_reward = None
                for r_span in reversed(reward_spans):
                    if r_span.sequence_id < llm_span.sequence_id:
                        closest_reward = get_reward_value(r_span)
                        break
                
                if closest_reward is not None:
                    matched[llm_span.span_id] = closest_reward
        
        return matched
    
    def _span_to_triplet(
        self,
        span: Span,
        reward: Optional[float],
    ) -> Triplet:
        """스팬 → 트리플렛 변환"""
        attrs = span.attributes or {}
        
        # 프롬프트 추출
        prompt: Dict[str, Any] = {}
        
        # 토큰 ID 우선
        prompt_ids = attrs.get("llm.prompt.token_ids")
        if prompt_ids:
            prompt["token_ids"] = prompt_ids
        
        # 메시지 형식
        messages = attrs.get("llm.request.messages")
        if messages:
            if isinstance(messages, str):
                try:
                    messages = json.loads(messages)
                except json.JSONDecodeError:
                    messages = None
            if messages:
                prompt["messages"] = messages
        
        # 텍스트 프롬프트
        prompt_text = attrs.get("llm.prompt", attrs.get("llm.prompt.text"))
        if prompt_text:
            prompt["text"] = prompt_text
        
        prompt["length"] = attrs.get("llm.prompt.length", 0)
        
        # 응답 추출
        response: Dict[str, Any] = {}
        
        response_ids = attrs.get("llm.response.token_ids")
        if response_ids:
            response["token_ids"] = response_ids
        
        response_text = attrs.get(
            "llm.response.content",
            attrs.get("llm.response", "")
        )
        if response_text:
            response["text"] = response_text
        
        response["length"] = attrs.get("llm.response.length", 0)
        
        # 토큰 사용량
        tokens = {
            "prompt": attrs.get("llm.tokens.prompt", 0),
            "completion": attrs.get("llm.tokens.completion", 0),
            "total": attrs.get("llm.tokens.total", 0),
        }
        
        # 메타데이터
        metadata = {
            "span_id": span.span_id,
            "sequence_id": span.sequence_id,
            "model": attrs.get("llm.model", "unknown"),
            "agent_name": span.agent_name,
            "duration_ms": span.duration_ms,
            "tokens": tokens,
        }
        
        return Triplet(
            prompt=prompt,
            response=response,
            reward=reward,
            metadata=metadata,
        )


# ============================================================================
# Trajectory Builder
# ============================================================================

def build_trajectory(
    spans: Sequence[Span],
    adapter: TracerTraceToTriplet,
    rollout_id: str,
    attempt_id: Optional[str] = None,
) -> Trajectory:
    """
    스팬으로부터 Trajectory 구축
    
    Args:
        spans: 스팬 시퀀스
        adapter: 어댑터
        rollout_id: 롤아웃 ID
        attempt_id: 어템프트 ID
        
    Returns:
        Trajectory
    """
    triplets = adapter.adapt(spans)
    
    trajectory = Trajectory(
        rollout_id=rollout_id,
        attempt_id=attempt_id,
    )
    
    for triplet in triplets:
        trajectory.add_triplet(triplet)
    
    return trajectory


# ============================================================================
# Export Helper
# ============================================================================

def export_triplets_to_jsonl(
    triplets: List[Triplet],
    filepath: str,
) -> int:
    """
    트리플렛을 JSONL 파일로 내보내기
    
    Args:
        triplets: 트리플렛 리스트
        filepath: 출력 파일 경로
        
    Returns:
        내보낸 행 수
    """
    import json
    
    count = 0
    with open(filepath, "w", encoding="utf-8") as f:
        for triplet in triplets:
            line = json.dumps(triplet.to_dict(), ensure_ascii=False)
            f.write(line + "\n")
            count += 1
    
    return count


def export_for_sft(
    triplets: List[Triplet],
    filepath: str,
    format: str = "alpaca",
) -> int:
    """
    SFT 학습용 형식으로 내보내기
    
    Args:
        triplets: 트리플렛 리스트
        filepath: 출력 파일 경로
        format: 출력 형식 ("alpaca", "sharegpt", "openai")
        
    Returns:
        내보낸 행 수
    """
    import json
    
    count = 0
    with open(filepath, "w", encoding="utf-8") as f:
        for triplet in triplets:
            if format == "alpaca":
                # Alpaca 형식
                data = {
                    "instruction": triplet.prompt.get("text", ""),
                    "input": "",
                    "output": triplet.response.get("text", ""),
                }
            elif format == "sharegpt":
                # ShareGPT 형식
                data = {
                    "conversations": [
                        {"from": "human", "value": triplet.prompt.get("text", "")},
                        {"from": "gpt", "value": triplet.response.get("text", "")},
                    ]
                }
            elif format == "openai":
                # OpenAI fine-tuning 형식
                messages = triplet.prompt.get("messages", [])
                if not messages:
                    messages = [{"role": "user", "content": triplet.prompt.get("text", "")}]
                messages.append({
                    "role": "assistant",
                    "content": triplet.response.get("text", ""),
                })
                data = {"messages": messages}
            else:
                data = triplet.to_dict()
            
            line = json.dumps(data, ensure_ascii=False)
            f.write(line + "\n")
            count += 1
    
    return count
