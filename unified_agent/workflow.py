#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Unified Agent Framework - 워크플로우 모듈 (Workflow Module)

================================================================================
📁 파일 위치: unified_agent/workflow.py
📋 역할: Node와 Graph 클래스를 통한 워크플로우 실행 관리
📅 최종 업데이트: 2026년 1월
================================================================================

🎯 주요 구성 요소:

    📌 Node (워크플로우 노드):
        - 단일 에이전트를 래핑하는 실행 단위
        - 조건부 라우팅 지원 (condition_func)
        - 엣지(edges)를 통한 다음 노드 지정
        - 실행 횟수 추적

    📌 Graph (워크플로우 그래프):
        - 노드들의 집합 및 실행 순서 관리
        - 조건부 엣지 추가 (add_conditional_edge)
        - 루프 노드 지정 및 무한 루프 방지
        - 체크포인트/롤백 지원
        - Mermaid 형식 시각화
        - 실행 통계 제공

🔧 워크플로우 실행 흐름:

    ┌───────────────────────────────────────────────────────┐
    │  [START] → [Node A] ───┬───→ [Node B] → [END]  │
    │                       │                          │
    │                       │ (condition: "need_review")  │
    │                       │                          │
    │                       └───→ [Node C] ────────┘  │
    └───────────────────────────────────────────────────────┘

📌 사용 예시:

    예제 1: 기본 워크플로우
    ----------------------------------------
    >>> from unified_agent.workflow import Node, Graph
    >>> from unified_agent.agents import SimpleAgent
    >>>
    >>> # 노드 생성
    >>> node_a = Node(name="greeting", agent=greeting_agent)
    >>> node_b = Node(name="response", agent=response_agent)
    >>>
    >>> # 그래프 생성 및 노드 추가
    >>> graph = Graph(name="chat_workflow")
    >>> graph.add_node(node_a)
    >>> graph.add_node(node_b)
    >>> graph.set_start_node("greeting")
    >>> graph.set_end_node("response")
    >>> graph.add_edge("greeting", "response")
    >>>
    >>> # 실행
    >>> result = await graph.run(initial_state, kernel)

    예제 2: 조건부 라우팅 (분기)
    ----------------------------------------
    >>> # 조건 함수 정의
    >>> async def route_by_intent(state, result):
    ...     if "code" in result.content.lower():
    ...         return "coding"
    ...     return "general"
    >>>
    >>> # 조건부 엣지 추가
    >>> graph.add_conditional_edge(
    ...     source="router",
    ...     condition_func=route_by_intent,
    ...     routes={"coding": "code_agent", "general": "chat_agent"}
    ... )

    예제 3: 루프 워크플로우 (반복)
    ----------------------------------------
    >>> # 루프 노드 지정 (reviewer는 반복 가능)
    >>> graph.set_loop_nodes(["reviewer"])
    >>> graph.max_iterations = 5  # 최대 5회 반복
    >>>
    >>> # 검토 완료 시 pass, 수정 필요 시 writer로 복귀
    >>> graph.add_conditional_edge(
    ...     source="reviewer",
    ...     routes={"pass": "end", "revise": "writer"}
    ... )

⚠️ 주요 기능:
    - 무한 루프 방지: max_iterations 설정
    - 체크포인트: 실행 중 상태 저장 및 복구
    - 시각화: visualize() 메서드로 Mermaid 다이어그램 생성
    - 통계: get_statistics()로 실행 통계 확인

🔗 참고:
    - LangGraph: https://github.com/langchain-ai/langgraph (영감)
    - Mermaid: https://mermaid.js.org/
"""

import logging
from dataclasses import dataclass, field
from typing import Dict, Set, Optional, Callable, Any

from semantic_kernel import Kernel

from .models import AgentState, NodeResult, ExecutionStatus
from .agents import Agent

# v3.3: SessionTree 통합
from .session_tree import SessionTree, SessionNode, NodeType

__all__ = [
    "Node",
    "Graph",
]


# ============================================================================
# Node - 워크플로우 노드
# ============================================================================

class Node:
    """
    워크플로우 노드

    주요 기능:
    1. condition_func: 조건부 라우팅 지원 (LangGraph 패턴)
    2. execution_count: 실행 횟수 추적
    """
    __slots__ = ('name', 'agent', 'edges', 'condition_func', 'execution_count')

    def __init__(
        self,
        name: str,
        agent: Agent,
        edges: Optional[Dict[str, str]] = None,
        condition_func: Optional[Callable] = None
    ):
        """
        노드 초기화

        Args:
            name: 노드 이름
            agent: 실행할 에이전트
            edges: 다음 노드 매핑 (condition -> node_name)
            condition_func: 조건부 라우팅 함수
        """
        self.name = name
        self.agent = agent
        self.edges = edges or {}
        self.condition_func = condition_func
        self.execution_count = 0

    async def execute(self, state: AgentState, kernel: Kernel) -> NodeResult:
        """노드 실행"""
        logging.info(f"📍 노드 실행: {self.name} (#{self.execution_count + 1})")

        result = await self.agent.execute(state, kernel)
        self.execution_count += 1

        # 조건부 라우팅
        if not result.next_node and self.edges:
            if self.condition_func:
                # 조건 함수로 다음 노드 결정
                next_node = await self.condition_func(state, result)
                result.next_node = self.edges.get(next_node, self.edges.get("default"))
            else:
                result.next_node = self.edges.get("default", None)

        state.visited_nodes.append(self.name)
        return result


# ============================================================================
# Graph - 워크플로우 그래프
# ============================================================================

class Graph:
    """
    워크플로우 그래프 - 조건부 라우팅 및 루프 지원

    주요 기능:
    1. loop_nodes: 루프 가능한 노드 집합
    2. add_conditional_edge(): 조건부 엣지 추가
    3. 무한 루프 방지 로직
    4. 상세한 실행 로그
    5. get_statistics(): 그래프 통계
    6. visualize(): Mermaid 형식 시각화
    7. v3.3: SessionTree 자동 분기 생성
    """

    def __init__(self, name: str = "workflow", enable_session_tree: bool = True):
        """
        그래프 초기화

        Args:
            name: 워크플로우 이름
            enable_session_tree: v3.3 SessionTree 기능 활성화 여부
        """
        self.name = name
        self.nodes: Dict[str, Node] = {}
        self.start_node: Optional[str] = None
        self.end_nodes: Set[str] = set()
        self.loop_nodes: Set[str] = set()
        
        # v3.3: SessionTree 통합
        self._enable_session_tree = enable_session_tree
        self._session_tree: Optional[SessionTree] = None
        self._current_session_node_id: Optional[str] = None
    
    def set_session_tree(self, session_tree: SessionTree):
        """v3.3: SessionTree 설정"""
        self._session_tree = session_tree
        self._logger_info(f"SessionTree connected to workflow: {self.name}")
    
    def _logger_info(self, msg: str):
        """로깅 헬퍼"""
        logging.info(f"[{self.name}] {msg}")

    def add_node(self, node: Node, allow_loop: bool = False):
        """
        노드 추가

        Args:
            node: 추가할 노드
            allow_loop: 루프 허용 여부
        """
        self.nodes[node.name] = node
        if allow_loop:
            self.loop_nodes.add(node.name)
        logging.info(f"✅ 노드 추가: {node.name}")

    def add_edge(self, from_node: str, to_node: str, condition: str = "default"):
        """
        엣지 추가

        Args:
            from_node: 시작 노드 이름
            to_node: 도착 노드 이름
            condition: 조건 키
        """
        if from_node not in self.nodes:
            raise ValueError(f"노드 '{from_node}'가 존재하지 않습니다.")
        self.nodes[from_node].edges[condition] = to_node
        logging.info(f"✅ 엣지 추가: {from_node} --[{condition}]--> {to_node}")

    def add_conditional_edge(self, from_node: str, condition_func: Callable):
        """
        조건부 엣지 추가

        LangGraph의 조건부 라우팅 패턴 구현

        사용 예시:
            async def route_by_complexity(state, result):
                if "simple" in result.output.lower():
                    return "simple"
                return "complex"

            graph.add_conditional_edge("analyzer", route_by_complexity)

        Args:
            from_node: 시작 노드 이름
            condition_func: 조건 결정 함수 (async)
        """
        if from_node not in self.nodes:
            raise ValueError(f"노드 '{from_node}'가 존재하지 않습니다.")
        self.nodes[from_node].condition_func = condition_func
        logging.info(f"✅ 조건부 엣지 추가: {from_node}")

    def set_start(self, node_name: str):
        """시작 노드 설정"""
        self.start_node = node_name
        logging.info(f"✅ 시작 노드: {node_name}")

    def set_end(self, node_name: str):
        """종료 노드 설정"""
        self.end_nodes.add(node_name)
        logging.info(f"✅ 종료 노드: {node_name}")

    async def execute(
        self,
        state: AgentState,
        kernel: Kernel,
        max_iterations: int = 10
    ) -> AgentState:
        """
        그래프 실행

        주요 기능:
        1. 승인 대기 처리
        2. 무한 루프 방지 (loop_nodes 체크)
        3. 상세한 로그 출력
        4. 실행 메트릭 수집
        5. v3.3: SessionTree 분기 자동 생성

        Args:
            state: 에이전트 상태
            kernel: Semantic Kernel 인스턴스
            max_iterations: 최대 반복 횟수

        Returns:
            업데이트된 에이전트 상태
        """
        if not self.start_node:
            raise ValueError("시작 노드가 설정되지 않았습니다.")

        current_node = self.start_node
        iterations = 0
        
        # v3.3: SessionTree 워크플로우 루트 노드 생성
        if self._enable_session_tree and self._session_tree:
            root_session_node = self._session_tree.add_node(
                content=f"Workflow: {self.name} started",
                role="system",
                node_type=NodeType.WORKFLOW,
                metadata={"workflow_name": self.name, "start_node": self.start_node}
            )
            self._current_session_node_id = root_session_node.id

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

            # 무한 루프 방지 (같은 노드 재방문 체크)
            if current_node in state.visited_nodes and current_node not in self.loop_nodes:
                logging.warning(f"⚠️ 노드 재방문 감지: {current_node}")
            
            # v3.3: SessionTree에 노드 실행 기록
            if self._enable_session_tree and self._session_tree and self._current_session_node_id:
                session_node = self._session_tree.add_node(
                    content=f"Execute node: {current_node}",
                    role="agent",
                    node_type=NodeType.AGENT,
                    parent_id=self._current_session_node_id,
                    metadata={"node_name": current_node, "iteration": iterations}
                )
                self._current_session_node_id = session_node.id

            result = await node.execute(state, kernel)
            state.metadata[f"{current_node}_result"] = result.model_dump()
            
            # v3.3: SessionTree에 결과 기록
            if self._enable_session_tree and self._session_tree and self._current_session_node_id:
                result_type = NodeType.BRANCH if result.next_node else NodeType.DECISION
                self._session_tree.add_node(
                    content=f"Result: {result.output[:100]}..." if len(result.output) > 100 else f"Result: {result.output}",
                    role="system",
                    node_type=result_type,
                    parent_id=self._current_session_node_id,
                    metadata={
                        "success": result.success,
                        "next_node": result.next_node,
                        "duration_ms": result.duration_ms
                    }
                )

            # 승인 대기 처리
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

        # 실행 통계
        state.metrics["total_iterations"] = iterations
        state.metrics["visited_nodes"] = len(state.visited_nodes)
        state.metrics["workflow_name"] = self.name

        return state

    def visualize(self) -> str:
        """
        그래프 시각화 (Mermaid 형식)

        Returns:
            Mermaid 다이어그램 문자열
        """
        lines = []
        lines.append("```mermaid")
        lines.append("graph TD")

        # 노드 정의
        for node_name, node in self.nodes.items():
            if node_name == self.start_node:
                shape = f"{node_name}([🎬 START: {node_name}])"
            elif node_name in self.end_nodes:
                shape = f"{node_name}[🏁 END: {node_name}]"
            elif node_name in self.loop_nodes:
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
        그래프 통계 반환

        Returns:
            통계 정보 딕셔너리
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
