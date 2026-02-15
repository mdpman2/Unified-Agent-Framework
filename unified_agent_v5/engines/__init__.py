#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Unified Agent Framework v5 — Engines

================================================================================
v4.1 대응: universal_bridge.py + 7개 브릿지 모듈 (16개 지원) → 3개 엔진
축소 이유: 실무 사용 빈도 기준 Top 3 선정.
          16개 브릿지 유지보수 불가 → 가장 잘 동작하는 3개만 집중.
================================================================================

Top 3 엔진 + Direct API:
    1. direct   — OpenAI API 직접 호출 (가장 가볍고 빠름)
    2. langchain — LangChain/LangGraph 체인/RAG 구성 (생태계 최대)
    3. crewai   — CrewAI 멀티 에이전트 협업 (API 안정)

나머지 13개 프레임워크는 plugins/ 디렉토리로 분리하거나 삭제.
"""

from __future__ import annotations

from .base import EngineProtocol, get_engine

__all__ = ["EngineProtocol", "get_engine"]
