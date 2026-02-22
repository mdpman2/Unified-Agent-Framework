#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Unified Agent Framework v5 — Plugins

================================================================================
v4.1 대응: 제거된 40개 모듈 중 선택적 마이그레이션 대상
축소 이유: 핵심 기능(runner, types, config, memory, tools, callback, engines)에
          해당하지 않는 기능들을 플러그인으로 분리.
          필요한 사용자만 선택적으로 설치/사용.
================================================================================

사용 가능한 플러그인 (optional):
    - security_guardrails: 프롬프트 보안 (← v4.1 security_guardrails.py)
      대안: Azure Content Safety API 직접 호출 권장
    - structured_output: 구조화된 출력 (← v4.1 structured_output.py)
      대안: openai 패키지 response_format 직접 사용
    - persistent_memory: 영속 메모리 (← v4.1 persistent_memory.py)
      대안: Memory.to_json() + Redis/CosmosDB
    - session_tree: Git 스타일 세션 분기 (← v4.1 session_tree.py)
      대안: 대부분 실무에서 미사용
    - browser_automation: Playwright + CUA (← v4.1 browser_use.py)
      대안: browser-use 패키지 직접 사용
    - deep_research: 다단계 자율 연구 (← v4.1 deep_research.py)
      대안: 별도 에이전트로 구현

설치/활성화:
    >>> from unified_agent_v5.plugins import security_guardrails
    >>> # 각 플러그인은 독립적으로 import 가능
"""

# 플러그인은 각각 독립 모듈로 존재하며
# 필요 시 lazy import됩니다.
