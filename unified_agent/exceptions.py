#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Unified Agent Framework - 예외 클래스 모듈 (Exceptions Module)

================================================================================
📁 파일 위치: unified_agent/exceptions.py
📋 역할: 프레임워크 전용 커스텀 예외 클래스 정의
📅 최종 업데이트: 2026년 1월
================================================================================

🎯 예외 계층 구조:

    Exception (Python 내장)
        └── FrameworkError (기본 예외)
            ├── ConfigurationError (설정 오류)
            ├── WorkflowError (워크플로우 실행 오류)
            ├── AgentError (에이전트 실행 오류)
            ├── ApprovalError (승인 처리 오류)
            └── RAIValidationError (RAI 검증 실패)

📌 예외별 설명:

    FrameworkError:
        모든 프레임워크 예외의 기본 클래스.
        이 예외를 잡으면 모든 프레임워크 관련 오류를 처리할 수 있습니다.

    ConfigurationError:
        설정 관련 오류 (API 키 누락, 잘못된 설정 등).
        주로 FrameworkConfig.validate() 실패 시 발생.

    WorkflowError:
        워크플로우 실행 중 오류 (노드 찾기 실패, 무한 루프 등).

    AgentError:
        에이전트 실행 중 오류 (LLM 호출 실패, 타임아웃 등).

    ApprovalError:
        승인 처리 중 오류 (승인 거부, 타임아웃 등).

    RAIValidationError:
        AI 출력이 RAI(Responsible AI) 검증을 통과하지 못했을 때 발생.
        유해 콘텐츠, PII 노출 등이 감지되면 발생.

📌 사용 예시:

    >>> from unified_agent.exceptions import (
    ...     FrameworkError, ConfigurationError, RAIValidationError
    ... )
    >>>
    >>> # 예제 1: 설정 오류 처리
    >>> try:
    ...     config = FrameworkConfig.from_env()
    ...     config.validate()
    ... except ConfigurationError as e:
    ...     print(f"❌ 설정 오류: {e}")
    ...     print("💡 .env 파일을 확인하세요.")
    >>>
    >>> # 예제 2: RAI 검증 오류 처리
    >>> try:
    ...     result = validator.validate(user_input)
    ...     if not result.is_safe:
    ...         raise RAIValidationError(
    ...             "안전하지 않은 콘텐츠",
    ...             category=result.category,
    ...             result=result
    ...         )
    ... except RAIValidationError as e:
    ...     print(f"⚠️ RAI 검증 실패: {e}")
    ...     print(f"   카테고리: {e.category}")
    >>>
    >>> # 예제 3: 모든 프레임워크 오류 처리
    >>> try:
    ...     result = await framework.run(session_id, workflow, input_text)
    ... except FrameworkError as e:
    ...     # 모든 프레임워크 관련 예외를 잡음
    ...     logger.error(f"프레임워크 오류: {e}")
    ...     await notify_admin(str(e))

⚠️ 주의사항:
    - 가능한 한 구체적인 예외를 잡으세요 (ConfigurationError > FrameworkError).
    - RAIValidationError는 category와 result 속성으로 상세 정보를 제공합니다.
    - 예외 메시지는 사용자 친화적인 한글로 제공됩니다.

🔗 참고:
    - Python 예외: https://docs.python.org/3/tutorial/errors.html
    - Azure Content Safety: https://learn.microsoft.com/azure/ai-services/content-safety/
"""

__all__ = [
    "FrameworkError",
    "ConfigurationError",
    "WorkflowError",
    "AgentError",
    "ApprovalError",
    "RAIValidationError",
]


class FrameworkError(Exception):
    """
    프레임워크 기본 예외 클래스

    모든 Unified Agent Framework 예외의 기본 클래스입니다.
    이 예외를 잡으면 프레임워크에서 발생하는 모든 오류를 처리할 수 있습니다.

    예시:
        >>> try:
        ...     await framework.run(...)
        ... except FrameworkError as e:
        ...     logger.error(f"프레임워크 오류: {e}")
    """
    pass


class ConfigurationError(FrameworkError):
    """
    설정 관련 예외

    다음 상황에서 발생합니다:
    - 필수 환경변수 누락 (API 키, 엔드포인트, 배포명)
    - 잘못된 설정 값
    - 지원하지 않는 모델명
    - .env 파일 로드 실패

    해결 방법:
    - .env 파일 확인 및 수정
    - 환경변수 설정 확인
    - Settings.SUPPORTED_MODELS 목록 참고
    """
    pass


class WorkflowError(FrameworkError):
    """
    워크플로우 실행 예외

    다음 상황에서 발생합니다:
    - 시작 노드가 설정되지 않음
    - 존재하지 않는 노드로 라우팅
    - 무한 루프 감지
    - 그래프 구조 오류

    해결 방법:
    - 노드 이름 확인
    - 엣지 설정 확인
    - loop_nodes 및 max_iterations 확인
    """
    pass


class AgentError(FrameworkError):
    """
    에이전트 실행 예외

    다음 상황에서 발생합니다:
    - LLM API 호출 실패
    - 타임아웃
    - 잘못된 응답 형식
    - 도구 실행 실패

    해결 방법:
    - API 키 및 열결 확인
    - CircuitBreaker 상태 확인
    - 모델 가용성 확인
    """
    pass


class ApprovalError(FrameworkError):
    """
    승인 처리 예외

    다음 상황에서 발생합니다:
    - 사용자가 승인을 거부함
    - 승인 대기 타임아웃
    - 승인 콜백 실패

    해결 방법:
    - 사용자에게 거부 사유 안내
    - 타임아웃 설정 조정
    - 자동 승인 규칙 검토
    """
    pass


class RAIValidationError(FrameworkError):
    """
    RAI 검증 예외

    AI 출력이 안전성 검증을 통과하지 못했을 때 발생합니다.
    """
    def __init__(self, message: str, category=None, result=None):
        super().__init__(message)
        self.category = category
        self.result = result
