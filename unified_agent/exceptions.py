#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Unified Agent Framework - 예외 클래스 모듈

커스텀 예외 클래스들을 정의합니다.
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
    """프레임워크 기본 예외 클래스"""
    pass


class ConfigurationError(FrameworkError):
    """설정 관련 예외"""
    pass


class WorkflowError(FrameworkError):
    """워크플로우 실행 예외"""
    pass


class AgentError(FrameworkError):
    """에이전트 실행 예외"""
    pass


class ApprovalError(FrameworkError):
    """승인 처리 예외"""
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
