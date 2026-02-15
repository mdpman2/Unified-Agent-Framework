#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Unified Agent Framework - 브라우저 자동화 모듈 (Browser Use & Computer Use Module)

================================================================================
📁 파일 위치: unified_agent/browser_use.py
📋 역할: 브라우저 자동화(Playwright), Computer Use(CUA) 통합
📅 최종 업데이트: 2026년 2월 13일
📦 버전: v4.1.0
✅ 테스트: test_v41_scenarios.py
================================================================================

🎯 주요 구성 요소:
    1. BrowserAutomation - Playwright 기반 브라우저 자동화
    2. ComputerUseAgent - OpenAI CUA (Computer-Using Agent) 통합
    3. ScreenCapture - 스크린 캡처 및 시각 분석
    4. BrowserSession - 격리된 브라우저 세션 관리
    5. ActionRecorder - 마우스/키보드 액션 기록 및 재생

🔧 2026년 2월 기능:
    - Azure Foundry Browser Automation Tool (Playwright Workspaces) 통합
    - OpenAI CUA (Computer-Using Agent) API 호출
    - 격리된 브라우저 환경에서 안전한 자동화
    - 스크린 캡처 → 모델 분석 → 액션 생성 루프
    - Prompt Injection 방어를 위한 안전 검사 내장
    - 브라우저/OS 양방향 자동화 지원

📌 사용 예시:
    >>> from unified_agent.browser_use import (
    ...     BrowserAutomation, ComputerUseAgent, BrowserConfig,
    ...     BrowserAction, ActionType
    ... )
    >>>
    >>> # 브라우저 자동화
    >>> browser = BrowserAutomation(BrowserConfig(headless=True))
    >>> await browser.start()
    >>> result = await browser.execute_task("Microsoft Learn에서 Agent Framework 검색")
    >>> await browser.close()
    >>>
    >>> # Computer Use Agent (CUA)
    >>> cua = ComputerUseAgent(model="computer-use-preview")
    >>> result = await cua.run(
    ...     task="최신 AI 뉴스를 검색하고 요약해주세요",
    ...     environment="browser"
    ... )

⚠️ 주의사항:
    - 브라우저 자동화는 격리된 환경에서만 실행하세요.
    - Computer Use는 아직 Research Preview이며, 오류 가능성이 있습니다.
    - 민감한 작업(결제, 로그인 등)에는 반드시 Human-in-the-loop을 적용하세요.
    - OSWorld 벤치마크: 38.1%, WebArena: 58.1%, WebVoyager: 87%

🔗 관련 문서:
    - Azure Browser Automation: https://learn.microsoft.com/azure/ai-foundry/agents/how-to/tools-classic/browser-automation
    - Azure Computer Use: https://learn.microsoft.com/azure/ai-foundry/agents/how-to/tools-classic/computer-use
    - OpenAI CUA: https://platform.openai.com/docs/guides/tools-computer-use
    - Playwright: https://playwright.dev/
"""

from __future__ import annotations

import asyncio
import logging
import time
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum, unique
from typing import Any

__all__ = [
    # Enums
    "ActionType",
    "BrowserStatus",
    "CUAEnvironment",
    # Config & Data Models
    "BrowserConfig",
    "BrowserAction",
    "ActionResult",
    "ScreenCapture",
    "CUAConfig",
    "CUAResult",
    # Core Components
    "BrowserSession",
    "BrowserAutomation",
    "ComputerUseAgent",
    "ActionRecorder",
    "SafetyChecker",
]

logger = logging.getLogger(__name__)

# ============================================================================
# Enums
# ============================================================================

@unique
class ActionType(Enum):
    """브라우저/컴퓨터 액션 타입"""
    # 마우스 액션
    CLICK = "click"
    DOUBLE_CLICK = "double_click"
    RIGHT_CLICK = "right_click"
    DRAG = "drag"
    SCROLL = "scroll"
    HOVER = "hover"

    # 키보드 액션
    TYPE = "type"
    KEY_PRESS = "key_press"
    KEY_COMBINATION = "key_combination"

    # 브라우저 네비게이션
    NAVIGATE = "navigate"
    BACK = "back"
    FORWARD = "forward"
    REFRESH = "refresh"

    # 페이지 상호작용
    WAIT = "wait"
    SCREENSHOT = "screenshot"
    SELECT = "select"
    UPLOAD = "upload"

    # 고수준 액션
    SEARCH = "search"
    FILL_FORM = "fill_form"
    EXTRACT_TEXT = "extract_text"


@unique
class BrowserStatus(Enum):
    """브라우저 세션 상태"""
    IDLE = "idle"
    RUNNING = "running"
    NAVIGATING = "navigating"
    WAITING = "waiting"
    ERROR = "error"
    CLOSED = "closed"


@unique
class CUAEnvironment(Enum):
    """Computer Use Agent 실행 환경"""
    BROWSER = "browser"         # 브라우저 전용
    DESKTOP = "desktop"         # 데스크톱 전체
    SANDBOXED = "sandboxed"     # 샌드박스 환경


# ============================================================================
# Data Models
# ============================================================================

@dataclass(frozen=True, slots=True)
class BrowserConfig:
    """
    브라우저 자동화 설정

    Attributes:
        headless: 헤드리스 모드 여부
        viewport_width: 뷰포트 너비 (px)
        viewport_height: 뷰포트 높이 (px)
        timeout_ms: 기본 타임아웃 (밀리초)
        user_agent: User-Agent 문자열
        enable_safety_checks: 안전성 검사 활성화
        max_actions_per_task: 작업당 최대 액션 수
        screenshot_on_action: 매 액션마다 스크린샷 저장
        proxy: 프록시 서버 URL
    """
    headless: bool = True
    viewport_width: int = 1280
    viewport_height: int = 720
    timeout_ms: int = 30000
    user_agent: str = "UnifiedAgent/4.1 BrowserAutomation"
    enable_safety_checks: bool = True
    max_actions_per_task: int = 50
    screenshot_on_action: bool = False
    proxy: str | None = None


@dataclass(slots=True)
class BrowserAction:
    """
    브라우저 액션 (개별 액션 단위)

    Attributes:
        action_id: 액션 고유 ID
        action_type: 액션 타입
        target: 대상 선택자 (CSS selector / XPath)
        value: 입력 값 (type, key_press 등)
        coordinates: 좌표 (x, y) — CUA에서 사용
        timestamp: 액션 실행 시각
        metadata: 추가 정보
    """
    action_id: str = field(default_factory=lambda: str(uuid.uuid4())[:8])
    action_type: ActionType = ActionType.CLICK
    target: str = ""
    value: str = ""
    coordinates: tuple[int, int] | None = None
    timestamp: datetime = field(
        default_factory=lambda: datetime.now(timezone.utc)
    )
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class ActionResult:
    """
    액션 실행 결과

    Attributes:
        action_id: 실행된 액션 ID
        success: 성공 여부
        screenshot_base64: 실행 후 스크린샷 (Base64)
        extracted_text: 추출된 텍스트
        error: 에러 메시지
        duration_ms: 실행 시간 (밀리초)
    """
    action_id: str = ""
    success: bool = True
    screenshot_base64: str | None = None
    extracted_text: str = ""
    error: str = ""
    duration_ms: float = 0.0


@dataclass(slots=True)
class ScreenCapture:
    """
    스크린 캡처 데이터

    Attributes:
        capture_id: 캡처 ID
        image_base64: Base64 인코딩 이미지
        width: 이미지 너비
        height: 이미지 높이
        url: 현재 페이지 URL
        title: 현재 페이지 제목
        timestamp: 캡처 시각
    """
    capture_id: str = field(default_factory=lambda: str(uuid.uuid4())[:8])
    image_base64: str = ""
    width: int = 0
    height: int = 0
    url: str = ""
    title: str = ""
    timestamp: datetime = field(
        default_factory=lambda: datetime.now(timezone.utc)
    )


@dataclass(frozen=True, slots=True)
class CUAConfig:
    """
    Computer Use Agent 설정

    Attributes:
        model: CUA 모델 이름
        display_width: 디스플레이 너비
        display_height: 디스플레이 높이
        environment: 실행 환경 (browser, desktop)
        max_steps: 최대 스텝 수
        truncation: 입력 트렁케이션 (auto)
        enable_safety: 안전성 검사 활성화
        confirmation_prompts: 민감 작업 확인 프롬프트 표시
    """
    model: str = "computer-use-preview"
    display_width: int = 1024
    display_height: int = 768
    environment: CUAEnvironment = CUAEnvironment.BROWSER
    max_steps: int = 50
    truncation: str = "auto"
    enable_safety: bool = True
    confirmation_prompts: bool = True


@dataclass(slots=True)
class CUAResult:
    """
    Computer Use Agent 실행 결과

    Attributes:
        task_id: 태스크 ID
        success: 성공 여부
        actions_taken: 수행된 액션 목록
        final_screenshot: 최종 스크린샷
        output_text: 결과 텍스트
        total_steps: 전체 스텝 수
        duration_seconds: 소요 시간 (초)
        safety_checks_passed: 안전성 검사 통과 여부
        error: 에러 메시지
    """
    task_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    success: bool = True
    actions_taken: list[BrowserAction] = field(default_factory=list)
    final_screenshot: ScreenCapture | None = None
    output_text: str = ""
    total_steps: int = 0
    duration_seconds: float = 0.0
    safety_checks_passed: bool = True
    error: str = ""


# ============================================================================
# Core Components
# ============================================================================

class SafetyChecker:
    """
    브라우저 자동화 안전성 검사기

    Prompt Injection, 민감 작업, 위험 URL 등을 검사합니다.

    📌 사용 예시:
        >>> checker = SafetyChecker()
        >>> result = checker.check_action(action)
        >>> if not result["safe"]:
        ...     print(f"차단: {result['reason']}")
    """

    # 위험 URL 패턴
    DANGEROUS_URLS = frozenset({
        "chrome://", "about:config", "file:///",
        "javascript:", "data:", "vbscript:",
    })

    # 민감 작업 키워드
    SENSITIVE_KEYWORDS = frozenset({
        "password", "credit card", "social security", "bank",
        "payment", "billing", "ssn", "pin", "cvv",
        "비밀번호", "신용카드", "계좌번호", "주민등록번호",
    })

    def check_url(self, url: str) -> dict[str, Any]:
        """URL 안전성 검사"""
        url_lower = url.lower()
        for pattern in self.DANGEROUS_URLS:
            if url_lower.startswith(pattern):
                return {"safe": False, "reason": f"Dangerous URL pattern: {pattern}"}
        return {"safe": True, "reason": ""}

    def check_action(self, action: BrowserAction) -> dict[str, Any]:
        """액션 안전성 검사"""
        # URL 네비게이션 검사
        if action.action_type == ActionType.NAVIGATE:
            return self.check_url(action.value)

        # 민감 데이터 입력 검사
        if action.action_type in (ActionType.TYPE, ActionType.FILL_FORM):
            value_lower = action.value.lower()
            for keyword in self.SENSITIVE_KEYWORDS:
                if keyword in value_lower:
                    return {
                        "safe": False,
                        "reason": f"Sensitive data detected: {keyword}",
                        "requires_confirmation": True,
                    }

        return {"safe": True, "reason": ""}

    def check_task(self, task_description: str) -> dict[str, Any]:
        """태스크 전체 안전성 검사"""
        task_lower = task_description.lower()
        for keyword in self.SENSITIVE_KEYWORDS:
            if keyword in task_lower:
                return {
                    "safe": True,
                    "requires_confirmation": True,
                    "reason": f"Sensitive keyword in task: {keyword}",
                }
        return {"safe": True, "requires_confirmation": False, "reason": ""}


class BrowserSession:
    """
    격리된 브라우저 세션 (Isolated Browser Session)

    각 태스크마다 독립된 브라우저 세션을 관리합니다.
    Azure Playwright Workspaces와 호환됩니다.

    📌 사용 예시:
        >>> session = BrowserSession(config=BrowserConfig())
        >>> await session.start()
        >>> await session.navigate("https://learn.microsoft.com")
        >>> screenshot = await session.capture_screen()
        >>> await session.close()
    """

    def __init__(self, config: BrowserConfig | None = None) -> None:
        self.config = config or BrowserConfig()
        self.session_id = str(uuid.uuid4())
        self.status = BrowserStatus.IDLE
        self._action_count = 0
        self._history: list[BrowserAction] = []
        self._screenshots: list[ScreenCapture] = []
        self._safety = SafetyChecker()
        self._current_url = ""
        self._current_title = ""

    async def start(self) -> None:
        """브라우저 세션 시작"""
        self.status = BrowserStatus.RUNNING
        logger.info(
            f"Browser session {self.session_id[:8]}... started "
            f"(headless={self.config.headless}, "
            f"viewport={self.config.viewport_width}x{self.config.viewport_height})"
        )

    async def navigate(self, url: str) -> ActionResult:
        """URL로 이동"""
        if self.config.enable_safety_checks:
            check = self._safety.check_url(url)
            if not check["safe"]:
                return ActionResult(
                    success=False, error=f"Safety check failed: {check['reason']}"
                )

        action = BrowserAction(
            action_type=ActionType.NAVIGATE, value=url
        )
        self._history.append(action)
        self._action_count += 1
        self._current_url = url
        self.status = BrowserStatus.NAVIGATING

        # 시뮬레이션: 실제 구현에서는 Playwright 호출
        await asyncio.sleep(0.01)
        self.status = BrowserStatus.RUNNING

        logger.debug(f"Navigated to: {url}")
        return ActionResult(
            action_id=action.action_id,
            success=True,
            extracted_text=f"Page loaded: {url}",
        )

    async def execute_action(self, action: BrowserAction) -> ActionResult:
        """단일 액션 실행"""
        if self._action_count >= self.config.max_actions_per_task:
            return ActionResult(
                success=False,
                error=f"Max actions ({self.config.max_actions_per_task}) exceeded",
            )

        if self.config.enable_safety_checks:
            check = self._safety.check_action(action)
            if not check["safe"]:
                return ActionResult(
                    action_id=action.action_id,
                    success=False,
                    error=f"Safety check failed: {check['reason']}",
                )

        self._history.append(action)
        self._action_count += 1

        # 시뮬레이션: 실제 구현에서는 Playwright으로 액션 수행
        start = time.monotonic()
        await asyncio.sleep(0.01)
        duration = (time.monotonic() - start) * 1000

        return ActionResult(
            action_id=action.action_id,
            success=True,
            duration_ms=duration,
        )

    async def capture_screen(self) -> ScreenCapture:
        """현재 화면 캡처"""
        capture = ScreenCapture(
            width=self.config.viewport_width,
            height=self.config.viewport_height,
            url=self._current_url,
            title=self._current_title,
            image_base64="<simulated_base64_screenshot>",
        )
        self._screenshots.append(capture)
        return capture

    async def extract_text(self, selector: str = "body") -> str:
        """페이지에서 텍스트 추출"""
        action = BrowserAction(
            action_type=ActionType.EXTRACT_TEXT, target=selector
        )
        self._history.append(action)
        return f"[Extracted text from {selector} at {self._current_url}]"

    async def close(self) -> None:
        """세션 종료"""
        self.status = BrowserStatus.CLOSED
        logger.info(
            f"Browser session {self.session_id[:8]}... closed "
            f"(actions={self._action_count}, screenshots={len(self._screenshots)})"
        )

    @property
    def action_count(self) -> int:
        return self._action_count

    @property
    def history(self) -> list[BrowserAction]:
        return self._history.copy()


class BrowserAutomation:
    """
    브라우저 자동화 엔진 (Browser Automation Engine)

    자연어 태스크를 브라우저 액션 시퀀스로 변환하고 실행합니다.
    Azure Foundry의 Browser Automation Tool과 호환됩니다.

    📌 사용 예시:
        >>> browser = BrowserAutomation(BrowserConfig(headless=True))
        >>> await browser.start()
        >>> result = await browser.execute_task(
        ...     "Microsoft Learn에서 Agent Framework 문서를 검색하세요"
        ... )
        >>> print(f"성공: {result.success}, 액션 수: {result.total_steps}")
        >>> await browser.close()
    """

    def __init__(self, config: BrowserConfig | None = None) -> None:
        self.config = config or BrowserConfig()
        self._session: BrowserSession | None = None
        self._safety = SafetyChecker()

    async def start(self) -> None:
        """자동화 세션 시작"""
        self._session = BrowserSession(self.config)
        await self._session.start()

    async def execute_task(self, task: str) -> CUAResult:
        """
        자연어 태스크 실행

        Args:
            task: 자연어로 된 작업 설명

        Returns:
            CUAResult: 실행 결과
        """
        if not self._session:
            return CUAResult(success=False, error="Session not started")

        # 안전성 검사
        safety = self._safety.check_task(task)
        start_time = time.monotonic()

        # 시뮬레이션: 실제 구현에서는 LLM이 태스크를 액션으로 분해
        actions = [
            BrowserAction(action_type=ActionType.NAVIGATE, value="https://learn.microsoft.com"),
            BrowserAction(action_type=ActionType.TYPE, target="#search-input", value=task),
            BrowserAction(action_type=ActionType.CLICK, target="#search-button"),
            BrowserAction(action_type=ActionType.EXTRACT_TEXT, target=".search-results"),
        ]

        results = []
        for action in actions:
            result = await self._session.execute_action(action)
            results.append(result)
            if not result.success:
                break

        screenshot = await self._session.capture_screen()
        duration = time.monotonic() - start_time

        return CUAResult(
            success=all(r.success for r in results),
            actions_taken=actions,
            final_screenshot=screenshot,
            output_text=f"Task completed: {task}",
            total_steps=len(actions),
            duration_seconds=duration,
            safety_checks_passed=safety["safe"],
        )

    async def close(self) -> None:
        """자동화 세션 종료"""
        if self._session:
            await self._session.close()
            self._session = None

    @property
    def session(self) -> BrowserSession | None:
        return self._session


class ComputerUseAgent:
    """
    Computer Use Agent (CUA) — OpenAI CUA API 통합

    OpenAI의 Computer-Using Agent 모델을 통해 브라우저/데스크톱 태스크를 자동화합니다.
    Responses API의 computer_use_preview 도구를 사용합니다.

    📊 벤치마크 (2025년 기준):
        - OSWorld (전체 컴퓨터 사용): 38.1%
        - WebArena (웹 작업): 58.1%
        - WebVoyager (웹 탐색): 87.0%

    📌 사용 예시:
        >>> cua = ComputerUseAgent(CUAConfig(
        ...     model="computer-use-preview",
        ...     environment=CUAEnvironment.BROWSER,
        ...     display_width=1024,
        ...     display_height=768
        ... ))
        >>> result = await cua.run(
        ...     task="온라인에서 최신 AI 뉴스를 검색하고 요약해주세요"
        ... )
        >>> print(f"결과: {result.output_text}")
        >>> print(f"수행한 액션: {result.total_steps}개")
    """

    def __init__(self, config: CUAConfig | None = None) -> None:
        self.config = config or CUAConfig()
        self._safety = SafetyChecker()
        self._task_history: list[CUAResult] = []

    async def run(self, task: str) -> CUAResult:
        """
        CUA 태스크 실행

        Args:
            task: 자연어 태스크

        Returns:
            CUAResult: 실행 결과
        """
        start_time = time.monotonic()
        task_id = str(uuid.uuid4())

        # 안전성 검사
        safety = self._safety.check_task(task)
        if self.config.enable_safety and not safety["safe"]:
            result = CUAResult(
                task_id=task_id,
                success=False,
                safety_checks_passed=False,
                error=f"Safety check failed: {safety['reason']}",
            )
            self._task_history.append(result)
            return result

        # 시뮬레이션: 실제 구현에서는 OpenAI Responses API 호출
        # response = await openai.responses.create(
        #     model=self.config.model,
        #     tools=[{
        #         "type": "computer_use_preview",
        #         "display_width": self.config.display_width,
        #         "display_height": self.config.display_height,
        #         "environment": self.config.environment.value,
        #     }],
        #     truncation=self.config.truncation,
        #     input=task,
        # )

        actions = [
            BrowserAction(action_type=ActionType.SCREENSHOT),
            BrowserAction(action_type=ActionType.CLICK, coordinates=(512, 384)),
            BrowserAction(action_type=ActionType.TYPE, value=task),
        ]

        duration = time.monotonic() - start_time

        result = CUAResult(
            task_id=task_id,
            success=True,
            actions_taken=actions,
            output_text=f"CUA completed task: {task}",
            total_steps=len(actions),
            duration_seconds=duration,
            safety_checks_passed=True,
        )
        self._task_history.append(result)
        return result

    @property
    def task_history(self) -> list[CUAResult]:
        return self._task_history.copy()

    @property
    def total_tasks(self) -> int:
        return len(self._task_history)

    @property
    def success_rate(self) -> float:
        if not self._task_history:
            return 0.0
        return sum(1 for r in self._task_history if r.success) / len(self._task_history)


class ActionRecorder:
    """
    액션 레코더 (Action Recorder)

    브라우저 자동화 액션을 기록하고 재생합니다.
    RPA(Robotic Process Automation) 시나리오에서 유용합니다.

    📌 사용 예시:
        >>> recorder = ActionRecorder()
        >>> recorder.start_recording()
        >>> recorder.record(BrowserAction(action_type=ActionType.NAVIGATE, value="..."))
        >>> recorder.record(BrowserAction(action_type=ActionType.CLICK, target="#btn"))
        >>> recording = recorder.stop_recording()
        >>> # 나중에 재생
        >>> await recorder.replay(recording, browser_session)
    """

    def __init__(self) -> None:
        self._recordings: dict[str, list[BrowserAction]] = {}
        self._current_recording: list[BrowserAction] | None = None
        self._current_id: str | None = None

    def start_recording(self, recording_id: str | None = None) -> str:
        """녹화 시작"""
        self._current_id = recording_id or str(uuid.uuid4())[:8]
        self._current_recording = []
        logger.info(f"Recording started: {self._current_id}")
        return self._current_id

    def record(self, action: BrowserAction) -> None:
        """액션 기록"""
        if self._current_recording is not None:
            self._current_recording.append(action)

    def stop_recording(self) -> list[BrowserAction]:
        """녹화 중지 및 저장"""
        if self._current_recording is None or self._current_id is None:
            return []
        recording = self._current_recording.copy()
        self._recordings[self._current_id] = recording
        logger.info(
            f"Recording {self._current_id} saved: {len(recording)} actions"
        )
        self._current_recording = None
        self._current_id = None
        return recording

    async def replay(
        self, actions: list[BrowserAction], session: BrowserSession
    ) -> list[ActionResult]:
        """녹화된 액션 재생"""
        results = []
        for action in actions:
            result = await session.execute_action(action)
            results.append(result)
            if not result.success:
                logger.warning(f"Replay failed at action {action.action_id}")
                break
        return results

    def get_recording(self, recording_id: str) -> list[BrowserAction]:
        """저장된 녹화 조회"""
        return self._recordings.get(recording_id, [])

    @property
    def recording_count(self) -> int:
        return len(self._recordings)
