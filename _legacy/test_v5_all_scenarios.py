#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Unified Agent Framework v5 -- 전체 시나리오별 테스트
=====================================================
API 키 없이 실행 가능한 유닛/통합 테스트 (28개 시나리오)

실행: python test_v5_all_scenarios.py
"""
from __future__ import annotations

import asyncio
import json
import logging
import sys
import time
import traceback
import uuid
from dataclasses import fields as dc_fields
from io import StringIO
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

# UTF-8 stdout (Windows cp949 방지)
if sys.stdout.encoding != "utf-8":
    sys.stdout.reconfigure(encoding="utf-8")  # type: ignore[attr-defined]

# ── v5 패키지 임포트 ────────────────────────────────────────────
from unified_agent_v5 import (
    AgentConfig,
    AgentResult,
    CallbackHandler,
    CompositeCallbackHandler,
    LoggingCallbackHandler,
    Memory,
    Message,
    Role,
    Runner,
    Settings,
    StreamChunk,
    Tool,
    ToolCall,
    ToolRegistry,
    ToolResult,
    fire_callbacks,
    get_engine,
    mcp_tool,
    run_agent,
    stream_agent,
)

# ═══════════════════════════════════════════════════════════════════
# 테스트 유틸
# ═══════════════════════════════════════════════════════════════════
_results: list[tuple[str, bool, str]] = []


def _record(name: str, passed: bool, detail: str = ""):
    _results.append((name, passed, detail))
    status = "PASS" if passed else "FAIL"
    print(f"  [{status}] {name}")
    if not passed and detail:
        print(f"         -> {detail}")


async def _run(name: str, coro):
    """비동기 테스트 래퍼"""
    try:
        await coro
    except Exception as e:
        _record(name, False, f"Exception: {e}\n{traceback.format_exc()}")


# ═══════════════════════════════════════════════════════════════════
# 시나리오 1: Role Enum
# ═══════════════════════════════════════════════════════════════════
def test_01_role_enum():
    """Role Enum 값/비교/문자열 호환"""
    try:
        assert Role.SYSTEM == "system"
        assert Role.USER == "user"
        assert Role.ASSISTANT == "assistant"
        assert Role.TOOL == "tool"
        assert str(Role.USER) == "Role.USER" or Role.USER.value == "user"
        assert Role("user") == Role.USER
        _record("01 Role Enum 기본", True)
    except Exception as e:
        _record("01 Role Enum 기본", False, str(e))


# ═══════════════════════════════════════════════════════════════════
# 시나리오 2: Message 생성/직렬화
# ═══════════════════════════════════════════════════════════════════
def test_02_message():
    """Message 팩토리 메서드, to_dict, 필드 생략"""
    try:
        # 팩토리 메서드
        m1 = Message.system("You are helpful.")
        assert m1.role == Role.SYSTEM and m1.content == "You are helpful."

        m2 = Message.user("Hello")
        assert m2.role == Role.USER

        tc = ToolCall(id="tc-1", name="search", arguments={"q": "ai"})
        m3 = Message.assistant("OK", tool_calls=[tc])
        assert m3.tool_calls == [tc]

        m4 = Message.tool("result", tool_call_id="tc-1")
        assert m4.tool_call_id == "tc-1" and m4.role == Role.TOOL

        # to_dict: None 필드 생략
        d = m2.to_dict()
        assert "name" not in d
        assert "tool_call_id" not in d
        assert d["role"] == "user"
        assert d["content"] == "Hello"

        # tool_calls 있으면 포함
        d3 = m3.to_dict()
        assert "tool_calls" in d3

        _record("02 Message 생성/직렬화", True)
    except Exception as e:
        _record("02 Message 생성/직렬화", False, str(e))


# ═══════════════════════════════════════════════════════════════════
# 시나리오 3: ToolCall / ToolResult
# ═══════════════════════════════════════════════════════════════════
def test_03_toolcall_toolresult():
    """ToolCall.to_dict, ToolResult.to_message"""
    try:
        tc = ToolCall(id="tc-1", name="get_weather", arguments={"city": "Seoul"})
        d = tc.to_dict()
        assert d["type"] == "function"
        assert d["function"]["name"] == "get_weather"
        # arguments가 JSON 문자열로 직렬화
        args_str = d["function"]["arguments"]
        assert json.loads(args_str) == {"city": "Seoul"}

        tr = ToolResult(tool_call_id="tc-1", name="get_weather", content="맑음 22C")
        msg = tr.to_message()
        assert msg.role == Role.TOOL
        assert msg.content == "맑음 22C"
        assert msg.tool_call_id == "tc-1"

        # is_error
        tr_err = ToolResult(tool_call_id="tc-2", name="fail", content="Error", is_error=True)
        assert tr_err.is_error is True

        _record("03 ToolCall/ToolResult", True)
    except Exception as e:
        _record("03 ToolCall/ToolResult", False, str(e))


# ═══════════════════════════════════════════════════════════════════
# 시나리오 4: AgentResult
# ═══════════════════════════════════════════════════════════════════
def test_04_agent_result():
    """AgentResult 기본값, request_id 자동 생성"""
    try:
        r = AgentResult(content="Hello", model="gpt-5.2", engine="direct")
        assert r.content == "Hello"
        assert isinstance(r.request_id, str) and len(r.request_id) > 0
        assert r.usage == {}
        assert r.tool_calls == []
        assert r.messages == []
        assert r.duration_ms == 0.0
        assert r.metadata == {}

        _record("04 AgentResult 기본값", True)
    except Exception as e:
        _record("04 AgentResult 기본값", False, str(e))


# ═══════════════════════════════════════════════════════════════════
# 시나리오 5: StreamChunk
# ═══════════════════════════════════════════════════════════════════
def test_05_stream_chunk():
    """StreamChunk 필드"""
    try:
        c1 = StreamChunk(content="hello")
        assert c1.is_final is False
        c2 = StreamChunk(content="", is_final=True)
        assert c2.is_final is True
        _record("05 StreamChunk", True)
    except Exception as e:
        _record("05 StreamChunk", False, str(e))


# ═══════════════════════════════════════════════════════════════════
# 시나리오 6: Settings (환경 변수 / Reasoning 모델)
# ═══════════════════════════════════════════════════════════════════
def test_06_settings():
    """Settings.supports_temperature, REASONING_MODELS"""
    try:
        # 일반 모델 -> temperature 지원
        assert Settings.supports_temperature("gpt-4o") is True
        assert Settings.supports_temperature("gpt-4o-mini") is True

        # Reasoning 모델 -> temperature 비지원
        assert Settings.supports_temperature("o1-preview") is False
        assert Settings.supports_temperature("o3-mini") is False
        assert Settings.supports_temperature("deepseek-r1") is False

        # gpt-5, gpt-5.1, gpt-5.2 -> reasoning
        assert Settings.supports_temperature("gpt-5") is False
        assert Settings.supports_temperature("gpt-5.1") is False
        assert Settings.supports_temperature("gpt-5.2") is False

        _record("06 Settings/Reasoning 모델", True)
    except Exception as e:
        _record("06 Settings/Reasoning 모델", False, str(e))


# ═══════════════════════════════════════════════════════════════════
# 시나리오 7: AgentConfig 기본값 / post_init
# ═══════════════════════════════════════════════════════════════════
def test_07_agent_config_defaults():
    """AgentConfig 환경 변수 폴백, slots, post_init"""
    try:
        cfg = AgentConfig()
        assert cfg.model == Settings.DEFAULT_MODEL
        assert cfg.engine == Settings.DEFAULT_ENGINE
        assert cfg.max_tokens == Settings.DEFAULT_MAX_TOKENS

        # from_env
        cfg2 = AgentConfig.from_env()
        assert cfg2.model == Settings.DEFAULT_MODEL

        # slots=True 확인
        assert hasattr(AgentConfig, "__slots__")

        _record("07 AgentConfig 기본값", True)
    except Exception as e:
        _record("07 AgentConfig 기본값", False, str(e))


# ═══════════════════════════════════════════════════════════════════
# 시나리오 8: AgentConfig reasoning 모델 temperature 자동 None
# ═══════════════════════════════════════════════════════════════════
def test_08_config_reasoning_temperature():
    """Reasoning 모델에서 temperature 자동 None 처리 (명시 안 한 경우만)"""
    try:
        # temperature 미지정 → reasoning 모델은 None으로 설정
        cfg = AgentConfig(model="o3-mini")
        assert cfg.temperature is None, f"Expected None, got {cfg.temperature}"

        # temperature 명시적 지정 → reasoning 모델이라도 유지 (사용자 의도 존중)
        cfg1b = AgentConfig(model="o3-mini", temperature=0.5)
        assert cfg1b.temperature == 0.5, "Explicit temperature should be preserved"

        # 일반 모델 + temperature 미지정 → DEFAULT_TEMPERATURE
        cfg2 = AgentConfig(model="gpt-4o")
        assert cfg2.temperature == Settings.DEFAULT_TEMPERATURE

        # 일반 모델 + 명시적 temperature
        cfg3 = AgentConfig(model="gpt-4o", temperature=0.3)
        assert cfg3.temperature == 0.3

        _record("08 Config reasoning temperature", True)
    except Exception as e:
        _record("08 Config reasoning temperature", False, str(e))


# ═══════════════════════════════════════════════════════════════════
# 시나리오 9: AgentConfig max_tool_rounds 검증
# ═══════════════════════════════════════════════════════════════════
def test_09_config_max_tool_rounds_validation():
    """max_tool_rounds < 1 이면 ValueError"""
    try:
        try:
            AgentConfig(max_tool_rounds=0)
            _record("09 max_tool_rounds 검증", False, "ValueError가 발생해야 함")
            return
        except ValueError:
            pass

        try:
            AgentConfig(max_tool_rounds=-1)
            _record("09 max_tool_rounds 검증", False, "ValueError가 발생해야 함")
            return
        except ValueError:
            pass

        # 정상 값
        cfg = AgentConfig(max_tool_rounds=5)
        assert cfg.max_tool_rounds == 5

        _record("09 max_tool_rounds 검증", True)
    except Exception as e:
        _record("09 max_tool_rounds 검증", False, str(e))


# ═══════════════════════════════════════════════════════════════════
# 시나리오 10: AgentConfig get_api_key / get_base_url
# ═══════════════════════════════════════════════════════════════════
def test_10_config_api_key():
    """get_api_key: Azure 우선, get_base_url: Azure 우선 (환경 변수 격리)"""
    try:
        # 환경 변수 격리하여 테스트
        with patch.multiple(
            Settings,
            AZURE_ENDPOINT="",
            AZURE_API_KEY="",
            AZURE_DEPLOYMENT="",
            AZURE_API_VERSION="2025-12-01-preview",
            OPENAI_API_KEY="",
            OPENAI_BASE_URL="",
        ):
            cfg = AgentConfig(
                azure_api_key="az-key",
                azure_endpoint="https://az.endpoint/",
                openai_api_key="oai-key",
                openai_base_url="https://api.openai.com/v1",
            )
            assert cfg.get_api_key() == "az-key"
            assert cfg.get_base_url() == "https://az.endpoint/"

            cfg2 = AgentConfig(
                openai_api_key="oai-key",
                openai_base_url="https://custom.com/v1",
            )
            assert cfg2.get_api_key() == "oai-key", f"Got: {cfg2.get_api_key()}"
            assert cfg2.get_base_url() == "https://custom.com/v1", f"Got: {cfg2.get_base_url()}"

        _record("10 Config API Key/URL", True)
    except Exception as e:
        _record("10 Config API Key/URL", False, str(e))


# ═══════════════════════════════════════════════════════════════════
# 시나리오 11: Memory CRUD
# ═══════════════════════════════════════════════════════════════════
def test_11_memory_crud():
    """Memory 추가/조회/삭제/길이"""
    try:
        mem = Memory(system_prompt="System", max_messages=100)
        assert len(mem) == 1  # system prompt만

        mem.add_user("Hello")
        mem.add_assistant("Hi there!")
        assert len(mem) == 3

        # messages (system 포함)
        assert mem.messages[0].role == Role.SYSTEM
        assert mem.messages[1].content == "Hello"

        # history (system 제외)
        assert len(mem.history) == 2
        assert mem.history[0].role == Role.USER

        # clear
        mem.clear()
        assert len(mem) == 1  # system만 남음
        assert len(mem.history) == 0

        _record("11 Memory CRUD", True)
    except Exception as e:
        _record("11 Memory CRUD", False, str(e))


# ═══════════════════════════════════════════════════════════════════
# 시나리오 12: Memory 슬라이딩 윈도우
# ═══════════════════════════════════════════════════════════════════
def test_12_memory_sliding_window():
    """max_messages 초과 시 오래된 메시지 삭제"""
    try:
        mem = Memory(system_prompt="Sys", max_messages=3)
        for i in range(10):
            mem.add_user(f"msg-{i}")

        # history는 max_messages개 이하
        assert len(mem.history) <= 3

        # 가장 최근 메시지가 남아야 함
        last = mem.history[-1]
        assert "msg-9" in last.content

        _record("12 Memory 슬라이딩 윈도우", True)
    except Exception as e:
        _record("12 Memory 슬라이딩 윈도우", False, str(e))


# ═══════════════════════════════════════════════════════════════════
# 시나리오 13: Memory JSON 직렬화/역직렬화
# ═══════════════════════════════════════════════════════════════════
def test_13_memory_json():
    """to_json / from_json 라운드트립"""
    try:
        mem = Memory(system_prompt="Expert", max_messages=50)
        mem.add_user("Question")
        mem.add_assistant("Answer")

        data = mem.to_json()
        mem2 = Memory.from_json(data)

        assert mem2.system_prompt == "Expert"
        assert len(mem2.history) == 2
        assert mem2.history[0].content == "Question"
        assert mem2.history[1].content == "Answer"

        _record("13 Memory JSON 라운드트립", True)
    except Exception as e:
        _record("13 Memory JSON 라운드트립", False, str(e))


# ═══════════════════════════════════════════════════════════════════
# 시나리오 14: Memory JSON - tool_calls 복원
# ═══════════════════════════════════════════════════════════════════
def test_14_memory_json_tool_calls():
    """from_json에서 tool_calls (str/dict arguments) 올바르게 복원"""
    try:
        mem = Memory(system_prompt="Sys")
        tc = ToolCall(id="tc-1", name="search", arguments={"q": "test"})
        msg = Message.assistant("Calling tool", tool_calls=[tc])
        mem.add(msg)

        data = mem.to_json()
        mem2 = Memory.from_json(data)

        restored = mem2.history[0]
        assert restored.tool_calls is not None
        assert len(restored.tool_calls) == 1
        assert restored.tool_calls[0].name == "search"
        assert restored.tool_calls[0].arguments == {"q": "test"}

        _record("14 Memory JSON tool_calls", True)
    except Exception as e:
        _record("14 Memory JSON tool_calls", False, str(e))


# ═══════════════════════════════════════════════════════════════════
# 시나리오 15: Memory get_messages 최대 제한
# ═══════════════════════════════════════════════════════════════════
def test_15_memory_get_messages():
    """get_messages(max_messages=N) 읽기 시 제한"""
    try:
        mem = Memory(system_prompt="Sys", max_messages=100)
        for i in range(20):
            mem.add_user(f"msg-{i}")

        # 전체
        all_msgs = mem.get_messages()
        assert all_msgs[0]["role"] == "system"
        assert len(all_msgs) == 21  # 1 system + 20 user

        # 제한
        limited = mem.get_messages(max_messages=5)
        assert len(limited) == 6  # 1 system + 5 recent
        assert limited[0]["role"] == "system"

        _record("15 Memory get_messages 제한", True)
    except Exception as e:
        _record("15 Memory get_messages 제한", False, str(e))


# ═══════════════════════════════════════════════════════════════════
# 시나리오 16: Tool 기본 / execute
# ═══════════════════════════════════════════════════════════════════
def test_16_tool_basic():
    """Tool 생성, execute, fn=None 시 에러"""
    try:
        async def search(query: str) -> str:
            return f"Result: {query}"

        tool = Tool(name="search", description="Search", fn=search)
        result = asyncio.run(tool.execute(query="AI"))
        assert result == "Result: AI"

        # fn=None → RuntimeError
        tool_no_fn = Tool(name="noop", description="No function")
        try:
            asyncio.run(tool_no_fn.execute())
            _record("16 Tool 기본/execute", False, "RuntimeError가 발생해야 함")
            return
        except RuntimeError:
            pass

        _record("16 Tool 기본/execute", True)
    except Exception as e:
        _record("16 Tool 기본/execute", False, str(e))


# ═══════════════════════════════════════════════════════════════════
# 시나리오 17: Tool OpenAI 스키마
# ═══════════════════════════════════════════════════════════════════
def test_17_tool_openai_schema():
    """to_openai_schema 구조 검증, 캐싱"""
    try:
        tool = Tool(
            name="calc",
            description="Calculator",
            parameters={
                "a": {"type": "integer", "description": "First number"},
                "b": {"type": "integer", "description": "Second number"},
            },
        )
        schema = tool.to_openai_schema()
        assert schema["type"] == "function"
        assert schema["function"]["name"] == "calc"
        assert "a" in schema["function"]["parameters"]["properties"]
        assert "a" in schema["function"]["parameters"]["required"]

        # 캐싱: 두 번째 호출 시 같은 객체
        schema2 = tool.to_openai_schema()
        assert schema is schema2

        _record("17 Tool OpenAI 스키마", True)
    except Exception as e:
        _record("17 Tool OpenAI 스키마", False, str(e))


# ═══════════════════════════════════════════════════════════════════
# 시나리오 18: @mcp_tool 데코레이터
# ═══════════════════════════════════════════════════════════════════
def test_18_mcp_tool_decorator():
    """@mcp_tool 타입 추론, 설명 추출, optional 파라미터"""
    try:
        @mcp_tool(description="날씨 조회")
        async def get_weather(city: str, unit: str = "celsius") -> str:
            """날씨를 조회합니다.

            city: 도시 이름
            unit: 온도 단위
            """
            return f"{city}: 맑음"

        # 반환 타입은 Tool
        assert isinstance(get_weather, Tool)
        assert get_weather.name == "get_weather"
        assert get_weather.description == "날씨 조회"

        # 파라미터 추출
        schema = get_weather.to_openai_schema()
        params = schema["function"]["parameters"]
        assert "city" in params["properties"]
        assert params["properties"]["city"]["type"] == "string"

        # optional 파라미터: required에 없어야 함
        assert "city" in params["required"]
        assert "unit" not in params["required"]

        # 실행
        r = asyncio.run(get_weather.execute(city="Seoul"))
        assert "Seoul" in r

        _record("18 @mcp_tool 데코레이터", True)
    except Exception as e:
        _record("18 @mcp_tool 데코레이터", False, str(e))


# ═══════════════════════════════════════════════════════════════════
# 시나리오 19: @mcp_tool 독스트링 파라미터 설명
# ═══════════════════════════════════════════════════════════════════
def test_19_mcp_tool_docstring():
    """독스트링에서 파라미터 설명 추출"""
    try:
        @mcp_tool()
        async def translate(text: str, target_lang: str = "ko") -> str:
            """텍스트를 번역합니다.

            text: 번역할 텍스트
            target_lang: 대상 언어 코드
            """
            return f"Translated: {text}"

        schema = translate.to_openai_schema()
        props = schema["function"]["parameters"]["properties"]
        assert "description" in props["text"]
        assert "번역" in props["text"]["description"]

        # 함수 설명 = 독스트링 첫 줄
        assert "번역" in schema["function"]["description"]

        _record("19 @mcp_tool 독스트링", True)
    except Exception as e:
        _record("19 @mcp_tool 독스트링", False, str(e))


# ═══════════════════════════════════════════════════════════════════
# 시나리오 20: ToolRegistry
# ═══════════════════════════════════════════════════════════════════
def test_20_tool_registry():
    """ToolRegistry 등록/조회/삭제/스키마"""
    try:
        reg = ToolRegistry()

        @mcp_tool(description="도구A")
        async def tool_a(x: str) -> str:
            return x

        @mcp_tool(description="도구B")
        async def tool_b(y: int) -> str:
            return str(y)

        reg.register(tool_a)
        reg.register(tool_b)

        assert len(reg) == 2
        assert "tool_a" in reg
        assert reg.get("tool_a") is tool_a

        schemas = reg.get_openai_schemas()
        assert len(schemas) == 2

        reg.unregister("tool_a")
        assert len(reg) == 1
        assert "tool_a" not in reg

        # 없는 도구 삭제 (에러 없음)
        reg.unregister("nonexistent")

        _record("20 ToolRegistry", True)
    except Exception as e:
        _record("20 ToolRegistry", False, str(e))


# ═══════════════════════════════════════════════════════════════════
# 시나리오 21: CallbackHandler 기본 (no-op)
# ═══════════════════════════════════════════════════════════════════
async def test_21_callback_noop():
    """CallbackHandler 모든 메서드가 no-op으로 동작"""
    try:
        cb = CallbackHandler()
        await cb.on_agent_start("test")
        await cb.on_llm_start("model", [])
        await cb.on_llm_end("content", {})
        tc = ToolCall(id="t1", name="tool", arguments={})
        await cb.on_tool_start(tc)
        await cb.on_tool_end(ToolResult(tool_call_id="t1", name="tool", content="ok"))
        await cb.on_agent_end(AgentResult(content="done", model="m", engine="e"))
        await cb.on_agent_error(Exception("test"))
        _record("21 CallbackHandler no-op", True)
    except Exception as e:
        _record("21 CallbackHandler no-op", False, str(e))


# ═══════════════════════════════════════════════════════════════════
# 시나리오 22: LoggingCallbackHandler
# ═══════════════════════════════════════════════════════════════════
async def test_22_logging_callback():
    """LoggingCallbackHandler가 로그를 출력하는지 확인"""
    try:
        handler = logging.StreamHandler(StringIO())
        handler.setLevel(logging.DEBUG)
        logger = logging.getLogger("unified_agent_v5.callback")
        logger.addHandler(handler)
        logger.setLevel(logging.DEBUG)

        cb = LoggingCallbackHandler(level=logging.INFO)
        await cb.on_agent_start("Test task")
        await cb.on_llm_start("gpt-4o", [{"role": "user", "content": "hi"}])
        await cb.on_llm_end("response", {"total_tokens": 42})
        await cb.on_agent_end(AgentResult(content="done", model="gpt-4o", engine="direct"))

        output = handler.stream.getvalue()
        assert "Agent started" in output or "agent" in output.lower()

        logger.removeHandler(handler)
        _record("22 LoggingCallbackHandler", True)
    except Exception as e:
        _record("22 LoggingCallbackHandler", False, str(e))


# ═══════════════════════════════════════════════════════════════════
# 시나리오 23: CompositeCallbackHandler
# ═══════════════════════════════════════════════════════════════════
async def test_23_composite_callback():
    """CompositeCallbackHandler: 다중 핸들러, 에러 격리"""
    try:
        events = []

        class TrackingHandler(CallbackHandler):
            async def on_agent_start(self, task, config=None, **kwargs):
                events.append(("start", task))

            async def on_agent_end(self, result, **kwargs):
                events.append(("end", result.content))

        class FailingHandler(CallbackHandler):
            async def on_agent_start(self, task, config=None, **kwargs):
                raise RuntimeError("Intentional failure")

        comp = CompositeCallbackHandler([TrackingHandler(), FailingHandler(), TrackingHandler()])
        await comp.on_agent_start("hello")

        # FailingHandler 에러가 나도 다른 핸들러는 실행
        starts = [e for e in events if e[0] == "start"]
        assert len(starts) == 2, f"Expected 2 starts, got {len(starts)}"

        r = AgentResult(content="done", model="m", engine="e")
        await comp.on_agent_end(r)
        ends = [e for e in events if e[0] == "end"]
        assert len(ends) == 2

        _record("23 CompositeCallbackHandler", True)
    except Exception as e:
        _record("23 CompositeCallbackHandler", False, str(e))


# ═══════════════════════════════════════════════════════════════════
# 시나리오 24: fire_callbacks 에러 격리
# ═══════════════════════════════════════════════════════════════════
async def test_24_fire_callbacks():
    """fire_callbacks: 개별 핸들러 에러가 전파되지 않음"""
    try:
        events = []

        class GoodHandler(CallbackHandler):
            async def on_llm_start(self, model, messages, **kwargs):
                events.append(model)

        class BadHandler(CallbackHandler):
            async def on_llm_start(self, model, messages, **kwargs):
                raise ValueError("boom")

        handlers = [GoodHandler(), BadHandler(), GoodHandler()]
        await fire_callbacks(handlers, "on_llm_start", "gpt-4o", [])

        assert len(events) == 2, f"Expected 2, got {len(events)}"
        _record("24 fire_callbacks 에러 격리", True)
    except Exception as e:
        _record("24 fire_callbacks 에러 격리", False, str(e))


# ═══════════════════════════════════════════════════════════════════
# 시나리오 25: OTelCallbackHandler 그레이스풀 디그레이드
# ═══════════════════════════════════════════════════════════════════
async def test_25_otel_graceful():
    """OpenTelemetry 미설치 시 에러 없이 동작"""
    try:
        from unified_agent_v5.callback import OTelCallbackHandler
        otel = OTelCallbackHandler()
        # OTEL이 설치되어 있든 없든 에러 없이 동작해야 함
        await otel.on_agent_start("test")
        await otel.on_llm_start("model", [])
        await otel.on_llm_end("done", {})
        tc = ToolCall(id="t1", name="tool", arguments={})
        await otel.on_tool_start(tc)
        await otel.on_tool_end(ToolResult(tool_call_id="t1", name="tool", content="ok"))
        await otel.on_agent_end(AgentResult(content="done", model="m", engine="e"))
        await otel.on_agent_error(Exception("test"))
        _record("25 OTel 그레이스풀", True)
    except Exception as e:
        _record("25 OTel 그레이스풀", False, str(e))


# ═══════════════════════════════════════════════════════════════════
# 시나리오 26: Engine Registry
# ═══════════════════════════════════════════════════════════════════
def test_26_engine_registry():
    """get_engine: direct 항상 존재, 잘못된 이름 → ValueError"""
    try:
        engine = get_engine("direct")
        assert engine is not None
        # 프로토콜 확인
        assert hasattr(engine, "run")
        assert hasattr(engine, "stream")

        # 캐싱: 같은 인스턴스
        engine2 = get_engine("direct")
        assert engine is engine2

        # 없는 엔진 → ValueError
        try:
            get_engine("nonexistent_engine")
            _record("26 Engine Registry", False, "ValueError가 발생해야 함")
            return
        except ValueError as e:
            assert "nonexistent_engine" in str(e)

        _record("26 Engine Registry", True)
    except Exception as e:
        _record("26 Engine Registry", False, str(e))


# ═══════════════════════════════════════════════════════════════════
# 시나리오 27: DirectEngine 클라이언트 캐싱
# ═══════════════════════════════════════════════════════════════════
def test_27_direct_engine_client_cache():
    """동일 파라미터 → 같은 클라이언트 재사용"""
    try:
        from unified_agent_v5.engines.direct import DirectEngine

        engine = DirectEngine()
        kwargs1 = {"azure_endpoint": "https://test.openai.azure.com/", "azure_api_key": "key1"}
        client1 = engine._get_client(**kwargs1)
        client2 = engine._get_client(**kwargs1)
        assert client1 is client2, "같은 파라미터인데 다른 클라이언트 생성됨"

        # 다른 파라미터 → 다른 클라이언트
        kwargs2 = {"azure_endpoint": "https://test2.openai.azure.com/", "azure_api_key": "key2"}
        client3 = engine._get_client(**kwargs2)
        assert client3 is not client1

        _record("27 DirectEngine 클라이언트 캐싱", True)
    except Exception as e:
        _record("27 DirectEngine 클라이언트 캐싱", False, str(e))


# ═══════════════════════════════════════════════════════════════════
# 시나리오 28: DirectEngine.run() Mock 테스트 (도구 루프)
# ═══════════════════════════════════════════════════════════════════
async def test_28_direct_engine_tool_loop():
    """DirectEngine.run: 도구 호출 루프, 콜백, 결과 검증 (Mock API)"""
    try:
        from unified_agent_v5.engines.direct import DirectEngine

        # Mock: 첫 호출 → tool_call, 두 번째 호출 → 최종 응답
        call_count = {"n": 0}

        # 첫 번째 응답: tool_call
        tc_mock = MagicMock()
        tc_mock.id = "tc-100"
        tc_mock.function.name = "get_weather"
        tc_mock.function.arguments = '{"city": "Seoul"}'

        first_choice = MagicMock()
        first_choice.message.tool_calls = [tc_mock]
        first_choice.message.content = ""

        first_response = MagicMock()
        first_response.choices = [first_choice]

        # 두 번째 응답: 최종
        second_choice = MagicMock()
        second_choice.message.tool_calls = None
        second_choice.message.content = "Seoul weather: Sunny 22C"

        second_response = MagicMock()
        second_response.choices = [second_choice]
        second_response.usage.prompt_tokens = 100
        second_response.usage.completion_tokens = 20
        second_response.usage.total_tokens = 120

        async def mock_create(**kwargs):
            call_count["n"] += 1
            if call_count["n"] == 1:
                return first_response
            return second_response

        # 도구 정의
        @mcp_tool(description="날씨 조회")
        async def get_weather(city: str) -> str:
            return f"{city}: Sunny 22C"

        # 콜백 추적
        events = []

        class TrackingCB(CallbackHandler):
            async def on_llm_start(self, model, messages, **kwargs):
                events.append("llm_start")

            async def on_llm_end(self, content, usage=None, **kwargs):
                events.append(("llm_end", content))

            async def on_tool_start(self, tool_call, **kwargs):
                events.append(("tool_start", tool_call.name))

            async def on_tool_end(self, tool_result, **kwargs):
                events.append(("tool_end", tool_result.content))

        engine = DirectEngine()

        with patch.object(engine, "_get_client") as mock_get_client:
            mock_client = AsyncMock()
            mock_client.chat.completions.create = mock_create
            mock_get_client.return_value = mock_client

            result = await engine.run(
                messages=[{"role": "user", "content": "서울 날씨"}],
                model="gpt-4o",
                tools=[get_weather],
                callbacks=[TrackingCB()],
            )

        # 검증
        assert result.content == "Seoul weather: Sunny 22C"
        assert result.engine == "direct"
        assert len(result.tool_calls) == 1
        assert result.tool_calls[0].name == "get_weather"
        assert result.usage["total_tokens"] == 120
        assert result.duration_ms > 0

        # 콜백 순서 검증
        assert events[0] == "llm_start"
        assert events[1] == ("tool_start", "get_weather")
        assert "Sunny" in events[2][1]  # tool_end
        assert events[3][0] == "llm_end"

        # API 2회 호출
        assert call_count["n"] == 2

        _record("28 DirectEngine 도구 루프", True)
    except Exception as e:
        _record("28 DirectEngine 도구 루프", False, str(e))


# ═══════════════════════════════════════════════════════════════════
# 시나리오 29: DirectEngine tool not found
# ═══════════════════════════════════════════════════════════════════
async def test_29_direct_engine_tool_not_found():
    """존재하지 않는 도구 호출 시 에러 ToolResult 반환"""
    try:
        from unified_agent_v5.engines.direct import DirectEngine

        # Mock: unknown_tool 호출 → 최종 응답
        tc_mock = MagicMock()
        tc_mock.id = "tc-200"
        tc_mock.function.name = "unknown_tool"
        tc_mock.function.arguments = "{}"

        first_choice = MagicMock()
        first_choice.message.tool_calls = [tc_mock]
        first_choice.message.content = ""

        first_response = MagicMock()
        first_response.choices = [first_choice]

        second_choice = MagicMock()
        second_choice.message.tool_calls = None
        second_choice.message.content = "I couldn't find that tool."

        second_response = MagicMock()
        second_response.choices = [second_choice]
        second_response.usage.prompt_tokens = 10
        second_response.usage.completion_tokens = 5
        second_response.usage.total_tokens = 15

        call_n = {"n": 0}

        async def mock_create(**kwargs):
            call_n["n"] += 1
            return first_response if call_n["n"] == 1 else second_response

        engine = DirectEngine()
        with patch.object(engine, "_get_client") as mock_gc:
            mock_client = AsyncMock()
            mock_client.chat.completions.create = mock_create
            mock_gc.return_value = mock_client

            result = await engine.run(
                messages=[{"role": "user", "content": "test"}],
                model="gpt-4o",
                tools=[],  # 빈 도구 목록
            )

        assert len(result.tool_calls) == 1
        assert result.tool_calls[0].name == "unknown_tool"
        _record("29 DirectEngine tool not found", True)
    except Exception as e:
        _record("29 DirectEngine tool not found", False, str(e))


# ═══════════════════════════════════════════════════════════════════
# 시나리오 30: DirectEngine max_tool_rounds 초과
# ═══════════════════════════════════════════════════════════════════
async def test_30_max_tool_rounds():
    """max_tool_rounds 초과 시 [Max tool rounds exceeded] 반환"""
    try:
        from unified_agent_v5.engines.direct import DirectEngine

        # 항상 tool_calls 반환하는 mock
        tc_mock = MagicMock()
        tc_mock.id = "tc-loop"
        tc_mock.function.name = "loop_tool"
        tc_mock.function.arguments = "{}"

        choice = MagicMock()
        choice.message.tool_calls = [tc_mock]
        choice.message.content = ""

        response = MagicMock()
        response.choices = [choice]

        @mcp_tool(description="loops")
        async def loop_tool() -> str:
            return "looping"

        async def mock_create(**kwargs):
            return response

        engine = DirectEngine()
        with patch.object(engine, "_get_client") as mock_gc:
            mock_client = AsyncMock()
            mock_client.chat.completions.create = mock_create
            mock_gc.return_value = mock_client

            result = await engine.run(
                messages=[{"role": "user", "content": "loop"}],
                model="gpt-4o",
                tools=[loop_tool],
                max_tool_rounds=3,
            )

        assert "Max tool rounds exceeded" in result.content
        assert len(result.tool_calls) == 3  # 3 rounds

        _record("30 max_tool_rounds 초과", True)
    except Exception as e:
        _record("30 max_tool_rounds 초과", False, str(e))


# ═══════════════════════════════════════════════════════════════════
# 시나리오 31: DirectEngine JSON parse 오류 방어
# ═══════════════════════════════════════════════════════════════════
async def test_31_json_parse_error():
    """tool arguments JSON 파싱 실패 시 빈 dict로 진행"""
    try:
        from unified_agent_v5.engines.direct import DirectEngine

        tc_mock = MagicMock()
        tc_mock.id = "tc-bad"
        tc_mock.function.name = "bad_json_tool"
        tc_mock.function.arguments = "NOT VALID JSON {{{{"

        first_choice = MagicMock()
        first_choice.message.tool_calls = [tc_mock]
        first_choice.message.content = ""

        first_response = MagicMock()
        first_response.choices = [first_choice]

        second_choice = MagicMock()
        second_choice.message.tool_calls = None
        second_choice.message.content = "Handled gracefully"

        second_response = MagicMock()
        second_response.choices = [second_choice]
        second_response.usage.prompt_tokens = 5
        second_response.usage.completion_tokens = 3
        second_response.usage.total_tokens = 8

        @mcp_tool(description="bad json tool")
        async def bad_json_tool() -> str:
            return "executed"

        call_n = {"n": 0}

        async def mock_create(**kwargs):
            call_n["n"] += 1
            return first_response if call_n["n"] == 1 else second_response

        engine = DirectEngine()
        with patch.object(engine, "_get_client") as mock_gc:
            mock_client = AsyncMock()
            mock_client.chat.completions.create = mock_create
            mock_gc.return_value = mock_client

            result = await engine.run(
                messages=[{"role": "user", "content": "test"}],
                model="gpt-4o",
                tools=[bad_json_tool],
            )

        # JSON 파싱 실패해도 에러 없이 완료
        assert result.content == "Handled gracefully"
        assert result.tool_calls[0].arguments == {}

        _record("31 JSON parse 오류 방어", True)
    except Exception as e:
        _record("31 JSON parse 오류 방어", False, str(e))


# ═══════════════════════════════════════════════════════════════════
# 시나리오 32: Runner 기본 흐름 (Mock)
# ═══════════════════════════════════════════════════════════════════
async def test_32_runner_flow():
    """Runner.run: Memory 통합, 콜백, 결과 반환"""
    try:
        mock_result = AgentResult(
            content="Mocked response",
            model="gpt-4o",
            engine="direct",
            usage={"input_tokens": 10, "output_tokens": 5, "total_tokens": 15},
        )

        async def mock_engine_run(messages, model, tools=None, callbacks=None, **kwargs):
            return mock_result

        events = []

        class TrackCB(CallbackHandler):
            async def on_agent_start(self, task, config=None, **kwargs):
                events.append("start")

            async def on_agent_end(self, result, **kwargs):
                events.append("end")

        with patch("unified_agent_v5.runner.get_engine") as mock_ge:
            mock_engine = MagicMock()
            mock_engine.run = mock_engine_run
            mock_ge.return_value = mock_engine

            cfg = AgentConfig(model="gpt-4o", engine="direct")
            runner = Runner(cfg)
            mem = Memory(system_prompt="Test")

            result = await runner.run("Hello", memory=mem, callbacks=[TrackCB()])

        assert result.content == "Mocked response"
        assert "start" in events
        assert "end" in events

        # Memory에 대화가 기록됨
        assert len(mem.history) >= 2  # user + assistant
        assert mem.history[0].content == "Hello"
        assert mem.history[-1].content == "Mocked response"

        _record("32 Runner 기본 흐름", True)
    except Exception as e:
        _record("32 Runner 기본 흐름", False, str(e))


# ═══════════════════════════════════════════════════════════════════
# 시나리오 33: Runner 에러 시 on_agent_error 발생
# ═══════════════════════════════════════════════════════════════════
async def test_33_runner_error_callback():
    """엔진 에러 시 on_agent_error 호출 후 re-raise"""
    try:
        async def mock_engine_run(messages, model, tools=None, callbacks=None, **kwargs):
            raise ConnectionError("API unavailable")

        errors_caught = []

        class ErrorCB(CallbackHandler):
            async def on_agent_error(self, error, **kwargs):
                errors_caught.append(str(error))

        with patch("unified_agent_v5.runner.get_engine") as mock_ge:
            mock_engine = MagicMock()
            mock_engine.run = mock_engine_run
            mock_ge.return_value = mock_engine

            runner = Runner(AgentConfig(model="gpt-4o"))
            try:
                await runner.run("Hello", callbacks=[ErrorCB()])
                _record("33 Runner 에러 콜백", False, "ConnectionError가 발생해야 함")
                return
            except ConnectionError:
                pass

        assert len(errors_caught) == 1
        assert "API unavailable" in errors_caught[0]

        _record("33 Runner 에러 콜백", True)
    except Exception as e:
        _record("33 Runner 에러 콜백", False, str(e))


# ═══════════════════════════════════════════════════════════════════
# 시나리오 34: Runner.stream (Mock)
# ═══════════════════════════════════════════════════════════════════
async def test_34_runner_stream():
    """Runner.stream: 청크 수집, Memory 기록"""
    try:
        async def mock_engine_stream(messages, model, tools=None, callbacks=None, **kwargs):
            for text in ["Hello ", "World", "!"]:
                yield StreamChunk(content=text)
            yield StreamChunk(content="", is_final=True)

        with patch("unified_agent_v5.runner.get_engine") as mock_ge:
            mock_engine = MagicMock()
            mock_engine.stream = mock_engine_stream
            mock_ge.return_value = mock_engine

            cfg = AgentConfig(model="gpt-4o")
            runner = Runner(cfg)
            mem = Memory(system_prompt="Test")

            chunks = []
            async for chunk in runner.stream("Stream test", memory=mem):
                chunks.append(chunk)

        assert len(chunks) == 4  # 3 content + 1 final
        assert chunks[-1].is_final is True
        full = "".join(c.content for c in chunks)
        assert full == "Hello World!"

        # Memory에 기록
        assert mem.history[-1].content == "Hello World!"

        _record("34 Runner.stream", True)
    except Exception as e:
        _record("34 Runner.stream", False, str(e))


# ═══════════════════════════════════════════════════════════════════
# 시나리오 35: run_agent 편의 함수
# ═══════════════════════════════════════════════════════════════════
async def test_35_run_agent_convenience():
    """run_agent() 편의 함수가 Runner를 올바르게 생성하는지"""
    try:
        mock_result = AgentResult(content="OK", model="gpt-4o", engine="direct")

        with patch("unified_agent_v5.runner.Runner") as MockRunner:
            mock_instance = MagicMock()
            mock_run = AsyncMock(return_value=mock_result)
            mock_instance.run = mock_run
            MockRunner.return_value = mock_instance

            result = await run_agent("test question", model="gpt-4o")

        assert result.content == "OK"
        MockRunner.assert_called_once()

        _record("35 run_agent 편의 함수", True)
    except Exception as e:
        _record("35 run_agent 편의 함수", False, str(e))


# ═══════════════════════════════════════════════════════════════════
# 시나리오 36: __all__ export 완전성
# ═══════════════════════════════════════════════════════════════════
def test_36_all_exports():
    """__all__에 선언된 모든 심볼이 실제 import 가능"""
    try:
        import unified_agent_v5

        expected = set(unified_agent_v5.__all__)
        actual = {name for name in dir(unified_agent_v5) if not name.startswith("_")}

        missing = expected - actual
        assert not missing, f"__all__에 있지만 없는 심볼: {missing}"

        # 최소 20개
        assert len(expected) >= 20, f"__all__ 항목 수: {len(expected)}"

        _record("36 __all__ export 완전성", True)
    except Exception as e:
        _record("36 __all__ export 완전성", False, str(e))


# ═══════════════════════════════════════════════════════════════════
# 시나리오 37: Memory system_prompt 변경
# ═══════════════════════════════════════════════════════════════════
def test_37_memory_system_prompt_change():
    """system_prompt 프로퍼티 setter"""
    try:
        mem = Memory(system_prompt="Original")
        assert mem.system_prompt == "Original"

        mem.system_prompt = "Updated"
        assert mem.system_prompt == "Updated"

        # messages[0]이 변경됨
        assert mem.messages[0].content == "Updated"

        _record("37 Memory system_prompt 변경", True)
    except Exception as e:
        _record("37 Memory system_prompt 변경", False, str(e))


# ═══════════════════════════════════════════════════════════════════
# 시나리오 38: Memory add_tool_result
# ═══════════════════════════════════════════════════════════════════
def test_38_memory_tool_result():
    """Memory.add_tool_result"""
    try:
        mem = Memory()
        mem.add_tool_result("Weather: Sunny", tool_call_id="tc-1")

        last = mem.history[-1]
        assert last.role == Role.TOOL
        assert last.content == "Weather: Sunny"
        assert last.tool_call_id == "tc-1"

        _record("38 Memory add_tool_result", True)
    except Exception as e:
        _record("38 Memory add_tool_result", False, str(e))


# ═══════════════════════════════════════════════════════════════════
# RUNNER
# ═══════════════════════════════════════════════════════════════════
def main():
    print("=" * 70)
    print("  Unified Agent Framework v5 -- 전체 시나리오별 테스트")
    print("=" * 70)
    print()

    # ── 동기 테스트 ──
    sync_tests = [
        test_01_role_enum,
        test_02_message,
        test_03_toolcall_toolresult,
        test_04_agent_result,
        test_05_stream_chunk,
        test_06_settings,
        test_07_agent_config_defaults,
        test_08_config_reasoning_temperature,
        test_09_config_max_tool_rounds_validation,
        test_10_config_api_key,
        test_11_memory_crud,
        test_12_memory_sliding_window,
        test_13_memory_json,
        test_14_memory_json_tool_calls,
        test_15_memory_get_messages,
        test_16_tool_basic,
        test_17_tool_openai_schema,
        test_18_mcp_tool_decorator,
        test_19_mcp_tool_docstring,
        test_20_tool_registry,
        test_26_engine_registry,
        test_27_direct_engine_client_cache,
        test_36_all_exports,
        test_37_memory_system_prompt_change,
        test_38_memory_tool_result,
    ]

    # ── 비동기 테스트 ──
    async_tests = [
        ("21 CallbackHandler no-op", test_21_callback_noop),
        ("22 LoggingCallbackHandler", test_22_logging_callback),
        ("23 CompositeCallbackHandler", test_23_composite_callback),
        ("24 fire_callbacks 에러 격리", test_24_fire_callbacks),
        ("25 OTel 그레이스풀", test_25_otel_graceful),
        ("28 DirectEngine 도구 루프", test_28_direct_engine_tool_loop),
        ("29 DirectEngine tool not found", test_29_direct_engine_tool_not_found),
        ("30 max_tool_rounds 초과", test_30_max_tool_rounds),
        ("31 JSON parse 오류 방어", test_31_json_parse_error),
        ("32 Runner 기본 흐름", test_32_runner_flow),
        ("33 Runner 에러 콜백", test_33_runner_error_callback),
        ("34 Runner.stream", test_34_runner_stream),
        ("35 run_agent 편의 함수", test_35_run_agent_convenience),
    ]

    print("--- 동기 테스트 ---")
    for fn in sync_tests:
        fn()

    print()
    print("--- 비동기 테스트 ---")

    async def run_async_tests():
        for name, fn in async_tests:
            await _run(name, fn())

    asyncio.run(run_async_tests())

    # ── 결과 요약 ──
    print()
    print("=" * 70)
    total = len(_results)
    passed = sum(1 for _, ok, _ in _results if ok)
    failed = sum(1 for _, ok, _ in _results if not ok)

    if failed:
        print(f"\n  실패 목록:")
        for name, ok, detail in _results:
            if not ok:
                print(f"    [FAIL] {name}")
                if detail:
                    for line in detail.split("\n")[:5]:
                        print(f"           {line}")

    print()
    print(f"  결과: {passed}/{total} 통과" + (f"  ({failed}개 실패)" if failed else ""))
    print("=" * 70)

    return 0 if failed == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
