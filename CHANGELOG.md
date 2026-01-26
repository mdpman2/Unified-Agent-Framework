# 📝 Changelog

All notable changes to Unified Agent Framework will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [3.1.0] - 2026-01-26

### 🆕 Added

#### 최신 AI 모델 지원 (40+ 모델)
- **GPT-5.2 시리즈**: gpt-5.2, gpt-5.2-chat, gpt-5.2-codex (400K context)
- **GPT-5.1 Codex 시리즈**: gpt-5.1-codex, gpt-5.1-codex-mini, gpt-5.1-codex-max
- **Claude 4.5 시리즈**: claude-opus-4-5, claude-sonnet-4-5, claude-haiku-4-5 (Microsoft Foundry)
- **Grok-4 시리즈**: grok-4, grok-4-fast-reasoning (2M context), grok-4-fast-non-reasoning
- **o4-mini**: o3-mini 후속 Reasoning 모델
- **DeepSeek**: deepseek-v3.2, deepseek-v3.2-speciale, deepseek-r1-0528
- **Llama 4**: llama-4-maverick-17b, llama-4-scout-17b (10M context!)
- **Phi-4**: phi-4, phi-4-reasoning, phi-4-multimodal-instruct
- **Mistral**: mistral-large-3, mistral-medium-2505, mistral-small-2503

#### 유틸리티 함수
- `is_multimodal_model()`: 멀티모달(이미지/오디오 입력) 지원 모델 확인
- `is_large_context_model()`: 대용량 컨텍스트(100K+) 지원 모델 확인
- `get_model_context_window()`: 모델별 컨텍스트 윈도우 크기 반환

#### MCP (Model Context Protocol) 설정
- `ENABLE_MCP`: MCP 활성화 플래그
- `MCP_AUTO_CONNECT`: 자동 연결 설정
- `MCP_RECONNECT_ATTEMPTS`: 재연결 시도 횟수
- `MCP_REQUEST_TIMEOUT`: 요청 타임아웃
- `MCP_APPROVAL_MODE`: 승인 모드 (always/never/selective)

#### Multi-Agent 오케스트레이션 설정
- `ORCHESTRATION_MODE`: 오케스트레이션 모드 (supervisor/sequential/parallel/adaptive)
- `MAX_CONCURRENT_AGENTS`: 최대 동시 에이전트 수
- `ENABLE_HANDOFF`: 에이전트 간 Handoff 활성화
- `ENABLE_REFLECTION`: 반성(Reflection) 패턴 활성화

#### RAI (Responsible AI) 설정
- `ENABLE_RAI_VALIDATION`: RAI 검증 활성화
- `RAI_STRICT_MODE`: RAI 엄격 모드
- `RAI_CONTENT_SAFETY_LEVEL`: 콘텐츠 안전 레벨 (low/medium/high)
- `ENABLE_PII_DETECTION`: PII 감지 활성화

#### 상세 한글 주석
- 모든 12개 모듈에 상세한 한글 주석 추가
- 각 클래스/함수별 역할, 사용 예시, 주의사항 포함
- ASCII 다이어그램을 통한 상태 전환 설명 (CircuitBreaker)
- 참고 링크 및 관련 문서 연결

### 🔧 Changed

#### Adaptive Circuit Breaker 개선
- `success_threshold` 파라미터 추가 (HALF_OPEN → CLOSED 전환에 연속 성공 필요)
- `adaptive_timeout` 옵션 추가 (평균 응답 시간 기반 동적 타임아웃)
- 메트릭 수집 기능 추가 (`get_metrics()`)
- `reset()` 메서드 추가 (수동 리셋)

#### Settings 클래스 확장
- `DEFAULT_API_VERSION`: 2025-12-01-preview로 업데이트
- `DEFAULT_MAX_TOKENS`: 1000 → 4096으로 증가
- `DEFAULT_CONTEXT_WINDOW`: 200,000 토큰 기본값
- `MAX_SUPERVISOR_ROUNDS`: 5 → 10으로 증가
- `MAX_CACHE_SIZE`: 100 → 500으로 증가
- `MAX_MEMORY_TURNS`: 20 → 50으로 증가
- `SESSION_TTL_HOURS`: 24 → 72시간으로 증가
- `ENABLE_STREAMING`: False → True (기본 활성화)
- `ENABLE_PARALLEL_TOOLS`: 병렬 도구 호출 활성화
- `MAX_PARALLEL_TOOL_CALLS`: 최대 5개 동시 호출

### 📚 Documentation
- README.md 전면 업데이트 (2026년 1월 최신 모델 정보)
- 모든 docstring 한글 상세화
- GitHub 오픈소스 파일 추가 (LICENSE, CONTRIBUTING, etc.)

### ✅ Tests
- 79개 테스트 전체 통과 유지

---

## [3.0.0] - 2025-12-01

### 🆕 Added

#### 완전한 모듈화 아키텍처
- 12개 독립 모듈로 분리
- 67개 공개 API export
- 순환 참조 없는 깔끔한 구조

#### Microsoft Multi-Agent Engine 통합
- WebSocket 스트리밍 (WebSocketMessageType)
- MPlan 계획 시스템 (PlanStep, PlanStepStatus)
- ProxyAgent (사용자 명확화)
- RAIValidator (Responsible AI 검증)
- AgentFactory & OrchestrationManager

#### Skills 시스템
- Skill, SkillManager, SkillResource 클래스
- Progressive Disclosure 패턴
- 자동 스킬 활성화

#### Memory Hook Provider
- MemoryHookProvider 클래스
- ConversationMessage 모델
- MemorySessionManager

### 🔧 Changed
- 메인 파일 93.5% 코드 감소 (6,040줄 → 325줄)
- Settings 클래스로 중앙 설정 관리
- FrameworkConfig 데이터클래스 도입

### ✅ Tests
- 79개 단위 테스트 추가
- 100% 모듈 커버리지

---

## [2.0.0] - 2025-06-01

### 🆕 Added
- SupervisorAgent 추가
- CircuitBreaker 패턴 도입
- OpenTelemetry 통합
- GPT-5 시리즈 지원

### 🔧 Changed
- Pydantic v2 마이그레이션
- Python 3.10+ 요구사항

---

## [1.0.0] - 2025-01-01

### 🆕 Added
- 초기 릴리스
- SimpleAgent, RouterAgent
- Graph 워크플로우
- Azure OpenAI 통합
