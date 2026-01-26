# 🤝 Contributing to Unified Agent Framework

먼저, Unified Agent Framework에 기여해 주셔서 감사합니다! 🎉

이 문서는 프로젝트에 기여하는 방법을 안내합니다.

## 📋 목차

- [행동 강령](#행동-강령)
- [시작하기](#시작하기)
- [개발 환경 설정](#개발-환경-설정)
- [기여 방법](#기여-방법)
- [Pull Request 가이드라인](#pull-request-가이드라인)
- [코드 스타일](#코드-스타일)
- [테스트](#테스트)
- [문서화](#문서화)

## 📜 행동 강령

이 프로젝트는 [Contributor Covenant](CODE_OF_CONDUCT.md)를 따릅니다.
참여함으로써 이 행동 강령을 준수하는 것에 동의하게 됩니다.

## 🚀 시작하기

### 이슈 확인

기여하기 전에 다음을 확인하세요:

1. [기존 이슈](../../issues)를 확인하여 중복되지 않는지 확인
2. 새로운 기능이나 버그 수정은 먼저 이슈를 생성하여 논의
3. `good first issue` 라벨이 붙은 이슈는 처음 기여하기 좋습니다

## 💻 개발 환경 설정

### 1. 저장소 Fork 및 Clone

```bash
# Fork 후 Clone
git clone https://github.com/YOUR_USERNAME/unified-agent-framework.git
cd unified-agent-framework

# 원본 저장소를 upstream으로 추가
git remote add upstream https://github.com/ORIGINAL_OWNER/unified-agent-framework.git
```

### 2. 가상 환경 생성

```bash
# Python 3.11+ 필요
python -m venv venv

# Windows
venv\Scripts\activate

# macOS/Linux
source venv/bin/activate
```

### 3. 의존성 설치

```bash
# 기본 의존성
pip install -r requirements.txt

# 개발 의존성
pip install -r requirements-dev.txt
```

### 4. 환경 변수 설정

```bash
# .env.example을 복사
cp .env.example .env

# .env 파일을 편집하여 API 키 설정
```

### 5. 테스트 실행

```bash
# 전체 테스트
python test_unified_agent.py

# 또는 pytest 사용
pytest tests/ -v
```

## 🎯 기여 방법

### 버그 리포트

버그를 발견하셨나요? 다음 정보와 함께 이슈를 생성해 주세요:

- **환경 정보**: Python 버전, OS, 의존성 버전
- **재현 단계**: 버그를 재현하는 구체적인 단계
- **예상 동작**: 어떻게 동작해야 하는지
- **실제 동작**: 실제로 어떻게 동작하는지
- **에러 메시지**: 전체 스택 트레이스 (있는 경우)

### 기능 제안

새로운 기능을 제안하시나요?

1. 먼저 이슈를 생성하여 아이디어를 공유
2. 커뮤니티와 논의 후 구현 방향 결정
3. PR 생성 시 해당 이슈 참조

### 코드 기여

1. **브랜치 생성**
   ```bash
   git checkout -b feature/your-feature-name
   # 또는
   git checkout -b fix/your-bug-fix
   ```

2. **코드 작성**
   - 코드 스타일 가이드라인 준수
   - 적절한 테스트 추가
   - 문서 업데이트 (필요한 경우)

3. **커밋**
   ```bash
   git add .
   git commit -m "feat: add new feature description"
   ```

4. **Push 및 PR 생성**
   ```bash
   git push origin feature/your-feature-name
   ```

## 📝 Pull Request 가이드라인

### PR 체크리스트

- [ ] 코드가 기존 스타일과 일관성 있음
- [ ] 새로운 기능에 대한 테스트 추가
- [ ] 모든 테스트 통과 (`python test_unified_agent.py`)
- [ ] 문서 업데이트 (README, docstring 등)
- [ ] 커밋 메시지가 명확함

### 커밋 메시지 규칙

[Conventional Commits](https://www.conventionalcommits.org/) 규칙을 따릅니다:

```
<type>(<scope>): <description>

[optional body]

[optional footer]
```

**Type:**
- `feat`: 새로운 기능
- `fix`: 버그 수정
- `docs`: 문서 변경
- `style`: 코드 포맷팅 (기능 변경 없음)
- `refactor`: 리팩토링
- `test`: 테스트 추가/수정
- `chore`: 빌드 프로세스, 도구 변경

**예시:**
```
feat(agents): add SupervisorAgent for multi-agent orchestration

- Implement supervisor pattern for agent coordination
- Add support for MPlan execution
- Include human-in-the-loop approval

Closes #123
```

## 🎨 코드 스타일

### Python 스타일 가이드

- [PEP 8](https://peps.python.org/pep-0008/) 준수
- [Black](https://github.com/psf/black) 포매터 사용 (line-length: 100)
- [isort](https://pycqa.github.io/isort/) 로 import 정렬
- Type hints 적극 활용

### Docstring 스타일

```python
def example_function(param1: str, param2: int = 10) -> bool:
    """
    함수에 대한 간단한 설명

    ================================================================================
    📋 역할: 함수의 목적을 설명
    📅 최종 업데이트: 2026년 1월
    ================================================================================

    Args:
        param1 (str): 첫 번째 파라미터 설명
        param2 (int): 두 번째 파라미터 설명 (기본: 10)

    Returns:
        bool: 반환값 설명

    Raises:
        ValueError: 예외 발생 조건

    📌 사용 예시:
        >>> result = example_function("test", 20)
        >>> print(result)
        True

    ⚠️ 주의사항:
        - 주의할 점 1
        - 주의할 점 2
    """
    pass
```

### 파일 구조

```python
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
모듈 설명

================================================================================
📁 파일 위치: unified_agent/module_name.py
📋 역할: 모듈의 역할
📅 최종 업데이트: 2026년 1월
================================================================================
"""

# Standard library imports
import os
import sys

# Third-party imports
import pydantic

# Local imports
from .exceptions import FrameworkError

__all__ = ["ExportedClass", "exported_function"]


# Code here...
```

## 🧪 테스트

### 테스트 작성

```python
def test_feature_name():
    """테스트 설명"""
    # Given (준비)
    input_data = ...

    # When (실행)
    result = function_under_test(input_data)

    # Then (검증)
    assert result == expected_output
```

### 테스트 실행

```bash
# 전체 테스트
python test_unified_agent.py

# 특정 테스트만
pytest tests/test_specific.py -v

# 커버리지 포함
pytest --cov=unified_agent tests/
```

## 📚 문서화

### README 업데이트

새로운 기능을 추가할 때:
1. README.md의 해당 섹션 업데이트
2. 사용 예시 추가
3. 필요한 경우 새 섹션 추가

### API 문서

- 모든 public 함수/클래스에 docstring 필수
- 예시 코드 포함 권장
- Type hints 필수

## 🙏 감사합니다!

여러분의 기여가 Unified Agent Framework를 더 좋게 만듭니다!

질문이 있으시면 [Discussions](../../discussions)에서 자유롭게 질문해 주세요.
