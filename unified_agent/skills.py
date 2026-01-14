#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Unified Agent Framework - Skills 시스템 모듈

Anthropic Skills 패턴 구현 - 모듈화된 지식/워크플로우/도구 패키지
"""

import re
import logging
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Any

try:
    import yaml
    YAML_AVAILABLE = True
except ImportError:
    YAML_AVAILABLE = False

__all__ = [
    "SkillResource",
    "Skill",
    "SkillManager",
]


@dataclass(slots=True)
class SkillResource:
    """
    스킬 번들 리소스

    스킬에 포함되는 추가 리소스를 정의합니다:
    - scripts/: 실행 가능한 스크립트 (Python, Bash 등)
    - references/: 참조 문서 (마크다운, 텍스트 등)
    - assets/: 템플릿, 이미지 등 출력용 파일
    """
    resource_type: str  # 'script', 'reference', 'asset'
    name: str
    path: str
    content: Optional[str] = None
    description: Optional[str] = None


@dataclass
class Skill:
    """
    Anthropic Skills 패턴 구현

    Skills는 AI의 능력을 확장하는 모듈화된 패키지입니다.
    특정 도메인의 지식, 워크플로우, 도구를 제공합니다.

    구조:
    ```
    skill-name/
    ├── SKILL.md (필수)
    │   ├── YAML frontmatter (name, description)
    │   └── Markdown 지침
    └── Bundled Resources (선택)
        ├── scripts/      - 실행 코드
        ├── references/   - 참조 문서
        └── assets/       - 템플릿, 아이콘 등
    ```
    """
    name: str
    description: str
    instructions: str = ""
    triggers: List[str] = field(default_factory=list)
    resources: List[SkillResource] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    enabled: bool = True
    priority: int = 0  # 높을수록 우선순위 높음

    # Progressive Disclosure 관련
    always_loaded: bool = False  # True면 항상 컨텍스트에 포함
    max_context_lines: int = 500  # SKILL.md 최대 라인 수

    @classmethod
    def from_file(cls, filepath: str) -> 'Skill':
        """
        SKILL.md 파일에서 스킬 로드
        """
        path = Path(filepath)
        if not path.exists():
            raise FileNotFoundError(f"스킬 파일을 찾을 수 없습니다: {filepath}")

        content = path.read_text(encoding='utf-8')
        return cls._parse_skill_content(content, filepath)

    @classmethod
    def from_directory(cls, dirpath: str) -> 'Skill':
        """
        스킬 디렉토리에서 스킬 로드 (리소스 포함)
        """
        dirpath = Path(dirpath)
        skill_file = dirpath / "SKILL.md"

        if not skill_file.exists():
            raise FileNotFoundError(f"SKILL.md를 찾을 수 없습니다: {skill_file}")

        # 기본 스킬 로드
        skill = cls.from_file(str(skill_file))

        # 리소스 로드
        skill._load_resources(dirpath)

        return skill

    @classmethod
    def _parse_skill_content(cls, content: str, source: str = "") -> 'Skill':
        """SKILL.md 내용 파싱"""
        # YAML frontmatter 추출
        frontmatter = {}
        body = content

        if content.startswith('---'):
            parts = content.split('---', 2)
            if len(parts) >= 3:
                if YAML_AVAILABLE:
                    try:
                        frontmatter = yaml.safe_load(parts[1]) or {}
                    except yaml.YAMLError:
                        frontmatter = cls._parse_simple_yaml(parts[1])
                else:
                    frontmatter = cls._parse_simple_yaml(parts[1])
                body = parts[2].strip()

        name = frontmatter.get('name', Path(source).stem if source else 'unnamed-skill')
        description = frontmatter.get('description', '')

        # triggers 추출
        triggers = frontmatter.get('triggers', [])
        if not triggers and description:
            triggers = cls._extract_triggers(description)

        # priority 추출
        priority = frontmatter.get('priority', 0)
        if isinstance(priority, str):
            try:
                priority = int(priority)
            except ValueError:
                priority = 0

        return cls(
            name=name,
            description=description,
            instructions=body,
            triggers=triggers,
            priority=priority,
            metadata={
                'source': source,
                'license': frontmatter.get('license', ''),
                **{k: v for k, v in frontmatter.items() if k not in ['name', 'description', 'triggers', 'license', 'priority']}
            }
        )

    @staticmethod
    def _parse_simple_yaml(text: str) -> Dict[str, Any]:
        """간단한 YAML 파싱 (yaml 라이브러리 없을 때)"""
        result = {}
        for line in text.strip().split('\n'):
            if ':' in line:
                key, value = line.split(':', 1)
                key = key.strip()
                value = value.strip()
                if value.startswith('"') and value.endswith('"'):
                    value = value[1:-1]
                elif value.startswith("'") and value.endswith("'"):
                    value = value[1:-1]
                result[key] = value
        return result

    @staticmethod
    def _extract_triggers(description: str) -> List[str]:
        """설명에서 트리거 키워드 추출"""
        keywords = []
        parens = re.findall(r'\(([^)]+)\)', description)
        for paren in parens:
            keywords.extend([k.strip() for k in paren.split(',')])

        words = re.findall(r'\b[A-Za-z가-힣]{3,}\b', description)
        stop_words = {'the', 'and', 'for', 'use', 'when', 'with', 'this', 'that', 'from', 'have', 'are'}
        keywords.extend([w.lower() for w in words if w.lower() not in stop_words][:5])

        return list(set(keywords))[:10]

    def _load_resources(self, dirpath: Path):
        """디렉토리에서 리소스 로드"""
        for res_type, folder_name in [("script", "scripts"), ("reference", "references"), ("asset", "assets")]:
            res_dir = dirpath / folder_name
            if res_dir.exists():
                for res_file in res_dir.glob("*"):
                    if res_file.is_file():
                        self.resources.append(SkillResource(
                            resource_type=res_type,
                            name=res_file.name,
                            path=str(res_file),
                            description=f"{res_type.title()}: {res_file.name}"
                        ))

    def get_resource(self, name: str) -> Optional[SkillResource]:
        """이름으로 리소스 찾기"""
        for resource in self.resources:
            if resource.name == name:
                return resource
        return None

    def load_resource_content(self, resource: SkillResource) -> str:
        """리소스 내용 로드"""
        if resource.content:
            return resource.content

        path = Path(resource.path)
        if path.exists() and path.is_file():
            try:
                resource.content = path.read_text(encoding='utf-8')
                return resource.content
            except Exception as e:
                logging.warning(f"리소스 로드 실패: {resource.path} - {e}")
        return ""

    def matches(self, query: str) -> float:
        """쿼리와의 매칭 점수 계산 (0.0 ~ 1.0)"""
        query_lower = query.lower()
        score = 0.0

        if self.name.lower() in query_lower:
            score += 0.5

        for trigger in self.triggers:
            if trigger.lower() in query_lower:
                score += 0.3
                break

        desc_words = self.description.lower().split()
        query_words = query_lower.split()
        common_words = set(desc_words) & set(query_words)
        if common_words:
            score += min(len(common_words) * 0.1, 0.2)

        return min(score, 1.0)

    def get_prompt_section(self, include_full: bool = False) -> str:
        """프롬프트에 포함할 스킬 섹션 생성"""
        if include_full:
            return f"""
## Skill: {self.name}

**Description:** {self.description}

{self.instructions}

---
"""
        else:
            return f"- **{self.name}**: {self.description}\n"

    def to_dict(self) -> Dict[str, Any]:
        """딕셔너리로 변환"""
        return {
            "name": self.name,
            "description": self.description,
            "instructions": self.instructions,
            "triggers": self.triggers,
            "resources": [
                {"type": r.resource_type, "name": r.name, "path": r.path}
                for r in self.resources
            ],
            "metadata": self.metadata,
            "enabled": self.enabled,
            "priority": self.priority
        }


class SkillManager:
    """
    스킬 관리자 - 스킬 등록, 검색, 활성화 관리

    주요 기능:
    - 스킬 등록 및 해제
    - 쿼리 기반 스킬 매칭 (Progressive Disclosure)
    - 디렉토리에서 스킬 일괄 로드
    - 스킬 우선순위 관리
    """

    def __init__(self, skill_dirs: Optional[List[str]] = None):
        self.skills: Dict[str, Skill] = {}
        self.skill_history: List[Dict[str, Any]] = []

        if skill_dirs:
            for skill_dir in skill_dirs:
                self.load_skills_from_directory(skill_dir)

    def register_skill(self, skill: Skill) -> bool:
        """스킬 등록"""
        if skill.name in self.skills:
            logging.warning(f"스킬 '{skill.name}'이 이미 존재합니다. 덮어씁니다.")

        self.skills[skill.name] = skill
        logging.info(f"✅ 스킬 등록: {skill.name}")
        return True

    def unregister_skill(self, name: str) -> bool:
        """스킬 해제"""
        if name in self.skills:
            del self.skills[name]
            logging.info(f"🗑️ 스킬 해제: {name}")
            return True
        return False

    def get_skill(self, name: str) -> Optional[Skill]:
        """이름으로 스킬 가져오기"""
        return self.skills.get(name)

    def list_skills(self, enabled_only: bool = True) -> List[Skill]:
        """등록된 스킬 목록"""
        skills = list(self.skills.values())
        if enabled_only:
            skills = [s for s in skills if s.enabled]
        return sorted(skills, key=lambda s: -s.priority)

    def load_skills_from_directory(self, dirpath: str) -> int:
        """디렉토리에서 스킬 일괄 로드"""
        dirpath = Path(dirpath)
        if not dirpath.exists():
            logging.warning(f"스킬 디렉토리가 존재하지 않습니다: {dirpath}")
            return 0

        loaded = 0
        for skill_dir in dirpath.iterdir():
            if skill_dir.is_dir():
                skill_file = skill_dir / "SKILL.md"
                if skill_file.exists():
                    try:
                        skill = Skill.from_directory(str(skill_dir))
                        self.register_skill(skill)
                        loaded += 1
                    except Exception as e:
                        logging.error(f"스킬 로드 실패: {skill_dir} - {e}")

        logging.info(f"📦 {loaded}개 스킬 로드 완료 from {dirpath}")
        return loaded

    def match_skills(
        self,
        query: str,
        threshold: float = 0.2,
        max_skills: int = 3
    ) -> List[Skill]:
        """쿼리에 매칭되는 스킬 찾기"""
        matched = []

        for skill in self.list_skills():
            if skill.always_loaded:
                matched.append((skill, 1.0))
                continue

            score = skill.matches(query)
            if score >= threshold:
                matched.append((skill, score))

        matched.sort(key=lambda x: (-x[1], -x[0].priority))

        result = [skill for skill, _ in matched[:max_skills]]

        self.skill_history.append({
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "query": query,
            "matched": [s.name for s in result]
        })

        return result

    def build_system_prompt(
        self,
        skills: List[Skill],
        base_prompt: str = "",
        include_full: bool = True
    ) -> str:
        """스킬을 포함한 시스템 프롬프트 생성"""
        prompt_parts = []

        if base_prompt:
            prompt_parts.append(base_prompt)

        if skills:
            prompt_parts.append("\n# Active Skills\n")
            for skill in skills:
                prompt_parts.append(skill.get_prompt_section(include_full=include_full))

        other_skills = [s for s in self.list_skills() if s not in skills]
        if other_skills:
            prompt_parts.append("\n# Available Skills (activate by mentioning)\n")
            for skill in other_skills[:5]:
                prompt_parts.append(skill.get_prompt_section(include_full=False))

        return "\n".join(prompt_parts)

    def get_usage_stats(self) -> Dict[str, Any]:
        """스킬 사용 통계"""
        stats = defaultdict(int)
        for record in self.skill_history:
            for skill_name in record.get("matched", []):
                stats[skill_name] += 1

        return {
            "total_queries": len(self.skill_history),
            "skill_usage": dict(stats),
            "registered_skills": len(self.skills),
            "enabled_skills": len([s for s in self.skills.values() if s.enabled])
        }

    def create_skill_template(self, name: str, output_dir: str) -> str:
        """새 스킬 템플릿 생성"""
        output_path = Path(output_dir) / name
        output_path.mkdir(parents=True, exist_ok=True)

        skill_md = f"""---
name: {name}
description: [TODO: 이 스킬이 무엇을 하는지, 언제 사용해야 하는지 설명하세요]
---

# {name.replace('-', ' ').title()}

## Overview

[TODO: 1-2문장으로 이 스킬이 무엇을 가능하게 하는지 설명]

## When to Use

이 스킬은 다음과 같은 경우에 사용합니다:
- [TODO: 사용 시나리오 1]
- [TODO: 사용 시나리오 2]

## Instructions

[TODO: AI가 따라야 할 지침을 작성하세요]

## Examples

### Example 1
[TODO: 예시 추가]

## Resources

- scripts/: 실행 가능한 스크립트
- references/: 참조 문서
- assets/: 템플릿 및 에셋
"""

        (output_path / "SKILL.md").write_text(skill_md, encoding='utf-8')

        (output_path / "scripts").mkdir(exist_ok=True)
        (output_path / "references").mkdir(exist_ok=True)
        (output_path / "assets").mkdir(exist_ok=True)

        example_script = f'''#!/usr/bin/env python3
"""
Example script for {name}
"""

def main():
    print("Hello from {name}!")

if __name__ == "__main__":
    main()
'''
        (output_path / "scripts" / "example.py").write_text(example_script, encoding='utf-8')

        logging.info(f"✅ 스킬 템플릿 생성: {output_path}")
        return str(output_path)
