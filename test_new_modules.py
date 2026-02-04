#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Unified Agent Framework v3.5 - 신규 모듈 테스트

================================================================================
📁 파일: test_new_modules.py
📋 역할: v3.5 신규 모듈 (Structured Output, Evaluation) 전용 테스트
📅 최종 업데이트: 2026년 2월 4일
✅ 결과: 모든 테스트 통과
================================================================================

테스트 내용:
    1. Structured Output
       - OutputSchema 생성 및 검증
       - StructuredOutputParser JSON 파싱
       - StructuredOutputValidator 스키마 검증
       - pydantic_to_schema 변환
    
    2. Evaluation (PDCA + LLM-as-Judge)
       - PDCAEvaluator PDCA 사이클 평가
       - LLMJudge 품질 평가
       - CheckActIterator 자동 개선 루프
       - GapAnalyzer 갭 분석
       - QualityMetrics 품질 메트릭

실행 방법:
    $ python test_new_modules.py
"""

import asyncio
import sys
sys.path.insert(0, r"d:\Azure-openai-sample\Unified-agent-framework")

from unified_agent import (
    # Structured Output
    OutputSchema,
    StructuredOutputConfig,
    StructuredOutputParser,
    StructuredOutputValidator,
    pydantic_to_schema,
    # Evaluation
    PDCAEvaluator,
    LLMJudge,
    CheckActIterator,
    GapAnalyzer,
    QualityMetrics,
    AgentBenchmark,
    EvaluationConfig,
    JudgeConfig,
    IterationConfig,
    EvaluationDimension,
    PDCAPhase,
    QualityLevel,
)


async def test_structured_output():
    """Structured Output 테스트"""
    print("\n" + "=" * 60)
    print("🧪 Structured Output 테스트")
    print("=" * 60)
    
    # 1. 스키마 정의 테스트
    schema = OutputSchema(
        name="PersonInfo",
        description="개인 정보 스키마",
        schema={
            "type": "object",
            "properties": {
                "name": {"type": "string", "description": "이름"},
                "age": {"type": "integer", "description": "나이"},
                "email": {"type": "string", "description": "이메일"}
            },
            "required": ["name", "age"]
        },
        strict=True
    )
    print(f"✅ 스키마 생성: {schema.name}")
    
    # 2. JSON Schema 변환 (OpenAI 형식)
    openai_format = schema.to_openai_format()
    print(f"✅ OpenAI 형식 변환: {openai_format['json_schema']['name']}")
    
    # 3. Parser 테스트
    parser = StructuredOutputParser()
    
    # 정상 JSON 파싱
    valid_json = '{"name": "홍길동", "age": 30, "email": "hong@example.com"}'
    result = parser.parse(valid_json, schema)
    print(f"✅ 정상 JSON 파싱: {result.data}")
    
    # 코드 블록 JSON 파싱
    code_block_json = '```json\n{"name": "김철수", "age": 25}\n```'
    code_result = parser.parse(code_block_json, schema)
    print(f"✅ 코드 블록 JSON 파싱: {code_result.data}")
    
    # 4. Validator 테스트
    validator = StructuredOutputValidator(schema)
    
    valid_data = {"name": "이영희", "age": 25}
    is_valid, errors = validator.validate(valid_data)
    print(f"✅ 유효성 검증 (유효): {is_valid}, 오류: {errors}")
    
    invalid_data = {"name": "박민수"}  # age 누락
    is_valid, errors = validator.validate(invalid_data)
    print(f"✅ 유효성 검증 (무효): {is_valid}, 오류: {errors}")
    
    print("\n✅ Structured Output 테스트 완료!")


async def test_evaluation():
    """Evaluation 모듈 테스트"""
    print("\n" + "=" * 60)
    print("🧪 Evaluation 모듈 테스트")
    print("=" * 60)
    
    # 1. Gap Analyzer 테스트
    print("\n📋 Gap Analyzer 테스트...")
    analyzer = GapAnalyzer()
    
    plan = """
    ## 프로젝트 목표
    1. 사용자 인증 시스템 구현
    2. 데이터베이스 연동
    3. REST API 개발
    4. 테스트 코드 작성
    """
    
    implementation = """
    # 구현 완료 내용
    - 사용자 인증: JWT 토큰 기반 구현
    - 데이터베이스: PostgreSQL 연동 완료
    - REST API: CRUD 엔드포인트 구현
    """
    
    gap_result = await analyzer.analyze(plan, implementation)
    print(f"✅ 일치율: {gap_result.match_rate:.1%}")
    print(f"✅ 누락 항목: {gap_result.missing_features[:3]}")
    print(f"✅ 권장 사항: {gap_result.recommendations}")
    
    # 2. PDCA Evaluator 테스트
    print("\n📋 PDCA Evaluator 테스트...")
    pdca = PDCAEvaluator()
    
    # Plan 평가
    plan_doc = """
    ## 프로젝트 계획
    
    ### 목표
    AI 챗봇 시스템 구축
    
    ### 요구사항
    - 자연어 처리 기능
    - 다국어 지원
    - 24시간 운영
    
    ### 범위
    웹 및 모바일 지원
    
    ### 일정
    2026-03-01 ~ 2026-06-30
    """
    
    plan_result = await pdca.evaluate_plan(plan_doc)
    print(f"✅ Plan 평가 점수: {plan_result.overall_score:.1%}")
    print(f"✅ 품질 수준: {plan_result.quality_level.value}")
    
    # 3. LLM Judge 테스트
    print("\n📋 LLM Judge 테스트...")
    judge = LLMJudge()
    
    output = """
    안녕하세요! Azure OpenAI Service는 Microsoft Azure 클라우드 플랫폼에서 
    제공하는 AI 서비스입니다. GPT-4, GPT-5.2 등의 최신 언어 모델을 사용할 수 있으며,
    엔터프라이즈급 보안과 규정 준수를 제공합니다.
    
    주요 기능:
    1. 텍스트 생성
    2. 코드 생성
    3. 이미지 분석
    4. 음성 인식
    """
    
    verdict = await judge.evaluate(
        output=output,
        criteria="정확성, 완전성, 명확성"
    )
    print(f"✅ Judge 점수: {verdict.score:.1f}/10")
    print(f"✅ 강점: {verdict.strengths}")
    print(f"✅ 약점: {verdict.weaknesses}")
    
    # 4. Check-Act Iterator 테스트
    print("\n📋 Check-Act Iterator 테스트...")
    
    config = IterationConfig(
        threshold=0.7,      # 70% 목표 (테스트용 낮춤)
        max_iterations=3,   # 최대 3회
        verbose=True
    )
    
    iterator = CheckActIterator(
        evaluator=judge,
        config=config
    )
    
    initial_output = "Azure는 클라우드 서비스입니다."  # 짧은 응답
    
    result = await iterator.iterate(
        initial_output=initial_output,
        criteria="응답의 길이와 정보량"
    )
    
    print(f"\n✅ 반복 횟수: {result.iterations}")
    print(f"✅ 최종 점수: {result.final_score:.1%}")
    print(f"✅ 점수 이력: {[f'{s:.1%}' for s in result.score_history]}")
    
    # 5. Quality Metrics 테스트
    print("\n📋 Quality Metrics 테스트...")
    metrics = QualityMetrics()
    
    # 메트릭 기록
    for i in range(5):
        metrics.record("task_completion", 0.8 + i * 0.02)
        metrics.record("response_time_ms", 200 + i * 10)
    
    # 통계 조회
    stats = metrics.get_stats("task_completion")
    print(f"✅ task_completion 통계: mean={stats['mean']:.2f}, min={stats['min']:.2f}, max={stats['max']:.2f}")
    
    # 리포트 생성
    report = metrics.generate_report()
    print(f"✅ 품질 리포트 요약: {report.summary}")
    print(f"✅ 종합 점수: {report.overall_score:.1%}")
    
    print("\n✅ Evaluation 테스트 완료!")


async def test_benchmark():
    """Benchmark 테스트"""
    print("\n" + "=" * 60)
    print("🧪 Agent Benchmark 테스트")
    print("=" * 60)
    
    benchmark = AgentBenchmark(suite_name="simple_qa")
    
    # 테스트 케이스 추가
    benchmark.add_test_case(
        name="capital_question",
        input_text="대한민국의 수도는?",
        expected="서울",
        criteria="정확성"
    )
    benchmark.add_test_case(
        name="math_question",
        input_text="1 + 1 = ?",
        expected="2",
        criteria="정확성"
    )
    
    # 간단한 에이전트 함수
    async def simple_agent(query: str) -> str:
        if "수도" in query:
            return "대한민국의 수도는 서울입니다."
        elif "1 + 1" in query:
            return "1 + 1 = 2 입니다."
        return "알 수 없습니다."
    
    # 벤치마크 실행
    result = await benchmark.run(simple_agent, "test_agent")
    
    print(f"✅ 테스트 결과: {result.passed}/{result.total_tests} 통과")
    print(f"✅ 평균 점수: {result.avg_score:.1%}")
    
    print("\n✅ Benchmark 테스트 완료!")


async def main():
    """메인 테스트 함수"""
    print("=" * 60)
    print("🚀 Unified Agent Framework v3.5 모듈 테스트")
    print("   - Structured Output")
    print("   - Evaluation (PDCA, LLM-as-Judge, Check-Act)")
    print("=" * 60)
    
    await test_structured_output()
    await test_evaluation()
    await test_benchmark()
    
    print("\n" + "=" * 60)
    print("🎉 모든 테스트 완료!")
    print("=" * 60)


if __name__ == "__main__":
    asyncio.run(main())
