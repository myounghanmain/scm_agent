#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
SQL Agent 실행 스크립트
"""

from main import app
import os

if __name__ == "__main__":
    # 테스트 쿼리 (텍스트 + 그래프)
    questions = [
        "2009년 가장 많은 매출을 올린 영업 사원이름을 말해.",
        "2009년부터 2011년까지 총 매출액은 얼마인가?",
        "2013년에 가장 많은 고객을 확보한 국가 3개를 작성해",
        "Steve Johnson의 2009 월별 매출을 그래프로 그려줘",
        "상위 10개 국가의 총 매출을 막대 그래프로 표시해줘",
        "3명 영업사원 각각의 총 매출을 파이 차트로 그려줘"
    ]

    for question in questions:
        print(f"\n{'='*60}")
        print(f"❓ 질문: {question}")
        print(f"{'='*60}\n")

        # 그래프 파일 초기화
        if os.path.exists('chart.png'):
            os.remove('chart.png')

        try:
            step = 1
            final_answer = None

            for output in app.stream(
                {"messages": [("user", question)]},
                config={"recursion_limit": 20}
            ):
                # 각 노드의 출력 표시
                for node, values in output.items():
                    if node != "__end__":
                        print(f"📍 Step {step}: [{node}]")

                        # 마지막 메시지 확인
                        if "messages" in values:
                            last_msg = values["messages"][-1]

                            # 도구 호출 확인
                            if hasattr(last_msg, 'tool_calls'):
                                for tc in (last_msg.tool_calls or []):
                                    if tc.get('name') == 'SubmitFinalAnswer':
                                        final_answer = tc['args']['final_answer']
                                        print(f"   ✅ 최종 답변: {final_answer}\n")
                                    elif tc.get('name') == 'db_query_tool':
                                        query = tc['args'].get('query', '')
                                        print(
                                            f"   🔍 SQL 쿼리: {query[:80]}...\n")
                                    elif tc.get('name') == 'model_check_query':
                                        print(f"   ✔️  쿼리 검증 중...\n")
                                    elif tc.get('name') == 'python_repl':
                                        print(f"   📊 그래프 생성 중...\n")
                            elif hasattr(last_msg, 'content') and last_msg.content:
                                print(f"   💬 {last_msg.content[:100]}...\n")

                        step += 1

            # 결과 확인
            if final_answer:
                print(f"✅ 텍스트 답변 완료")
            if os.path.exists('chart.png'):
                size = os.path.getsize('chart.png')
                print(f"✅ 그래프 생성 완료 ({size} bytes)")

        except Exception as e:
            print(f"❌ 에러: {e}")
            import traceback
            traceback.print_exc()
