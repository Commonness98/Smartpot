"""Agent Core: 자연어 질의 → (의도 분류 → 추론·계획) → 자연어 권고안.

의도 분류(intent routing):
  - operation : 전력/설비 가동 관련 → 예측·요금 데이터를 근거로 추천
  - general   : 인사·자기소개·일반 질문 → 역할 안내(추천 데이터 주입 안 함)

LM Studio/Ollama LLM이 있으면 자연어 생성을, 없으면 템플릿 폴백을 사용한다.
"""
from __future__ import annotations

import json

from . import llm_client
from .planner import build_facts

# 전력/설비 운영 의도로 판단할 키워드
OPERATION_KEYWORDS = [
    "펌프", "관수", "냉난방", "냉방", "난방", "환기", "조명", "가동", "운영",
    "전력", "전기", "요금", "소비", "절감", "절약", "켜", "켤", "돌리", "가둠",
    "방제", "정비", "내일", "에너지", "부하",
]

SYSTEM_OPERATION = (
    "당신은 스마트팜 운영을 돕는 한국어 AI 에이전트입니다. "
    "주어진 예측·요금 데이터(JSON)에 근거해, 농업인이 이해하기 쉬운 "
    "운영 권고안을 2~4문장으로 제시하세요.\n"
    "규칙:\n"
    "- 숫자와 단위는 데이터에 있는 값 그대로만 사용하세요.\n"
    "- 데이터에 없는 생육·환경·작물 정보는 절대 지어내지 마세요.\n"
    "- 데이터는 일(day) 단위 예측이라 시간대(몇 시)는 알 수 없습니다. "
    "'내일 에너지 수요가 평소보다 높은지/낮은지'를 근거로 방제·정비 등 "
    "비필수 작업을 오늘/내일 중 언제 하는 게 좋을지 권고하세요."
)

SYSTEM_GENERAL = (
    "당신은 스마트팜 운영을 돕는 한국어 AI 에이전트입니다. "
    "온실의 일일 에너지(난방+전기) 수요 예측과 전기요금을 근거로, 방제·정비 등 "
    "비필수 작업을 언제(오늘/내일) 진행하면 좋을지 안내하는 역할을 합니다. "
    "인사·자기소개·일반 질문에는 1~3문장으로 간단히 답하고, "
    "전력·설비 운영과 무관한 내용에 대해 수치를 지어내지 마세요. "
    "필요하면 사용자가 물어볼 수 있는 예시 질문을 안내하세요."
)

HELP_TEXT = (
    "안녕하세요! 저는 스마트팜 에너지 운영을 돕는 AI 에이전트입니다. "
    "온실의 일일 에너지(난방+전기) 수요를 예측해 방제·정비 같은 비필수 작업을 "
    "언제 진행하면 좋을지 알려드립니다.\n\n"
    "예를 들어 이렇게 물어보세요:\n"
    "- \"내일 에너지 수요가 어느 정도일까?\"\n"
    "- \"방제 작업은 오늘 할까 내일 할까?\""
)


def _classify(question: str) -> str:
    q = question.lower()
    return "operation" if any(k in q for k in OPERATION_KEYWORDS) else "general"


def _fallback_operation(facts: dict) -> str:
    fc = facts["forecast"]
    sched = facts["schedule"]
    return (
        f"내일({fc['horizon_days']}일 뒤) 예상 에너지 수요는 약 {fc['predicted_energy']} "
        f"(최근 평균 {fc['recent_avg_energy']}, 수준: {sched['level']})입니다. "
        f"{sched['advice']} 근사 비용은 평균 단가 {sched['avg_rate']}원/kWh 기준 "
        f"약 {sched['estimated_cost']}원입니다."
    )


def answer(question: str, model_path=None) -> dict:
    """사용자 질문의 의도를 분류하여 적절한 응답을 생성한다."""
    intent = _classify(question)

    # --- 일반/인사 질문: 추천 데이터 주입 없이 역할 안내 ---
    if intent == "general":
        if llm_client.available():
            try:
                text = llm_client.chat([
                    {"role": "system", "content": SYSTEM_GENERAL},
                    {"role": "user", "content": question},
                ])
                return {"text": text, "facts": None, "intent": intent, "used_llm": True}
            except Exception:
                pass
        return {"text": HELP_TEXT, "facts": None, "intent": intent, "used_llm": False}

    # --- 전력/설비 운영 질문: 예측·요금 데이터 기반 추천 ---
    facts = build_facts(model_path=model_path)

    if llm_client.available():
        try:
            text = llm_client.chat([
                {"role": "system", "content": SYSTEM_OPERATION},
                {"role": "user", "content":
                    f"[질문]\n{question}\n\n[근거 데이터(JSON)]\n"
                    + json.dumps(facts, ensure_ascii=False, default=str)},
            ])
            return {"text": text, "facts": facts, "intent": intent, "used_llm": True}
        except Exception as e:
            fb = _fallback_operation(facts)
            return {"text": fb + f"\n\n(LLM 호출 실패로 규칙 기반 응답: {e})",
                    "facts": facts, "intent": intent, "used_llm": False}

    return {"text": _fallback_operation(facts),
            "facts": facts, "intent": intent, "used_llm": False}
