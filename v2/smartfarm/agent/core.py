"""Agent Core: 자연어 질의 → (의도 분류 → 추론·계획) → 자연어 권고안.

의도 분류(intent routing)
  - history   : 과거 실측값 조회("5월 3일 얼마 썼어?") → 예측 없이 DB 값 그대로
  - operation : 전력·설비·작업 일정 관련 → 예측·근거 데이터를 바탕으로 권고
  - general   : 인사·자기소개·범위 밖 질문 → 역할 안내(추천 데이터 주입 안 함)

로컬 LLM(Ollama/LM Studio)이 있으면 자연어 생성을, 없으면 템플릿 폴백을 사용한다.

v2 변경점
- 의도 분류를 규칙 1차 + 저신뢰 시 LLM 보정 하이브리드로 전환. 기존에는 키워드
  22개 단순 포함검사여서 "모레는 어때?" 같은 운영 질문이 general로 오분류되어
  예측 데이터가 주입되지 않고 도움말만 반환되었다.
- 대화 이력(history)을 인자로 받아 후속 질문의 맥락을 유지한다. 기존에는 현재
  질문만 받아 이전 대화를 참조할 수 없었다.
- 시간대 언급 금지를 프롬프트에서 명시하고, 근거 데이터에서도 시간대 항목을
  제거했다(planner.build_facts). 존재하지 않는 시간대를 생성하던 문제 대응.
- 예측 신뢰도가 낮으면 단정 표현을 쓰지 않도록 지시한다.
- 과거 실측 조회(history) 의도를 추가했다. 기존에는 특정 날짜·기간을 물어도
  예측 경로로 흘러가 답하지 못했다. 실측값은 모델을 거치지 않으므로 예측
  신뢰도 문제와 무관하다.
"""
from __future__ import annotations

import json
import re

from . import history as history_mod
from . import knowledge, llm_client
from .planner import build_facts

# --- 규칙 기반 1차 분류 ---------------------------------------------------
# 운영 질문에서 자주 나타나는 어휘. 어느 한 그룹이라도 걸리면 operation 후보.
_ENERGY_WORDS = [
    "전력", "전기", "에너지", "요금", "비용", "소비", "부하", "난방", "냉방",
    "가동", "운영", "절감", "절약", "펌프", "관수", "조명", "돌리", "켜", "켤",
]
_TASK_WORDS = [
    "방제", "정비", "점검", "수리", "교체", "청소", "환기", "통풍", "작업",
    "살포", "약제", "농약", "소독",
]
# 시점 표현. "모레는 어때?"처럼 시점만 나오는 후속 질문을 놓치지 않기 위함.
_TIME_WORDS = [
    "오늘", "내일", "모레", "글피", "이번 주", "이번주", "다음 주", "다음주",
    "주말", "며칠", "언제", "날짜", "요일",
]
# 인사·자기소개 등 명백한 일반 질문
_GENERAL_WORDS = [
    "안녕", "반가", "누구", "뭐야", "뭐 할 수", "뭘 할 수", "소개", "도움말",
    "사용법", "어떻게 써", "고마", "감사",
]

_SYSTEM_OPERATION = (
    "당신은 스마트팜 운영을 돕는 한국어 AI 에이전트입니다. "
    "주어진 근거 데이터(JSON)에 기반해 농업인이 이해하기 쉬운 운영 권고안을 "
    "제시하세요.\n"
    "규칙:\n"
    "- 사용자가 물어본 것에만 답하세요. 최대 3문장이며, 같은 내용을 반복하지 "
    "마세요. 묻지 않은 항목(요금, 최근 평균 등)은 덧붙이지 마세요.\n"
    "- 숫자와 단위는 근거 데이터에 있는 값만 사용하세요. 없는 값은 만들지 마세요.\n"
    "- 데이터는 일(day) 단위입니다. 시간대(몇 시, 오전/오후, 새벽 등)는 데이터에 "
    "존재하지 않으므로 절대 언급하지 마세요.\n"
    "- 생육·병해충·작물 상태 등 데이터에 없는 정보는 지어내지 마세요.\n"
    "- 예측은 forecast.horizon_days일 뒤 하루치만 존재합니다. 그보다 먼 날짜"
    "(모레·이번 주 등)를 물으면 해당 날짜의 예측은 아직 없다고 먼저 밝히고, "
    "있는 정보만으로 답하세요. 없는 날짜의 수요를 추정하지 마세요.\n"
    "- 반드시 예측값을 먼저 제시하세요. 신뢰도가 낮더라도 숫자를 생략하지 말고, "
    "범위가 있으면 '예상 범위 A~B'로 함께 밝히세요. '단정하기 어렵다'는 말만 "
    "하고 값을 빼놓으면 안 됩니다.\n"
    "- 범위를 '신뢰도'라고 부르지 마세요. 신뢰도는 높음/보통/낮음 등급입니다.\n"
    "- JSON의 키 이름(forecast, predicted_energy, confidence 등)을 답변에 그대로 "
    "쓰지 마세요. 사람이 읽는 자연스러운 한국어로만 쓰세요.\n"
    "- 신뢰도가 '낮음'이면 값을 제시한 뒤 오차가 클 수 있다고 덧붙이세요.\n"
    "- 어떤 방법으로 산출했는지 한 번 언급하세요.\n"
    "- drivers가 있으면 수요가 높거나 낮은 이유로 함께 설명하되, 각 항목의 "
    "summary 문장을 그대로 인용하세요. 숫자를 직접 해석해 방향(높음/낮음)을 "
    "바꿔 쓰지 마세요.\n"
    "- task_check.findings가 있으면 해당 작업의 제약으로 안내하되, "
    "is_provisional이 true인 항목은 '참고 기준'임을 밝히세요."
)

_SYSTEM_GENERAL = (
    "당신은 스마트팜 운영을 돕는 한국어 AI 에이전트입니다. "
    "온실의 일일 에너지(난방+전기) 수요 예측을 근거로, 방제·정비 등 비필수 작업을 "
    "언제 진행하면 좋을지 안내하는 역할을 합니다. "
    "인사·자기소개·일반 질문에는 1~3문장으로 간단히 답하고, "
    "전력·설비 운영과 무관한 내용에 대해 수치를 지어내지 마세요. "
    "필요하면 사용자가 물어볼 수 있는 예시 질문을 안내하세요."
)

_SYSTEM_HISTORY = (
    "당신은 스마트팜 운영을 돕는 한국어 AI 에이전트입니다. "
    "사용자가 과거 실측 데이터를 물었고, 조회 결과(JSON)가 주어집니다.\n"
    "규칙:\n"
    "- summary 문장의 숫자와 날짜를 그대로 사용해 **1~2문장으로 짧게** 답하세요. "
    "같은 내용을 반복하거나 묻지 않은 내용을 덧붙이지 마세요.\n"
    "- 조회 결과에 없는 값은 절대 만들지 마세요.\n"
    "- 이 값은 예측이 아니라 실제 측정값입니다. 예측·전망으로 표현하지 마세요.\n"
    "- found가 false면 해당 기간의 데이터가 없다는 점과 조회 가능 기간을 안내하세요."
)

# 예측 수치·요금을 실제로 물어본 질문에서만 지표 카드를 노출한다.
# ("방제해도 될까?" 같은 권고 질문에 예측/평균/요금 카드까지 붙으면 산만하다.)
_NUMERIC_ASK = [
    "얼마", "몇", "수요", "요금", "비용", "가격", "예측", "전망", "사용량",
    "소비량", "kwh", "평균", "지표", "숫자",
]


def wants_numbers(question: str) -> bool:
    q = question.lower()
    return any(w in q for w in _NUMERIC_ASK)


_CLASSIFY_SYSTEM = (
    "너는 스마트팜 에이전트의 질문 분류기다. 사용자 질문이 다음 중 무엇인지 판단해 "
    "한 단어만 출력한다.\n"
    "operation: 에너지/전력/요금/설비 가동, 또는 방제·정비 같은 농작업을 "
    "언제 할지에 관한 질문. 시점(오늘/내일/모레/이번 주)만 언급한 후속 질문 포함.\n"
    "general: 인사, 자기소개, 사용법 문의, 그 외 잡담.\n"
    "반드시 operation 또는 general 중 하나만 출력한다."
)

HELP_TEXT = (
    "안녕하세요! 저는 스마트팜 에너지 운영을 돕는 AI 에이전트입니다. "
    "온실의 일일 에너지(난방+전기) 수요를 예측해 방제·정비 같은 비필수 작업을 "
    "언제 진행하면 좋을지 알려드립니다.\n\n"
    "예를 들어 이렇게 물어보세요:\n"
    "- \"내일 에너지 수요가 어느 정도일까?\"\n"
    "- \"방제 작업은 오늘 할까 내일 할까?\""
)

# 대화 이력에서 LLM에 넘길 최대 턴 수(사용자+어시스턴트 합계)
_HISTORY_TURNS = 6


def _rule_classify(question: str) -> tuple[str, float]:
    """규칙 기반 1차 분류. (의도, 신뢰도 0~1)를 반환한다."""
    q = question.lower().strip()

    # 과거 실측 조회가 최우선. 구체적 날짜·최대/최소·지난 기간 등이 신호이며,
    # 예측을 거치지 않으므로 다른 의도보다 먼저 판정한다.
    if history_mod.looks_like_history(question):
        return "history", 0.9

    hit_energy = any(w in q for w in _ENERGY_WORDS)
    hit_task = any(w in q for w in _TASK_WORDS)
    hit_time = any(w in q for w in _TIME_WORDS)
    hit_general = any(w in q for w in _GENERAL_WORDS)

    # 에너지/작업 어휘는 운영 질문의 강한 신호
    if hit_energy or hit_task:
        return "operation", 0.9
    # 시점 표현만 있는 경우("모레는 어때?") — 후속 질문일 가능성이 높다
    if hit_time and not hit_general:
        return "operation", 0.7
    if hit_general:
        return "general", 0.85
    # 판단 근거 없음 → LLM 보정 대상
    return "general", 0.3


def _llm_classify(question: str, history: list[dict] | None) -> str | None:
    """저신뢰 질문을 LLM으로 재분류한다. 실패 시 None."""
    msgs = [{"role": "system", "content": _CLASSIFY_SYSTEM}]
    if history:
        # 직전 대화를 함께 넘겨야 "모레는?" 같은 생략 질문을 판단할 수 있다
        msgs += [{"role": m["role"], "content": m["content"]}
                 for m in history[-2:]]
    msgs.append({"role": "user", "content": question})
    try:
        out = llm_client.chat(msgs, temperature=0.0, max_tokens=8).lower()
    except Exception:
        return None
    if "operation" in out:
        return "operation"
    if "general" in out:
        return "general"
    return None


def classify(question: str, history: list[dict] | None = None) -> dict:
    """의도 분류. 규칙 1차 → 저신뢰 시 LLM 보정."""
    intent, conf = _rule_classify(question)
    source = "rule"

    if conf < 0.6 and llm_client.available():
        refined = _llm_classify(question, history)
        if refined:
            intent, conf, source = refined, 0.75, "llm"

    return {"intent": intent, "confidence": conf, "source": source}


def _history_messages(history: list[dict] | None) -> list[dict]:
    """LLM 입력용 대화 이력. 최근 _HISTORY_TURNS개만 사용한다."""
    if not history:
        return []
    return [{"role": m["role"], "content": m["content"]}
            for m in history[-_HISTORY_TURNS:]
            if m.get("role") in ("user", "assistant") and m.get("content")]


_NUM_RE = re.compile(r"\d+(?:\.\d+)?")

# 근거에 없어도 허용할 일반 수치(문장 표현에 흔히 쓰임)
_NUM_WHITELIST = {0.0, 1.0, 2.0, 3.0, 100.0}


def unsupported_numbers(text: str, facts: dict) -> list[float]:
    """응답에 등장하지만 근거 데이터에 없는 수치를 찾아낸다.

    소형 로컬 모델이 조회 결과 뒤에 없는 날짜·기간을 덧붙이는 사례가 있어
    (예: 실제로는 5/29까지인데 "5월 31일까지 확인 가능"), 생성 결과를 근거와
    대조해 검증한다.
    """
    allowed = {float(x) for x in _NUM_RE.findall(
        json.dumps(facts, ensure_ascii=False, default=str))}
    allowed |= _NUM_WHITELIST
    found = {float(x) for x in _NUM_RE.findall(text)}
    return sorted(found - allowed)


def _verified_or_fallback(text: str, facts: dict, fallback: str,
                          require_value: float | None = None) -> tuple[str, bool]:
    """검증에 실패하면 근거 그대로의 폴백 문장을 쓴다. (텍스트, 검증통과여부)

    require_value: 반드시 답변에 포함되어야 하는 수치. 사용자가 값을 물었는데
                   소형 모델이 "예측하기 어렵습니다"로 숫자를 빼는 경우가 있어,
                   프롬프트 지시에만 의존하지 않고 코드로 확인한다.
    """
    if unsupported_numbers(text, facts):
        return fallback, False
    if require_value is not None and str(require_value) not in text:
        return fallback, False
    return text, True


def _fallback_operation(facts: dict) -> str:
    """LLM이 없을 때의 템플릿 응답. 신뢰도·근거·작업 제약을 함께 반영한다."""
    fc = facts["forecast"]
    sched = facts["schedule"]
    conf = facts["confidence"]

    unit = fc.get("energy_unit", "")
    rng = conf.get("range")
    band = f" (오차 감안 {rng[0]}~{rng[1]})" if rng else ""

    head = (f"내일({fc['horizon_days']}일 뒤) 예상 에너지 수요는 "
            f"약 {fc['predicted_energy']} {unit}{band}이고, "
            f"최근 평균 {fc['recent_avg_energy']} 대비 {sched['level']} 수준입니다.")

    # 물어보지 않은 항목(산출 방법·요금·평년 편차)까지 나열하면 장황해지므로,
    # 예측값 / 권고 / 작업 제약만 담는다. 나머지는 화면 지표와 근거 데이터에 있다.
    parts = [head, sched["advice"]]

    tc = facts.get("task_check")
    if tc and tc.get("findings"):
        f0 = tc["findings"][0]
        note = "참고 기준" if f0["is_provisional"] else "기준"
        parts.append(f"{tc['label']} 관련: {f0['reason']} "
                     f"(현재 {f0['label']} {f0['value']}{f0['unit']}, "
                     f"{note} {f0['threshold']}{f0['unit']}).")
    return " ".join(parts)


def answer(question: str, model_path=None,
           history: list[dict] | None = None) -> dict:
    """사용자 질문의 의도를 분류하여 적절한 응답을 생성한다.

    history: [{"role": "user"|"assistant", "content": str}, ...] 형태의 이전 대화.
             후속 질문("모레는?")의 맥락 유지에 사용한다.
    """
    cls = classify(question, history)
    intent = cls["intent"]
    hist_msgs = _history_messages(history)

    # --- 과거 실측 조회: 예측 모델을 거치지 않고 DB 값을 그대로 답한다 ---
    if intent == "history":
        record = history_mod.query(question, model_path=model_path)
        if record is None:
            # 기간 해석 실패 → 운영 질문으로 넘겨 평소 로직을 태운다
            intent = "operation"
        else:
            facts = {"history": record}
            if llm_client.available():
                try:
                    raw = llm_client.chat(
                        [{"role": "system", "content": _SYSTEM_HISTORY}]
                        + hist_msgs
                        + [{"role": "user", "content":
                            f"[질문]\n{question}\n\n[조회 결과(JSON)]\n"
                            + json.dumps(record, ensure_ascii=False, default=str)}])
                    text, ok = _verified_or_fallback(raw, facts, record["summary"])
                    return {"text": text, "facts": facts, "intent": "history",
                            "intent_meta": cls, "used_llm": True, "verified": ok}
                except Exception:
                    pass
            return {"text": record["summary"], "facts": facts, "intent": "history",
                    "intent_meta": cls, "used_llm": False}

    # --- 일반/인사 질문: 추천 데이터 주입 없이 역할 안내 ---
    if intent == "general":
        if llm_client.available():
            try:
                text = llm_client.chat(
                    [{"role": "system", "content": _SYSTEM_GENERAL}]
                    + hist_msgs
                    + [{"role": "user", "content": question}])
                return {"text": text, "facts": None, "intent": intent,
                        "intent_meta": cls, "used_llm": True}
            except Exception:
                pass
        return {"text": HELP_TEXT, "facts": None, "intent": intent,
                "intent_meta": cls, "used_llm": False}

    # --- 운영 질문: 예측·근거 데이터 기반 추천 ---
    task = knowledge.find_task(question)
    facts = build_facts(model_path=model_path, task=task)

    if llm_client.available():
        try:
            raw = llm_client.chat(
                [{"role": "system", "content": _SYSTEM_OPERATION}]
                + hist_msgs
                + [{"role": "user", "content":
                    f"[질문]\n{question}\n\n[근거 데이터(JSON)]\n"
                    + json.dumps(facts, ensure_ascii=False, default=str)}])
            # 수치를 물어본 질문이면 예측값이 답변에 반드시 들어가야 한다.
            # (권고 질문까지 강제하면 대부분 폴백으로 빠져 답이 획일화된다.
            #  단위 혼동은 근거에서 온실 전체 환산값을 빼는 것으로 예방한다.)
            need = (facts["forecast"]["predicted_energy"]
                    if wants_numbers(question) else None)
            text, ok = _verified_or_fallback(raw, facts,
                                             _fallback_operation(facts), need)
            return {"text": text, "facts": facts, "intent": intent,
                    "intent_meta": cls, "task": task, "used_llm": True,
                    "verified": ok, "show_metrics": wants_numbers(question)}
        except Exception as e:
            fb = _fallback_operation(facts)
            return {"text": fb + f"\n\n(LLM 호출 실패로 규칙 기반 응답: {e})",
                    "facts": facts, "intent": intent, "intent_meta": cls,
                    "task": task, "used_llm": False,
                    "show_metrics": wants_numbers(question)}

    return {"text": _fallback_operation(facts), "facts": facts,
            "intent": intent, "intent_meta": cls, "task": task,
            "used_llm": False, "show_metrics": wants_numbers(question)}
