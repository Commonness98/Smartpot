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

# --- 답변 범위 밖 주제 -----------------------------------------------------
# 이 시스템이 가진 데이터는 에너지 소비 실측과 환경 센서 실측뿐이다. 그 밖의
# 주제를 물으면 LLM이 손에 쥔 에너지 수치를 질문에 억지로 끼워맞춰 답을 지어낸다
# (예: "병해충 있어?" -> 어제 에너지값 0.26을 "병해충 예측"이라고 제시).
# 프롬프트의 금지 지시만으로는 막히지 않아, 아예 LLM에 넘기지 않는다.
_OUT_OF_SCOPE = {
    "병해충": ["병해충", "해충", "벌레", "진딧물", "응애", "총채", "곰팡이",
              "흰가루", "잿빛", "감염", "발병", "병징"],
    "생육·수확": ["생육", "수확", "착과", "개화", "열매", "당도", "품질",
                "상품성", "출하", "파종", "정식"],
    "날씨 예보": ["날씨", "예보", "강수", "비 와", "비가 와", "눈이 와", "태풍"],
    "토양·양액": ["토양", "양액", "비료", "시비", "관수량", "급액", "배지"],
}

_SCOPE_NOTE = (
    "이 시스템이 다루는 데이터는 온실의 **일일 에너지 소비 실측값**과 "
    "**환경 센서 실측값**(실내 온도·습도·CO2·일사량, 외부 기온·습도·풍속)입니다.\n\n"
    "다음은 답변드릴 수 있습니다:\n"
    "- 과거 특정 날짜·기간의 에너지 사용량\n"
    "- 내일 에너지 수요 예측과 예상 전기요금\n"
    "- 방제·정비 등 비필수 작업의 시점 권고"
)


def detect_out_of_scope(question: str) -> str | None:
    """보유 데이터로 답할 수 없는 주제인지 판별한다. 해당 분야명 또는 None."""
    q = question.lower()
    # 작업 시점 질문은 범위 안이다("병해충 방제 언제 할까?" 등)
    if any(w in q for w in _TASK_WORDS):
        return None
    for topic, words in _OUT_OF_SCOPE.items():
        if any(w in q for w in words):
            return topic
    return None


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
    "history: 이미 지나간 기간의 실측값을 묻는 질문. 과거 시제이거나 특정 날짜·"
    "기간을 가리킨다. 예) '어제는 얼마나 사용했어', '5월 3일 얼마 썼어', "
    "'지난주 평균은', '제일 많이 쓴 날은', '1월이랑 5월 비교해줘'\n"
    "operation: 앞으로의 에너지/전력/요금 전망, 또는 방제·정비 같은 농작업을 "
    "언제 할지에 관한 질문. 예) '내일 방제해도 될까', '모레는 어때'\n"
    "general: 인사, 자기소개, 사용법 문의, 그 외 잡담.\n"
    "핵심 기준: 이미 지나간 일이면 history, 앞으로의 일이면 operation이다.\n"
    "반드시 history, operation, general 중 하나만 출력한다."
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

    # 보유 데이터로 답할 수 없는 주제는 다른 판정보다 먼저 걸러낸다.
    # 시점 단어("언제", "오늘")만으로 운영 질문이 되어 엉뚱한 답을 만들던 경로 차단.
    if detect_out_of_scope(question):
        return "out_of_scope", 0.95

    hit_energy = any(w in q for w in _ENERGY_WORDS)
    hit_task = any(w in q for w in _TASK_WORDS)
    hit_time = any(w in q for w in _TIME_WORDS)
    hit_general = any(w in q for w in _GENERAL_WORDS)

    # 과거 실측 조회가 최우선. 예측을 거치지 않으므로 다른 의도보다 먼저 판정한다.
    if history_mod.looks_like_history(question):
        # 데이터 보유 기간 질문("언제부터 언제까지 있어?")은 시점 단어를 포함하지만
        # 미래 예측과 무관하므로 아래 혼동 판정에서 제외한다.
        if history_mod.is_coverage_question(question):
            return "history", 0.95
        # 과거와 미래가 섞인 질문("어제 방제했는데 내일도 해야 하나?")은 규칙으로
        # 단정하기 어렵다. 신뢰도를 낮춰 LLM 판단에 맡긴다.
        if hit_time:
            return "history", 0.5
        return "history", 0.9

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
    # history를 먼저 확인한다. 과거 조회는 예측을 거치지 않아야 하므로,
    # 애매할 때 operation으로 흘려보내면 잘못된 예측값을 답하게 된다.
    for label in ("history", "operation", "general"):
        if label in out:
            return label
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


def trim_sentences(text: str, limit: int) -> str:
    """문장 수를 제한한다.

    프롬프트로 길이를 지시해도 소형 모델이 자주 초과하고, 뒤로 갈수록 같은 말을
    반복하거나 근거 없는 문장을 덧붙이는 경향이 있어 코드로 잘라낸다.
    """
    parts = re.split(r"(?<=[.!?])\s+", text.strip())
    parts = [p for p in parts if p]
    if len(parts) <= limit:
        return text.strip()
    return " ".join(parts[:limit])


_RE_DATE_ISO = re.compile(r"(\d{4})-(\d{1,2})-(\d{1,2})")
_RE_DATE_KO = re.compile(r"(\d{4})\s*년\s*(\d{1,2})\s*월\s*(\d{1,2})\s*일")


def _dates_in(text: str) -> set[str]:
    """텍스트에 등장하는 날짜를 YYYY-MM-DD로 정규화해 모은다."""
    out = set()
    for rx in (_RE_DATE_ISO, _RE_DATE_KO):
        for m in rx.finditer(text):
            y, mo, d = (int(g) for g in m.groups())
            out.add(f"{y:04d}-{mo:02d}-{d:02d}")
    return out


def unsupported_dates(text: str, facts: dict) -> list[str]:
    """응답에 등장하지만 근거에 없는 날짜를 찾아낸다.

    수치 검증만으로는 걸러지지 않는 오류가 있다. 예를 들어 실제 조회 결과가
    5월 28일인데 모델이 5월 29일이라 답해도, 29가 데이터 기간 끝으로 근거에
    존재하므로 숫자 검증은 통과해 버린다. 날짜는 따로 대조한다.
    """
    allowed = _dates_in(json.dumps(facts, ensure_ascii=False, default=str))
    return sorted(_dates_in(text) - allowed)


def _verified_or_fallback(text: str, facts: dict, fallback: str,
                          require_value: float | None = None,
                          require_date: str | None = None) -> tuple[str, bool]:
    """검증에 실패하면 근거 그대로의 폴백 문장을 쓴다. (텍스트, 검증통과여부)

    require_value: 반드시 답변에 포함되어야 하는 수치. 사용자가 값을 물었는데
                   소형 모델이 "예측하기 어렵습니다"로 숫자를 빼는 경우가 있어,
                   프롬프트 지시에만 의존하지 않고 코드로 확인한다.
    require_date:  조회 대상 날짜. 답변이 날짜를 언급한다면 반드시 이 날짜여야 한다.
                   조회 결과는 5월 28일인데 5월 29일이라 답하는 사례가 있었고,
                   그 날짜도 데이터 기간 안이라 일반 날짜 대조로는 걸러지지 않는다.
    """
    if unsupported_numbers(text, facts) or unsupported_dates(text, facts):
        return fallback, False
    if require_value is not None and str(require_value) not in text:
        return fallback, False
    if require_date:
        mentioned = _dates_in(text)
        if mentioned and require_date not in mentioned:
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

    # --- 답변 범위 밖: LLM을 거치지 않고 고정 문구로 답한다 ---
    # 근거가 없는 주제라 생성에 맡기면 반드시 지어낸다. 정직하게 한계를 밝힌다.
    if intent == "out_of_scope":
        topic = detect_out_of_scope(question) or "해당 주제"
        return {"text": (f"{topic}에 대한 정보는 저장된 데이터에 없어 "
                         f"답변드릴 수 없습니다.\n\n{_SCOPE_NOTE}"),
                "facts": None, "intent": intent, "intent_meta": cls,
                "used_llm": False, "show_metrics": False}

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
                    # 대화 이력은 넘기지 않는다. 조회 결과만으로 답이 완결되는데,
                    # 이전 대화가 섞이면 앞서 조회한 기간을 "데이터가 없다"고
                    # 단정하는 등 사실과 반대되는 문장을 만들어냈다.
                    raw = llm_client.chat(
                        [{"role": "system", "content": _SYSTEM_HISTORY},
                         {"role": "user", "content":
                            f"[질문]\n{question}\n\n[조회 결과(JSON)]\n"
                            + json.dumps(record, ensure_ascii=False, default=str)}])
                    raw = trim_sentences(raw, 2)
                    text, ok = _verified_or_fallback(
                        raw, facts, record["summary"],
                        require_date=record.get("date"))
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
                text = trim_sentences(text, 2)
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
            raw = trim_sentences(raw, 3)
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
