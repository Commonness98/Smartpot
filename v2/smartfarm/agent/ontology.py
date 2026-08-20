"""온실 작업 의사결정 온톨로지 (OWL).

작업 유형과 환경 제약을 OWL 클래스 계층으로 정의한다. 이전 구현은 작업마다
성질을 모두 나열한 평면 JSON이어서, 작업을 추가할 때 같은 성질을 매번 다시
써야 했고 "방제는 미룰 수 있는가" 같은 판단이 명시된 항목에만 의존했다.

계층으로 표현하면 상위 개념에 한 번 정의한 성질을 하위가 물려받는다.
    방제 ⊑ 개방작업 ⊑ 비필수작업 ⊑ 농작업
따라서 "방제는 미룰 수 있다"를 어디에도 적지 않아도 계층에서 도출된다.

[범위] 완전한 기술논리(DL) 추론기(HermiT 등)는 Java 런타임을 요구하므로 쓰지
않는다. 여기서 활용하는 것은 클래스 계층에 기반한 상속 추론이며, owlready2가
순수 파이썬으로 처리한다. 보고서에는 "온톨로지 기반 지식 표현과 계층 추론"으로
기술하는 것이 정확하다.

[주의] 임계값은 도메인 전문가 검증 전의 잠정값이다. verified=False 인 제약은
호출부에서 '참고 기준'으로만 제시하고 단정하지 않는다.
"""
from __future__ import annotations

from functools import lru_cache
from pathlib import Path

from owlready2 import DataProperty, ObjectProperty, Thing, get_ontology

ONTOLOGY_IRI = "http://smartfarm.gitrc/ontology/greenhouse-task.owl"
OWL_PATH = Path(__file__).resolve().parent / "knowledge" / "greenhouse_task.owl"

_onto = get_ontology(ONTOLOGY_IRI)

with _onto:
    # --- 개념 계층 -------------------------------------------------------
    class 농작업(Thing):
        """온실에서 수행하는 모든 작업의 최상위 개념."""

    class 필수작업(농작업):
        """생육 유지에 반드시 필요해 시점을 미룰 수 없는 작업(관수·난방 등)."""

    class 비필수작업(농작업):
        """생육에 즉각적 영향이 없어 시점 조정이 가능한 작업.
        에너지 수요가 높은 날에는 미루는 것을 권고할 수 있다."""

    class 개방작업(비필수작업):
        """출입·환기로 온실을 여는 작업.
        난방 수요가 큰 날에는 유출된 열을 다시 데워야 해 추가 비용이 발생한다."""

    class 밀폐작업(비필수작업):
        """온실을 열지 않고 수행 가능한 작업. 난방 부하와 무관하다."""

    # --- 개별 작업 -------------------------------------------------------
    # 아래 클래스에는 성질을 따로 쓰지 않는다. 상위에서 물려받는다.
    class 방제(개방작업):
        """약제 살포. 살포 후 환기가 필요하다."""

    class 정비(개방작업):
        """설비 점검·수리. 자재 반입으로 출입이 잦다."""

    class 환기작업(개방작업):
        """창 개폐를 통한 강제 환기."""

    class 점검기록(밀폐작업):
        """생육 기록·계측 등 온실을 열지 않는 관리 작업."""

    class 관수(필수작업):
        """급액. 생육에 직결되어 미룰 수 없다."""

    # --- 환경 제약 -------------------------------------------------------
    class 환경제약(Thing):
        """특정 환경 변수 값이 기준을 넘으면 작업 수행에 영향을 주는 조건."""

    class 제약있음(ObjectProperty):
        """작업 ─ 환경제약 연결. 하위 클래스로 상속된다."""
        domain = [농작업]
        range = [환경제약]

    class 대상변수(DataProperty):
        """제약이 참조하는 표준 스키마 역할명(windspeed 등)."""
        domain = [환경제약]
        range = [str]

    class 비교연산(DataProperty):
        """gt(초과) 또는 lt(미만)."""
        domain = [환경제약]
        range = [str]

    class 임계값(DataProperty):
        domain = [환경제약]
        range = [float]

    class 단위(DataProperty):
        domain = [환경제약]
        range = [str]

    class 심각도(DataProperty):
        """block(부적합) 또는 caution(주의)."""
        domain = [환경제약]
        range = [str]

    class 사유(DataProperty):
        domain = [환경제약]
        range = [str]

    class 검증됨(DataProperty):
        """도메인 전문가 검증 여부. False면 잠정값이다."""
        domain = [환경제약]
        range = [bool]

    class 질의어(DataProperty):
        """질문에서 이 작업을 식별하는 표현."""
        domain = [농작업]
        range = [str]

    class 표시명(DataProperty):
        domain = [농작업]
        range = [str]


def _constraint(name, feature, op, threshold, unit, severity, reason, verified=False):
    c = 환경제약(name)
    c.대상변수 = [feature]
    c.비교연산 = [op]
    c.임계값 = [float(threshold)]
    c.단위 = [unit]
    c.심각도 = [severity]
    c.사유 = [reason]
    c.검증됨 = [verified]
    return c


with _onto:
    # --- 제약 개체 -------------------------------------------------------
    풍속과다 = _constraint(
        "풍속과다", "windspeed", "gt", 5.0, "m/s", "block",
        "풍속이 높으면 약제가 흩날려 방제 효과가 떨어지고 작업자 노출 위험이 있습니다")
    고습도 = _constraint(
        "고습도", "indoor_humid", "gt", 90.0, "%", "caution",
        "습도가 매우 높으면 약액이 마르지 않아 약해가 발생할 수 있습니다")
    실내고온 = _constraint(
        "실내고온", "indoor_temp", "gt", 35.0, "℃", "caution",
        "실내 온도가 높으면 작업자 온열 부담이 커집니다")
    외기저온 = _constraint(
        "외기저온", "out_temp", "lt", 5.0, "℃", "caution",
        "외부 기온이 낮을 때 환기하면 난방 부하가 크게 늘어납니다")

    # --- 작업별 개별 제약 및 표기 ----------------------------------------
    방제.제약있음 = [풍속과다, 고습도]
    방제.표시명 = ["방제(약제 살포)"]
    방제.질의어 = ["방제", "약제", "살포", "농약", "소독"]

    정비.제약있음 = [실내고온]
    정비.표시명 = ["설비 정비"]
    정비.질의어 = ["정비", "점검", "수리", "교체", "청소"]

    환기작업.제약있음 = [외기저온]
    환기작업.표시명 = ["환기 작업"]
    환기작업.질의어 = ["환기", "통풍", "창 개폐"]

    점검기록.표시명 = ["생육 기록·계측"]
    점검기록.질의어 = ["기록", "계측", "측정", "관찰"]

    관수.표시명 = ["관수(급액)"]
    관수.질의어 = ["관수", "급액", "물주기", "관주"]


# --- 조회 API --------------------------------------------------------------
# 표준 스키마 역할명 → 사람이 읽는 이름
FEATURE_LABELS = {
    "power_target": "에너지 수요", "out_temp": "외부 기온",
    "out_humid": "외부 습도", "indoor_temp": "실내 온도",
    "indoor_humid": "실내 습도", "co2": "CO2 농도",
    "light": "일사량", "windspeed": "풍속", "pressure": "기압",
}


# 에너지 수요와 작업 시점을 잇는 근거. 개별 작업이 아니라 **개념 단위**로 둔다.
# 하위 작업은 계층을 따라 물려받으므로 작업을 추가해도 다시 쓰지 않는다.
_CONCEPT_NOTES = {
    "개방작업": ("출입·환기로 온실을 열게 되어, 난방 수요가 큰 날에는 유출된 열을 "
               "다시 데우는 만큼 비용이 더 듭니다"),
    "밀폐작업": "온실을 열지 않아 난방 부하에 거의 영향을 주지 않습니다",
    "필수작업": "생육에 직결되어 에너지 수요와 무관하게 예정대로 수행해야 합니다",
}


def _first(values, default=None):
    return values[0] if values else default


@lru_cache(maxsize=1)
def _task_classes() -> dict:
    """질의어가 정의된 구체 작업 클래스 목록."""
    return {c.name: c for c in 농작업.descendants()
            if c is not 농작업 and getattr(c, "질의어", None)}


def task_names() -> list[str]:
    return list(_task_classes().keys())


def find_task(question: str) -> str | None:
    """질문에서 작업 유형을 식별한다."""
    q = question.lower()
    for name, cls in _task_classes().items():
        for kw in cls.질의어:
            if kw.lower() in q:
                return name
    return None


def feature_label(role: str) -> str:
    return FEATURE_LABELS.get(role, role)


def ancestors_of(task: str) -> list[str]:
    """상위 개념 목록(자기 자신 제외). 계층 추론 결과를 확인·설명하는 데 쓴다."""
    cls = _task_classes().get(task)
    if not cls:
        return []
    return [c.name for c in cls.ancestors()
            if c not in (cls, Thing) and c.name != "Thing"]


def _inherits(task: str, concept) -> bool:
    cls = _task_classes().get(task)
    return bool(cls) and concept in cls.ancestors()


def is_deferrable(task: str) -> bool:
    """시점을 미룰 수 있는 작업인가. 비필수작업 상속 여부로 판단한다."""
    return _inherits(task, 비필수작업)


def needs_ventilation(task: str) -> bool:
    """온실을 여는 작업인가. 개방작업 상속 여부로 판단한다."""
    return _inherits(task, 개방작업)


def constraints_of(task: str) -> list[dict]:
    """작업에 적용되는 환경 제약. 상위 클래스에 정의된 것도 함께 모은다."""
    cls = _task_classes().get(task)
    if not cls:
        return []
    seen, out = set(), []
    for ancestor in [cls] + [a for a in cls.ancestors() if a is not cls]:
        for c in getattr(ancestor, "제약있음", []) or []:
            if c.name in seen:
                continue
            seen.add(c.name)
            out.append({
                "name": c.name,
                "feature": _first(c.대상변수),
                "op": _first(c.비교연산),
                "threshold": _first(c.임계값),
                "unit": _first(c.단위, ""),
                "severity": _first(c.심각도, "caution"),
                "reason": _first(c.사유, ""),
                "verified": bool(_first(c.검증됨, False)),
            })
    return out


def energy_note(task: str) -> str | None:
    """에너지 수요가 이 작업에 왜 영향을 주는지에 대한 설명.

    개념 단위로 정의된 설명을 계층을 따라 올라가며 찾는다. 개별 작업에는
    적혀 있지 않지만 상위 개념에서 물려받는다.
    """
    cls = _task_classes().get(task)
    if not cls:
        return None
    for ancestor in cls.ancestors():
        if ancestor.name in _CONCEPT_NOTES:
            return _CONCEPT_NOTES[ancestor.name]
    return None


def label_of(task: str) -> str:
    cls = _task_classes().get(task)
    return _first(getattr(cls, "표시명", None), task) if cls else task


def hierarchy_lines() -> list[str]:
    """계층 구조를 들여쓰기 문자열로 출력한다(문서·발표 자료용)."""
    lines = []

    def walk(cls, depth=0):
        lines.append("  " * depth + cls.name)
        for sub in sorted(cls.subclasses(), key=lambda c: c.name):
            walk(sub, depth + 1)

    walk(농작업)
    return lines


def save_owl(path=OWL_PATH) -> str:
    """온톨로지를 표준 OWL 파일로 저장한다(Protégé 등에서 열람 가능)."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    _onto.save(file=str(path), format="rdfxml")
    return str(path)
