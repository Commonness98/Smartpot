"""온톨로지 기반 의사결정 지식 계층.

작업 유형(방제·정비·환기 등)과 환경 변수 사이의 제약 관계를 코드가 아닌
지식 파일(knowledge/tasks.json)로 분리해 정의하고, 이를 평가하는 경량 규칙
엔진을 제공한다. 지식을 추가·수정할 때 코드를 고칠 필요가 없다.

정식 온톨로지 스택(RDF/OWL + 추론기)은 본 과제 규모 대비 과도하므로,
'개념(작업) - 속성(환경 변수) - 제약(조건)' 구조를 JSON으로 표현하는
규칙 기반 지식표현으로 경량 구현했다.

[주의] tasks.json의 threshold는 도메인 전문가 검증 전 잠정값이다.
verified=false 인 조건은 is_provisional=True로 표시되어 나가며,
호출부(planner/core)는 이를 단정하지 않고 '참고' 수준으로 다뤄야 한다.
"""
from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path

_KNOWLEDGE_PATH = Path(__file__).resolve().parent / "knowledge" / "tasks.json"


@lru_cache(maxsize=1)
def _kb() -> dict:
    with open(_KNOWLEDGE_PATH, encoding="utf-8") as f:
        return json.load(f)


def reload() -> None:
    """지식 파일을 수정한 뒤 캐시를 비운다(앱 재시작 없이 반영)."""
    _kb.cache_clear()


def feature_label(role: str) -> str:
    """표준 스키마 역할명 → 사람이 읽는 이름."""
    return _kb().get("feature_labels", {}).get(role, role)


def task_names() -> list[str]:
    return list(_kb().get("tasks", {}).keys())


def find_task(question: str) -> str | None:
    """질문 문장에서 작업 유형을 식별한다. 없으면 None."""
    q = question.lower()
    for name, spec in _kb().get("tasks", {}).items():
        for kw in spec.get("keywords", []):
            if kw.lower() in q:
                return name
    return None


def _violates(value: float, op: str, threshold: float) -> bool:
    if op == "gt":
        return value > threshold
    if op == "lt":
        return value < threshold
    return False


def evaluate(task: str, observations: dict[str, float]) -> dict:
    """작업 유형 제약을 현재 관측값에 대해 평가한다.

    observations: {역할명: 값} (예: {"windspeed": 6.1, "indoor_humid": 88})
                  값이 없는 역할은 그냥 건너뛴다.

    반환: {task, label, essential, findings[], blocked, has_caution, provisional}
      findings 각 항목: {feature, label, value, threshold, unit, severity,
                        reason, is_provisional}
    """
    spec = _kb().get("tasks", {}).get(task)
    if not spec:
        return {"task": task, "label": task, "essential": None,
                "findings": [], "blocked": False, "has_caution": False,
                "provisional": False}

    findings = []
    for cond in spec.get("conditions", []):
        role = cond["feature"]
        if role not in observations or observations[role] is None:
            continue
        value = float(observations[role])
        if not _violates(value, cond["op"], float(cond["threshold"])):
            continue
        findings.append({
            "feature": role,
            "label": feature_label(role),
            "value": round(value, 2),
            "threshold": cond["threshold"],
            "unit": cond.get("unit", ""),
            "severity": cond.get("severity", "caution"),
            "reason": cond.get("reason", ""),
            "is_provisional": not cond.get("verified", False),
        })

    return {
        "task": task,
        "label": spec.get("label", task),
        "essential": spec.get("essential"),
        # 개방(출입·환기)이 필요한 작업인지. 에너지 수요와 작업 시점을 잇는 근거로 쓴다.
        "needs_ventilation": spec.get("needs_ventilation", False),
        "energy_note": spec.get("energy_note"),
        "findings": findings,
        "blocked": any(f["severity"] == "block" for f in findings),
        "has_caution": any(f["severity"] == "caution" for f in findings),
        "provisional": any(f["is_provisional"] for f in findings),
    }
