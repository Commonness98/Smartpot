"""온톨로지 기반 의사결정 지식 계층.

작업 유형과 환경 제약의 정의는 ontology.py(OWL)에 있고, 이 모듈은 그 지식을
현재 관측값에 적용해 판단 결과를 만든다. 지식(무엇이 참인가)과 평가(지금 상태에
적용하면 어떻게 되는가)를 분리한 구조다.

이전 구현은 작업마다 성질을 모두 나열한 평면 JSON이었다. OWL 계층으로 옮기면서
"미룰 수 있는가", "온실을 여는가", "에너지 수요가 왜 영향을 주는가"는 개별 작업이
아니라 상위 개념에 한 번만 정의되고, 하위 작업이 상속받아 도출된다.

[주의] 임계값은 도메인 전문가 검증 전의 잠정값이다. is_provisional=True 인 항목은
호출부에서 '참고 기준'으로만 제시하고 단정하지 않는다.
"""
from __future__ import annotations

from . import ontology

# 지식 정의는 ontology 모듈이 갖고 있다. 기존 호출부가 knowledge를 통해 쓰던
# 조회 함수는 그대로 위임해 인터페이스를 유지한다.
feature_label = ontology.feature_label
task_names = ontology.task_names
find_task = ontology.find_task
hierarchy_lines = ontology.hierarchy_lines
save_owl = ontology.save_owl


def reload() -> None:
    """지식 캐시를 비운다(온톨로지 수정 후 앱 재시작 없이 반영)."""
    ontology._task_classes.cache_clear()


def _violates(value: float, op: str, threshold: float) -> bool:
    if op == "gt":
        return value > threshold
    if op == "lt":
        return value < threshold
    return False


def evaluate(task: str, observations: dict[str, float]) -> dict:
    """작업 제약을 현재 관측값에 대해 평가한다.

    observations: {역할명: 값} (예: {"windspeed": 6.1, "indoor_humid": 88})
                  값이 없는 역할은 건너뛴다.

    반환에는 계층에서 도출된 성질(deferrable, needs_ventilation, energy_note)이
    함께 담긴다. 개별 작업에 적혀 있지 않아도 상위 개념에서 상속된 값이다.
    """
    if task not in ontology.task_names():
        return {"task": task, "label": task, "known": False,
                "findings": [], "blocked": False, "has_caution": False,
                "provisional": False}

    findings = []
    for c in ontology.constraints_of(task):
        role = c["feature"]
        if role not in observations or observations[role] is None:
            continue
        value = float(observations[role])
        if not _violates(value, c["op"], float(c["threshold"])):
            continue
        findings.append({
            "constraint": c["name"],
            "feature": role,
            "label": ontology.feature_label(role),
            "value": round(value, 2),
            "threshold": c["threshold"],
            "unit": c["unit"],
            "severity": c["severity"],
            "reason": c["reason"],
            "is_provisional": not c["verified"],
        })

    return {
        "task": task,
        "label": ontology.label_of(task),
        "known": True,
        # --- 계층에서 도출된 성질 (개별 작업에 명시하지 않음) ---
        "concepts": ontology.ancestors_of(task),
        "deferrable": ontology.is_deferrable(task),
        "needs_ventilation": ontology.needs_ventilation(task),
        "energy_note": ontology.energy_note(task),
        # --- 현재 관측값에 대한 평가 ---
        "findings": findings,
        "blocked": any(f["severity"] == "block" for f in findings),
        "has_caution": any(f["severity"] == "caution" for f in findings),
        "provisional": any(f["is_provisional"] for f in findings),
    }
