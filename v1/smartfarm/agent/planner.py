"""추론·계획(Reasoning & Planning) 엔진.

예측 모델 결과 + 전기요금(TOU) 정보를 취합해 '구조화된 권고 사실(facts)'을 계산한다.
자연어 변환은 agent.core가 담당. model_path를 지정하면 해당(활성) 데이터셋 모델을 사용한다.

실측 에너지 소비(난방+전기)가 실제 온실 데이터에서는 일(day) 단위로만 제공되어
시간대별(0~23시) 수요 분포를 알 수 없다. 그래서 기존의 "몇 시~몇 시에 가동하라"는
시간창 추천 대신, "내일 수요가 평소보다 높은지 낮은지"를 보고 방제·정비 같은
비필수 작업을 당길지 미룰지 권고하는 일(day) 단위 방식으로 동작한다.
"""
from __future__ import annotations

from .. import config, registry
from ..models.predict import DEFAULT_MODEL_PATH, forecast_next, recent_series

# 최근 평균 대비 이 비율 이상 벗어나면 "높음/낮음"으로 분류
_LEVEL_THRESHOLD = 0.1


def _model_path(model_path=None):
    if model_path:
        return model_path
    active = registry.get_active()
    return active["model_path"] if active else DEFAULT_MODEL_PATH


def _level(predicted: float, recent_avg: float) -> str:
    if recent_avg <= 0:
        return "보통"
    diff = (predicted - recent_avg) / recent_avg
    if diff <= -_LEVEL_THRESHOLD:
        return "낮음"
    if diff >= _LEVEL_THRESHOLD:
        return "높음"
    return "보통"


def next_day_outlook(model_path=None) -> dict:
    """내일(HORIZON일 뒤) 예측 에너지와 최근 평균 대비 수준·근사 비용을 계산."""
    mp = _model_path(model_path)
    fc = forecast_next(mp)
    level = _level(fc["predicted_energy"], fc["recent_avg_energy"])
    avg_rate = config.avg_tou_rate()  # 시간대별 실측 분포를 몰라 평균 단가로 근사
    return {
        **fc,
        "level": level,
        "avg_rate": avg_rate,
        "estimated_cost": round(fc["predicted_energy"] * avg_rate, 1),
    }


def recommend_schedule(model_path=None) -> dict:
    """예측 수준에 따라 방제·정비 등 비필수 작업 일정을 권고."""
    outlook = next_day_outlook(model_path)
    level = outlook["level"]
    if level == "낮음":
        advice = ("내일은 에너지 수요가 평소보다 낮을 것으로 예상됩니다. "
                  "방제·정비 등 비필수 작업을 진행하기 좋은 날입니다.")
    elif level == "높음":
        advice = ("내일은 에너지 수요가 평소보다 높을 것으로 예상됩니다. "
                  "비필수 작업은 다른 날로 미루는 것을 권장합니다.")
    else:
        advice = "내일 에너지 수요는 평소와 비슷할 것으로 예상됩니다. 평소대로 작업을 진행해도 무방합니다."
    return {**outlook, "advice": advice}


def build_facts(model_path=None) -> dict:
    """LLM에 전달할 구조화된 근거 데이터 묶음."""
    mp = _model_path(model_path)
    sched = recommend_schedule(mp)
    return {
        "forecast": {k: sched[k] for k in
                    ("as_of", "horizon_days", "predicted_energy", "recent_avg_energy")},
        "schedule": {k: sched[k] for k in ("level", "advice", "estimated_cost", "avg_rate")},
        "recent_series": recent_series(mp, days=30),
        "tariff_bands": {
            k: {"hours": v["hours"], "rate": v["rate"]}
            for k, v in config.TOU_TARIFF.items()
        },
    }
