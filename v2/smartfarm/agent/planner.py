"""추론·계획(Reasoning & Planning) 엔진.

예측 모델 결과와 도메인 지식을 취합해 '구조화된 권고 사실(facts)'을 계산한다.
자연어 변환은 agent.core가 담당한다. model_path를 지정하면 해당(활성) 데이터셋
모델을 사용한다.

실측 에너지 소비(난방+전기)가 일(day) 단위로만 제공되어 시간대별(0~23시) 수요
분포를 알 수 없다. 따라서 "몇 시에 가동하라"가 아니라 "내일 수요가 평소보다
높은지 낮은지"를 근거로 비필수 작업을 당길지 미룰지 권고한다.

v2 변경점
- 근거 데이터 정합성: 시간대별 요금표(TOU hours)를 facts에서 제거했다. 일 단위
  예측이라 시간대는 근거가 없는데도 이를 LLM에 넘겨, 존재하지 않는 시간대
  ("저녁 9시~새벽 2시" 등)를 생성하는 원인이 되었다.
- 예측 신뢰도 판정: 모델의 기록된 테스트 성능과 지속성(내일=오늘) 기준선을 비교해
  신뢰도를 산출한다. 낮으면 단정하지 않도록 core가 참고한다.
- 판단 근거 제시: 최근 환경 변수가 평년 대비 얼마나 벗어났는지 계산해 함께 제공한다.
- 온톨로지 규칙 연동: 질문에 담긴 작업 유형의 제약을 knowledge 모듈로 평가한다.
"""
from __future__ import annotations

import numpy as np

import torch

from .. import config, registry
from ..db import read_sensor_data
from ..models.power_lstm import StandardScalerNP, add_time_features
from ..models.predict import DEFAULT_MODEL_PATH, forecast_next, recent_series
from ..models.predict import _load as _load_ckpt   # 체크포인트(table 등) 조회용
from ..schema import BASE_FEATURE_ROLES, TARGET_FEATURE
from . import knowledge

# 최근 평균 대비 이 비율 이상 벗어나면 "높음/낮음"으로 분류
_LEVEL_THRESHOLD = 0.1

# 근거로 제시할 환경 변수 개수와 최소 편차(표준편차 배수)
_MAX_DRIVERS = 3
_DRIVER_MIN_Z = 0.8

# 최근 구간 길이(일). 평년 대비 비교의 '최근' 정의.
_RECENT_WINDOW = 7


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


# --- 데이터 접근 -----------------------------------------------------------

def dataset_frame(model_path):
    """해당 모델이 학습한 테이블의 표준 컬럼 프레임을 읽어온다."""
    _, ckpt = _load_ckpt(model_path)
    return read_sensor_data(table=ckpt.get("table") or config.SENSOR_TABLE)


def _dataset_metrics(model_path) -> dict:
    """레지스트리에서 해당 모델의 기록된 학습 지표를 찾는다."""
    for ds in registry.list_datasets():
        if str(ds.get("model_path")) == str(model_path):
            return ds.get("metrics", {}) or {}
    active = registry.get_active()
    return (active or {}).get("metrics", {}) or {}


# --- 판단 근거(왜 높은가/낮은가) -------------------------------------------

def latest_observations(model_path=None) -> dict:
    """가장 최근 관측일의 환경 변수 값. 온톨로지 규칙 평가 입력으로 쓴다."""
    df = dataset_frame(_model_path(model_path))
    if df.empty:
        return {}
    last = df.iloc[-1]
    obs = {r: float(last[r]) for r in BASE_FEATURE_ROLES
           if r in df.columns and last[r] is not None}

    # 파생 변수: 실내외 온도차. 온실을 열었을 때 빠져나가는 열의 크기를 좌우하며,
    # 본 데이터에서 에너지 소비와의 상관이 0.71로 단일 변수 중 가장 높다.
    if "indoor_temp" in obs and "out_temp" in obs:
        obs["temp_gap"] = round(obs["indoor_temp"] - obs["out_temp"], 2)
    return obs


def _energy_unit(model_path=None) -> str:
    """활성 데이터셋에 기록된 에너지 타깃 단위. 모르면 빈 문자열."""
    mp = str(_model_path(model_path))
    for ds in registry.list_datasets():
        if str(ds.get("model_path")) == mp:
            return (ds.get("units") or {}).get(TARGET_FEATURE, "")
    active = registry.get_active() or {}
    return (active.get("units") or {}).get(TARGET_FEATURE, "")


def dataset_quantile(model_path=None):
    """활성 데이터셋의 분포 백분위를 구하는 함수를 돌려준다.

    온톨로지의 백분위 기준 제약이 데이터셋마다 다른 임계값을 갖도록 하는 데 쓴다.
    값이 없거나 상수인 항목은 None을 반환해 해당 제약을 건너뛰게 한다.
    """
    df = dataset_frame(_model_path(model_path))
    if "indoor_temp" in df.columns and "out_temp" in df.columns:
        df = df.assign(temp_gap=df["indoor_temp"].astype(float)
                       - df["out_temp"].astype(float))

    def q(role: str, p: float):
        if role not in df.columns or df.empty:
            return None
        s = df[role].astype(float)
        if s.nunique() <= 1:
            return None
        return float(s.quantile(p))

    return q


def demand_drivers(model_path=None) -> list[dict]:
    """최근 구간의 환경 변수가 평년 대비 어느 방향으로 벗어났는지 산출.

    수요가 높거나 낮은 '이유'를 설명하기 위한 근거로, 표준편차 기준 편차가 큰
    변수부터 제시한다. 인과관계가 아니라 동시 관측된 편차임에 유의한다.
    """
    df = dataset_frame(_model_path(model_path))
    if len(df) < _RECENT_WINDOW + 1:
        return []

    drivers = []
    for role in BASE_FEATURE_ROLES:
        if role == TARGET_FEATURE or role not in df.columns:
            continue
        series = df[role].astype(float)
        if series.nunique() <= 1:            # 상수 컬럼(해당 역할 데이터 없음)
            continue
        std = float(series.std())
        if std <= 0:
            continue
        recent = float(series.tail(_RECENT_WINDOW).mean())
        overall = float(series.mean())
        z = (recent - overall) / std
        if abs(z) < _DRIVER_MIN_Z:
            continue
        label = knowledge.feature_label(role)
        direction = "높습니다" if z > 0 else "낮습니다"
        drivers.append({
            "feature": role,
            "label": label,
            "recent_avg": round(recent, 2),
            "normal_avg": round(overall, 2),
            "direction": "높음" if z > 0 else "낮음",
            "z": round(float(z), 2),
            # LLM이 숫자를 해석하다 방향을 뒤집는 사례가 있어, 그대로 인용할 수 있는
            # 완성 문장을 함께 제공한다.
            "summary": (f"최근 {label}이(가) 평년 {round(overall, 2)} 대비 "
                        f"{round(recent, 2)}로 {direction}"),
        })

    drivers.sort(key=lambda d: abs(d["z"]), reverse=True)
    return drivers[:_MAX_DRIVERS]


# --- 예측 방법 선택(백테스트) ----------------------------------------------
# 최근 구간에서 두 방법을 실제로 겨뤄보고 더 정확한 쪽을 채택한다.
#   model       : 학습된 LSTM
#   persistence : 내일 = 오늘(직전 실측값). 단순하지만 강력한 시계열 기준선
# 학습 시점과 운영 시점의 분포가 달라져 모델이 무너지는 경우(계절 이동 등)에도
# 사용자에게 "모르겠다" 대신 최선의 추정치를 제공하기 위한 장치다.
_BACKTEST_DAYS = 30

# 모델을 채택하려면 기준선보다 이 비율 이상 오차가 작아야 한다.
_MODEL_WIN_MARGIN = 0.95

_backtest_cache: dict = {}


def _metrics(pred: np.ndarray, truth: np.ndarray) -> dict:
    err = pred - truth
    ss_tot = float(np.sum((truth - truth.mean()) ** 2)) + 1e-8
    return {
        "rmse": round(float(np.sqrt(np.mean(err ** 2))), 3),
        "mae": round(float(np.mean(np.abs(err))), 3),
        "r2": round(1.0 - float(np.sum(err ** 2)) / ss_tot, 3),
    }


def backtest(model_path=None, days: int = _BACKTEST_DAYS) -> dict:
    """최근 days일에 대해 LSTM과 지속성 기준선의 예측 정확도를 비교한다."""
    mp = str(_model_path(model_path))
    key = (mp, days)
    if key in _backtest_cache:
        return _backtest_cache[key]

    model, ckpt = _load_ckpt(mp)
    scaler = StandardScalerNP(ckpt["scaler_mean"], ckpt["scaler_std"])
    tgt, seq_len = ckpt["target_idx"], ckpt["seq_len"]
    horizon = ckpt.get("horizon", 1)
    feat = ckpt["feature_cols"]

    df = add_time_features(dataset_frame(mp)).dropna().reset_index(drop=True)
    arr = df[feat].to_numpy(dtype=np.float32)
    n = len(arr)

    first = max(seq_len + horizon, n - days)
    idx = list(range(first, n))
    if len(idx) < 5:
        result = {"available": False, "reason": "백테스트할 최근 데이터가 부족합니다."}
        _backtest_cache[key] = result
        return result

    # i번째 실측을 맞히는 입력 윈도우: [i-horizon-seq_len+1, i-horizon]
    windows = np.stack([arr[i - horizon - seq_len + 1: i - horizon + 1] for i in idx])
    truth = np.array([arr[i, tgt] for i in idx], dtype=np.float32)
    persist = np.array([arr[i - horizon, tgt] for i in idx], dtype=np.float32)

    with torch.no_grad():
        x = torch.tensor(scaler.transform(windows), dtype=torch.float32)
        scaled = model(x).cpu().numpy()
    model_pred = scaler.inverse_target(scaled, tgt).astype(np.float32)

    m_stat, p_stat = _metrics(model_pred, truth), _metrics(persist, truth)
    better = ("model" if m_stat["rmse"] <= p_stat["rmse"] * _MODEL_WIN_MARGIN
              else "persistence")
    ratio = (round(m_stat["rmse"] / p_stat["rmse"], 1)
             if p_stat["rmse"] > 0 else None)

    result = {
        "available": True, "days": len(idx),
        "model": m_stat, "persistence": p_stat,
        "better": better, "model_over_persistence": ratio,
    }
    _backtest_cache[key] = result
    return result


# --- 예측 신뢰도 -----------------------------------------------------------

def forecast_confidence(predicted: float, model_path=None,
                        method: str = "model") -> dict:
    """채택된 예측 방법의 신뢰도를 최근 백테스트 성적으로 판정한다.

    학습 시점 지표(레지스트리의 test R²)가 아니라 **최근 구간에서 실제로 맞혔는지**를
    기준으로 삼는다. 방법을 기준선으로 교체했다면 그 기준선의 성적으로 평가해야
    한다("모델이 나쁘다"는 이유로 좋은 기준선까지 깎아내리지 않기 위함).

    또한 오차 크기를 함께 반환해, 단일 숫자 대신 범위로 안내할 수 있게 한다.
    """
    mp = _model_path(model_path)
    bt = backtest(mp)

    if not bt.get("available"):
        return {"level": "낮음", "method": method,
                "reasons": [bt.get("reason", "최근 성능을 확인할 수 없습니다.")],
                "expected_error": None, "range": None}

    stat = bt["persistence"] if method == "persistence" else bt["model"]
    r2, mae = stat["r2"], stat["mae"]

    if r2 >= 0.3:
        level = "높음"
    elif r2 >= 0:
        level = "보통"
    else:
        level = "낮음"

    reasons = [
        f"최근 {bt['days']}일 검증 기준 R² {r2}, 평균 오차 약 {mae} "
        f"{config.ENERGY_UNIT}입니다."
    ]
    if method == "persistence":
        reasons.append(bt and
                       f"학습 모델(RMSE {bt['model']['rmse']})보다 정확한 "
                       f"기준선(RMSE {bt['persistence']['rmse']})을 사용했습니다.")

    # 평균 오차를 이용한 실용적 범위. 음수 소비는 없으므로 0에서 자른다.
    lo, hi = max(0.0, predicted - mae), predicted + mae
    return {
        "level": level,
        "method": method,
        "recent_r2": r2,
        "expected_error": mae,
        "range": [round(lo, 2), round(hi, 2)],
        "reasons": reasons,
    }


# --- 전망 및 권고 ----------------------------------------------------------

def next_day_outlook(model_path=None) -> dict:
    """내일(HORIZON일 뒤) 예측 에너지와 최근 평균 대비 수준·근사 비용을 계산.

    최근 구간 백테스트에서 LSTM이 지속성 기준선보다 정확하지 않으면, 기준선
    (내일 = 직전 실측)을 채택해 최선의 추정치를 제시한다. 어느 쪽을 썼는지는
    method 필드로 함께 알린다.
    """
    mp = _model_path(model_path)
    fc = forecast_next(mp)
    bt = backtest(mp)

    model_pred = fc["predicted_energy"]
    df = dataset_frame(mp)
    persist = (round(float(df[TARGET_FEATURE].astype(float).iloc[-1]), 2)
               if TARGET_FEATURE in df.columns and len(df) else None)

    # config.FORECAST_METHOD == "auto" 일 때만 기준선으로 대체한다. 기본값은
    # "model"이며, 과제의 예측 모델(LSTM)을 그대로 사용한다. 성능 차이는 값을
    # 바꾸는 대신 신뢰도(forecast_confidence)로 알린다.
    use_persist = (config.FORECAST_METHOD == "auto"
                   and bt.get("available") and bt["better"] == "persistence"
                   and persist is not None)

    if use_persist:
        method = "persistence"
        method_label = "최근 실측 기반(내일 = 직전 실측)"
        chosen = persist
        method_reason = (
            f"최근 {bt['days']}일 검증에서 학습 모델의 오차(RMSE {bt['model']['rmse']})가 "
            f"단순 기준선(RMSE {bt['persistence']['rmse']})보다 커서, 더 정확한 "
            f"기준선 값을 채택했습니다.")
    else:
        method = "model"
        method_label = "학습 모델(LSTM)"
        chosen = model_pred
        method_reason = "과제 예측 모델(LSTM)의 예측값입니다."

    fc = {**fc, "predicted_energy": chosen}
    level = _level(chosen, fc["recent_avg_energy"])
    avg_rate = config.avg_tou_rate()   # 시간대별 분포를 몰라 일 평균 단가로 근사
    area = config.GREENHOUSE_AREA_M2

    # 요금은 타깃이 면적당 kWh일 때만 의미가 있다. 업로드된 데이터셋은 단위를
    # 알 수 없으므로(Wh일 수도, 총량일 수도 있다) 요금을 계산하지 않는다.
    unit = _energy_unit(mp)
    cost_ok = "kwh" in unit.lower() and "m²" in unit
    total_kwh = fc["predicted_energy"] * area if cost_ok else None
    return {
        **fc,
        "energy_unit": unit,
        "level": level,
        "method": method,
        "method_label": method_label,
        "method_reason": method_reason,
        "model_prediction": model_pred,      # 참고용(채택 여부와 무관)
        "persistence_prediction": persist,
        "backtest": bt,
        "avg_rate": avg_rate,
        "area_m2": area if cost_ok else None,
        "predicted_energy_total_kwh": round(total_kwh, 1) if cost_ok else None,
        "estimated_cost": round(total_kwh * avg_rate, 1) if cost_ok else None,
        "cost_basis": ((f"면적 {area}m² × 평균 단가 {avg_rate}원/kWh 기준 근사값. "
                        f"시간대별 실측 분포가 없어 일 평균 단가를 사용")
                       if cost_ok else
                       "이 데이터셋은 에너지 단위를 알 수 없어 요금을 계산하지 않습니다"),
    }


def recommend_schedule(model_path=None, task: str | None = None) -> dict:
    """예측 수준·신뢰도·작업 제약을 종합해 비필수 작업 일정을 권고."""
    mp = _model_path(model_path)
    outlook = next_day_outlook(mp)
    conf = forecast_confidence(outlook["predicted_energy"], mp,
                               method=outlook["method"])

    level = outlook["level"]
    if conf["level"] == "낮음":
        advice = ("내일 수요를 신뢰할 만한 수준으로 예측하기 어렵습니다. "
                  "에너지 수요만으로 작업 시점을 판단하기보다 현장 상황을 함께 "
                  "고려하시기 바랍니다.")
    elif level == "낮음":
        advice = ("내일은 에너지 수요가 평소보다 낮을 것으로 예상됩니다. "
                  "방제·정비 등 비필수 작업을 진행하기 좋은 날입니다.")
    elif level == "높음":
        advice = ("내일은 에너지 수요가 평소보다 높을 것으로 예상됩니다. "
                  "비필수 작업은 다른 날로 미루는 것을 권장합니다.")
    else:
        advice = ("내일 에너지 수요는 평소와 비슷할 것으로 예상됩니다. "
                  "평소대로 작업을 진행해도 무방합니다.")

    result = {**outlook, "advice": advice, "confidence": conf,
              "drivers": demand_drivers(mp)}

    # 작업 유형이 지목된 경우 온톨로지에서 성질과 제약을 도출해 함께 평가한다.
    # deferrable / needs_ventilation / energy_note는 개별 작업이 아니라 상위 개념
    # (비필수작업·개방작업 등)에 정의된 것을 계층에서 물려받은 값이다.
    if task:
        check = knowledge.evaluate(task, latest_observations(mp),
                                   quantile=dataset_quantile(mp))
        result["task_check"] = check
        note = check.get("energy_note") or ""

        if check.get("known") and not check.get("deferrable"):
            # 필수작업은 수요와 무관하게 수행해야 하므로 연기를 권하지 않는다
            result["advice"] = (
                f"{check['label']}는 시점을 미룰 수 있는 작업이 아닙니다. "
                f"{note}.")
        elif check.get("needs_ventilation") and conf["level"] != "낮음":
            # 온실을 여는 작업만 난방 수요의 영향을 받는다
            if level == "높음":
                result["advice"] = (
                    f"내일은 에너지 수요가 평소보다 높을 것으로 예상됩니다. "
                    f"{note}. {check['label']}은(는) 수요가 낮은 날로 미루는 것이 "
                    f"난방비 측면에서 유리합니다.")
            elif level == "낮음":
                result["advice"] = (
                    f"내일은 에너지 수요가 평소보다 낮을 것으로 예상됩니다. "
                    f"난방 부담이 적어 {check['label']}을(를) 진행하기 좋은 날입니다.")
        elif check.get("known") and not check.get("needs_ventilation"):
            # 밀폐작업: 미룰 수는 있으나 에너지와는 무관하다
            result["advice"] = (
                f"{check['label']}는 온실을 열지 않는 작업이라 {note}. "
                f"에너지 수요와 무관하게 편한 시점에 진행하셔도 됩니다.")
    return result


def build_facts(model_path=None, task: str | None = None) -> dict:
    """LLM에 전달할 구조화된 근거 데이터 묶음.

    일(day) 단위 예측 태스크에 정합한 항목만 담는다. 시간대별 요금표는 의도적으로
    제외했다(v2 변경점 참고).
    """
    mp = _model_path(model_path)
    sched = recommend_schedule(mp, task=task)

    facts = {
        "granularity": "일(day) 단위. 시간대(몇 시) 정보는 데이터에 존재하지 않음",
        "forecast": {k: sched[k] for k in
                     ("as_of", "horizon_days", "predicted_energy", "recent_avg_energy",
                      "energy_unit")},
        "confidence": sched["confidence"],
        "method": {k: sched[k] for k in
                   ("method", "method_label", "method_reason",
                    "model_prediction", "persistence_prediction")},
        # predicted_energy_total_kwh(온실 전체 환산값)는 화면 표시용이며 여기에
        # 넣지 않는다. 근거에 비슷한 에너지 수치가 여럿 있으면 소형 모델이
        # 면적당 값 대신 전체 값을 인용하는 오류가 발생한다.
        "schedule": {k: sched[k] for k in
                     ("level", "advice", "estimated_cost", "avg_rate", "cost_basis",
                      "area_m2")},
        "drivers": sched["drivers"],
        "recent_series": recent_series(mp, days=14),
    }
    if "task_check" in sched:
        facts["task_check"] = sched["task_check"]
    return facts
