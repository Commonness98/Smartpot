"""표준(정규) 스키마: 물리 컬럼 ↔ 논리 역할 분리.

데이터셋마다 컬럼 이름·개수가 달라도, 역할(role) 기반으로 **고정된 표준 특징 벡터**를
만들어 모델 입력 차원을 안정화한다. 같은 역할의 컬럼이 여러 개면 평균 집계한다.
"""
from __future__ import annotations

import warnings

import numpy as np
import pandas as pd

# --- 논리 역할 ---
ROLE_TIMESTAMP = "timestamp"
ROLE_POWER = "power_target"
ROLE_LIGHT = "light"
ROLE_INDOOR_TEMP = "indoor_temp"
ROLE_INDOOR_HUMID = "indoor_humid"
ROLE_OUT_TEMP = "out_temp"
ROLE_OUT_HUMID = "out_humid"
ROLE_PRESSURE = "pressure"
ROLE_WINDSPEED = "windspeed"
ROLE_CO2 = "co2"
ROLE_IGNORE = "ignore"

# UI 셀렉트박스에 노출할 역할 목록
ALL_ROLES = [
    ROLE_TIMESTAMP, ROLE_POWER, ROLE_INDOOR_TEMP, ROLE_INDOOR_HUMID,
    ROLE_OUT_TEMP, ROLE_OUT_HUMID, ROLE_PRESSURE, ROLE_WINDSPEED,
    ROLE_LIGHT, ROLE_CO2, ROLE_IGNORE,
]

# 표준 특징(순서 고정). 시계열 예측 입력으로 사용.
BASE_FEATURE_ROLES = [
    ROLE_POWER, ROLE_LIGHT, ROLE_INDOOR_TEMP, ROLE_INDOOR_HUMID,
    ROLE_OUT_TEMP, ROLE_OUT_HUMID, ROLE_PRESSURE, ROLE_WINDSPEED, ROLE_CO2,
]
# 일 단위(day) 데이터 기준 주기성 특징: 시간 개념이 없으므로 요일·연중 계절성만 사용.
TIME_FEATURES = ["dow_sin", "dow_cos", "doy_sin", "doy_cos"]
CANONICAL_FEATURES = BASE_FEATURE_ROLES + TIME_FEATURES
TARGET_FEATURE = ROLE_POWER  # 표준 벡터에서 예측 타깃 컬럼명

# 역할별 컬럼명 키워드(한/영). 휴리스틱 분류에 사용.
ROLE_KEYWORDS = {
    ROLE_TIMESTAMP: ["date", "time", "ts", "datetime", "시각", "일시", "날짜", "타임"],
    ROLE_LIGHT: ["light", "lux", "par", "조명", "조도", "일사", "광량"],
    ROLE_POWER: ["power", "appliance", "kwh", "wh", "watt", "energy", "load",
                 "전력", "소비전력", "전력계", "에너지", "소비", "부하"],
    ROLE_PRESSURE: ["press", "hpa", "mmhg", "기압", "압력"],
    ROLE_WINDSPEED: ["wind", "풍속", "바람"],
    ROLE_CO2: ["co2", "이산화탄소"],
    ROLE_OUT_TEMP: ["외기온", "외부온도", "외기 온도", "outdoor temp", "t_out", "temp_out", "외부 기온"],
    ROLE_OUT_HUMID: ["외부습도", "외기습도", "rh_out", "humid_out", "외부 습도"],
    ROLE_INDOOR_TEMP: ["temp", "온도", "기온", "t1", "t2", "t3"],
    ROLE_INDOOR_HUMID: ["humid", "humidity", "rh", "습도"],
}
_OUTDOOR_HINTS = ["out", "외기", "외부", "실외"]


def _looks_like_time(series: pd.Series) -> bool:
    sample = series.dropna().astype(str).head(20)
    if sample.empty:
        return False
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")          # 형식 추론 경고 억제
            parsed = pd.to_datetime(sample, errors="coerce", format="mixed")
        return parsed.notna().mean() > 0.8
    except Exception:
        return False


def heuristic_roles(df: pd.DataFrame) -> dict[str, dict]:
    """각 컬럼의 역할을 규칙 기반으로 추정한다.

    반환: {컬럼명: {"role": ..., "confidence": 0~1, "reason": ...}}
    """
    result: dict[str, dict] = {}
    for col in df.columns:
        name = str(col).lower()
        series = df[col]

        # 1) 타임스탬프: 값 파싱 우선
        if _looks_like_time(series) or any(k in name for k in ROLE_KEYWORDS[ROLE_TIMESTAMP]):
            result[col] = {"role": ROLE_TIMESTAMP, "confidence": 0.95,
                           "reason": "날짜/시각 형식"}
            continue

        outdoor = any(h in name for h in _OUTDOOR_HINTS)
        role, conf, reason = ROLE_IGNORE, 0.2, "매칭 키워드 없음"

        # 2) 키워드 매칭(우선순위: 조명 > 전력 > CO2 > 기압 > 풍속 > 온도/습도)
        for r in (ROLE_LIGHT, ROLE_POWER, ROLE_CO2, ROLE_PRESSURE, ROLE_WINDSPEED,
                  ROLE_INDOOR_TEMP, ROLE_INDOOR_HUMID):
            if any(k in name for k in ROLE_KEYWORDS[r]):
                role, conf, reason = r, 0.85, f"컬럼명에 '{r}' 키워드"
                # 실내/실외 구분
                if r == ROLE_INDOOR_TEMP and outdoor:
                    role = ROLE_OUT_TEMP
                elif r == ROLE_INDOOR_HUMID and outdoor:
                    role = ROLE_OUT_HUMID
                break

        # 3) 값 범위 보조(습도 0~100, 온도 -50~60 등)
        if role == ROLE_IGNORE and pd.api.types.is_numeric_dtype(series):
            s = pd.to_numeric(series, errors="coerce").dropna()
            if not s.empty:
                lo, hi = s.min(), s.max()
                if 0 <= lo and hi <= 100 and s.mean() > 30:
                    role, conf, reason = ROLE_INDOOR_HUMID, 0.4, "값 범위(0~100)로 습도 추정"

        result[col] = {"role": role, "confidence": conf, "reason": reason}
    return result


def build_canonical(df: pd.DataFrame, mapping: dict[str, str]) -> pd.DataFrame:
    """물리 컬럼 df + 매핑(컬럼→역할) → 표준 컬럼(ts + BASE_FEATURE_ROLES) DataFrame.

    같은 역할 컬럼이 여러 개면 평균 집계, 없는 역할은 상수 0으로 채운다.
    """
    ts_cols = [c for c, r in mapping.items() if r == ROLE_TIMESTAMP and c in df.columns]
    if not ts_cols:
        raise ValueError("timestamp 역할 컬럼이 지정되지 않았습니다.")

    out = pd.DataFrame()
    out["ts"] = pd.to_datetime(df[ts_cols[0]], errors="coerce")

    for role in BASE_FEATURE_ROLES:
        cols = [c for c, r in mapping.items() if r == role and c in df.columns]
        if cols:
            vals = df[cols].apply(pd.to_numeric, errors="coerce")
            out[role] = vals.mean(axis=1)
        else:
            out[role] = np.nan  # 없는 역할 → 아래에서 처리

    out = out.dropna(subset=["ts"]).sort_values("ts").reset_index(drop=True)

    # 결측 처리: 데이터가 있는 역할은 평균 보간, 아예 없는 역할은 상수 0
    for role in BASE_FEATURE_ROLES:
        if out[role].notna().any():
            out[role] = out[role].fillna(out[role].mean())
        else:
            out[role] = 0.0
    return out
