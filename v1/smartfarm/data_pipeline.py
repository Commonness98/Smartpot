"""데이터 파이프라인: 원본 CSV → 정제 → DB 적재.

문서 목표① "IoT 센서 데이터 수집·처리"를, 프로토타입에서는 공개 데이터셋
(UCI Appliances Energy Prediction) 적재로 대체한다. 실제 라즈베리파이/Jetson
연동 시 이 모듈의 load_raw()만 센서 수집 함수로 교체하면 된다.
"""
from __future__ import annotations

import pandas as pd

from . import config, schema
from .db import get_engine

# UCI 원본 컬럼 → 프로젝트 표준 컬럼(스마트팜 의미)
COLUMN_MAP = {
    "date": "ts",
    "Appliances": "appliances",     # 전력 수요(Wh) — 예측 타깃
    "lights": "lights",             # 조명 전력(Wh)
    "T1": "temp_zone1", "RH_1": "humid_zone1",
    "T2": "temp_zone2", "RH_2": "humid_zone2",
    "T3": "temp_zone3", "RH_3": "humid_zone3",
    "T4": "temp_zone4", "RH_4": "humid_zone4",
    "T5": "temp_zone5", "RH_5": "humid_zone5",
    "T6": "temp_zone6", "RH_6": "humid_zone6",
    "T7": "temp_zone7", "RH_7": "humid_zone7",
    "T8": "temp_zone8", "RH_8": "humid_zone8",
    "T9": "temp_zone9", "RH_9": "humid_zone9",
    "T_out": "temp_out", "RH_out": "humid_out",     # 외부 기상
    "Press_mm_hg": "pressure", "Windspeed": "windspeed",
    "Visibility": "visibility", "Tdewpoint": "dewpoint",
}


def load_raw(csv_path=None) -> pd.DataFrame:
    """원본 CSV를 읽어 표준 컬럼으로 정제한다."""
    csv_path = csv_path or config.RAW_CSV
    df = pd.read_csv(csv_path)
    df = df.rename(columns=COLUMN_MAP)
    df = df.drop(columns=[c for c in ("rv1", "rv2") if c in df.columns])  # 난수 컬럼 제거
    df["ts"] = pd.to_datetime(df["ts"])
    # 숫자형 강제 변환(공백 포함 문자열 방지)
    for c in df.columns:
        if c != "ts":
            df[c] = pd.to_numeric(df[c], errors="coerce")
    df = df.dropna().sort_values("ts").reset_index(drop=True)
    return df


def uci_mapping() -> dict[str, str]:
    """기본 UCI 데이터셋(sensor_readings 표준 컬럼)의 컬럼→역할 매핑."""
    m = {
        "ts": schema.ROLE_TIMESTAMP,
        "appliances": schema.ROLE_POWER,
        "lights": schema.ROLE_LIGHT,
        "temp_out": schema.ROLE_OUT_TEMP,
        "humid_out": schema.ROLE_OUT_HUMID,
        "pressure": schema.ROLE_PRESSURE,
        "windspeed": schema.ROLE_WINDSPEED,
        "visibility": schema.ROLE_IGNORE,
        "dewpoint": schema.ROLE_IGNORE,
    }
    for i in range(1, 10):
        m[f"temp_zone{i}"] = schema.ROLE_INDOOR_TEMP
        m[f"humid_zone{i}"] = schema.ROLE_INDOOR_HUMID
    return m


def ingest_dataframe(df: pd.DataFrame, mapping: dict[str, str], table: str,
                     if_exists: str = "replace") -> int:
    """임의 원본 df를 매핑에 따라 표준 컬럼으로 변환 후 지정 테이블에 적재.

    저장 시점에 매핑을 1회 적용하므로, 이후 학습/예측은 표준 컬럼만 읽으면 된다.
    """
    canon = schema.build_canonical(df, mapping)
    eng = get_engine()
    canon.to_sql(table, eng, if_exists=if_exists, index=False)
    return len(canon)


def ingest_to_db(csv_path=None, table: str | None = None,
                 if_exists: str = "replace") -> int:
    """기본 UCI 데이터셋을 표준 컬럼으로 변환해 DB에 적재하고 행 수를 반환한다.

    (참고용) 기본 프로토타입 데이터셋은 이제 WUR 온실 데이터(ingest_wur_to_db)이며,
    이 함수는 UCI 데이터를 수동으로 다시 써보고 싶을 때를 위해 남겨둔다.
    """
    df = load_raw(csv_path)
    return ingest_dataframe(df, uci_mapping(), table or config.SENSOR_TABLE, if_exists)


# --- WUR Autonomous Greenhouse Challenge, 2nd Edition (2019) ---
# 실제 상업용 유리온실(6개 구획) 6개월치 기후·자원소비 데이터.
# 출처: https://data.4tu.nl/datasets/8b35675d-7549-49cc-a8c3-dedec8d3f23e
# 구성: 팀별 폴더(GreenhouseClimate.csv 5분 간격 기후, Resources.csv 일 단위 자원소비) +
#      공용 Weather/Weather.csv(5분 간격 외부기상). 실측 에너지(난방+전기)가 일 단위로만
#      존재해, 표준 스텝 단위를 "일(day)"로 삼아 기후는 일 평균으로 집계해 맞춘다.
WUR_DIR = config.RAW_DIR / "wur_greenhouse"
WUR_TEAM = "Reference"  # WUR 자체 기준(대조군) 온실 구획 — 실험적 제어가 적어 기본값으로 적합


def _excel_serial_to_ts(series: pd.Series) -> pd.Series:
    """엑셀/로터스 일련번호(1899-12-30 기준 경과일)를 datetime으로 변환."""
    return pd.to_datetime(pd.to_numeric(series, errors="coerce"),
                          unit="D", origin="1899-12-30")


def load_wur_raw(team: str = WUR_TEAM) -> pd.DataFrame:
    """WUR 온실 데이터(팀 1곳)를 일(day) 단위로 집계한 표준 컬럼 프레임으로 변환한다."""
    base = WUR_DIR / team

    clim = pd.read_csv(base / "GreenhouseClimate.csv", low_memory=False)
    clim["ts"] = _excel_serial_to_ts(clim["%time"])
    clim_cols = ["Tair", "Rhair", "CO2air", "Tot_PAR"]
    for c in clim_cols:
        clim[c] = pd.to_numeric(clim[c], errors="coerce")
    clim["date"] = clim["ts"].dt.floor("D")
    clim_daily = clim.groupby("date")[clim_cols].mean().reset_index()

    weather = pd.read_csv(WUR_DIR / "Weather" / "Weather.csv", low_memory=False)
    weather["ts"] = _excel_serial_to_ts(weather["%time"])
    weather_cols = ["Tout", "Rhout", "Windsp"]
    for c in weather_cols:
        weather[c] = pd.to_numeric(weather[c], errors="coerce")
    weather["date"] = weather["ts"].dt.floor("D")
    weather_daily = weather.groupby("date")[weather_cols].mean().reset_index()

    res = pd.read_csv(base / "Resources.csv")
    res.columns = [c.strip() for c in res.columns]     # "%Time " 컬럼명 앞뒤 공백 제거
    res["date"] = _excel_serial_to_ts(res["%Time"]).dt.floor("D")
    res["energy_total"] = (
        pd.to_numeric(res["Heat_cons"], errors="coerce").fillna(0)
        + pd.to_numeric(res["ElecHigh"], errors="coerce").fillna(0)
        + pd.to_numeric(res["ElecLow"], errors="coerce").fillna(0)
    )

    df = (clim_daily.merge(weather_daily, on="date", how="left")
                    .merge(res[["date", "energy_total"]], on="date", how="inner"))
    df = df.rename(columns={"date": "ts"}).sort_values("ts").reset_index(drop=True)
    return df.dropna(subset=["energy_total"])


def wur_mapping() -> dict[str, str]:
    """WUR 온실 데이터(load_wur_raw 결과)의 컬럼→역할 매핑."""
    return {
        "ts": schema.ROLE_TIMESTAMP,
        "energy_total": schema.ROLE_POWER,
        "Tot_PAR": schema.ROLE_LIGHT,
        "Tair": schema.ROLE_INDOOR_TEMP,
        "Rhair": schema.ROLE_INDOOR_HUMID,
        "CO2air": schema.ROLE_CO2,
        "Tout": schema.ROLE_OUT_TEMP,
        "Rhout": schema.ROLE_OUT_HUMID,
        "Windsp": schema.ROLE_WINDSPEED,
    }


def ingest_wur_to_db(team: str = WUR_TEAM, table: str | None = None,
                     if_exists: str = "replace") -> int:
    """WUR 온실 데이터를 표준 컬럼으로 변환해 DB에 적재하고 행 수를 반환한다."""
    df = load_wur_raw(team)
    return ingest_dataframe(df, wur_mapping(), table or config.SENSOR_TABLE, if_exists)
