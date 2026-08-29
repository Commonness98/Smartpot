"""전역 설정. 환경변수(.env)로 오버라이드 가능.

문서 스택: MariaDB + LM Studio(로컬 LLM). 프로토타입 단계에서는
MariaDB 서버가 없을 수 있으므로 SQLite로 자동 폴백한다(코드 변경 없이 .env만 수정).
"""
from __future__ import annotations

import os
from pathlib import Path

try:
    from dotenv import load_dotenv
    load_dotenv()
except Exception:  # python-dotenv 미설치 시 무시
    pass

# --- 경로 ---
ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"
MODEL_DIR = ROOT / "models_store"
for _d in (RAW_DIR, PROCESSED_DIR, MODEL_DIR):
    _d.mkdir(parents=True, exist_ok=True)

RAW_CSV = RAW_DIR / "energydata_complete.csv"

# --- 데이터베이스 ---
# MariaDB 사용 예: DB_URL=mysql+pymysql://user:pw@localhost:3306/smartfarm
# 미지정 시 SQLite 파일로 폴백하여 서버 없이도 동작한다.
DB_URL = os.getenv("DB_URL", f"sqlite:///{(DATA_DIR / 'smartfarm.db').as_posix()}")

# 센서 데이터 테이블명
SENSOR_TABLE = "sensor_readings"

# --- LM Studio (OpenAI 호환 로컬 LLM) ---
# LM Studio 기본 서버: http://localhost:1234/v1
LM_STUDIO_BASE_URL = os.getenv("LM_STUDIO_BASE_URL", "http://localhost:1234/v1")
LM_STUDIO_MODEL = os.getenv("LM_STUDIO_MODEL", "local-model")
LM_STUDIO_API_KEY = os.getenv("LM_STUDIO_API_KEY", "lm-studio")  # LM Studio는 임의값 허용

# --- 예측 모델 ---
# 표준 스키마의 전력 타깃 컬럼명(power_target). schema.TARGET_FEATURE와 동일.
# 실측 에너지 소비(난방+전기)가 일(day) 단위로만 제공되는 실제 온실 데이터 특성상,
# 표준 스텝 단위는 "일"이다(과거 코드의 10분 간격 가정과 다름).
TARGET_COL = "power_target"    # 일일 총 에너지 소비 예측 타깃(난방+전기, 온실 면적당)
SEQ_LEN = 14                   # 입력 시퀀스 길이 (과거 14일)
HORIZON = 1                    # 예측 지평 (다음 1일 후)

# 전기요금 시간대(원/kWh) — 계획(Planning)에서 최적 가동시점 계산에 사용.
# 한국 산업용 계절/시간대 요금을 단순화한 예시값(프로토타입).
TOU_TARIFF = {
    "경부하": {"hours": list(range(23, 24)) + list(range(0, 9)), "rate": 70},
    "중간부하": {"hours": [9, 10] + list(range(12, 17)) + [22], "rate": 110},
    "최대부하": {"hours": [11, 17, 18, 19, 20, 21], "rate": 190},
}


def rate_at_hour(hour: int) -> int:
    """해당 시(hour)의 전기요금 단가(원/kWh)를 반환."""
    for band in TOU_TARIFF.values():
        if hour in band["hours"]:
            return band["rate"]
    return TOU_TARIFF["중간부하"]["rate"]


def avg_tou_rate() -> float:
    """시간대별 요금을 시간 수로 가중 평균한 일 평균 단가(원/kWh).

    예측 타깃이 일(day) 단위라 시간대별 실측 수요 분포를 알 수 없으므로,
    일일 예상 비용은 이 평균 단가로 근사(대략치)한다.
    """
    total_hours = sum(len(b["hours"]) for b in TOU_TARIFF.values())
    weighted = sum(len(b["hours"]) * b["rate"] for b in TOU_TARIFF.values())
    return round(weighted / total_hours, 1)
