"""원본 CSV를 DB에 적재한다.

사용법:
    python scripts/01_load_data.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from smartfarm import config
from smartfarm.data_pipeline import WUR_DIR, ingest_wur_to_db

if __name__ == "__main__":
    if not (WUR_DIR / "Reference" / "GreenhouseClimate.csv").exists():
        raise SystemExit(
            f"원본 데이터가 없습니다: {WUR_DIR}\n"
            "먼저 scripts/00_download_data.py 를 실행하세요."
        )
    n = ingest_wur_to_db()
    print(f"[OK] {n:,}행을 '{config.SENSOR_TABLE}' 테이블에 적재했습니다. (DB={config.DB_URL})")
