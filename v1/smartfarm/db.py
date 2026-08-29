"""데이터베이스 접근 계층 (SQLAlchemy).

MariaDB를 타깃으로 하되, DB_URL 미지정 시 SQLite로 폴백한다.
"""
from __future__ import annotations

import pandas as pd
from sqlalchemy import create_engine, text
from sqlalchemy.engine import Engine

from . import config

_engine: Engine | None = None


def get_engine() -> Engine:
    """싱글톤 엔진 반환."""
    global _engine
    if _engine is None:
        _engine = create_engine(config.DB_URL, future=True)
    return _engine


def read_sensor_data(limit: int | None = None, table: str | None = None) -> pd.DataFrame:
    """센서 테이블 전체(또는 최근 N행)를 시간순으로 읽어온다."""
    table = table or config.SENSOR_TABLE
    eng = get_engine()
    q = f"SELECT * FROM {table} ORDER BY ts"
    if limit:
        q = (
            f"SELECT * FROM (SELECT * FROM {table} "
            f"ORDER BY ts DESC LIMIT {int(limit)}) sub ORDER BY ts"
        )
    df = pd.read_sql(q, eng, parse_dates=["ts"])
    return df


def table_exists(table: str | None = None) -> bool:
    from sqlalchemy import inspect
    table = table or config.SENSOR_TABLE
    return table in inspect(get_engine()).get_table_names()


def row_count(table: str | None = None) -> int:
    table = table or config.SENSOR_TABLE
    eng = get_engine()
    with eng.connect() as c:
        return c.execute(text(f"SELECT COUNT(*) FROM {table}")).scalar_one()
