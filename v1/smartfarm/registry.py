"""데이터셋 레지스트리: 여러 데이터셋/모델을 등록·선택·삭제한다.

각 데이터셋은 표준 컬럼이 적재된 DB 테이블과 학습된 모델 체크포인트를 갖는다.
메타데이터는 models_store/registry.json에 저장한다.
"""
from __future__ import annotations

import json
import re
import uuid
from datetime import datetime

from sqlalchemy import text

from . import config
from .db import get_engine

REGISTRY_PATH = config.MODEL_DIR / "registry.json"

# 기본(대체 불가) 데이터셋 id. UI에서 삭제 버튼으로부터 보호된다.
DEFAULT_DATASET_ID = "smartfarm_default"


def _load() -> dict:
    if REGISTRY_PATH.exists():
        return json.loads(REGISTRY_PATH.read_text(encoding="utf-8"))
    return {"active": None, "datasets": {}}


def _save(reg: dict) -> None:
    REGISTRY_PATH.write_text(json.dumps(reg, ensure_ascii=False, indent=2),
                             encoding="utf-8")


def _slug(name: str) -> str:
    # MariaDB 호환 위해 ASCII 영숫자만 유지(한글 등은 제거, uuid로 유일성 보장)
    s = re.sub(r"[^0-9a-zA-Z]+", "_", name).strip("_").lower()
    return (s or "ds")[:20]


def new_ids(name: str) -> tuple[str, str, str]:
    """(dataset_id, table_name, model_path) 생성."""
    ds_id = f"{_slug(name)}_{uuid.uuid4().hex[:6]}"
    table = f"ds_{ds_id}"
    model_path = str(config.MODEL_DIR / f"power_lstm_{ds_id}.pt")
    return ds_id, table, model_path


def add_dataset(ds_id: str, name: str, table: str, mapping: dict,
                model_path: str, metrics: dict, rows: int,
                make_active: bool = True) -> dict:
    reg = _load()
    entry = {
        "id": ds_id, "name": name, "table": table, "mapping": mapping,
        "model_path": model_path, "metrics": metrics, "rows": rows,
        "created_at": datetime.now().isoformat(timespec="seconds"),
    }
    reg["datasets"][ds_id] = entry
    if make_active or reg.get("active") is None:
        reg["active"] = ds_id
    _save(reg)
    return entry


def list_datasets() -> list[dict]:
    return list(_load()["datasets"].values())


def get_dataset(ds_id: str) -> dict | None:
    return _load()["datasets"].get(ds_id)


def set_active(ds_id: str) -> None:
    reg = _load()
    if ds_id in reg["datasets"]:
        reg["active"] = ds_id
        _save(reg)


def get_active() -> dict | None:
    reg = _load()
    active = reg.get("active")
    return reg["datasets"].get(active) if active else None


def delete_dataset(ds_id: str, drop_table: bool = True) -> None:
    reg = _load()
    entry = reg["datasets"].pop(ds_id, None)
    if entry is None:
        return
    # 모델 파일 삭제
    from pathlib import Path
    mp = Path(entry["model_path"])
    if mp.exists():
        mp.unlink()
    # DB 테이블 삭제(기본 UCI 테이블은 보존)
    if drop_table and entry["table"] != config.SENSOR_TABLE:
        with get_engine().begin() as c:
            c.execute(text(f"DROP TABLE IF EXISTS {entry['table']}"))
    # active 재설정
    if reg.get("active") == ds_id:
        reg["active"] = next(iter(reg["datasets"]), None)
    _save(reg)


def ensure_default() -> dict:
    """기본 WUR 온실 데이터셋을 레지스트리에 등록(없으면). 등록된 항목 반환."""
    from .data_pipeline import wur_mapping
    reg = _load()
    for e in reg["datasets"].values():
        if e["table"] == config.SENSOR_TABLE:
            return e
    entry = add_dataset(
        ds_id=DEFAULT_DATASET_ID, name="WUR 온실 챌린지 (기본 예제)",
        table=config.SENSOR_TABLE, mapping=wur_mapping(),
        model_path=str(config.MODEL_DIR / "power_lstm.pt"),
        metrics={}, rows=0, make_active=True,
    )
    return entry
