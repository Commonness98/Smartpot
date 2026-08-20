"""학습된 LSTM으로 전력 수요를 예측하고, 시간대별 수요 프로파일을 제공한다.

체크포인트에 저장된 feature_cols·table을 진실의 원천으로 사용하므로,
어떤 데이터셋으로 학습했든 예측이 학습과 일치한다.
"""
from __future__ import annotations

import numpy as np
import torch

from .. import config
from ..db import read_sensor_data
from ..schema import TARGET_FEATURE
from .power_lstm import LSTMForecaster, StandardScalerNP, add_time_features

DEFAULT_MODEL_PATH = config.MODEL_DIR / "power_lstm.pt"
_cache: dict = {}


def _load(model_path=DEFAULT_MODEL_PATH):
    key = str(model_path)
    if key in _cache:
        return _cache[key]
    ckpt = torch.load(model_path, map_location="cpu", weights_only=False)
    # 자동 튜닝 에이전트가 고른 아키텍처를 그대로 복원(구버전 체크포인트는 기본값 폴백).
    model = LSTMForecaster(n_features=len(ckpt["feature_cols"]),
                           hidden=ckpt.get("hidden", 64),
                           layers=ckpt.get("layers", 2),
                           dropout=ckpt.get("dropout", 0.2))
    model.load_state_dict(ckpt["state_dict"])
    model.eval()
    _cache[key] = (model, ckpt)
    return model, ckpt


def _canonical_frame(ckpt):
    """체크포인트가 학습한 테이블을 읽어 시간특징까지 붙인 표준 프레임 반환."""
    df = read_sensor_data(table=ckpt.get("table") or config.SENSOR_TABLE)
    return add_time_features(df).dropna().reset_index(drop=True)


def forecast_next(model_path=DEFAULT_MODEL_PATH) -> dict:
    """가장 최근 관측 시퀀스로부터 HORIZON일 뒤 일일 에너지 소비를 예측."""
    model, ckpt = _load(model_path)
    scaler = StandardScalerNP(ckpt["scaler_mean"], ckpt["scaler_std"])
    tgt = ckpt["target_idx"]
    seq_len = ckpt["seq_len"]
    feat = ckpt["feature_cols"]           # ← 학습 시 특징 목록(버그 방지)

    df = _canonical_frame(ckpt)
    window = df[feat].to_numpy(dtype=np.float32)[-seq_len:]
    x = torch.tensor(scaler.transform(window)[None, ...], dtype=torch.float32)
    with torch.no_grad():
        pred_scaled = model(x).item()
    pred = float(scaler.inverse_target(np.array(pred_scaled), tgt))
    recent = df[TARGET_FEATURE].iloc[-14:]
    return {
        "as_of": df["ts"].iloc[-1],
        "horizon_days": ckpt["horizon"],
        "predicted_energy": round(pred, 2),
        "recent_avg_energy": round(float(recent.mean()), 2),
    }


def recent_series(model_path=DEFAULT_MODEL_PATH, days: int = 60) -> dict:
    """최근 N일간의 실측 일일 에너지 소비 시계열(날짜문자열 → 값)을 반환."""
    _, ckpt = _load(model_path)
    df = read_sensor_data(table=ckpt.get("table") or config.SENSOR_TABLE)
    tail = df.tail(days)
    return {d.strftime("%Y-%m-%d"): round(float(v), 2)
            for d, v in zip(tail["ts"], tail[TARGET_FEATURE])}
