"""전력 수요 예측용 LSTM 모델 + 특징 엔지니어링/시퀀스 생성.

many-to-one 구조: 과거 SEQ_LEN 스텝의 다변량 특징으로 HORIZON 스텝 뒤의
전력 수요(appliances)를 예측한다.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

# 모델 입력 특징 = 표준 스키마의 정규 특징(데이터셋 무관 고정 차원)
from ..schema import CANONICAL_FEATURES as FEATURE_COLS  # noqa: F401


def add_time_features(df: pd.DataFrame) -> pd.DataFrame:
    """날짜로부터 주기성(요일/연중 계절) 특징을 추가한다.

    일 단위(day) 데이터라 시각(hour) 개념이 없으므로, 요일 주기(7일)와
    연중 계절 주기(365.25일, 온실 에너지 소비의 계절성 반영)를 사용한다.
    """
    df = df.copy()
    dow = df["ts"].dt.dayofweek
    df["dow_sin"] = np.sin(2 * np.pi * dow / 7)
    df["dow_cos"] = np.cos(2 * np.pi * dow / 7)
    doy = df["ts"].dt.dayofyear
    df["doy_sin"] = np.sin(2 * np.pi * doy / 365.25)
    df["doy_cos"] = np.cos(2 * np.pi * doy / 365.25)
    return df


def make_sequences(arr: np.ndarray, target_idx: int,
                   seq_len: int, horizon: int):
    """(N, F) 배열 → (X:[n,seq_len,F], y:[n]) 시퀀스로 변환."""
    X, y = [], []
    last = len(arr) - seq_len - horizon + 1
    for i in range(last):
        X.append(arr[i:i + seq_len])
        y.append(arr[i + seq_len + horizon - 1, target_idx])
    return np.asarray(X, dtype=np.float32), np.asarray(y, dtype=np.float32)


class LSTMForecaster(nn.Module):
    def __init__(self, n_features: int, hidden: int = 64, layers: int = 2,
                 dropout: float = 0.2):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=n_features, hidden_size=hidden, num_layers=layers,
            batch_first=True, dropout=dropout if layers > 1 else 0.0,
        )
        self.head = nn.Sequential(
            nn.Linear(hidden, hidden // 2), nn.ReLU(),
            nn.Linear(hidden // 2, 1),
        )

    def forward(self, x):
        out, _ = self.lstm(x)          # (B, T, H)
        return self.head(out[:, -1, :]).squeeze(-1)  # 마지막 스텝 → 스칼라


class StandardScalerNP:
    """넘파이 기반 표준화(모델 파일에 함께 저장하기 위한 경량 스케일러)."""
    def __init__(self, mean=None, std=None):
        self.mean, self.std = mean, std

    def fit(self, x: np.ndarray):
        self.mean = x.mean(axis=0)
        self.std = x.std(axis=0) + 1e-8
        return self

    def transform(self, x):
        return (x - self.mean) / self.std

    def inverse_target(self, y_scaled, target_idx: int):
        return y_scaled * self.std[target_idx] + self.mean[target_idx]
