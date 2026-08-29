"""전력 수요 예측 LSTM 학습 스크립트.

표준 스키마(canonical) 컬럼이 적재된 테이블을 읽어 학습한다. 데이터셋마다
컬럼이 달라도 입력 차원이 고정되므로 코드 변경 없이 재학습만 하면 된다.

사용법:
    python -m smartfarm.models.train --epochs 15            # 기본 데이터셋
    python -m smartfarm.models.train --table ds_xxx --model models_store/power_lstm_ds_xxx.pt

자동 하이퍼파라미터 튜닝이 필요하면 smartfarm.agent.tuner를 사용한다.
"""
from __future__ import annotations

import argparse

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

from .. import config
from ..db import read_sensor_data
from ..schema import CANONICAL_FEATURES, TARGET_FEATURE
from .power_lstm import (LSTMForecaster, StandardScalerNP, add_time_features,
                         make_sequences)

DEFAULT_MODEL_PATH = config.MODEL_DIR / "power_lstm.pt"


def _prepare(table: str, seq_len: int, horizon: int):
    df = read_sensor_data(table=table)
    df = add_time_features(df).dropna().reset_index(drop=True)
    arr = df[CANONICAL_FEATURES].to_numpy(dtype=np.float32)
    n = len(arr)
    split = int(n * 0.8)                       # 시계열 순서 유지 분할
    scaler = StandardScalerNP().fit(arr[:split])
    arr_s = scaler.transform(arr)
    tgt = CANONICAL_FEATURES.index(TARGET_FEATURE)
    Xtr, ytr = make_sequences(arr_s[:split], tgt, seq_len, horizon)
    Xte, yte = make_sequences(arr_s[split:], tgt, seq_len, horizon)
    return scaler, tgt, (Xtr, ytr), (Xte, yte)


def _train_once(table: str = config.SENSOR_TABLE, mapping: dict | None = None,
                epochs: int = 15, batch: int = 64, lr: float = 1e-3,
                hidden: int = 64, layers: int = 2, dropout: float = 0.2,
                seq_len: int | None = None, horizon: int | None = None,
                ) -> tuple[dict, dict]:
    """1회 학습을 실행하고 (metrics, checkpoint dict)를 반환한다(디스크 저장 없음).

    자동 파라미터 튜닝 에이전트(smartfarm.agent.tuner)가 여러 하이퍼파라미터
    조합을 저장 없이 빠르게 비교할 수 있도록 학습 로직만 분리한 함수다.
    """
    seq_len = config.SEQ_LEN if seq_len is None else seq_len
    horizon = config.HORIZON if horizon is None else horizon
    device = "cuda" if torch.cuda.is_available() else "cpu"
    scaler, tgt, (Xtr, ytr), (Xte, yte) = _prepare(table, seq_len, horizon)
    print(f"table={table}  train seqs={len(Xtr):,}  test seqs={len(Xte):,}  device={device}  "
          f"hidden={hidden} layers={layers} dropout={dropout} lr={lr} seq_len={seq_len}")

    tr = DataLoader(TensorDataset(torch.tensor(Xtr), torch.tensor(ytr)),
                    batch_size=batch, shuffle=True)
    model = LSTMForecaster(n_features=len(CANONICAL_FEATURES), hidden=hidden,
                           layers=layers, dropout=dropout).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    lossf = torch.nn.SmoothL1Loss()

    Xte_t = torch.tensor(Xte).to(device)
    best_rmse, best_mae, best_r2 = float("inf"), float("inf"), 0.0
    best_state, best_ep = None, 0
    for ep in range(1, epochs + 1):
        model.train()
        tot = 0.0
        for xb, yb in tr:
            xb, yb = xb.to(device), yb.to(device)
            opt.zero_grad()
            loss = lossf(model(xb), yb)
            loss.backward()
            opt.step()
            tot += loss.item() * len(xb)
        model.eval()
        with torch.no_grad():
            pred = model(Xte_t).cpu().numpy()
        pr = scaler.inverse_target(pred, tgt)
        gt = scaler.inverse_target(yte, tgt)
        rmse = float(np.sqrt(np.mean((pr - gt) ** 2)))
        mae = float(np.mean(np.abs(pr - gt)))
        ss_res = float(np.sum((pr - gt) ** 2))
        ss_tot = float(np.sum((gt - gt.mean()) ** 2)) + 1e-8
        r2 = 1.0 - ss_res / ss_tot
        flag = ""
        if rmse < best_rmse:
            best_rmse, best_mae, best_r2, best_ep = rmse, mae, r2, ep
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            flag = "  <- best"
        print(f"epoch {ep:2d}  train_loss={tot/len(Xtr):.4f}  "
              f"test_RMSE={rmse:.2f}  test_MAE={mae:.2f}  test_R2={r2:.3f}{flag}")

    print(f"[best] epoch {best_ep}  test_RMSE={best_rmse:.2f}  test_R2={best_r2:.3f}")
    ckpt = {
        "state_dict": best_state,
        "feature_cols": CANONICAL_FEATURES,
        "target_idx": tgt,
        "scaler_mean": scaler.mean,
        "scaler_std": scaler.std,
        "seq_len": seq_len,
        "horizon": horizon,
        "hidden": hidden,
        "layers": layers,
        "dropout": dropout,
        "table": table,
        "mapping": mapping,
    }
    metrics = {"best_rmse": round(best_rmse, 2), "best_mae": round(best_mae, 2),
               "best_r2": round(best_r2, 3), "epoch": best_ep}
    return metrics, ckpt


def train(table: str = config.SENSOR_TABLE, model_path=DEFAULT_MODEL_PATH,
          mapping: dict | None = None, epochs: int = 15, batch: int = 64,
          lr: float = 1e-3, hidden: int = 64, layers: int = 2,
          dropout: float = 0.2, seq_len: int | None = None,
          horizon: int | None = None) -> dict:
    """학습 후 체크포인트를 저장하고 {best_rmse, best_mae, best_r2, epoch} 지표를 반환한다."""
    metrics, ckpt = _train_once(table, mapping, epochs, batch, lr, hidden, layers,
                                dropout, seq_len, horizon)
    torch.save(ckpt, model_path)
    print(f"[OK] 모델 저장: {model_path}")
    return metrics


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--table", default=config.SENSOR_TABLE)
    ap.add_argument("--model", default=str(DEFAULT_MODEL_PATH))
    ap.add_argument("--epochs", type=int, default=15)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--hidden", type=int, default=64)
    ap.add_argument("--layers", type=int, default=2)
    ap.add_argument("--dropout", type=float, default=0.2)
    a = ap.parse_args()
    train(table=a.table, model_path=a.model, epochs=a.epochs, batch=a.batch, lr=a.lr,
          hidden=a.hidden, layers=a.layers, dropout=a.dropout)
