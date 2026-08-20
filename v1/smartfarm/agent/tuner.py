"""자동 파라미터 튜닝 에이전트.

새 데이터로 (재)학습할 때 LSTM 하이퍼파라미터(hidden size, layers, dropout,
learning rate, seq_len)를 사람이 고정값으로 넣는 대신, 여러 조합을 무작위로
탐색해 검증 RMSE와 R²(결정계수)를 함께 반영한 점수가 가장 좋은 조합을 채택한다.
외부 튜닝 라이브러리(Optuna 등) 없이 기존 학습 루프(_train_once)만 재사용하는
경량 랜덤 서치다.

사용법:
    python -m smartfarm.agent.tuner --trials 6 --epochs 8            # 기본 데이터셋
    python -m smartfarm.agent.tuner --table ds_xxx --model models_store/power_lstm_ds_xxx.pt
"""
from __future__ import annotations

import argparse
import itertools
import random
from typing import Callable

import torch

from .. import config
from ..models.train import _train_once

# 탐색 공간. horizon/batch는 예측 태스크 정의(내일 하루 뒤 예측)를 안정적으로
# 유지하기 위해 고정값을 쓰고 탐색 대상에서 제외한다.
SEARCH_SPACE = {
    "hidden": [32, 64, 128],
    "layers": [1, 2],
    "dropout": [0.0, 0.2],
    "lr": [2e-3, 1e-3, 5e-4],
    "seq_len": [7, 14, 21],
}


def _score(rmse: float, r2: float) -> float:
    """RMSE와 R²를 함께 반영한 점수(낮을수록 좋음).

    RMSE만 보면 근소하게 오차가 작지만 설명력(R²)이 떨어지는 조합을 고를 수 있어,
    R²가 낮을수록 점수가 커지도록(=불리해지도록) 나눠준다.
    """
    return rmse / max(r2, 0.01)


def auto_train(table: str = config.SENSOR_TABLE, model_path=None,
              mapping: dict | None = None, epochs: int = 8, n_trials: int = 6,
              batch: int = 64, seed: int = 42,
              progress_cb: Callable[[int, int, dict, dict], None] | None = None,
              ) -> dict:
    """무작위 하이퍼파라미터 조합을 탐색해 가장 좋은 모델만 저장하고 지표를 반환한다.

    progress_cb(trial_idx, n_trials, combo, metrics)를 넘기면 매 트라이얼 직후 호출되어
    UI에서 진행 상황을 표시할 수 있다.
    """
    model_path = model_path or (config.MODEL_DIR / "power_lstm.pt")
    keys = list(SEARCH_SPACE.keys())
    all_combos = list(itertools.product(*(SEARCH_SPACE[k] for k in keys)))
    n_trials = min(n_trials, len(all_combos))
    rng = random.Random(seed)
    chosen = rng.sample(all_combos, n_trials)

    best_score, best_metrics, best_ckpt, best_combo = float("inf"), None, None, None
    for i, values in enumerate(chosen, 1):
        combo = dict(zip(keys, values))
        metrics, ckpt = _train_once(
            table=table, mapping=mapping, epochs=epochs, batch=batch,
            lr=combo["lr"], hidden=combo["hidden"], layers=combo["layers"],
            dropout=combo["dropout"], seq_len=combo["seq_len"],
        )
        score = _score(metrics["best_rmse"], metrics["best_r2"])
        print(f"[trial {i}/{n_trials}] {combo} -> RMSE={metrics['best_rmse']} "
              f"R2={metrics['best_r2']} score={score:.3f}")
        if progress_cb:
            progress_cb(i, n_trials, combo, metrics)
        if score < best_score:
            best_score, best_metrics, best_ckpt, best_combo = score, metrics, ckpt, combo

    torch.save(best_ckpt, model_path)
    print(f"[OK] 최적 조합 {best_combo} 저장: {model_path}")
    return {**best_metrics, "params": best_combo}


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--table", default=config.SENSOR_TABLE)
    ap.add_argument("--model", default=str(config.MODEL_DIR / "power_lstm.pt"))
    ap.add_argument("--trials", type=int, default=6)
    ap.add_argument("--epochs", type=int, default=8)
    a = ap.parse_args()
    auto_train(table=a.table, model_path=a.model, epochs=a.epochs, n_trials=a.trials)
