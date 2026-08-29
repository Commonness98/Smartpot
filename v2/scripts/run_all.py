"""전체 파이프라인을 한 번에 실행한다.

  1) 공개 데이터셋 다운로드
  2) DB 적재
  3) 자동 파라미터 튜닝 에이전트로 전력수요 LSTM 학습

사용법:
    python scripts/run_all.py                # 기본 트라이얼당 15 epochs
    python scripts/run_all.py --epochs 20 --trials 8

이후 웹 UI 실행:
    python -m streamlit run app/app.py
"""
import argparse
import os
import subprocess
import sys
from pathlib import Path

# Windows 콘솔(cp949)에서 이모지/한글 출력 시 크래시 방지
try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

ROOT = Path(__file__).resolve().parent.parent
PY = sys.executable
# 하위 프로세스도 UTF-8로 출력하도록 강제
CHILD_ENV = {**os.environ, "PYTHONUTF8": "1", "PYTHONIOENCODING": "utf-8"}


def step(title: str, args: list[str]):
    print("\n" + "=" * 60)
    print(f"▶ {title}")
    print("=" * 60)
    r = subprocess.run([PY, *args], cwd=ROOT, env=CHILD_ENV)
    if r.returncode != 0:
        raise SystemExit(f"[중단] '{title}' 단계 실패 (exit={r.returncode})")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=15)
    ap.add_argument("--trials", type=int, default=6)
    a = ap.parse_args()

    step("1/3 데이터 다운로드", ["scripts/00_download_data.py"])
    step("2/3 DB 적재", ["scripts/01_load_data.py"])
    step("3/3 자동 파라미터 튜닝 학습", ["-m", "smartfarm.agent.tuner",
                                     "--epochs", str(a.epochs), "--trials", str(a.trials)])

    print("\n✅ 전체 파이프라인 완료.")
    print("   웹 UI 실행:  python -m streamlit run app/app.py")
    print("   되돌리기  :  python scripts/reset.py")
