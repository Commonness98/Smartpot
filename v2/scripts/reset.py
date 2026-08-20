"""실행으로 생성된 산출물을 삭제하여 '코드만 남은' 초기 상태로 되돌린다.

삭제 대상(모두 .gitignore 대상이라 코드/이력에는 영향 없음):
  - data/raw/*.csv           다운로드한 원본 데이터
  - data/processed/*         중간 산출물
  - data/smartfarm.db        SQLite DB
  - models_store/*.pt        학습된 모델

사용법:
    python scripts/reset.py           # 무엇이 지워질지 먼저 보여주고 확인
    python scripts/reset.py --yes     # 확인 없이 즉시 삭제
    python scripts/reset.py --keep-data   # 데이터는 두고 모델만 삭제
"""
import argparse
import shutil
import sys
from pathlib import Path

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from smartfarm import config  # noqa: E402


def _targets(keep_data: bool) -> list[Path]:
    items: list[Path] = [config.MODEL_DIR]                 # 모델
    if not keep_data:
        items += [
            config.RAW_DIR,                                # 원본 데이터
            config.PROCESSED_DIR,                          # 중간 산출물
            config.DATA_DIR / "smartfarm.db",              # SQLite DB
        ]
    return [p for p in items if p.exists()]


def _describe(p: Path) -> str:
    if p.is_dir():
        files = list(p.rglob("*"))
        size = sum(f.stat().st_size for f in files if f.is_file())
        return f"{p}  (폴더 내용, {size/1e6:.1f} MB)"
    return f"{p}  ({p.stat().st_size/1e6:.1f} MB)"


def _clear(p: Path):
    """파일은 삭제, 폴더는 내용만 비우고 폴더 자체는 유지."""
    if p.is_dir():
        for child in p.iterdir():
            if child.is_dir():
                shutil.rmtree(child, ignore_errors=True)
            else:
                child.unlink(missing_ok=True)
    else:
        p.unlink(missing_ok=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--yes", action="store_true", help="확인 없이 삭제")
    ap.add_argument("--keep-data", action="store_true", help="데이터는 유지, 모델만 삭제")
    a = ap.parse_args()

    targets = _targets(a.keep_data)
    if not targets:
        print("삭제할 산출물이 없습니다. 이미 초기 상태입니다.")
        raise SystemExit(0)

    print("다음 산출물을 삭제합니다:")
    for p in targets:
        print("  -", _describe(p))

    if not a.yes:
        ans = input("\n정말 삭제할까요? [y/N] ").strip().lower()
        if ans not in ("y", "yes"):
            print("취소했습니다.")
            raise SystemExit(0)

    for p in targets:
        _clear(p)
        print("  삭제됨:", p)
    print("\n✅ 초기 상태로 되돌렸습니다. 다시 실행하려면: python scripts/run_all.py")
