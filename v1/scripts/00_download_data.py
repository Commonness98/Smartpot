"""공개 데이터셋(WUR Autonomous Greenhouse Challenge, 2nd Edition) 다운로드.

실제 상업용 유리온실 6개 구획에서 6개월간 수집된 기후·자원소비(난방·전기) 데이터다.
출처: https://data.4tu.nl/datasets/8b35675d-7549-49cc-a8c3-dedec8d3f23e

사용법:
    python scripts/00_download_data.py
"""
import sys
import urllib.request
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from smartfarm import config
from smartfarm.data_pipeline import WUR_DIR

URL = (
    "https://data.4tu.nl/file/8b35675d-7549-49cc-a8c3-dedec8d3f23e/"
    "12ed099a-41b1-4a1c-a7be-12b588f2e34a"
)
ARCHIVE_PATH = config.RAW_DIR / "AutonomousGreenhouseChallenge_edition2.7z"


def _extract(archive_path: Path, dest: Path) -> None:
    import py7zr
    dest.mkdir(parents=True, exist_ok=True)
    with py7zr.SevenZipFile(archive_path, mode="r") as z:
        z.extractall(path=dest)


if __name__ == "__main__":
    if (WUR_DIR / "Reference" / "GreenhouseClimate.csv").exists():
        print(f"[SKIP] 이미 압축 해제됨: {WUR_DIR}")
    else:
        if not ARCHIVE_PATH.exists():
            print(f"다운로드 중: {URL}")
            try:
                urllib.request.urlretrieve(URL, ARCHIVE_PATH)
            except Exception as e:
                raise SystemExit(
                    f"[실패] 다운로드에 실패했습니다: {e}\n"
                    f"4TU.ResearchData 페이지에서 수동으로 내려받아 다음 경로에 저장한 뒤 "
                    f"다시 실행하세요: {ARCHIVE_PATH}\n"
                    "(https://data.4tu.nl/datasets/8b35675d-7549-49cc-a8c3-dedec8d3f23e)"
                )
            print(f"[OK] 저장됨: {ARCHIVE_PATH} ({ARCHIVE_PATH.stat().st_size:,} bytes)")
        print(f"압축 해제 중 → {WUR_DIR}")
        _extract(ARCHIVE_PATH, WUR_DIR)
        print(f"[OK] 압축 해제 완료: {WUR_DIR}")
