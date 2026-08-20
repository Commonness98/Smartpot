"""과거 실측 데이터 조회.

"5월 3일 얼마 썼어?", "지난주 평균은?", "제일 많이 쓴 날이 언제야?" 같은 질문을
처리한다. 예측 모델을 전혀 거치지 않고 DB에 적재된 실측값을 그대로 조회하므로,
예측 신뢰도 문제와 무관하며 값 자체가 실제 측정치다.

날짜 기준(today)은 실제 오늘이 아니라 **데이터의 마지막 관측일**이다. 프로토타입
데이터셋이 과거 기간(2019-12 ~ 2020-05)이라, "지난주" 같은 상대 표현을 실제
오늘 기준으로 풀면 데이터 범위를 벗어나기 때문이다.
"""
from __future__ import annotations

import re

import pandas as pd

from ..schema import TARGET_FEATURE
from .planner import _model_path, dataset_frame

# --- 질문에서 과거 조회 의도를 식별하는 신호 ---
_PAST_MARKERS = ["지난", "저번", "과거", "이전", "그동안", "여태", "지금까지"]

# 과거를 가리키는 상대 날짜 표현. "오늘"은 운영 질문("오늘 방제할까?")과
# 겹치므로 제외한다.
_PAST_DAYS = {"어제": 1, "그저께": 2, "그제": 2, "엊그제": 2, "그끄제": 3}

# 한글 종성 ㅆ의 인덱스. 과거 시제 선어말어미(았/었/였/했/썼…)의 표지다.
_JONG_SS = 20
_PAST_ENDINGS = "어다니나지는"
# 최상급 표현. "제일 에너지를 많이 쓴 날"처럼 수식어와 형용사 사이에 다른 말이
# 끼어드는 경우가 흔해, 단순 문자열 포함 대신 사이 간격을 허용하는 정규식을 쓴다.
_RE_EXTREME_MAX = re.compile(r"(제일|가장)[^.?!]{0,15}?(많|높|크)|최대|최고|피크")
_RE_EXTREME_MIN = re.compile(r"(제일|가장)[^.?!]{0,15}?(적|낮|작)|최소|최저")
_AGGREGATE = ["평균", "합계", "총", "통계", "추이", "비교", "사용", "소비", "썼"]

# --- 날짜 표현 ---
_RE_FULL_DATE = re.compile(r"(\d{4})\s*[-/.년]\s*(\d{1,2})\s*[-/.월]\s*(\d{1,2})")
_RE_MD_KO = re.compile(r"(\d{1,2})\s*월\s*(\d{1,2})\s*일")
_RE_MD_SLASH = re.compile(r"(?<!\d)(\d{1,2})\s*/\s*(\d{1,2})(?!\d)")
_RE_MONTH_ONLY = re.compile(r"(\d{1,2})\s*월(?!\s*\d)")
_RE_RECENT_N = re.compile(r"(?:최근|지난)\s*(\d{1,3})\s*일")

# 보유 데이터 기간 자체를 묻는 질문
_RE_COVERAGE = re.compile(
    r"언제\s*부터|언제\s*까지|어느\s*기간|기간이\s*(어떻게|얼마)|"
    r"데이터\s*(가\s*)?(언제|어느|기간|범위)|자료\s*(가\s*)?(언제|어느|기간|범위)")


def _target(df: pd.DataFrame) -> pd.Series:
    return df[TARGET_FEATURE].astype(float)


def data_range(model_path=None) -> dict:
    """적재된 실측 데이터의 기간·건수."""
    df = dataset_frame(_model_path(model_path))
    if df.empty:
        return {"rows": 0}
    return {
        "rows": len(df),
        "start": str(df["ts"].min().date()),
        "end": str(df["ts"].max().date()),
        "note": "상대 표현(지난주 등)은 이 마지막 관측일을 기준으로 해석합니다.",
    }


def has_past_tense(text: str) -> bool:
    """한국어 과거 시제 표현이 있는지 판단한다.

    "사용했어", "썼나", "어땠지"처럼 활용형이 무한히 많아 어휘 목록으로는
    계속 누락된다. 과거 선어말어미가 종성 ㅆ으로 실현되는 규칙을 이용해
    'ㅆ 받침 + 어미'를 찾는다.
    """
    for i, ch in enumerate(text[:-1]):
        code = ord(ch) - 0xAC00
        if not (0 <= code < 11172):        # 한글 음절이 아님
            continue
        if code % 28 != _JONG_SS:
            continue
        if ch == "있":                      # "있어"는 현재형이므로 제외
            continue
        if text[i + 1] in _PAST_ENDINGS:
            return True
    return False


def is_coverage_question(question: str) -> bool:
    """보유 데이터 기간 자체를 묻는 질문인지.

    "언제부터 언제까지 데이터 있어?"처럼 시점 단어를 포함하지만 미래 예측과는
    무관하다. 호출부에서 시점 단어로 인한 의도 혼동을 피하는 데 쓴다.
    """
    return bool(_RE_COVERAGE.search(question))


def looks_like_history(question: str) -> bool:
    """과거 실측 조회 질문으로 볼 만한지 판단."""
    q = question
    if any(w in q for w in _PAST_MARKERS):
        return True
    if any(w in q for w in _PAST_DAYS):
        return True
    if _RE_EXTREME_MAX.search(q) or _RE_EXTREME_MIN.search(q):
        return True
    if _RE_COVERAGE.search(q):
        return True
    if has_past_tense(q) and any(w in q for w in _AGGREGATE + ["얼마"]):
        return True
    # 구체적 날짜/월이 등장하면 과거 조회로 본다(미래 예측은 1일치뿐이므로)
    if _RE_FULL_DATE.search(q) or _RE_MD_KO.search(q) or _RE_MONTH_ONLY.search(q):
        return True
    if _RE_RECENT_N.search(q):
        return True
    return False


def _resolve_year(month: int, day: int | None, df: pd.DataFrame) -> int | None:
    """데이터 범위 안에서 해당 월/일이 존재하는 연도를 찾는다."""
    years = sorted(df["ts"].dt.year.unique())
    for y in years:
        m = df[(df["ts"].dt.year == y) & (df["ts"].dt.month == month)]
        if m.empty:
            continue
        if day is None or not m[m["ts"].dt.day == day].empty:
            return int(y)
    return None


def _find_dates(question: str, df: pd.DataFrame) -> list:
    """질문에 등장하는 모든 날짜를 나온 순서대로 찾는다.

    "1월 3일부터 2월 3일까지"처럼 날짜가 둘 이상이면 기간 조회로 해석해야 한다.
    첫 날짜만 보고 하루치를 답하면 기간 평균을 물었는데 단일 값을 주게 된다.
    반환 항목은 (Timestamp) 또는 범위 밖이면 ("out", 표기문자열).
    """
    found: list[tuple[int, object]] = []
    spans: list[tuple[int, int]] = []

    for m in _RE_FULL_DATE.finditer(question):
        y, mo, d = (int(g) for g in m.groups())
        found.append((m.start(), pd.Timestamp(y, mo, d)))
        spans.append((m.start(), m.end()))

    for rx in (_RE_MD_KO, _RE_MD_SLASH):
        for m in rx.finditer(question):
            # 연-월-일 표기 안에 포함된 부분 일치는 건너뛴다
            if any(s <= m.start() < e for s, e in spans):
                continue
            mo, d = int(m.group(1)), int(m.group(2))
            y = _resolve_year(mo, d, df)
            found.append((m.start(),
                          pd.Timestamp(y, mo, d) if y else ("out", f"{mo}월 {d}일")))
            spans.append((m.start(), m.end()))

    found.sort(key=lambda t: t[0])
    return [v for _, v in found]


def parse_period(question: str, df: pd.DataFrame) -> dict | None:
    """질문에서 조회 기간을 해석한다. 실패 시 None."""
    last = df["ts"].max()

    # 보유 데이터 기간 자체를 묻는 질문
    if _RE_COVERAGE.search(question):
        return {"kind": "coverage"}

    dates = _find_dates(question, df)
    bad = [d for d in dates if isinstance(d, tuple)]
    if bad:
        return {"kind": "out_of_range", "text": bad[0][1]}
    if len(dates) >= 2:
        start, end = sorted(dates[:2])
        return {"kind": "range", "start": start, "end": end,
                "label": f"{start.date()}~{end.date()}"}
    if len(dates) == 1:
        return {"kind": "point", "date": dates[0]}

    # 월 단위 (여러 개면 비교)
    months = [int(x) for x in _RE_MONTH_ONLY.findall(question)]
    if months:
        resolved = []
        for mo in months:
            y = _resolve_year(mo, None, df)
            if y:
                resolved.append((y, mo))
        if len(resolved) >= 2:
            return {"kind": "compare_months", "months": resolved[:2]}
        if resolved:
            return {"kind": "month", "year": resolved[0][0], "month": resolved[0][1]}
        return {"kind": "out_of_range", "text": f"{months[0]}월"}

    m = _RE_RECENT_N.search(question)
    if m:
        n = int(m.group(1))
        return {"kind": "range", "start": last - pd.Timedelta(days=n - 1),
                "end": last, "label": f"최근 {n}일"}

    if "지난주" in question or "저번주" in question or "지난 주" in question:
        return {"kind": "range", "start": last - pd.Timedelta(days=6),
                "end": last, "label": "지난 7일"}

    # 상대 날짜(어제·그저께 등)는 데이터 마지막 관측일을 기준으로 센다
    for word, back in _PAST_DAYS.items():
        if word in question:
            return {"kind": "point", "date": last - pd.Timedelta(days=back)}

    if _RE_EXTREME_MAX.search(question):
        return {"kind": "extreme", "which": "max"}
    if _RE_EXTREME_MIN.search(question):
        return {"kind": "extreme", "which": "min"}

    if "전체" in question or "전 기간" in question or "그동안" in question:
        return {"kind": "range", "start": df["ts"].min(), "end": last,
                "label": "전체 기간"}
    return None


def _stats(sub: pd.DataFrame) -> dict:
    t = _target(sub)
    return {"days": int(len(sub)), "avg": round(float(t.mean()), 2),
            "min": round(float(t.min()), 2), "max": round(float(t.max()), 2),
            "total": round(float(t.sum()), 2)}


def query(question: str, model_path=None) -> dict | None:
    """과거 실측 조회 결과. 해석 실패 시 None.

    반환에는 LLM이 그대로 인용할 수 있는 완성 문장(summary)을 포함한다.
    """
    df = dataset_frame(_model_path(model_path))
    if df.empty:
        return None
    period = parse_period(question, df)
    if not period:
        return None

    rng = data_range(model_path)
    base = {"source": "실측값(예측 아님)", "data_range": rng}

    if period["kind"] == "coverage":
        t = _target(df)
        return {**base, "kind": "coverage", "found": True,
                "start": rng["start"], "end": rng["end"], "days": rng["rows"],
                "summary": (f"보유 데이터는 {rng['start']}부터 {rng['end']}까지 "
                            f"총 {rng['rows']}일치입니다. 이 기간 실측 에너지는 "
                            f"평균 {round(float(t.mean()), 2)}, "
                            f"최소 {round(float(t.min()), 2)}, "
                            f"최대 {round(float(t.max()), 2)}입니다.")}

    if period["kind"] == "out_of_range":
        return {**base, "kind": "out_of_range", "found": False,
                "summary": (f"{period['text']}은(는) 보유 데이터 기간을 벗어납니다. "
                            f"조회 가능한 기간은 {rng['start']} ~ {rng['end']}입니다.")}

    if period["kind"] == "point":
        d = pd.Timestamp(period["date"]).normalize()
        row = df[df["ts"].dt.normalize() == d]
        if row.empty:
            return {**base, "kind": "point", "found": False,
                    "summary": (f"{d.date()}의 실측 데이터는 없습니다. "
                                f"보유 기간은 {rng['start']} ~ {rng['end']}입니다.")}
        v = round(float(_target(row).iloc[0]), 2)
        return {**base, "kind": "point", "found": True, "date": str(d.date()),
                "value": v,
                "summary": f"{d.date()}의 실측 에너지 소비는 {v}입니다."}

    if period["kind"] == "range":
        sub = df[(df["ts"] >= period["start"]) & (df["ts"] <= period["end"])]
        if sub.empty:
            return {**base, "kind": "range", "found": False,
                    "summary": "해당 기간의 실측 데이터가 없습니다."}
        s = _stats(sub)
        label = period.get("label", "해당 기간")
        span = f"{period['start'].date()}~{period['end'].date()}"
        # label이 이미 날짜 범위면 중복 표기하지 않는다
        head = span if label == span else f"{label}({span})"
        return {**base, "kind": "range", "found": True, "label": label,
                "start": str(period["start"].date()), "end": str(period["end"].date()),
                **s,
                "summary": (f"{head} {s['days']}일간의 실측 에너지는 "
                            f"평균 {s['avg']}, 최소 {s['min']}, 최대 {s['max']}입니다.")}

    if period["kind"] == "month":
        y, mo = period["year"], period["month"]
        sub = df[(df["ts"].dt.year == y) & (df["ts"].dt.month == mo)]
        s = _stats(sub)
        return {**base, "kind": "month", "found": True, "year": y, "month": mo, **s,
                "summary": (f"{y}년 {mo}월({s['days']}일)의 실측 에너지는 "
                            f"평균 {s['avg']}, 최소 {s['min']}, 최대 {s['max']}입니다.")}

    if period["kind"] == "compare_months":
        parts, detail = [], []
        for y, mo in period["months"]:
            sub = df[(df["ts"].dt.year == y) & (df["ts"].dt.month == mo)]
            s = _stats(sub)
            detail.append({"year": y, "month": mo, **s})
            parts.append(f"{y}년 {mo}월 평균 {s['avg']}({s['days']}일)")
        gap = round(detail[0]["avg"] - detail[1]["avg"], 2)
        higher = detail[0] if gap > 0 else detail[1]
        return {**base, "kind": "compare_months", "found": True, "months": detail,
                "diff": abs(gap),
                "summary": (f"{' / '.join(parts)}. "
                            f"{higher['month']}월이 {abs(gap)}만큼 더 높습니다.")}

    if period["kind"] == "extreme":
        t = _target(df)
        idx = t.idxmax() if period["which"] == "max" else t.idxmin()
        row = df.loc[idx]
        word = "가장 많이" if period["which"] == "max" else "가장 적게"
        v = round(float(row[TARGET_FEATURE]), 2)
        return {**base, "kind": "extreme", "found": True,
                "which": period["which"], "date": str(row["ts"].date()), "value": v,
                "summary": (f"보유 기간 중 에너지를 {word} 쓴 날은 "
                            f"{row['ts'].date()}이며 {v}입니다.")}

    return None
