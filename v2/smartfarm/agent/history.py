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
_RE_YEAR_MONTH = re.compile(r"(\d{4})\s*년\s*(\d{1,2})\s*월")
_RE_MONTH_ONLY = re.compile(r"(\d{1,2})\s*월(?!\s*\d)")

# 상대 월 표현. 데이터의 마지막 관측월을 '이번 달'로 본다.
_REL_MONTH = {
    "지지난달": 2, "지지난 달": 2,
    "지난달": 1, "지난 달": 1, "저번달": 1, "저번 달": 1, "전달": 1, "전 달": 1,
    "이번달": 0, "이번 달": 0, "이달": 0,
}
_RE_RECENT_N = re.compile(r"(?:최근|지난)\s*(\d{1,3})\s*일")

# 현재 시점 표현. 데이터가 과거 기간이라 실제 오늘 값은 없으므로
# 마지막 관측일로 답하되 그 사실을 함께 알린다.
_RE_NOW = re.compile(r"오늘|지금|현재|요즘|근래")

# 보유 데이터 기간 자체를 묻는 질문
_RE_COVERAGE = re.compile(
    r"언제\s*부터|언제\s*까지|어느\s*기간|기간이\s*(어떻게|얼마)|"
    r"데이터\s*(가\s*)?(언제|어느|기간|범위)|자료\s*(가\s*)?(언제|어느|기간|범위)")


# --- 조회 가능한 측정 항목 ---
# 이름(label)과 인식어(words)는 표준 스키마 역할 기준이라 데이터셋과 무관하다.
# 단위는 데이터셋마다 다르므로(같은 일사량이라도 W/m²·lux·µmol/m²s 등) 여기
# 고정하지 않고 레지스트리에 보관된 값을 조회한다. 모르면 표기하지 않는다.
METRICS: dict[str, dict] = {
    "power_target": {"label": "에너지 소비",
                     "words": ["에너지", "전력", "전기", "소비", "사용량",
                               "사용률", "요금", "kwh"]},
    "light":        {"label": "일사량",
                     "words": ["일사", "광량", "햇빛", "조도", "일조", "par"]},
    "indoor_temp":  {"label": "실내 온도",
                     "words": ["실내 온도", "실내온도", "온실 온도", "온실온도",
                               "내부 온도", "내부온도"]},
    "indoor_humid": {"label": "실내 습도",
                     "words": ["실내 습도", "실내습도", "온실 습도", "온실습도",
                               "내부 습도", "내부습도"]},
    "out_temp":     {"label": "외부 기온",
                     "words": ["외부 기온", "외부기온", "외기온", "외부 온도",
                               "외부온도", "바깥 온도", "바깥온도", "바깥 기온"]},
    "out_humid":    {"label": "외부 습도",
                     "words": ["외부 습도", "외부습도", "외기 습도", "바깥 습도"]},
    "windspeed":    {"label": "풍속",
                     "words": ["풍속", "바람"]},
    "co2":          {"label": "CO2 농도",
                     "words": ["co2", "이산화탄소", "탄산가스"]},
    "pressure":     {"label": "기압",
                     "words": ["기압"]},
}

# 수식어 없이 "온도"·"습도"만 물으면 온실 내부를 뜻하는 것으로 본다.
_BARE_FALLBACK = {"온도": "indoor_temp", "습도": "indoor_humid"}

# 데이터가 비어 있는(전부 동일값) 항목을 조회했을 때 알린다.
_EMPTY_NOTE = "원본 데이터에 값이 없어 조회할 수 없습니다"


def _has_batchim(word: str) -> bool:
    """마지막 글자에 받침이 있는지. 조사(은/는) 선택에 쓴다."""
    for ch in reversed(word.strip()):
        code = ord(ch) - 0xAC00
        if 0 <= code < 11172:              # 한글 음절
            return code % 28 != 0
        if ch.isdigit():                   # 숫자로 끝나면 읽는 소리 기준
            return ch in "0136780"
        if ch.isalpha():                   # 영문·기호는 받침 없는 것으로 본다
            return False
    return False


def _josa(word: str, with_batchim: str, without: str) -> str:
    """단어 뒤에 알맞은 조사를 붙인다(일사량은 / 실내 온도는)."""
    return word + (with_batchim if _has_batchim(word) else without)


def _match_metrics(question: str) -> list[str]:
    """질문에 명시된 측정 항목. 없으면 빈 목록(기본값을 넣지 않는다)."""
    q = question.lower()
    found = [role for role, spec in METRICS.items()
             if any(w in q for w in spec["words"])]
    if not found:
        # "5월 27일 온도는?" 처럼 수식어가 없는 경우
        found = [role for word, role in _BARE_FALLBACK.items() if word in q]
    return found


def detect_metrics(question: str) -> list[str]:
    """질문이 어떤 측정 항목을 묻는지 찾는다. 없으면 에너지를 기본값으로 쓴다."""
    return _match_metrics(question) or [TARGET_FEATURE]


def sensor_metrics(question: str) -> list[str]:
    """에너지 외 센서 항목이 명시적으로 언급됐는지.

    예측 대상은 에너지 수요뿐이다. 일사량·온도 같은 센서 항목은 예측하지 않으므로,
    이런 항목이 언급되면 조회로 처리해야 한다. 그러지 않으면 시점 단어("오늘")
    때문에 예측 경로로 흘러가 에너지 예측값을 해당 항목인 양 답하게 된다.
    """
    return [m for m in _match_metrics(question) if m != TARGET_FEATURE]


def unit_of(role: str) -> str:
    """활성 데이터셋에 기록된 단위. 모르면 빈 문자열(표기하지 않음)."""
    from .. import registry
    active = registry.get_active() or {}
    return (active.get("units") or {}).get(role, "")


def _fmt(role: str, value: float) -> str:
    label = METRICS.get(role, {"label": role})["label"]
    unit = unit_of(role)
    tail = f" {unit}" if unit else ""
    return f"{_josa(label, '은', '는')} {round(float(value), 2)}{tail}"


def _compose(head: str, value_parts: list[str], empty_parts: list[str]) -> str:
    """값 문장과 '데이터 없음' 안내를 자연스럽게 잇는다."""
    out = ""
    if value_parts:
        out = f"{head} " + ", ".join(value_parts) + "입니다."
    if empty_parts:
        note = ", ".join(_josa(p, "은", "는") for p in empty_parts)
        out = (out + " " if out else "") + f"{note} {_EMPTY_NOTE}."
    return out.strip()


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
    if any(w in q for w in _REL_MONTH):
        return True
    if any(w in q for w in _PAST_DAYS):
        return True
    if _RE_EXTREME_MAX.search(q) or _RE_EXTREME_MIN.search(q):
        return True
    if _RE_COVERAGE.search(q):
        return True
    # 센서 항목은 예측 대상이 아니므로 언제를 묻든 조회로 처리한다
    if sensor_metrics(q):
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

    # 연도까지 명시된 월("2021년 1월"). 연도를 무시하고 데이터에 있는 해로
    # 답해버리면 묻지 않은 기간을 답하게 되므로 먼저 처리한다.
    m = _RE_YEAR_MONTH.search(question)
    if m:
        y, mo = int(m.group(1)), int(m.group(2))
        if df[(df["ts"].dt.year == y) & (df["ts"].dt.month == mo)].empty:
            return {"kind": "out_of_range", "text": f"{y}년 {mo}월"}
        return {"kind": "month", "year": y, "month": mo}

    # 상대 월 표현(지난달·이번달 등). 데이터 마지막 관측월이 기준이다.
    for word, back in _REL_MONTH.items():
        if word in question:
            anchor = (last.to_period("M") - back)
            y, mo = anchor.year, anchor.month
            if df[(df["ts"].dt.year == y) & (df["ts"].dt.month == mo)].empty:
                return {"kind": "out_of_range", "text": f"{y}년 {mo}월"}
            return {"kind": "month", "year": y, "month": mo}

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

    # "오늘·지금·현재"는 실제 오늘이 아니라 마지막 관측일로 답하되, 그 사실을 밝힌다.
    # 데이터가 과거 기간(2019-12~2020-05)이라 실제 오늘 값은 존재하지 않는다.
    if _RE_NOW.search(question):
        return {"kind": "point", "date": last, "is_latest": True}

    if _RE_EXTREME_MAX.search(question):
        return {"kind": "extreme", "which": "max"}
    if _RE_EXTREME_MIN.search(question):
        return {"kind": "extreme", "which": "min"}

    if "전체" in question or "전 기간" in question or "그동안" in question:
        return {"kind": "range", "start": df["ts"].min(), "end": last,
                "label": "전체 기간"}
    return None


def _stats(sub: pd.DataFrame, role: str = TARGET_FEATURE) -> dict:
    t = sub[role].astype(float)
    return {"days": int(len(sub)), "avg": round(float(t.mean()), 2),
            "min": round(float(t.min()), 2), "max": round(float(t.max()), 2),
            "total": round(float(t.sum()), 2)}


def _stats_text(sub: pd.DataFrame, metrics: list[str], df: pd.DataFrame):
    """요청된 항목별 평균·최소·최대를 한 문장으로 만든다."""
    parts, empty = [], []
    for role in metrics:
        if role not in sub.columns:
            continue
        if df[role].astype(float).nunique() <= 1:
            empty.append(METRICS[role]["label"])
            continue
        st = _stats(sub, role)
        label = METRICS.get(role, {"label": role})["label"]
        unit = unit_of(role)
        tail = f" {unit}" if unit else ""
        parts.append(f"{_josa(label, '은', '는')} 평균 {st['avg']}{tail}, "
                     f"최소 {st['min']}, 최대 {st['max']}")
    return parts, empty


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
    metrics = detect_metrics(question)
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
        values, parts, empty = {}, [], []
        for role in metrics:
            if role not in row.columns:
                continue
            if df[role].astype(float).nunique() <= 1:   # 원본에 값이 없는 항목
                empty.append(METRICS[role]["label"])
                continue
            v = round(float(row[role].iloc[0]), 2)
            values[role] = v
            parts.append(_fmt(role, v))
        # "오늘"을 물었을 때는 실제 오늘이 아니라 마지막 관측일임을 밝힌다
        head = (f"가장 최근 기록일인 {d.date()}의 실측"
                if period.get("is_latest") else f"{d.date()}의 실측")
        summary = _compose(head, parts, empty)
        if period.get("is_latest") and parts:
            summary += " (보유 데이터가 여기까지라 이후 값은 없습니다.)"
        return {**base, "kind": "point", "found": True, "date": str(d.date()),
                "metrics": metrics, "values": values,
                "is_latest": bool(period.get("is_latest")),
                # 단일 항목 조회 시 기존 호출부 호환을 위해 value도 함께 둔다
                "value": values.get(TARGET_FEATURE, next(iter(values.values()), None)),
                "summary": summary}

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
                "metrics": metrics,
                "summary": _compose(f"{head} {s['days']}일간의 실측",
                                    *_stats_text(sub, metrics, df))}

    if period["kind"] == "month":
        y, mo = period["year"], period["month"]
        sub = df[(df["ts"].dt.year == y) & (df["ts"].dt.month == mo)]
        s = _stats(sub)
        return {**base, "kind": "month", "found": True, "year": y, "month": mo, **s,
                "metrics": metrics,
                "summary": _compose(f"{y}년 {mo}월({s['days']}일)의 실측",
                                    *_stats_text(sub, metrics, df))}

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
