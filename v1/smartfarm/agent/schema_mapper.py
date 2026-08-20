"""하이브리드 컬럼 분류기: 규칙(휴리스틱) 1차 + LLM 보정.

업로드된 데이터프레임의 각 컬럼이 어떤 논리 역할(전력타깃/실내온도/습도/외부기상 등)
인지 추정한다. LLM(LM Studio/Ollama)이 없으면 휴리스틱 결과만 사용한다.
"""
from __future__ import annotations

import json
import re

import pandas as pd

from .. import schema
from . import llm_client

_LOW_CONF = 0.6  # 이 값 미만이면 LLM 보정 대상


def _column_summary(df: pd.DataFrame, col: str, n: int = 5) -> dict:
    s = df[col]
    info = {"column": str(col), "dtype": str(s.dtype),
            "samples": [str(v) for v in s.dropna().head(n).tolist()]}
    if pd.api.types.is_numeric_dtype(s):
        sn = pd.to_numeric(s, errors="coerce").dropna()
        if not sn.empty:
            info["min"] = round(float(sn.min()), 2)
            info["max"] = round(float(sn.max()), 2)
            info["mean"] = round(float(sn.mean()), 2)
    return info


def _llm_refine(df: pd.DataFrame, base: dict[str, dict],
                targets: list[str]) -> dict[str, str]:
    """저신뢰 컬럼만 LLM에 보내 역할을 재추정한다. {컬럼: 역할} 반환."""
    cols_info = [_column_summary(df, c) for c in targets]
    prompt = (
        "다음은 스마트팜 시계열 데이터의 일부 컬럼 요약이다. "
        "각 컬럼을 아래 역할 중 하나로 분류하라.\n"
        f"역할: {', '.join(schema.ALL_ROLES)}\n"
        "- timestamp: 날짜/시각\n- power_target: 전력 수요/사용량(예측 대상)\n"
        "- indoor_temp/indoor_humid: 실내(온실) 온도/습도\n"
        "- out_temp/out_humid: 외부 기온/습도\n"
        "- pressure: 기압, windspeed: 풍속, light: 조명/조도\n"
        "- ignore: 위에 해당 없음\n\n"
        f"컬럼 요약(JSON): {json.dumps(cols_info, ensure_ascii=False)}\n\n"
        "반드시 {\"컬럼명\": \"역할\", ...} 형태의 JSON만 출력하라."
    )
    text = llm_client.chat(
        [{"role": "system", "content": "너는 데이터 스키마 분류기다. JSON만 출력한다."},
         {"role": "user", "content": prompt}],
        temperature=0.0, max_tokens=400)
    m = re.search(r"\{.*\}", text, re.DOTALL)
    if not m:
        return {}
    try:
        raw = json.loads(m.group(0))
    except json.JSONDecodeError:
        return {}
    # 유효 역할만 채택
    return {k: v for k, v in raw.items()
            if k in targets and v in schema.ALL_ROLES}


def propose_mapping(df: pd.DataFrame, use_llm: bool = True) -> list[dict]:
    """컬럼별 역할 제안 목록을 반환한다.

    각 항목: {column, role, confidence, reason}
    """
    base = schema.heuristic_roles(df)

    if use_llm and llm_client.available():
        low = [c for c, g in base.items()
               if g["confidence"] < _LOW_CONF and g["role"] != schema.ROLE_TIMESTAMP]
        if low:
            try:
                refined = _llm_refine(df, base, low)
                for col, role in refined.items():
                    base[col] = {"role": role, "confidence": 0.7,
                                 "reason": "LLM 보정"}
            except Exception:
                pass  # LLM 실패 시 휴리스틱 유지

    return [{"column": str(c), **g} for c, g in base.items()]


def mapping_dict(proposals: list[dict]) -> dict[str, str]:
    """제안/확정 목록을 build_canonical용 {컬럼: 역할} 딕셔너리로 변환."""
    return {p["column"]: p["role"] for p in proposals}
