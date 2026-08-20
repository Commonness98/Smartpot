"""로컬 LLM(Ollama / LM Studio) 연동 클라이언트.

두 백엔드 모두 OpenAI 호환 chat completions API를 제공하므로 동일하게 다룬다.
서버가 없으면 available()가 False를 반환하며, 호출부는 규칙 기반 응답으로 폴백한다.

v2 변경점
- available() 결과를 짧게 캐시. 기존에는 호출할 때마다 HTTP 요청이 나가
  Streamlit 리런마다 중복 요청되던 문제가 있었다(app.py 사이드바 + core.py 2곳).
- 설정된 백엔드가 응답하지 않으면 알려진 로컬 백엔드를 자동 탐지한다
  (Jetson 등 엣지 이식 시 백엔드가 바뀌어도 설정 수정 없이 동작).
- chat() 재시도 처리 추가. 일시적 실패로 폴백되던 경우를 줄인다.
"""
from __future__ import annotations

import time

import requests

from .. import config

# 서버 생존 확인 결과 캐시 유효시간(초).
_PROBE_TTL = 30.0

# 자동 탐지 대상 백엔드. config 설정값을 먼저 시도한 뒤 순서대로 확인한다.
_FALLBACK_BASE_URLS = [
    "http://localhost:11434/v1",   # Ollama
    "http://localhost:1234/v1",    # LM Studio
]

_probe: dict = {"at": 0.0, "ok": False, "base_url": None, "models": []}


def _headers() -> dict:
    return {"Authorization": f"Bearer {config.LM_STUDIO_API_KEY}"}


def _try_backend(base_url: str, timeout: float) -> list[str] | None:
    """해당 주소가 응답하면 사용 가능한 모델 id 목록을, 아니면 None을 반환."""
    try:
        r = requests.get(f"{base_url}/models", timeout=timeout, headers=_headers())
        if r.status_code != 200:
            return None
        return [m.get("id", "") for m in r.json().get("data", [])]
    except Exception:
        return None


def available(timeout: float = 1.5, force: bool = False) -> bool:
    """로컬 LLM 서버 사용 가능 여부. 결과는 _PROBE_TTL초 동안 캐시된다."""
    now = time.monotonic()
    if not force and (now - _probe["at"]) < _PROBE_TTL:
        return _probe["ok"]

    candidates = [config.LM_STUDIO_BASE_URL]
    candidates += [u for u in _FALLBACK_BASE_URLS if u != config.LM_STUDIO_BASE_URL]

    for url in candidates:
        models = _try_backend(url, timeout)
        if models is not None:
            _probe.update(at=now, ok=True, base_url=url, models=models)
            return True

    _probe.update(at=now, ok=False, base_url=None, models=[])
    return False


def active_base_url() -> str:
    """실제 사용 중인 백엔드 주소(자동 탐지 결과 반영)."""
    return _probe["base_url"] or config.LM_STUDIO_BASE_URL


def active_model() -> str:
    """실제 사용할 모델 id.

    설정된 모델이 서버에 없으면(백엔드가 자동 전환된 경우 등) 서버가 제공하는
    첫 모델로 대체한다. 임베딩 전용 모델은 대화에 쓸 수 없으므로 제외한다.
    """
    want = config.LM_STUDIO_MODEL
    models = _probe.get("models") or []
    if not models or want in models:
        return want
    usable = [m for m in models if "embed" not in m.lower()]
    return usable[0] if usable else want


def chat(messages: list[dict], temperature: float = 0.3, max_tokens: int = 512,
         timeout: float = 60, retries: int = 1) -> str:
    """chat completion 요청. 실패 시 retries회까지 재시도한다."""
    if not _probe["ok"]:
        available()                      # 백엔드 주소·모델 확정

    payload = {
        "model": active_model(),
        "messages": messages,
        "temperature": temperature,
        "max_tokens": max_tokens,
        "stream": False,
    }

    last_err: Exception | None = None
    for attempt in range(retries + 1):
        try:
            r = requests.post(f"{active_base_url()}/chat/completions",
                              json=payload, timeout=timeout, headers=_headers())
            r.raise_for_status()
            return r.json()["choices"][0]["message"]["content"].strip()
        except Exception as e:
            last_err = e
            if attempt < retries:
                time.sleep(0.5)

    _probe["at"] = 0.0                   # 다음 호출에서 백엔드 재탐지
    raise last_err
