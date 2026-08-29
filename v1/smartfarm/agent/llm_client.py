"""LM Studio (OpenAI 호환 로컬 LLM) 연동 클라이언트.

LM Studio가 실행 중이 아니면 available()가 False를 반환하며,
호출부는 템플릿 기반 자연어 응답으로 폴백한다.
"""
from __future__ import annotations

import requests

from .. import config


def available(timeout: float = 1.5) -> bool:
    """LM Studio 서버가 응답 가능한지 확인."""
    try:
        r = requests.get(f"{config.LM_STUDIO_BASE_URL}/models", timeout=timeout,
                         headers={"Authorization": f"Bearer {config.LM_STUDIO_API_KEY}"})
        return r.status_code == 200
    except Exception:
        return False


def chat(messages: list[dict], temperature: float = 0.3,
         max_tokens: int = 512, timeout: float = 60) -> str:
    """LM Studio에 chat completion 요청."""
    payload = {
        "model": config.LM_STUDIO_MODEL,
        "messages": messages,
        "temperature": temperature,
        "max_tokens": max_tokens,
        "stream": False,
    }
    r = requests.post(
        f"{config.LM_STUDIO_BASE_URL}/chat/completions",
        json=payload, timeout=timeout,
        headers={"Authorization": f"Bearer {config.LM_STUDIO_API_KEY}"},
    )
    r.raise_for_status()
    return r.json()["choices"][0]["message"]["content"].strip()
