"""Thin client for free-tier LLM APIs: Google Gemini, Groq and OpenRouter (set at least one key in .env)."""
from __future__ import annotations

import json
import os
import re


class LLMUnavailable(RuntimeError):
    """No API key configured."""


class LLMError(RuntimeError):
    """Provider returned an error."""


KEYS = {"gemini": "GEMINI_API_KEY", "groq": "GROQ_API_KEY", "openrouter": "OPENROUTER_API_KEY"}
DEFAULT_MODELS = {
    "gemini": ("GEMINI_MODEL", "gemini-2.5-flash"),
    "groq": ("GROQ_MODEL", "llama-3.3-70b-versatile"),
    "openrouter": ("OPENROUTER_MODEL", "meta-llama/llama-3.3-70b-instruct:free"),
}
OPENAI_STYLE = {
    "groq": "https://api.groq.com/openai/v1/chat/completions",
    "openrouter": "https://openrouter.ai/api/v1/chat/completions",
}


def configured() -> list[str]:
    preferred = os.getenv("LLM_PROVIDER", "").lower()
    names = [p for p, k in KEYS.items() if os.getenv(k)]
    if preferred in names:
        names.remove(preferred)
        names.insert(0, preferred)
    return names


def active_provider() -> str | None:
    names = configured()
    return names[0] if names else None


def _model(provider: str) -> str:
    env, default = DEFAULT_MODELS[provider]
    return os.getenv(env, default)


def _call(provider: str, prompt: str, system: str, temperature: float, max_tokens: int, json_mode: bool) -> str:
    import httpx  # lazy import: the analytics code and tests work without it

    key = os.environ[KEYS[provider]]
    if provider == "gemini":
        url = f"https://generativelanguage.googleapis.com/v1beta/models/{_model(provider)}:generateContent"
        cfg = {"temperature": temperature, "maxOutputTokens": max_tokens}
        if json_mode:
            cfg["responseMimeType"] = "application/json"
        body = {"contents": [{"role": "user", "parts": [{"text": prompt}]}], "generationConfig": cfg}
        if system:
            body["systemInstruction"] = {"parts": [{"text": system}]}
        r = httpx.post(url, headers={"x-goog-api-key": key}, json=body, timeout=50)
        r.raise_for_status()
        parts = r.json()["candidates"][0]["content"]["parts"]
        return "".join(p.get("text", "") for p in parts).strip()

    messages = ([{"role": "system", "content": system}] if system else []) + [{"role": "user", "content": prompt}]
    body = {"model": _model(provider), "messages": messages, "temperature": temperature, "max_tokens": max_tokens}
    if json_mode and provider == "groq":
        body["response_format"] = {"type": "json_object"}
    r = httpx.post(OPENAI_STYLE[provider], headers={"Authorization": f"Bearer {key}"}, json=body, timeout=50)
    r.raise_for_status()
    return r.json()["choices"][0]["message"]["content"].strip()


def complete(prompt: str, system: str = "", temperature: float = 0.3, max_tokens: int = 2048, json_mode: bool = False) -> str:
    providers = configured()
    if not providers:
        raise LLMUnavailable(
            "No LLM API key configured. Add GEMINI_API_KEY, GROQ_API_KEY or OPENROUTER_API_KEY "
            "(see .env.example, or Vercel > Settings > Environment Variables)."
        )
    last: Exception | None = None
    for p in providers:
        try:
            return _call(p, prompt, system, temperature, max_tokens, json_mode)
        except Exception as exc:  # noqa: BLE001 - fall back to the next provider
            last = exc
    raise LLMError(f"LLM request failed: {last}")


def extract_json(text: str):
    """Parse a JSON object out of a model reply (handles ```json fences and stray prose)."""
    text = re.sub(r"^```(?:json)?|```$", "", text.strip(), flags=re.I | re.M).strip()
    start, end = text.find("{"), text.rfind("}")
    if start == -1 or end <= start:
        raise ValueError("Model reply did not contain JSON.")
    return json.loads(text[start : end + 1])
