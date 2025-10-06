# genai/main.py
from importlib import import_module
import os
import sys
import logging

logger = logging.getLogger(__name__)

# --- app.main を安全にインポート ---
try:
    _base = import_module("app.main")
    app = getattr(_base, "app")
except Exception as e:
    logger.error("Failed to import app.main: %s", e)
    from fastapi import FastAPI
    app = FastAPI()
    _base = None

# --- FastAPI の依存関数を安全に転送 ---
def _dummy_verify():
    return True

def _dummy_rate(_req=None):
    return True

verify_api_key = getattr(_base, "verify_api_key", _dummy_verify)
rate_limit = getattr(_base, "rate_limit", _dummy_rate)  # ← ★ここをrate_limitに統一！

# 🔹 モジュール属性として明示的に登録（テストが参照）
current_module = sys.modules[__name__]
setattr(current_module, "verify_api_key", verify_api_key)
setattr(current_module, "rate_limit", rate_limit)

# --- 定数類を転送 ---
EMOTION_KEYS = getattr(
    _base,
    "EMOTION_KEYS",
    ["楽しい", "悲しい", "怒り", "不安", "しんどい", "中立"],
)

ALIAS_MAP = getattr(
    _base,
    "ALIAS_MAP",
    {k: k for k in EMOTION_KEYS},
)

# --- OpenAI クライアント ---
try:
    from openai import OpenAI
    from openai import (
        OpenAIError,
        APIError,
        APIConnectionError,
        RateLimitError,
        BadRequestError,
        AuthenticationError,
    )
except Exception:
    OpenAI = None
    class DummyError(Exception): ...
    OpenAIError = APIError = APIConnectionError = RateLimitError = BadRequestError = AuthenticationError = DummyError

# --- LLM 分類ラッパ ---
def llm_classify(prompt: str) -> dict:
    """LLMが失敗してもAPI全体を落とさない安全ラッパ"""
    if OpenAI is None:
        return {"ok": False, "reason": "no_client"}
    key = os.getenv("OPENAI_API_KEY") or ""
    if not key:
        return {"ok": False, "reason": "no_api_key"}
    client = OpenAI(api_key=key)
    try:
        resp = client.chat.completions.create(
            model=os.getenv("OPENAI_MODEL", "gpt-4o-mini"),
            messages=[
                {"role": "system", "content": "You are a Japanese emotion classifier."},
                {"role": "user", "content": prompt},
            ],
            temperature=0.0,
        )
        if hasattr(_base, "parse_llm_output"):
            return _base.parse_llm_output(resp)
        return {"ok": True, "resp": resp}
    except (
        RateLimitError,
        APIConnectionError,
        BadRequestError,
        AuthenticationError,
        APIError,
        OpenAIError,
    ) as e:
        logger.warning("LLM fallback(OpenAI): %s: %s", type(e).__name__, e)
        return {"ok": False, "reason": "llm_error"}
    except Exception as e:
        logger.exception("LLM unexpected error: %s", e)
        return {"ok": False, "reason": "llm_error"}

__all__ = [
    "app",
    "verify_api_key",
    "rate_limit",
    "EMOTION_KEYS",
    "ALIAS_MAP",
    "llm_classify",
]
