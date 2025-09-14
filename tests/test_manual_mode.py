# tests/test_manual_mode.py
import os
from fastapi.testclient import TestClient
from genai.main import app, EMOTION_KEYS

def make_client():
    # 認証をオフにして実行（API_KEYが設定されていても無視）
    os.environ.pop("API_KEY", None)
    return TestClient(app)

def test_aliases_are_normalized():
    c = make_client()
    for alias in ["嬉しい", "楽しみ", "ok", "ふつう"]:
        r = c.post("/analyze", json={"prompt": "テスト", "selected_emotion": alias})
        assert r.status_code == 200, (alias, r.text)
        js = r.json()
        assert set(js["labels"].keys()) == set(EMOTION_KEYS)

def test_manual_only_requires_selected(monkeypatch):
    monkeypatch.setenv("NOLOOK_MANUAL_ONLY", "1")
    c = make_client()
    r = c.post("/ask", json={"prompt": "テスト"})
    assert r.status_code == 400
    assert "selected_emotion" in r.json().get("detail", "")

def test_accepts_exact_six_keys():
    c = make_client()
    for key in ["楽しい", "悲しい", "怒り", "不安", "しんどい", "中立"]:
        r = c.post("/ask", json={"prompt": "ok", "selected_emotion": key})
        assert r.status_code == 200, (key, r.text)
