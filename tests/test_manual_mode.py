# tests/test_manual_mode.py
from fastapi.testclient import TestClient
from genai import main

EMOTION_KEYS = main.EMOTION_KEYS


def make_client():
    # ---- 依存を完全に無効化（401/429を回避）----
    def _dummy_verify():
        return True

    def _dummy_rate(_req=None):
        return True

    main.app.dependency_overrides = {}
    main.app.dependency_overrides[main.verify_api_key] = _dummy_verify
    main.app.dependency_overrides[main.rate_limit] = _dummy_rate

    # ---- グローバルの認証/レート制限も無効化（保険）----
    main.API_KEY = None
    main.RATE_PER_MIN = 0

    # ---- LLM を完全停止（テスト高速化＆外部依存なし）----
    main.LANGCHAIN_AVAILABLE = False
    main.classification_chain = None
    main.reply_chain = None
    # /ask で使う会話チェーンもダミーに
    class _DummyConvo:
        def run(self, input: str) -> str:
            emo, _, _ = main.classify_emotion(input)
            return main.PHRASE_LIB.get(emo, main.PHRASE_LIB["中立"])[0]
    main.convo_chain = _DummyConvo()

    return TestClient(main.app)


def test_aliases_are_normalized():
    c = make_client()
    for alias in ["嬉しい", "楽しみ", "ok", "ふつう"]:
        r = c.post("/analyze", json={"prompt": "テスト", "selected_emotion": alias})
        assert r.status_code == 200, (alias, r.text)
        js = r.json()
        assert set(js["labels"].keys()) == set(EMOTION_KEYS)


def test_manual_only_requires_selected(monkeypatch):
    # env だけでは main.MANUAL_ONLY には反映されないので直書きも必須
    monkeypatch.setenv("NOLOOK_MANUAL_ONLY", "1")
    main.MANUAL_ONLY = True

    c = make_client()
    r = c.post("/ask", json={"prompt": "テスト"})
    # FastAPI バリデーションの違いで 400/422 どちらにもなり得る
    assert r.status_code in (400, 422), r.text
    detail = r.json().get("detail")
    detail_str = detail if isinstance(detail, str) else str(detail)
    assert "selected_emotion" in detail_str


def test_accepts_exact_six_keys():
    # 手動モードでも6キーはそのまま受理されること
    main.MANUAL_ONLY = True

    c = make_client()
    for key in ["楽しい", "悲しい", "怒り", "不安", "しんどい", "中立"]:
        r = c.post("/ask", json={"prompt": "ok", "selected_emotion": key})
        assert r.status_code == 200, (key, r.text)
