import os, sys
from pathlib import Path
from fastapi.testclient import TestClient

def make_client(tmpdir: Path):
    os.environ["DATABASE_URL"] = f"sqlite:///{tmpdir}/nolook_test.db"
    os.environ["NOLOOK_DISABLE_OPENAI"] = "1"
    os.environ["NOLOOK_RATE_LIMIT_PER_MIN"] = "0"
    os.environ["ALLOWED_ORIGINS"] = "http://localhost:3000"
    if "genai.main" in sys.modules:
        del sys.modules["genai.main"]
    import genai.main as main
    main.API_KEY = None
    client = TestClient(main.app)
    return main, client

def test_ask_sad(tmp_path):
    m, client = make_client(tmp_path)
    r = client.post("/ask", json={"prompt": "今日は萎えたー"})
    assert r.status_code == 200
    js = r.json()
    assert js["emotion"] == "悲しい"
    assert list(js["labels"].keys()) == m.EMOTION_KEYS

def test_ask_anger(tmp_path):
    m, client = make_client(tmp_path)
    r = client.post("/ask", json={"prompt": "俺のミスじゃないのに責められてイライラする"})
    assert r.status_code == 200
    assert r.json()["emotion"] == "怒り"

def test_long_input_ok(tmp_path):
    m, client = make_client(tmp_path)
    long_text = "あ" * 5000
    r = client.post("/ask", json={"prompt": long_text})
    assert r.status_code == 200

def test_analyze_and_summary(tmp_path):
    m, client = make_client(tmp_path)
    client.post("/analyze", json={"prompt": "部活で最悪…落ち込んだ"})
    client.post("/analyze", json={"prompt": "部活でムカついた"})
    r = client.get("/summary", params={"days": 1, "tz": "Asia/Tokyo"})
    assert r.status_code == 200
    js = r.json()
    assert js["days"] == 1
    day = js["daily"][0]["counts"]
    assert day["悲しい"] >= 1
    assert day["怒り"] >= 1
