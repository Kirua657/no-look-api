# NO LOOK API

生徒の入力から **やさしい一言返信** と **感情の数値化（統計）** を行う FastAPI バックエンド。
本文は保存せず、**統計のみ**をDBに保存します。

## エンドポイント
- POST /ask      … reply / emotion / score / labels を返す
- POST /analyze  … labels / signals（本文とreplyは返さない）
- GET  /summary  … 直近N日の感情件数（空白日は0埋め）

## セットアップ（ローカル）
py -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -U pip
python -m pip install -r requirements.txt
copy .env.example .env
python -m uvicorn genai.main:app --reload   # http://127.0.0.1:8000/docs

## 環境変数
OPENAI_API_KEY=sk-...
DATABASE_URL=sqlite:///C:/Users/Kirua/Documents/NoLook/nolook_dev.db

## 方針
- 本文は保存しない（統計のみ保存）
- 感情ラベルのキーは固定: ["楽しい","悲しい","怒り","不安","しんどい","中立"]
