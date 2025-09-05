"""
NO LOOK API (FastAPI)
- Endpoints: / (health), /ask, /analyze, /summary, /weekly_report
- DB: SQLAlchemy (SQLite by default)
- LLM: LangChain+OpenAI if available (hybrid classifier). 無効時は辞書ルールのみ。
- Security: API Key, simple rate limit, request logging

Run:
    python genai/main.py
or:
    uvicorn genai.main:app --host 0.0.0.0 --port 8000

Env:
    OPENAI_API_KEY=sk-...
    NOLOOK_DISABLE_OPENAI=1
    NOLOOK_LLM_WEIGHT=0.7
    API_KEY=devkey-123
    NOLOOK_RATE_LIMIT_PER_MIN=60
    ALLOWED_ORIGINS=http://localhost:3000,http://127.0.0.1:3000
"""
from __future__ import annotations

import json
import os
import re
import time
import logging
import uuid
from typing import Dict, List, Optional, Literal
from collections import defaultdict
from datetime import datetime, timedelta

from fastapi import FastAPI, HTTPException, Depends, Request, Security
from fastapi.security import APIKeyHeader
from fastapi.middleware.cors import CORSMiddleware
from fastapi.exceptions import RequestValidationError
from starlette.responses import JSONResponse
from starlette import status
from pydantic import BaseModel
from dotenv import load_dotenv

# ===== Env =====
load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
DISABLE_OPENAI = os.getenv("NOLOOK_DISABLE_OPENAI", "0") == "1"
LLM_WEIGHT = float(os.getenv("NOLOOK_LLM_WEIGHT", "0.7"))  # 0..1
API_KEY = os.getenv("API_KEY")
RATE_PER_MIN = int(os.getenv("NOLOOK_RATE_LIMIT_PER_MIN", "0") or 0)
ALLOWED_ORIGINS = [o.strip() for o in os.getenv(
    "ALLOWED_ORIGINS",
    "http://localhost:3000,http://127.0.0.1:3000"
).split(",") if o.strip()]

# ===== Optional LangChain / OpenAI (フォールバック対応) =====
LANGCHAIN_AVAILABLE = False
try:
    if not DISABLE_OPENAI and OPENAI_API_KEY:
        from langchain.prompts import PromptTemplate
        from langchain.chains import LLMChain
        from langchain_openai import ChatOpenAI
        LANGCHAIN_AVAILABLE = True
except Exception:
    LANGCHAIN_AVAILABLE = False

# ===== DB: SQLAlchemy =====
from sqlalchemy import (
    create_engine, Column, Integer, String, Float, Boolean, DateTime, JSON, func, select, and_
)
from sqlalchemy.orm import declarative_base, sessionmaker

DATABASE_URL = os.getenv("DATABASE_URL", "sqlite:///./nolook_dev.db")
connect_args = {"check_same_thread": False} if DATABASE_URL.startswith("sqlite") else {}
engine = create_engine(DATABASE_URL, connect_args=connect_args, future=True)
SessionLocal = sessionmaker(bind=engine, autoflush=False, autocommit=False, future=True)
Base = declarative_base()

# 固定キー（グラフ用に毎回そろえる）
EMOTION_KEYS = ["楽しい", "悲しい", "怒り", "不安", "しんどい", "中立"]

class StatsRecord(Base):
    __tablename__ = "stats_records"
    id = Column(Integer, primary_key=True, autoincrement=True)
    created_at = Column(DateTime(timezone=True), server_default=func.now(), nullable=False)
    class_id = Column(String, nullable=True)
    emotion = Column(String, nullable=False)
    score = Column(Float, nullable=False)
    labels = Column(JSON, nullable=False)            # 例: {"不安":0.5, ...}
    topic_tags = Column(JSON, nullable=False)        # 例: ["友だち","勉強"]
    relationship_mention = Column(Boolean, nullable=False)
    negation_index = Column(Float, nullable=False)
    avoidance = Column(Float, nullable=False)

Base.metadata.create_all(bind=engine)

# --- index ensure ---
with engine.begin() as conn:
    try:
        conn.exec_driver_sql("CREATE INDEX IF NOT EXISTS idx_stats_created_at ON stats_records (created_at)")
        conn.exec_driver_sql("CREATE INDEX IF NOT EXISTS idx_stats_emotion ON stats_records (emotion)")
        conn.exec_driver_sql("CREATE INDEX IF NOT EXISTS idx_stats_class_id ON stats_records (class_id)")
    except Exception:
        pass

# ===== Logging & Request ID =====
logger = logging.getLogger("nolook")
if not logger.handlers:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

# ===== Utils =====
def _full_labels(labels: dict) -> dict:
    """毎回同じキーを出す（欠けは0で埋める）"""
    return {k: float(labels.get(k, 0.0)) for k in EMOTION_KEYS}

def _normalize(labels: Dict[str, float]) -> Dict[str, float]:
    total = float(sum(max(0.0, v) for v in labels.values()))
    if total <= 0.0:
        return {k: (1.0 if k == "中立" else 0.0) for k in EMOTION_KEYS}
    return {k: round(max(0.0, labels.get(k, 0.0)) / total, 4) for k in EMOTION_KEYS}

# ====== 感情辞書ベース分類 ======
EMOTION_LEXICON: Dict[str, List[str]] = {
    "楽しい":  [
        "楽しい", "嬉しい", "うれしい", "わくわく", "最高", "よかった",
        "やった", "やったー", "やったぁ", "嬉しすぎ", "うれしすぎ", "最高すぎ",
        "自己ベスト", "自己ベスト更新", "過去最高", "記録更新", "満点", "満点取れた",
        "合格", "受かった", "当選", "優勝", "入賞", "勝てた", "勝った",
        "一位", "1位", "MVP", "表彰", "褒められた", "認められた",
        "成功", "達成", "できた", "できるようになった", "うまくいった",
        "楽しかった", "楽しすぎ", "楽しすぎた"
    ],
    "悲しい":  [
        "悲しい", "さみしい", "辛い", "つらい", "落ち込", "泣きたい", "憂鬱", "しょんぼり",
        "萎え", "萎えた", "萎える", "萎えたー", "萎えー", "萎えぇ", "へこむ", "凹む", "凹んだ"
    ],
    "怒り":    ["怒", "ムカつ", "むかつ", "腹立", "イライラ", "キレた"],
    "不安":    ["不安", "心配", "こわい", "怖い", "緊張", "ドキドキ", "心細い"],
    "しんどい": ["疲れ", "疲れた", "だる", "しんど", "眠い", "眠たい", "疲労"],
}

# Joyの文脈パターン（語彙が無くてもポジティブ達成/勝利/評価で加点）
JOY_CONTEXT_PATTERNS = [
    r"(テスト|模試|試験|返却|成績).*(最高|過去最高|自己ベスト|満点|良|高|上が|更新|とれた|取れた|できた)",
    r"(点数|スコア).*(最高|過去最高|自己ベスト|満点|高|上が|更新)",
    r"(合格|受かった|認定|合格発表|結果).*(出た|きた|勝|合格)",
    r"(優勝|入賞|MVP|表彰|メダル|トロフィー)",
    r"(試合|大会|コンテスト|コンクール).*(勝(った|てた)|優勝|入賞)",
    r"(うまくい(っ|た)|成功|達成|完走|やり切(っ|った)|できたー?)",
    r"(文化祭|体育祭|発表|合唱|演奏|ダンス).*(成功|盛り上が|楽しかった|よかった)",
    r"(チケット|抽選).*(当たった|当選)",
    r"(推し|ライブ|イベント).*(行けた|行ける|最高|神)",
]
NEG_NEAR_PATTERN = re.compile(r"(じゃない|じゃなかった|なくはない|ない|なかった)")

def _apply_context_boosts(text: str, counts: Dict[str, int]) -> None:
    t = text or ""
    if NEG_NEAR_PATTERN.search(t):
        return
    if any(re.search(p, t) for p in JOY_CONTEXT_PATTERNS):
        counts["楽しい"] += 2

def _lexicon_dist(text: str) -> Dict[str, float]:
    t = text or ""
    counts = defaultdict(int)
    for emo, kws in EMOTION_LEXICON.items():
        for kw in kws:
            counts[emo] += len(re.findall(re.escape(kw), t))
    _apply_context_boosts(t, counts)
    if sum(counts.values()) == 0:
        return {"中立": 1.0}
    labels = {emo: float(c) for emo, c in counts.items()}
    return _normalize(_full_labels(labels))

# ===== LLMベース分類（JSON厳格出力） =====
classification_chain = None
if LANGCHAIN_AVAILABLE:
    from langchain.prompts import PromptTemplate
    from langchain.chains import LLMChain
    from langchain_openai import ChatOpenAI

    _cls_prompt = PromptTemplate(
        input_variables=["text"],
        template=(
            """
あなたは中高生の日本語日記の感情分類器です。文脈に応じて6感情の分布を出してください。
感情キー: ["楽しい","悲しい","怒り","不安","しんどい","中立"]
出力要件: JSONのみで返す。例:
{"labels":{"楽しい":0.0,"悲しい":0.9,"怒り":0.05,"不安":0.05,"しんどい":0.0,"中立":0.0},"primary":"悲しい"}
- 値は0..1でおよそ合計1
- スラングや直喩は文脈で解釈（例: 萎えた→多くは悲しい）
- 例1: "最悪…ミスって落ち込んだ" → 悲しい優勢
- 例2: "最悪だ、あの審判ふざけてる" → 怒り優勢
- 例3: "ヤバいくらい眠い" → しんどい優勢
- 例4: "今日は萎えたー" → 悲しい優勢
テキスト: {text}
            """
        ),
    )
    classification_chain = LLMChain(
        llm=ChatOpenAI(model="gpt-4o-mini", temperature=0.2, openai_api_key=OPENAI_API_KEY),
        prompt=_cls_prompt,
        verbose=False,
    )

def _llm_dist(text: str) -> Optional[Dict[str, float]]:
    if not classification_chain:
        return None
    try:
        raw = classification_chain.run(text=text)
        data = json.loads(raw)
        labels = data.get("labels", {})
        labels = _normalize(_full_labels(labels))
        return labels
    except Exception:
        return None

# ===== ハイブリッド分類 =====
def classify_emotion(text: str):
    lex = _lexicon_dist(text)
    llm = _llm_dist(text) if (LANGCHAIN_AVAILABLE and classification_chain) else None
    if llm:
        combined = {k: (1.0 - LLM_WEIGHT) * lex.get(k, 0.0) + LLM_WEIGHT * llm.get(k, 0.0) for k in EMOTION_KEYS}
        labels = _normalize(combined)
    else:
        labels = _normalize(lex)
    top_emotion = max(labels.items(), key=lambda x: x[1])[0]
    top_score = float(labels[top_emotion])
    return top_emotion, top_score, labels

# ===== Signals =====
class Signals(BaseModel):
    topic_tags: List[str]
    relationship_mention: bool
    negation_index: float
    avoidance: float

TOPIC_LEXICON: Dict[str, List[str]] = {
    "友だち": ["友だち", "友達", "ともだち", "クラスメイト", "いじめ", "仲間", "先輩", "後輩"],
    "勉強": ["勉強", "テスト", "宿題", "成績", "授業", "課題", "受験"],
    "家庭": ["家", "家族", "親", "父", "母", "兄", "姉", "弟", "妹"],
    "部活": ["部活", "クラブ", "サークル", "試合", "大会", "練習"],
    "体調": ["体調", "熱", "風邪", "腹痛", "頭痛", "眠い", "疲れ", "しんどい"],
}
RELATIONSHIP_WORDS = ["友だち", "友達", "ともだち", "いじめ", "無視", "悪口", "仲間はずれ", "ぼっち"]
NEGATION_WORDS = ["ない", "できない", "無理", "嫌い", "いやだ", "ダメ", "もうやだ"]
AVOIDANCE_WORDS = ["別に", "なんでもない", "知らない", "まあいい", "どうでもいい"]

def compute_signals(text: str) -> Signals:
    t = text or ""
    tags: List[str] = []
    for tag, kws in TOPIC_LEXICON.items():
        if any(kw in t for kw in kws):
            tags.append(tag)
    rel = any(w in t for w in RELATIONSHIP_WORDS)

    def score_by_terms(terms: List[str]) -> float:
        hits = sum(t.count(w) for w in terms)
        return float(min(1.0, round(hits / 10.0, 2)))

    return Signals(
        topic_tags=tags,
        relationship_mention=rel,
        negation_index=score_by_terms(NEGATION_WORDS),
        avoidance=score_by_terms(AVOIDANCE_WORDS),
    )

# ===== Reply generation (LLM強化 + ルールFallback) =====
SUGGESTION_LIB: Dict[str, List[str]] = {
    "楽しい": [
        "今日のよかった点を3つメモしよう",
        "支えてくれた人に一言伝えよう",
        "次も続けるコツを1つだけ決めよう",
        "学びを1行で日記に残そう",
    ],
    "悲しい": [
        "3分だけ深呼吸して落ち着こう",
        "話せる相手に一行メッセ送ろう",
        "今日の救いを1つ探して書こう",
    ],
    "怒り": [
        "4-4-4呼吸で一息つこう",
        "事実・気持ち・要望を一文で書いてみよう",
        "距離を取れるなら少し離れよう",
    ],
    "不安": [
        "やることを1個だけに絞ろう",
        "5分だけ着手してみよう",
        "不安を書き出し対処を1つ決めよう",
    ],
    "しんどい": [
        "水を飲んで5分休もう",
        "今日は早寝アラームを設定しよう",
        "無理しない宣言を自分にしよう",
    ],
    "中立": [
        "今日のハイライトを1行で",
        "次に試したいことを1つ書こう",
    ],
}
TOPIC_TWEAKS: Dict[str, Dict[str, List[str]]] = {
    "勉強": {
        "楽しい": ["自己ベストの理由を1つメモしよう"],
        "不安": ["問題を1題だけ解いてみよう"],
    },
    "部活": {
        "楽しい": ["よかったプレーを1つ言語化しよう"],
        "怒り": ["次の練習で試す案を1つ書こう"],
    },
}
def _pick_step(emotion: str, sig: Signals) -> str:
    steps = list(SUGGESTION_LIB.get(emotion, SUGGESTION_LIB["中立"]))
    for tag in sig.topic_tags:
        for add in TOPIC_TWEAKS.get(tag, {}).get(emotion, []):
            steps.append(add)
    return steps[0] if steps else "次の一歩を1つだけ決めよう"

# LLMで文章を整えるチェーン（返信トーン強化）
reply_chain = None
if LANGCHAIN_AVAILABLE:
    from langchain.prompts import PromptTemplate as _PT
    from langchain.chains import LLMChain as _LC
    from langchain_openai import ChatOpenAI as _C

    _reply_prompt = _PT(
        input_variables=["text", "emotion", "step", "style", "length", "topics", "tone"],
        template="""
日本語で短い返答文を作成します。常に高い共感と温かいトーン。
順序: ①共感（文脈の鏡映）→ ②称賛/慰め（{emotion} に合う）→ ③次の一歩: {step} → ④一言の応援。
- 口調: {style}（buddy=フレンドリー / coach=前向き指導 / teacher=落ち着いた先生）
- 長さ: {length}（short=1文, medium=最大2文, 各120字以内）
- topics: {topics} をさりげなく反映。
- {tone}
- 末尾に自然な励まし語（例: 応援してるよ / 一緒に進もう）を入れる。
出力はテキストのみ。
ユーザー文: {text}
""",
    )
    reply_chain = _LC(
        llm=_C(model="gpt-4o-mini", temperature=0.4, openai_api_key=OPENAI_API_KEY),
        prompt=_reply_prompt,
        verbose=False,
    )

def _joy_big_achievement(text: str) -> bool:
    return any(re.search(p, text or "") for p in JOY_CONTEXT_PATTERNS)

def build_reply(
    text: str,
    emotion: str,
    sig: Signals,
    *,
    style: str = "buddy",
    length: str = "medium",
    followup: bool = False
) -> str:
    step = _pick_step(emotion, sig)

    # LLM整形（あれば）
    if reply_chain is not None:
        try:
            tone = {
                "楽しい": "努力や過程を具体的に称賛し、成果に喜びを共有する",
                "悲しい": "痛みを否定せず寄り添い、安心感を与える",
                "怒り":   "怒りの妥当性を認め、安全な対処へ導く",
                "不安":   "不安は自然だと伝え、最小の一歩に集中させる",
                "しんどい": "ねぎらいと休息の許可を出す",
                "中立":   "受容し、前向きな振り返りへ",
            }.get(emotion, "自然で温かい励まし")
            base = reply_chain.run(
                text=text,
                emotion=emotion,
                step=step,
                style=style,
                length=length,
                topics=",".join(sig.topic_tags),
                tone=tone,
            )
            if followup:
                q = {
                    "楽しい": "次に伸ばしたい所はどこ？",
                    "悲しい": "今、心が少し楽になる一歩は？",
                    "怒り":   "落ち着いたら最初に伝えたいことは？",
                    "不安":   "まず一歩だけ選ぶなら？",
                    "しんどい": "今すぐできる休みはどれ？",
                    "中立":   "今日のハイライトは？",
                }.get(emotion, "次は何を試そう？")
                return f"{base} — {q}"
            return base
        except Exception:
            pass  # 失敗時は下のフォールバックへ

    # フォールバック（OpenAI無しでも温かい文に）
    if emotion == "楽しい" and _joy_big_achievement(text):
        opening = "過去最高や自己ベスト、ほんとうにすごいね。努力が実ったね！"
    else:
        opening = {
            "楽しい": "それは嬉しいね。よく頑張ったね！",
            "悲しい": "その気持ち、つらかったね。ここまで話してくれてえらいよ。",
            "怒り":   "それは腹が立つよね。そう感じるのは自然だよ。",
            "不安":   "不安になるの、すごく分かるよ。",
            "しんどい": "よくここまで頑張ったね。まずは自分をいたわろう。",
            "中立":   "話してくれてありがとう。",
        }.get(emotion, "話してくれてありがとう。")
    encouragement = {
        "楽しい": "この調子でいこう。応援してるよ。",
        "悲しい": "少しずつで大丈夫。いっしょに進もう。",
        "怒り":   "安全第一で、あなたの味方だよ。",
        "不安":   "できた分だけで十分。応援してるよ。",
        "しんどい": "今日は休んでOK。味方だよ。",
        "中立":   "次の一歩、一緒に考えよう。",
    }
    body = f"{opening} {step}。{encouragement.get(emotion, '')}"
    if followup:
        tail = {
            "楽しい": "次に伸ばしたい所はどこ？",
            "悲しい": "今できる小さな助けは何かな？",
            "怒り":   "落ち着いたら何から伝える？",
            "不安":   "まず一歩だけ選ぶなら？",
            "しんどい": "今すぐ休める？",
            "中立":   "今日のハイライトは？",
        }.get(emotion, "次は何を試そう？")
        body += f" — {tail}"
    return body

# ===== Persistence helper =====
class _SignalsLike:
    topic_tags: List[str]
    relationship_mention: bool
    negation_index: float
    avoidance: float

def save_stats(*, labels: dict, signals: _SignalsLike, top_emotion: str, score: float, class_id: Optional[str] = None) -> int:
    rec = StatsRecord(
        class_id=class_id,
        emotion=top_emotion,
        score=score,
        labels=_full_labels(labels),
        topic_tags=list(getattr(signals, "topic_tags", [])),
        relationship_mention=bool(getattr(signals, "relationship_mention", False)),
        negation_index=float(getattr(signals, "negation_index", 0.0)),
        avoidance=float(getattr(signals, "avoidance", 0.0)),
    )
    with SessionLocal() as db:
        db.add(rec)
        db.commit()
        db.refresh(rec)
        return rec.id

# ===== Security: API Key (optional) + Rate Limit (optional) =====
api_key_header = APIKeyHeader(name="X-API-Key", auto_error=False)

async def verify_api_key(api_key: Optional[str] = Security(api_key_header)):
    if not API_KEY:
        return True  # 未設定なら認証なし
    if not api_key:
        raise HTTPException(status_code=401, detail="Missing API key")
    if api_key != API_KEY:
        raise HTTPException(status_code=403, detail="Invalid API key")
    return True

RATE_STORE: Dict[str, tuple] = {}  # key -> (window_start_epoch, count)

async def rate_limit(req: Request):
    if RATE_PER_MIN <= 0:
        return True  # 無効
    key = f"{req.client.host}:{req.url.path}"
    now = int(time.time())
    window = now // 60
    cur = RATE_STORE.get(key)
    if not cur or cur[0] != window:
        RATE_STORE[key] = (window, 1)
        return True
    cnt = cur[1] + 1
    RATE_STORE[key] = (window, cnt)
    if cnt > RATE_PER_MIN:
        raise HTTPException(status_code=429, detail="Rate limit exceeded")
    return True

# ===== FastAPI =====
app = FastAPI(title="NO LOOK API", version="0.4.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_credentials=True,
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["*"],
    max_age=3600,
)

# --- Request ID + logging middleware ---
@app.middleware("http")
async def add_request_id_and_log(request: Request, call_next):
    rid = str(uuid.uuid4())
    request.state.rid = rid
    start = time.time()
    try:
        response = await call_next(request)
    except Exception as e:
        logger.exception("rid=%s unhandled error: %s", rid, e)
        return JSONResponse(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            content={"error": {"type": "internal_error", "message": "Internal Server Error", "rid": rid}},
        )
    dur = int((time.time() - start) * 1000)
    response.headers["X-Request-ID"] = rid
    logger.info("rid=%s method=%s path=%s status=%s dur_ms=%s", rid, request.method, request.url.path, response.status_code, dur)
    return response

# --- Unified validation error (422) ---
@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request: Request, exc: RequestValidationError):
    rid = getattr(request.state, "rid", str(uuid.uuid4()))
    return JSONResponse(
        status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
        content={"error": {"type": "validation_error", "message": exc.errors(), "rid": rid}},
    )

# ===== Pydantic models =====
class AskRequest(BaseModel):
    prompt: str
    style: Optional[Literal["buddy", "coach", "teacher"]] = "buddy"
    length: Optional[Literal["short", "medium"]] = "medium"
    followup: Optional[bool] = False

class AskResponse(BaseModel):
    reply: str
    emotion: str
    score: float
    labels: Dict[str, float]

class AnalyzeRequest(BaseModel):
    prompt: str

class AnalyzeResponse(BaseModel):
    labels: Dict[str, float]
    signals: 'Signals'

class DayCounts(BaseModel):
    date: str
    counts: Dict[str, int]
    total: int

class SummaryResponse(BaseModel):
    days: int
    daily: List[DayCounts]
    totals: Dict[str, int]
    top_emotion: str

# ===== Routes =====
@app.get("/")
async def read_root(verify: bool = Depends(verify_api_key)):
    return {"message": "NO LOOK API running", "openai": LANGCHAIN_AVAILABLE, "llm_weight": LLM_WEIGHT}

@app.post("/ask", response_model=AskResponse)
async def ask_ai(request: AskRequest, _k=Depends(verify_api_key), _r=Depends(rate_limit)):
    text = (request.prompt or "").strip()
    if not text:
        raise HTTPException(status_code=400, detail="prompt が空です。")
    # 解析してから、文脈に合わせた返信を生成
    emotion, score, labels = classify_emotion(text)
    sig = compute_signals(text)
    reply = build_reply(
        text,
        emotion,
        sig,
        style=request.style or "buddy",
        length=request.length or "medium",
        followup=bool(request.followup),
    )
    return AskResponse(reply=reply, emotion=emotion, score=score, labels=labels)

@app.post("/analyze", response_model=AnalyzeResponse)
async def analyze(req: AnalyzeRequest, _k=Depends(verify_api_key), _r=Depends(rate_limit)):
    text = (req.prompt or "").strip()
    if not text:
        raise HTTPException(status_code=400, detail="prompt が空です。")
    top_emotion, score, labels = classify_emotion(text)
    sig = compute_signals(text)
    _ = save_stats(labels=labels, signals=sig, top_emotion=top_emotion, score=score, class_id=None)
    return AnalyzeResponse(labels=_full_labels(labels), signals=sig)

# ===== /summary（日別集計） =====
import zoneinfo
@app.get("/summary", response_model=SummaryResponse)
async def summary(
    days: int = 7,
    class_id: Optional[str] = None,
    tz: str = "Asia/Tokyo",
    include_empty_days: bool = True,
    _k: bool = Depends(verify_api_key),
    _r: bool = Depends(rate_limit),
):
    """直近days日を日別集計して返す。DBのcreated_atはUTC naiveとみなす。"""
    tzinfo = zoneinfo.ZoneInfo(tz)
    today_local = datetime.now(tzinfo).date()
    start_date_local = today_local - timedelta(days=days - 1)

    from datetime import timezone as _tz
    start_local_aware = datetime.combine(start_date_local, datetime.min.time(), tzinfo=tzinfo)
    start_dt = start_local_aware.astimezone(_tz.utc).replace(tzinfo=None)

    where_clause = [StatsRecord.created_at >= start_dt]
    if class_id:
        where_clause.append(StatsRecord.class_id == class_id)

    with SessionLocal() as db:
        rows = db.execute(
            select(StatsRecord.created_at, StatsRecord.emotion).where(and_(*where_clause))
        ).all()

    def to_local_date_str(dt: datetime) -> str:
        if dt.tzinfo is None:
            return dt.replace(tzinfo=_tz.utc).astimezone(tzinfo).date().isoformat()
        return dt.astimezone(tzinfo).date().isoformat()

    by_day: Dict[str, Dict[str, int]] = {}
    if include_empty_days:
        for i in range(days):
            d = (start_date_local + timedelta(days=i)).isoformat()
            by_day[d] = {k: 0 for k in EMOTION_KEYS}

    for created_at, emo in rows:
        d = to_local_date_str(created_at)
        if d < start_date_local.isoformat() or d > today_local.isoformat():
            continue
        if d not in by_day:
            by_day[d] = {k: 0 for k in EMOTION_KEYS}
        if emo not in by_day[d]:
            by_day[d][emo] = 0
        by_day[d][emo] += 1

    days_sorted = [
        (start_date_local + timedelta(days=i)).isoformat() for i in range(days)
    ] if include_empty_days else sorted(by_day.keys())

    daily: List[DayCounts] = []
    totals = {k: 0 for k in EMOTION_KEYS}
    for d in days_sorted:
        counts = {k: int(by_day.get(d, {}).get(k, 0)) for k in EMOTION_KEYS}
        total = sum(counts.values())
        daily.append(DayCounts(date=d, counts=counts, total=total))
        for k, v in counts.items():
            totals[k] += v

    top_emotion = "中立" if sum(totals.values()) == 0 else max(
        EMOTION_KEYS, key=lambda k: (totals.get(k, 0), -EMOTION_KEYS.index(k))
    )

    return SummaryResponse(days=len(daily), daily=daily, totals=totals, top_emotion=top_emotion)

# ===== /weekly_report（週次レポート） =====
class WeeklyReportResponse(BaseModel):
    start_date: str
    end_date: str
    tz: str
    days: int
    daily: List[DayCounts]
    totals: Dict[str, int]
    top_emotion: str
    trend: Dict[str, List[str]]
    summary: str
    suggestions: List[str]

@app.get("/weekly_report", response_model=WeeklyReportResponse)
async def weekly_report(
    days: int = 7,
    class_id: Optional[str] = None,
    tz: str = "Asia/Tokyo",
    include_empty_days: bool = True,
    _k: bool = Depends(verify_api_key),
    _r: bool = Depends(rate_limit),
):
    tzinfo = zoneinfo.ZoneInfo(tz)

    today_local = datetime.now(tzinfo).date()
    start_date_local = today_local - timedelta(days=days - 1)

    from datetime import timezone as _tz
    start_local_aware = datetime.combine(start_date_local, datetime.min.time(), tzinfo=tzinfo)
    start_dt = start_local_aware.astimezone(_tz.utc).replace(tzinfo=None)

    where_clause = [StatsRecord.created_at >= start_dt]
    if class_id:
        where_clause.append(StatsRecord.class_id == class_id)

    with SessionLocal() as db:
        rows = db.execute(
            select(StatsRecord.created_at, StatsRecord.emotion).where(and_(*where_clause))
        ).all()

    def to_local_date_str(dt: datetime) -> str:
        if dt.tzinfo is None:
            return dt.replace(tzinfo=_tz.utc).astimezone(tzinfo).date().isoformat()
        return dt.astimezone(tzinfo).date().isoformat()

    by_day: Dict[str, Dict[str, int]] = {}
    if include_empty_days:
        for i in range(days):
            d = (start_date_local + timedelta(days=i)).isoformat()
            by_day[d] = {k: 0 for k in EMOTION_KEYS}

    for created_at, emo in rows:
        d = to_local_date_str(created_at)
        if d < start_date_local.isoformat() or d > today_local.isoformat():
            continue
        if d not in by_day:
            by_day[d] = {k: 0 for k in EMOTION_KEYS}
        if emo not in by_day[d]:
            by_day[d][emo] = 0
        by_day[d][emo] += 1

    days_sorted = [
        (start_date_local + timedelta(days=i)).isoformat() for i in range(days)
    ] if include_empty_days else sorted(by_day.keys())

    daily: List[DayCounts] = []
    totals = {k: 0 for k in EMOTION_KEYS}
    for d in days_sorted:
        counts = {k: int(by_day.get(d, {}).get(k, 0)) for k in EMOTION_KEYS}
        total = sum(counts.values())
        daily.append(DayCounts(date=d, counts=counts, total=total))
        for k, v in counts.items():
            totals[k] += v

    grand_total = sum(totals.values())
    top_emotion = "中立" if grand_total == 0 else max(
        EMOTION_KEYS, key=lambda k: (totals.get(k, 0), -EMOTION_KEYS.index(k))
    )

    split = max(1, days // 2)
    older_days = days_sorted[:split]
    recent_days = days_sorted[split:]

    def sum_range(day_keys: List[str]) -> Dict[str, int]:
        x = {k: 0 for k in EMOTION_KEYS}
        for dd in day_keys:
            for k in EMOTION_KEYS:
                x[k] += int(by_day.get(dd, {}).get(k, 0))
        return x

    older = sum_range(older_days)
    recent = sum_range(recent_days)

    rising = [k for k in EMOTION_KEYS if recent.get(k, 0) > older.get(k, 0)]
    falling = [k for k in EMOTION_KEYS if recent.get(k, 0) < older.get(k, 0)]

    def pct(n: int) -> float:
        return (n / grand_total) if grand_total else 0.0

    p_sad = pct(totals.get("悲しい", 0))
    p_fear = pct(totals.get("不安", 0))
    p_ang = pct(totals.get("怒り", 0))
    p_tired = pct(totals.get("しんどい", 0))

    notes: List[str] = []
    if grand_total > 0:
        peak_day = max(daily, key=lambda r: r.total).date if daily else start_date_local.isoformat()
        notes.append(f"ピーク日: {peak_day}")
        notes.append(f"最多感情: {top_emotion} ({totals.get(top_emotion, 0)}件)")
        if rising:
            notes.append("増加: " + ", ".join(rising))
        if falling:
            notes.append("減少: " + ", ".join(falling))

    suggestions: List[str] = []
    if p_sad + p_fear >= 0.6 and grand_total >= 5:
        suggestions.append("不安・悲しみが目立つ週。相談先と低負荷の個別声かけを案内")
    if p_ang >= 0.3:
        suggestions.append("怒りが目立つ。対人トラブル/ルール確認の機会を設定")
    if p_tired >= 0.3:
        suggestions.append("しんどいが多め。睡眠・休息の促しと提出負荷の調整")
    if not suggestions and grand_total > 0:
        suggestions.append("小さな成功体験の共有と、次週も継続的に観察")

    summary_text = (
        "今週の投稿は少数でした。次週も観察を継続しましょう。" if grand_total == 0 else
        f"合計{grand_total}件。最多は『{top_emotion}』。直近で{', '.join(rising) if rising else '大きな増減は'}見られません。"
    )

    return WeeklyReportResponse(
        start_date=start_date_local.isoformat(),
        end_date=today_local.isoformat(),
        tz=tz,
        days=len(days_sorted),
        daily=daily,
        totals=totals,
        top_emotion=top_emotion,
        trend={"rising": rising, "falling": falling},
        summary=summary_text,
        suggestions=notes + suggestions,
    )

# ===== Lightweight smoke tests =====
if __name__ == "__main__":
    if os.getenv("RUN_TESTS") == "1":
        from fastapi.testclient import TestClient
        orig_key = API_KEY
        API_KEY = None
        client = TestClient(app)
        r = client.post("/ask", json={"prompt": "今日は萎えたー"})
        assert r.status_code == 200, r.text
        assert r.json()["emotion"] == "悲しい", r.json()
        r2 = client.post("/analyze", json={"prompt": "今日はテストで不安"})
        assert r2.status_code == 200
        keys = list(r2.json()["labels"].keys())
        assert keys == EMOTION_KEYS, keys
        r3 = client.get("/summary", params={"days": 7})
        assert r3.status_code == 200
        API_KEY = "test"
        r4 = client.post("/ask", json={"prompt": "ok"})
        assert r4.status_code == 401
        print("All tests passed")
        API_KEY = orig_key

    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=int(os.getenv("PORT", "8000")))
"""
NO LOOK API (FastAPI)
- Endpoints: / (health), /ask, /analyze, /summary, /weekly_report
- DB: SQLAlchemy (SQLite by default)
- LLM: LangChain+OpenAI if available (hybrid classifier). 無効時は辞書ルールのみ。
- Security: API Key, simple rate limit, request logging
- Step A+B 反映:
  * CORS allowlist (env: ALLOWED_ORIGINS)
  * DB index auto-ensure (created_at/emotion/class_id)
  * Unified error handler (422/500) with request-id
  * /ask の返答を“会話っぽく・短く・共感強め”に最適化（style/length/followup オプション）
  * Joy語彙＆文脈ブーストを追加（自己ベスト/優勝/合格などを Joy に）

Run (おすすめ):
    python genai/main.py

Uvicorn 直起動:
    uvicorn genai.main:app --host 0.0.0.0 --port 8000

Env:
    OPENAI_API_KEY=sk-...
    NOLOOK_DISABLE_OPENAI=1           # OpenAIを明示OFF（任意）
    NOLOOK_LLM_WEIGHT=0.7             # LLMと辞書のハイブリッド重み（0..1, 既定0.7）
    API_KEY=devkey-123                # (任意) APIキー。未設定なら認証なし
    NOLOOK_RATE_LIMIT_PER_MIN=60      # (任意) 1分あたりの許容回数（IP×パス）
    ALLOWED_ORIGINS=http://localhost:3000,http://127.0.0.1:3000  # CORS許可
"""
from __future__ import annotations

import json
import os
import re
import time
import logging
import uuid
from typing import Dict, List, Optional, Literal
from collections import defaultdict
from datetime import datetime, timedelta

from fastapi import FastAPI, HTTPException, Depends, Request, Security
from fastapi.security import APIKeyHeader
from fastapi.middleware.cors import CORSMiddleware
from fastapi.exceptions import RequestValidationError
from starlette.responses import JSONResponse
from starlette import status
from pydantic import BaseModel
from dotenv import load_dotenv

# ===== Env =====
load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
DISABLE_OPENAI = os.getenv("NOLOOK_DISABLE_OPENAI", "0") == "1"
LLM_WEIGHT = float(os.getenv("NOLOOK_LLM_WEIGHT", "0.7"))  # 0..1
API_KEY = os.getenv("API_KEY")
RATE_PER_MIN = int(os.getenv("NOLOOK_RATE_LIMIT_PER_MIN", "0") or 0)
ALLOWED_ORIGINS = [o.strip() for o in os.getenv(
    "ALLOWED_ORIGINS",
    "http://localhost:3000,http://127.0.0.1:3000"
).split(",") if o.strip()]

# ===== Optional LangChain / OpenAI (フォールバック対応) =====
LANGCHAIN_AVAILABLE = False
try:
    if not DISABLE_OPENAI and OPENAI_API_KEY:
        from langchain.prompts import PromptTemplate
        from langchain.chains import LLMChain
        from langchain_openai import ChatOpenAI
        LANGCHAIN_AVAILABLE = True
except Exception:
    LANGCHAIN_AVAILABLE = False

# ===== DB: SQLAlchemy =====
from sqlalchemy import (
    create_engine, Column, Integer, String, Float, Boolean, DateTime, JSON, func, select, and_
)
from sqlalchemy.orm import declarative_base, sessionmaker

DATABASE_URL = os.getenv("DATABASE_URL", "sqlite:///./nolook_dev.db")
connect_args = {"check_same_thread": False} if DATABASE_URL.startswith("sqlite") else {}
engine = create_engine(DATABASE_URL, connect_args=connect_args, future=True)
SessionLocal = sessionmaker(bind=engine, autoflush=False, autocommit=False, future=True)
Base = declarative_base()

# 固定キー（グラフ用に毎回そろえる）
EMOTION_KEYS = ["楽しい", "悲しい", "怒り", "不安", "しんどい", "中立"]

class StatsRecord(Base):
    __tablename__ = "stats_records"
    id = Column(Integer, primary_key=True, autoincrement=True)
    created_at = Column(DateTime(timezone=True), server_default=func.now(), nullable=False)
    class_id = Column(String, nullable=True)
    emotion = Column(String, nullable=False)
    score = Column(Float, nullable=False)
    labels = Column(JSON, nullable=False)            # 例: {"不安":0.5, ...}
    topic_tags = Column(JSON, nullable=False)        # 例: ["友だち","勉強"]
    relationship_mention = Column(Boolean, nullable=False)
    negation_index = Column(Float, nullable=False)
    avoidance = Column(Float, nullable=False)

Base.metadata.create_all(bind=engine)

# --- ensure indexes (SQLite: IF NOT EXISTS) ---
with engine.begin() as conn:
    try:
        conn.exec_driver_sql("CREATE INDEX IF NOT EXISTS idx_stats_created_at ON stats_records (created_at)")
        conn.exec_driver_sql("CREATE INDEX IF NOT EXISTS idx_stats_emotion ON stats_records (emotion)")
        conn.exec_driver_sql("CREATE INDEX IF NOT EXISTS idx_stats_class_id ON stats_records (class_id)")
    except Exception:
        pass

# ===== Logging & Request ID =====
logger = logging.getLogger("nolook")
if not logger.handlers:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

# ===== Utils =====

def _full_labels(labels: dict) -> dict:
    """毎回同じキーを出す（欠けは0で埋める）"""
    return {k: float(labels.get(k, 0.0)) for k in EMOTION_KEYS}


def _normalize(labels: Dict[str, float]) -> Dict[str, float]:
    total = float(sum(max(0.0, v) for v in labels.values()))
    if total <= 0.0:
        return {k: (1.0 if k == "中立" else 0.0) for k in EMOTION_KEYS}
    return {k: round(max(0.0, labels.get(k, 0.0)) / total, 4) for k in EMOTION_KEYS}

# ====== 感情辞書ベース分類 ======
EMOTION_LEXICON: Dict[str, List[str]] = {
    "楽しい":  [
        "楽しい", "嬉しい", "うれしい", "わくわく", "最高", "よかった",
        "やった", "やったー", "やったぁ", "嬉しすぎ", "うれしすぎ", "最高すぎ",
        "自己ベスト", "自己ベスト更新", "過去最高", "記録更新", "満点", "満点取れた",
        "合格", "受かった", "当選", "優勝", "入賞", "勝てた", "勝った",
        "一位", "1位", "MVP", "表彰", "褒められた", "認められた",
        "成功", "達成", "できた", "できるようになった", "うまくいった",
        "楽しかった", "楽しすぎ", "楽しすぎた"
    ],
    "悲しい":  [
        "悲しい", "さみしい", "辛い", "つらい", "落ち込", "泣きたい", "憂鬱", "しょんぼり",
        "萎え", "萎えた", "萎える", "萎えたー", "萎えー", "萎えぇ", "へこむ", "凹む", "凹んだ"
    ],
    "怒り":    ["怒", "ムカつ", "むかつ", "腹立", "イライラ", "キレた"],
    "不安":    ["不安", "心配", "こわい", "怖い", "緊張", "ドキドキ", "心細い"],
    "しんどい": ["疲れ", "疲れた", "だる", "しんど", "眠い", "眠たい", "疲労"],
}

# Joyの文脈パターン（語彙が無くてもポジティブ達成/勝利/評価で加点）
JOY_CONTEXT_PATTERNS = [
    r"(テスト|模試|試験|返却|成績).*(最高|過去最高|自己ベスト|満点|良|高|上が|更新|とれた|取れた|できた)",
    r"(点数|スコア).*(最高|過去最高|自己ベスト|満点|高|上が|更新)",
    r"(合格|受かった|認定|合格発表|結果).*(出た|きた|勝|合格)",
    r"(優勝|入賞|MVP|表彰|メダル|トロフィー)",
    r"(試合|大会|コンテスト|コンクール).*(勝(った|てた)|優勝|入賞)",
    r"(うまくい(っ|た)|成功|達成|完走|やり切(っ|った)|できたー?)",
    r"(文化祭|体育祭|発表|合唱|演奏|ダンス).*(成功|盛り上が|楽しかった|よかった)",
    r"(チケット|抽選).*(当たった|当選)",
    r"(推し|ライブ|イベント).*(行けた|行ける|最高|神)",
]
NEG_NEAR_PATTERN = re.compile(r"(じゃない|じゃなかった|なくはない|ない|なかった)")


def _apply_context_boosts(text: str, counts: Dict[str, int]) -> None:
    t = text or ""
    if NEG_NEAR_PATTERN.search(t):
        return
    if any(re.search(p, t) for p in JOY_CONTEXT_PATTERNS):
        counts["楽しい"] += 2


# ルールベースの確率分布

def _lexicon_dist(text: str) -> Dict[str, float]:
    t = text or ""
    counts = defaultdict(int)
    for emo, kws in EMOTION_LEXICON.items():
        for kw in kws:
            counts[emo] += len(re.findall(re.escape(kw), t))
    _apply_context_boosts(t, counts)
    if sum(counts.values()) == 0:
        return {"中立": 1.0}
    labels = {emo: float(c) for emo, c in counts.items()}
    return _normalize(_full_labels(labels))

# ===== LLMベース分類（JSON厳格出力） =====
classification_chain = None
if LANGCHAIN_AVAILABLE:
    from langchain.prompts import PromptTemplate
    from langchain.chains import LLMChain
    from langchain_openai import ChatOpenAI

    _cls_prompt = PromptTemplate(
        input_variables=["text"],
        template=(
            """
あなたは中高生の日本語日記の感情分類器です。文脈に応じて6感情の分布を出してください。
感情キー: ["楽しい","悲しい","怒り","不安","しんどい","中立"]
出力要件: JSONのみで返す。例:
{"labels":{"楽しい":0.0,"悲しい":0.9,"怒り":0.05,"不安":0.05,"しんどい":0.0,"中立":0.0},"primary":"悲しい"}
- 値は0..1でおよそ合計1
- スラングや直喩は文脈で解釈（例: 萎えた→多くは悲しい）
- 例1: "最悪…ミスって落ち込んだ" → 悲しい優勢
- 例2: "最悪だ、あの審判ふざけてる" → 怒り優勢
- 例3: "ヤバいくらい眠い" → しんどい優勢
- 例4: "今日は萎えたー" → 悲しい優勢
テキスト: {text}
            """
        ),
    )
    classification_chain = LLMChain(
        llm=ChatOpenAI(model="gpt-4o-mini", temperature=0.2, openai_api_key=OPENAI_API_KEY),
        prompt=_cls_prompt,
        verbose=False,
    )


def _llm_dist(text: str) -> Optional[Dict[str, float]]:
    if not classification_chain:
        return None
    try:
        raw = classification_chain.run(text=text)
        data = json.loads(raw)
        labels = data.get("labels", {})
        labels = _normalize(_full_labels(labels))
        return labels
    except Exception:
        return None

# ===== ハイブリッド分類 =====

def classify_emotion(text: str):
    """辞書 + (あれば)LLM を混ぜた分布を返す。primary と score も計算。"""
    lex = _lexicon_dist(text)
    llm = _llm_dist(text) if (LANGCHAIN_AVAILABLE and classification_chain) else None
    if llm:
        combined = {k: (1.0 - LLM_WEIGHT) * lex.get(k, 0.0) + LLM_WEIGHT * llm.get(k, 0.0) for k in EMOTION_KEYS}
        labels = _normalize(combined)
    else:
        labels = _normalize(lex)
    top_emotion = max(labels.items(), key=lambda x: x[1])[0]
    top_score = float(labels[top_emotion])
    return top_emotion, top_score, labels

# ===== Signals =====
class Signals(BaseModel):
    topic_tags: List[str]
    relationship_mention: bool
    negation_index: float
    avoidance: float

TOPIC_LEXICON: Dict[str, List[str]] = {
    "友だち": ["友だち", "友達", "ともだち", "クラスメイト", "いじめ", "仲間", "先輩", "後輩"],
    "勉強": ["勉強", "テスト", "宿題", "成績", "授業", "課題", "受験"],
    "家庭": ["家", "家族", "親", "父", "母", "兄", "姉", "弟", "妹"],
    "部活": ["部活", "クラブ", "サークル", "試合", "大会", "練習"],
    "体調": ["体調", "熱", "風邪", "腹痛", "頭痛", "眠い", "疲れ", "しんどい"],
}
RELATIONSHIP_WORDS = ["友だち", "友達", "ともだち", "いじめ", "無視", "悪口", "仲間はずれ", "ぼっち"]
NEGATION_WORDS = ["ない", "できない", "無理", "嫌い", "いやだ", "ダメ", "もうやだ"]
AVOIDANCE_WORDS = ["別に", "なんでもない", "知らない", "まあいい", "どうでもいい"]


def compute_signals(text: str) -> Signals:
    t = text or ""
    tags: List[str] = []
    for tag, kws in TOPIC_LEXICON.items():
        if any(kw in t for kw in kws):
            tags.append(tag)
    rel = any(w in t for w in RELATIONSHIP_WORDS)

    def score_by_terms(terms: List[str]) -> float:
        hits = sum(t.count(w) for w in terms)
        return float(min(1.0, round(hits / 10.0, 2)))

    return Signals(
        topic_tags=tags,
        relationship_mention=rel,
        negation_index=score_by_terms(NEGATION_WORDS),
        avoidance=score_by_terms(AVOIDANCE_WORDS),
    )

# ===== Reply generation (LLM強化 + ルールFallback) =====
# ぜんぶ短く・会話っぽく・命令調控えめ（タスク出し廃止）
PHRASE_LIB: Dict[str, List[str]] = {
    "楽しい": [
        "自己ベストってほんとうにすごいね。",  # 共感
        "がんばりが実ったね、誇っていいよ。",
        "この調子でいこ。応援してる。",
    ],
    "悲しい": [
        "それはつらかったね…。",
        "ここに話してくれてえらいよ。",
        "今は無理しないで。そばにいるよ。",
    ],
    "怒り": [
        "それは腹立つね…。",
        "気持ちはわかるよ。",
        "少し落ち着いたら、どうしたいか一緒に考えよ。",
    ],
    "不安": [
        "不安な気持ち、わかるよ。",
        "いまの君なら大丈夫。",
        "一歩ずつでいいからね。",
    ],
    "しんどい": [
        "しんどかったね…。",
        "今日は自分を労って。",
        "少し休めたら、また話そう。",
    ],
    "中立": [
        "話してくれてありがとう。",
        "状況はわかったよ。",
        "必要ならいつでも呼んで。",
    ],
}

# topicに沿った一言を軽く追加（任意）
TOPIC_TWEAKS: Dict[str, Dict[str, List[str]]] = {
    "勉強": {
        "楽しい": ["努力が結果に出たね。"],
        "不安": ["一緒に作戦立てよ。"],
    },
    "部活": {
        "楽しい": ["ナイスプレー！"],
        "怒り": ["次はうまくやれるよ。"],
    },
}


def _join_short(lines: List[str], max_sent: int) -> str:
    msg = " ".join(lines[:max_sent])
    return msg.strip()


# LLMで文章を整えるチェーン（なければNone）
reply_chain = None
if LANGCHAIN_AVAILABLE:
    from langchain.prompts import PromptTemplate as _PT
    from langchain.chains import LLMChain as _LC
    from langchain_openai import ChatOpenAI as _C
    _reply_prompt = _PT(
        input_variables=["text", "emotion", "style", "length", "topics"],
        template=(
            """
日本語で返答。口調: {style}（buddy=友だち風/coach=前向き/teacher=落ち着いた先生）。
絵文字・箇条書きなし。命令調のタスク出しはしない。誇張しすぎない。
長さ: {length}（short=1文/自然停止, medium=最大2文）。各文60字以内。
必ず含める:
- 共感のひとこと（ユーザーの内容を短く鏡映）
- ささやかな励まし（例: 応援してる/この調子）
- 可能なら topics: {topics} をさりげなく踏まえる
入力: {text}
            """
        ),
    )
    reply_chain = _LC(
        llm=_C(model="gpt-4o-mini", temperature=0.4, openai_api_key=OPENAI_API_KEY),
        prompt=_reply_prompt,
        verbose=False,
    )


def build_reply(text: str, emotion: str, sig: Signals, *, style: str = "buddy", length: str = "short", followup: bool = False) -> str:
    # LLMがあれば整形
    if reply_chain is not None:
        try:
            base = reply_chain.run(text=text, emotion=emotion, style=style, length=length, topics=",".join(sig.topic_tags))
            if followup:
                tail = {
                    "楽しい": "次もこの調子でいけそう？",
                    "悲しい": "今は何してると少し楽？",
                    "怒り": "落ち着いたらどう動きたい？",
                    "不安": "まず何からやれそう？",
                    "しんどい": "少し休めそう？",
                    "中立": "続きも聞かせて？",
                }.get(emotion, "続きも聞かせて？")
                return f"{base} {tail}"
            return base
        except Exception:
            pass
    # ルール整形（フォールバック）: 1-2短文
    base_lines = PHRASE_LIB.get(emotion, PHRASE_LIB["中立"]).copy()
    for tag in sig.topic_tags:
        base_lines.extend(TOPIC_TWEAKS.get(tag, {}).get(emotion, []))
    msg = _join_short(base_lines, 2 if length == "medium" else 1)
    if followup:
        tail = {
            "楽しい": "次もこの調子でいけそう？",
            "悲しい": "今は何してると少し楽？",
            "怒り": "落ち着いたらどう動きたい？",
            "不安": "まず何からやれそう？",
            "しんどい": "少し休めそう？",
            "中立": "続きも聞かせて？",
        }.get(emotion, "続きも聞かせて？")
        msg = f"{msg} {tail}"
    return msg

# ===== Persistence helper =====
class _SignalsLike:  # for type hinting when called from /analyze
    topic_tags: List[str]
    relationship_mention: bool
    negation_index: float
    avoidance: float


def save_stats(*, labels: dict, signals: _SignalsLike, top_emotion: str, score: float, class_id: Optional[str] = None) -> int:
    rec = StatsRecord(
        class_id=class_id,
        emotion=top_emotion,
        score=score,
        labels=_full_labels(labels),
        topic_tags=list(getattr(signals, "topic_tags", [])),
        relationship_mention=bool(getattr(signals, "relationship_mention", False)),
        negation_index=float(getattr(signals, "negation_index", 0.0)),
        avoidance=float(getattr(signals, "avoidance", 0.0)),
    )
    with SessionLocal() as db:
        db.add(rec)
        db.commit()
        db.refresh(rec)
        return rec.id

# ===== Security: API Key (optional) + Rate Limit (optional) =====
api_key_header = APIKeyHeader(name="X-API-Key", auto_error=False)


async def verify_api_key(api_key: Optional[str] = Security(api_key_header)):
    if not API_KEY:
        return True  # 未設定なら認証なし
    if not api_key:
        raise HTTPException(status_code=401, detail="Missing API key")
    if api_key != API_KEY:
        raise HTTPException(status_code=403, detail="Invalid API key")
    return True


RATE_STORE: Dict[str, tuple] = {}  # key -> (window_start_epoch, count)


async def rate_limit(req: Request):
    if RATE_PER_MIN <= 0:
        return True  # 無効
    key = f"{req.client.host}:{req.url.path}"
    now = int(time.time())
    window = now // 60
    cur = RATE_STORE.get(key)
    if not cur or cur[0] != window:
        RATE_STORE[key] = (window, 1)
        return True
    cnt = cur[1] + 1
    RATE_STORE[key] = (window, cnt)
    if cnt > RATE_PER_MIN:
        raise HTTPException(status_code=429, detail="Rate limit exceeded")
    return True

# ===== FastAPI =====
app = FastAPI(title="NO LOOK API", version="0.4.0")

# --- CORS allowlist ---
app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_credentials=True,
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["*"],
    max_age=3600,
)

# --- Request ID + logging middleware ---
@app.middleware("http")
async def add_request_id_and_log(request: Request, call_next):
    rid = str(uuid.uuid4())
    request.state.rid = rid
    start = time.time()
    try:
        response = await call_next(request)
    except Exception as e:
        logger.exception("rid=%s unhandled error: %s", rid, e)
        return JSONResponse(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            content={"error": {"type": "internal_error", "message": "Internal Server Error", "rid": rid}},
        )
    dur = int((time.time() - start) * 1000)
    response.headers["X-Request-ID"] = rid
    logger.info("rid=%s method=%s path=%s status=%s dur_ms=%s", rid, request.method, request.url.path, response.status_code, dur)
    return response

# --- Unified validation error (422) ---
@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request: Request, exc: RequestValidationError):
    rid = getattr(request.state, "rid", str(uuid.uuid4()))
    return JSONResponse(
        status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
        content={"error": {"type": "validation_error", "message": exc.errors(), "rid": rid}},
    )

# ===== 会話返信チェーン（未使用：フォールバックに移行） =====
if LANGCHAIN_AVAILABLE:
    from langchain.prompts import PromptTemplate
    from langchain.chains import LLMChain
    from langchain_openai import ChatOpenAI

    convo_chain = LLMChain(
        llm=ChatOpenAI(model="gpt-4o-mini", temperature=0.7, openai_api_key=OPENAI_API_KEY),
        prompt=PromptTemplate(
            input_variables=["input"],
            template=(
                """
ユーザーの一言に、丁寧で短い励ましの一言を日本語で返してください。60字以内。
文脈に共感し、次の一歩を1つだけ提案して。
入力: {input}
                """
            ),
        ),
        verbose=False,
    )
else:
    class _DummyConvo:
        def run(self, input: str) -> str:
            emo, _, _ = classify_emotion(input)
            return PHRASE_LIB.get(emo, PHRASE_LIB["中立"])[0]
    convo_chain = _DummyConvo()

# ===== Pydantic models =====
class AskRequest(BaseModel):
    prompt: str
    style: Optional[Literal["buddy", "coach", "teacher"]] = "buddy"  # 返信の声色
    length: Optional[Literal["short", "medium"]] = "short"            # 文量: short=1文 / medium=2文
    followup: Optional[bool] = False                                      # 末尾に短い問いかけを付けるか


class AskResponse(BaseModel):
    reply: str                 # 一言返信（自然文）
    emotion: str               # 代表感情
    score: float               # 代表感情の強さ（0〜1）
    labels: Dict[str, float]   # 全感情の分布（固定キー）


class AnalyzeRequest(BaseModel):
    prompt: str


class AnalyzeResponse(BaseModel):
    labels: Dict[str, float]
    signals: 'Signals'


class DayCounts(BaseModel):
    date: str
    counts: Dict[str, int]
    total: int


class SummaryResponse(BaseModel):
    days: int
    daily: List[DayCounts]
    totals: Dict[str, int]
    top_emotion: str

# ===== Security helpers =====
api_key_header = APIKeyHeader(name="X-API-Key", auto_error=False)


async def verify_api_key(api_key: Optional[str] = Security(api_key_header)):
    if not API_KEY:
        return True
    if not api_key:
        raise HTTPException(status_code=401, detail="Missing API key")
    if api_key != API_KEY:
        raise HTTPException(status_code=403, detail="Invalid API key")
    return True


RATE_STORE: Dict[str, tuple] = {}


async def rate_limit(req: Request):
    if RATE_PER_MIN <= 0:
        return True
    key = f"{req.client.host}:{req.url.path}"
    now = int(time.time())
    window = now // 60
    cur = RATE_STORE.get(key)
    if not cur or cur[0] != window:
        RATE_STORE[key] = (window, 1)
        return True
    cnt = cur[1] + 1
    RATE_STORE[key] = (window, cnt)
    if cnt > RATE_PER_MIN:
        raise HTTPException(status_code=429, detail="Rate limit exceeded")
    return True

# ===== Routes =====
@app.get("/")
async def read_root(verify: bool = Depends(verify_api_key)):
    return {"message": "NO LOOK API running", "openai": LANGCHAIN_AVAILABLE, "llm_weight": LLM_WEIGHT}


@app.post("/ask", response_model=AskResponse)
async def ask_ai(request: AskRequest, _k=Depends(verify_api_key), _r=Depends(rate_limit)):
    text = (request.prompt or "").strip()
    if not text:
        raise HTTPException(status_code=400, detail="prompt が空です。")
    emotion, score, labels = classify_emotion(text)
    sig = compute_signals(text)
    reply = build_reply(text, emotion, sig, style=request.style or "buddy", length=request.length or "short", followup=bool(request.followup))
    return AskResponse(reply=reply, emotion=emotion, score=score, labels=labels)


@app.post("/analyze", response_model=AnalyzeResponse)
async def analyze(req: AnalyzeRequest, _k=Depends(verify_api_key), _r=Depends(rate_limit)):
    text = (req.prompt or "").strip()
    if not text:
        raise HTTPException(status_code=400, detail="prompt が空です。")
    top_emotion, score, labels = classify_emotion(text)
    sig = compute_signals(text)
    _ = save_stats(labels=labels, signals=sig, top_emotion=top_emotion, score=score, class_id=None)
    return AnalyzeResponse(labels=_full_labels(labels), signals=sig)


# ===== Summary（日別集計） =====
import zoneinfo


@app.get("/summary", response_model=SummaryResponse)
async def summary(
    days: int = 7,
    class_id: Optional[str] = None,
    tz: str = "Asia/Tokyo",
    include_empty_days: bool = True,
    _k: bool = Depends(verify_api_key),
    _r: bool = Depends(rate_limit),
):
    """直近days日を日別集計して返す。DBの `created_at` は UTC naive とみなし、
    表示タイムゾーン `tz` に変換して集計。フィルタ下限も `tz` の start_date 00:00 を UTC→naive にして比較。
    """
    tzinfo = zoneinfo.ZoneInfo(tz)

    today_local = datetime.now(tzinfo).date()
    start_date_local = today_local - timedelta(days=days - 1)

    from datetime import timezone as _tz
    start_local_aware = datetime.combine(start_date_local, datetime.min.time(), tzinfo=tzinfo)
    start_dt = start_local_aware.astimezone(_tz.utc).replace(tzinfo=None)

    where_clause = [StatsRecord.created_at >= start_dt]
    if class_id:
        where_clause.append(StatsRecord.class_id == class_id)

    with SessionLocal() as db:
        rows = db.execute(select(StatsRecord.created_at, StatsRecord.emotion).where(and_(*where_clause))).all()

    def to_local_date_str(dt: datetime) -> str:
        if dt.tzinfo is None:
            return dt.replace(tzinfo=_tz.utc).astimezone(tzinfo).date().isoformat()
        return dt.astimezone(tzinfo).date().isoformat()

    by_day: Dict[str, Dict[str, int]] = {}
    if include_empty_days:
        for i in range(days):
            d = (start_date_local + timedelta(days=i)).isoformat()
            by_day[d] = {k: 0 for k in EMOTION_KEYS}

    for created_at, emo in rows:
        d = to_local_date_str(created_at)
        if d < start_date_local.isoformat() or d > today_local.isoformat():
            continue
        if d not in by_day:
            by_day[d] = {k: 0 for k in EMOTION_KEYS}
        if emo not in by_day[d]:
            by_day[d][emo] = 0
        by_day[d][emo] += 1

    days_sorted = [
        (start_date_local + timedelta(days=i)).isoformat() for i in range(days)
    ] if include_empty_days else sorted(by_day.keys())

    daily: List[DayCounts] = []
    totals = {k: 0 for k in EMOTION_KEYS}
    for d in days_sorted:
        counts = {k: int(by_day.get(d, {}).get(k, 0)) for k in EMOTION_KEYS}
        total = sum(counts.values())
        daily.append(DayCounts(date=d, counts=counts, total=total))
        for k, v in counts.items():
            totals[k] += v

    top_emotion = "中立" if sum(totals.values()) == 0 else max(
        EMOTION_KEYS, key=lambda k: (totals.get(k, 0), -EMOTION_KEYS.index(k))
    )

    return SummaryResponse(days=len(daily), daily=daily, totals=totals, top_emotion=top_emotion)


# ===== Weekly report =====
class WeeklyReportResponse(BaseModel):
    start_date: str
    end_date: str
    tz: str
    days: int
    daily: List[DayCounts]
    totals: Dict[str, int]
    top_emotion: str
    trend: Dict[str, List[str]]
    summary: str
    suggestions: List[str]


@app.get("/weekly_report", response_model=WeeklyReportResponse)
async def weekly_report(
    days: int = 7,
    class_id: Optional[str] = None,
    tz: str = "Asia/Tokyo",
    include_empty_days: bool = True,
    _k: bool = Depends(verify_api_key),
    _r: bool = Depends(rate_limit),
):
    tzinfo = zoneinfo.ZoneInfo(tz)
    today_local = datetime.now(tzinfo).date()
    start_date_local = today_local - timedelta(days=days - 1)

    from datetime import timezone as _tz
    start_local_aware = datetime.combine(start_date_local, datetime.min.time(), tzinfo=tzinfo)
    start_dt = start_local_aware.astimezone(_tz.utc).replace(tzinfo=None)

    where_clause = [StatsRecord.created_at >= start_dt]
    if class_id:
        where_clause.append(StatsRecord.class_id == class_id)

    with SessionLocal() as db:
        rows = db.execute(select(StatsRecord.created_at, StatsRecord.emotion).where(and_(*where_clause))).all()

    def to_local_date_str(dt: datetime) -> str:
        if dt.tzinfo is None:
            return dt.replace(tzinfo=_tz.utc).astimezone(tzinfo).date().isoformat()
        return dt.astimezone(tzinfo).date().isoformat()

    by_day: Dict[str, Dict[str, int]] = {}
    if include_empty_days:
        for i in range(days):
            d = (start_date_local + timedelta(days=i)).isoformat()
            by_day[d] = {k: 0 for k in EMOTION_KEYS}

    for created_at, emo in rows:
        d = to_local_date_str(created_at)
        if d < start_date_local.isoformat() or d > today_local.isoformat():
            continue
        if d not in by_day:
            by_day[d] = {k: 0 for k in EMOTION_KEYS}
        if emo not in by_day[d]:
            by_day[d][emo] = 0
        by_day[d][emo] += 1

    days_sorted = [
        (start_date_local + timedelta(days=i)).isoformat() for i in range(days)
    ] if include_empty_days else sorted(by_day.keys())

    daily: List[DayCounts] = []
    totals = {k: 0 for k in EMOTION_KEYS}
    for d in days_sorted:
        counts = {k: int(by_day.get(d, {}).get(k, 0)) for k in EMOTION_KEYS}
        total = sum(counts.values())
        daily.append(DayCounts(date=d, counts=counts, total=total))
        for k, v in counts.items():
            totals[k] += v

    grand_total = sum(totals.values())
    top_emotion = "中立" if grand_total == 0 else max(
        EMOTION_KEYS, key=lambda k: (totals.get(k, 0), -EMOTION_KEYS.index(k))
    )

    split = max(1, days // 2)
    older_days = days_sorted[:split]
    recent_days = days_sorted[split:]

    def sum_range(day_keys: List[str]) -> Dict[str, int]:
        x = {k: 0 for k in EMOTION_KEYS}
        for dd in day_keys:
            for k in EMOTION_KEYS:
                x[k] += int(by_day.get(dd, {}).get(k, 0))
        return x

    older = sum_range(older_days)
    recent = sum_range(recent_days)
    rising = [k for k in EMOTION_KEYS if recent.get(k, 0) > older.get(k, 0)]
    falling = [k for k in EMOTION_KEYS if recent.get(k, 0) < older.get(k, 0)]

    def pct(n: int) -> float:
        return (n / grand_total) if grand_total else 0.0

    p_sad = pct(totals.get("悲しい", 0))
    p_fear = pct(totals.get("不安", 0))
    p_ang = pct(totals.get("怒り", 0))
    p_tired = pct(totals.get("しんどい", 0))

    notes: List[str] = []
    if grand_total > 0:
        peak_day = max(daily, key=lambda r: r.total).date if daily else start_date_local.isoformat()
        notes.append(f"ピーク日: {peak_day}")
        notes.append(f"最多感情: {top_emotion} ({totals.get(top_emotion, 0)}件)")
        if rising:
            notes.append("増加: " + ", ".join(rising))
        if falling:
            notes.append("減少: " + ", ".join(falling))

    suggestions: List[str] = []
    if p_sad + p_fear >= 0.6 and grand_total >= 5:
        suggestions.append("不安・悲しみが目立つ週。相談先と低負荷の個別声かけを案内")
    if p_ang >= 0.3:
        suggestions.append("怒りが目立つ。対人トラブル/ルール確認の機会を設定")
    if p_tired >= 0.3:
        suggestions.append("しんどいが多め。睡眠・休息の促しと提出負荷の調整")
    if not suggestions and grand_total > 0:
        suggestions.append("小さな成功体験の共有と、次週も継続的に観察")

    summary_text = (
        "今週の投稿は少数でした。次週も観察を継続しましょう。" if grand_total == 0 else
        f"合計{grand_total}件。最多は『{top_emotion}』。直近で{', '.join(rising) if rising else '大きな増減は'}見られません。"
    )

    return WeeklyReportResponse(
        start_date=start_date_local.isoformat(),
        end_date=today_local.isoformat(),
        tz=tz,
        days=len(days_sorted),
        daily=daily,
        totals=totals,
        top_emotion=top_emotion,
        trend={"rising": rising, "falling": falling},
        summary=summary_text,
        suggestions=notes + suggestions,
    )

# ===== Lightweight smoke tests =====
if __name__ == "__main__":
    if os.getenv("RUN_TESTS") == "1":
        from fastapi.testclient import TestClient
        orig_key = API_KEY
        API_KEY = None
        client = TestClient(app)
        r = client.post("/ask", json={"prompt": "今日は萎えたー"})
        assert r.status_code == 200, r.text
        assert r.json()["emotion"] == "悲しい", r.json()
        r2 = client.post("/analyze", json={"prompt": "今日はテストで不安"})
        assert r2.status_code == 200
        keys = list(r2.json()["labels"].keys())
        assert keys == EMOTION_KEYS, keys
        r3 = client.get("/summary", params={"days": 7})
        assert r3.status_code == 200
        API_KEY = "test"
        r4 = client.post("/ask", json={"prompt": "ok"})
        assert r4.status_code == 401
        print("All tests passed")
        API_KEY = orig_key

    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=int(os.getenv("PORT", "8000")))
