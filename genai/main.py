from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Dict
from collections import defaultdict
import re
import os
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

def save_stats(*, labels: dict, signals, top_emotion: str, score: float, class_id: str | None = None) -> int:
    rec = StatsRecord(
        class_id=class_id,
        emotion=top_emotion,
        score=score,
        labels=_full_labels(labels),
        topic_tags=list(signals.topic_tags),
        relationship_mention=bool(signals.relationship_mention),
        negation_index=float(signals.negation_index),
        avoidance=float(signals.avoidance),
    )
    with SessionLocal() as db:
        db.add(rec)
        db.commit()
        db.refresh(rec)
        return rec.id

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# FastAPI インスタンス
app = FastAPI(title="NO LOOK API")

# メモリ付きのChatモデル
chat_model = ChatOpenAI(
    model="gpt-4o-mini",  # 軽量モデル
    temperature=0.7,
    openai_api_key=OPENAI_API_KEY
)

# 会話メモリ
memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)

# プロンプトテンプレート
prompt = PromptTemplate(
    input_variables=["chat_history", "input"],
    template="""
以下はユーザーとAIの会話です。
会話の履歴：
{chat_history}

ユーザー: {input}
AI:"""
)

# LLMChain 作成（メモリ付き）
conversation_chain = LLMChain(
    llm=chat_model,
    prompt=prompt,
    memory=memory
)

# ====== ここから 追加: モデル定義 と 感情分類 ======

# 入力モデル
class AskRequest(BaseModel):
    prompt: str

# 出力モデル（構造化JSON）
class AskResponse(BaseModel):
    reply: str                 # 一言返信（自然文）
    emotion: str               # 代表感情（例: 不安/楽しい/悲しい/怒り/しんどい/中立）
    score: float               # 代表感情の強さ（0〜1）
    labels: Dict[str, float]   # 全感情の分布（合計≈1.0）

# 超シンプルな辞書ベースの感情キーワード
EMOTION_LEXICON = {
    "楽しい":  ["楽しい", "嬉しい", "うれしい", "わくわく", "最高", "よかった"],
    "悲しい":  ["悲しい", "さみしい", "辛い", "つらい", "落ち込", "泣きたい", "憂鬱", "しょんぼり"],
    "怒り":    ["怒", "ムカつ", "むかつ", "腹立", "イライラ", "キレた"],
    "不安":    ["不安", "心配", "こわい", "怖い", "緊張", "ドキドキ", "心細い"],
    "しんどい": ["疲れ", "疲れた", "だる", "しんど", "眠い", "眠たい", "疲労"],
}

def classify_emotion(text: str):
    """投稿（ユーザー入力）から感情分布を作り、代表感情とスコアを返す。"""
    text = text or ""
    counts = defaultdict(int)
    for emo, kws in EMOTION_LEXICON.items():
        for kw in kws:
            counts[emo] += len(re.findall(re.escape(kw), text))
    total = sum(counts.values())

    if total == 0:
        labels = {"中立": 1.0}
        return "中立", 1.0, labels

    labels = {emo: round(c / total, 4) for emo, c in counts.items()}
    top_emotion, top_score = max(labels.items(), key=lambda x: x[1])
    return top_emotion, float(top_score), labels

# ====== ここまで 追加 ======


# ルート確認用
@app.get("/")
def read_root():
    return {"message": "LangChain連携APIが起動しています"}


# /ask: 構造化JSONで返す
@app.post("/ask", response_model=AskResponse)
async def ask_ai(request: AskRequest):
    text = (request.prompt or "").strip()
    if not text:
        raise HTTPException(status_code=400, detail="prompt が空です。")

    # 1) 会話LLMで一言返信（既存の会話体験はそのまま）
    reply = conversation_chain.run(input=text)

    # 2) 入力テキストを簡易分類（あとでLLM分類に差し替え可）
    emotion, score, labels = classify_emotion(text)

    # 3) 構造化して返す
    return AskResponse(
        reply=reply,
        emotion=emotion,
        score=score,
        labels=labels
    )
# ====== ここから /analyze 追加 ======
from typing import List

# 統計シグナルのPydanticモデル
class Signals(BaseModel):
    topic_tags: List[str]        # 話題タグ（友だち/勉強/家庭/部活/体調）
    relationship_mention: bool   # 人間関係に触れているか
    negation_index: float        # 否定表現の強さ(0-1)
    avoidance: float             # 回避/そっけなさ(0-1)

class AnalyzeRequest(BaseModel):
    prompt: str

class AnalyzeResponse(BaseModel):
    labels: Dict[str, float]
    signals: Signals

# 簡易辞書（必要に応じて増やせます）
TOPIC_LEXICON = {
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
        negation_index=neg_idx,
        avoidance=avoid
    )

@app.post("/analyze", response_model=AnalyzeResponse)
async def analyze(req: AnalyzeRequest):
    text = (req.prompt or "").strip()
    if not text:
        raise HTTPException(status_code=400, detail="prompt が空です。")

    # 感情分布 & シグナル
    top_emotion, score, labels = classify_emotion(text)
    sig = compute_signals(text)

    # ★ 生テキストは保存せず、統計だけ保存
    _ = save_stats(labels=labels, signals=sig, top_emotion=top_emotion, score=score, class_id=None)

    # 返すのも統計のみ（labelsは固定キーで）
    return AnalyzeResponse(labels=_full_labels(labels), signals=sig)

# ====== /analyze 追加ここまで ======

# ====== /summary  ======
from datetime import datetime, timedelta
from typing import List, Dict, Optional
import zoneinfo
from sqlalchemy import select, and_

# 既に定義済なら再定義不要
EMOTION_KEYS = ["楽しい","悲しい","怒り","不安","しんどい","中立"]

class DayCounts(BaseModel):
    date: str                 # "YYYY-MM-DD"
    counts: Dict[str, int]    # 感情ごとの件数（キーは EMOTION_KEYS の順）
    total: int

class SummaryResponse(BaseModel):
    days: int
    daily: List[DayCounts]
    totals: Dict[str, int]    # 指定期間の合計（感情別）
    top_emotion: str          # 合計で最多の感情（同数なら EMOTION_KEYS の順で決定）

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
