from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Dict
from collections import defaultdict
import re
import os
from dotenv import load_dotenv

# --- LangChain / OpenAI ---
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain.memory import ConversationBufferMemory
from langchain_openai import ChatOpenAI

# 環境変数読み込み
load_dotenv()
# --- DB: SQLAlchemy セットアップ ---
from sqlalchemy import create_engine, Column, Integer, String, Float, Boolean, DateTime, JSON, func
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
    class_id = Column(String, nullable=True)   # 学級IDなど、無ければ None でOK
    emotion = Column(String, nullable=False)
    score = Column(Float, nullable=False)
    labels = Column(JSON, nullable=False)            # 例: {"不安":0.5, ...}
    topic_tags = Column(JSON, nullable=False)        # 例: ["友だち","勉強"]
    relationship_mention = Column(Boolean, nullable=False)
    negation_index = Column(Float, nullable=False)
    avoidance = Column(Float, nullable=False)

Base.metadata.create_all(bind=engine)

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
    # 話題タグ
    tags = []
    for tag, kws in TOPIC_LEXICON.items():
        if any(kw in t for kw in kws):
            tags.append(tag)
    if not tags:
        tags = []

    # 人間関係の言及
    rel = any(w in t for w in RELATIONSHIP_WORDS)

    # 否定/回避のスコア（0〜1に丸め）
    def score_by_terms(terms: List[str]) -> float:
        hits = sum(t.count(w) for w in terms)
        length = max(1, len(t))
        # ヒット数をざっくり正規化（文字長で割って上限1.0）
        return float(min(1.0, round(hits / 10.0, 2)))

    neg_idx = score_by_terms(NEGATION_WORDS)
    avoid = score_by_terms(AVOIDANCE_WORDS)

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
def summary(
    days: int = 7,
    class_id: Optional[str] = None,
    tz: str = "Asia/Tokyo",
    include_empty_days: bool = True,
):
    """
    直近days日を日別に集計して返す。
    - include_empty_days=True ならデータが無い日も0件で返す
    - tz は表示上の集計タイムゾーン（例: "Asia/Tokyo"）
    """
    tzinfo = zoneinfo.ZoneInfo(tz)

    # 期間レンジ（今日を含む直近N日）
    today_local = datetime.now(tzinfo).date()
    start_date_local = today_local - timedelta(days=days - 1)

    # 期間の開始(00:00)をnaiveに落としてDB比較用に使う（SQLiteのローカル時刻前提）
    start_dt = datetime.combine(start_date_local, datetime.min.time())

    # 期間内の全レコード（created_at, emotion）を取得して Python 側で日付集計
    where_clause = [StatsRecord.created_at >= start_dt]
    if class_id:
        where_clause.append(StatsRecord.class_id == class_id)

    with SessionLocal() as db:
        rows = db.execute(
            select(StatsRecord.created_at, StatsRecord.emotion).where(and_(*where_clause))
        ).all()

    # 日付キーを "YYYY-MM-DD" でそろえる（tz付きなら変換、naiveならそのまま）
    def to_local_date_str(dt: datetime) -> str:
        if dt.tzinfo is not None:
            return dt.astimezone(tzinfo).date().isoformat()
        # naiveはローカル生成想定。必要なら tz を付与して扱う:
        # return dt.replace(tzinfo=tzinfo).date().isoformat()
        return dt.date().isoformat()

    # 初期化（0埋め）
    by_day: Dict[str, Dict[str, int]] = {}
    if include_empty_days:
        for i in range(days):
            d = (start_date_local + timedelta(days=i)).isoformat()
            by_day[d] = {k: 0 for k in EMOTION_KEYS}

    # 集計
    for created_at, emo in rows:
        d = to_local_date_str(created_at)
        if d < start_date_local.isoformat() or d > today_local.isoformat():
            continue
        if d not in by_day:
            by_day[d] = {k: 0 for k in EMOTION_KEYS}
        if emo not in by_day[d]:
            # 想定外のラベルが来た場合でも受け止める
            by_day[d][emo] = 0
        by_day[d][emo] += 1

    # 並びを日付昇順で安定化
    days_sorted = [ (start_date_local + timedelta(days=i)).isoformat() for i in range(days) ] \
                  if include_empty_days else sorted(by_day.keys())

    daily: List[DayCounts] = []
    totals = {k: 0 for k in EMOTION_KEYS}
    for d in days_sorted:
        counts = {k: int(by_day.get(d, {}).get(k, 0)) for k in EMOTION_KEYS}
        total = sum(counts.values())
        daily.append(DayCounts(date=d, counts=counts, total=total))
        for k, v in counts.items():
            totals[k] += v

    # 合計が同数のときは EMOTION_KEYS の順で決める
    if sum(totals.values()) == 0:
        top_emotion = "中立"
    else:
        top_emotion = max(EMOTION_KEYS, key=lambda k: (totals.get(k, 0), -EMOTION_KEYS.index(k)))

    return SummaryResponse(days=len(daily), daily=daily, totals=totals, top_emotion=top_emotion)
# ====== /summary  ======



