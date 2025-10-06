import os, json, sys
from datetime import datetime
from pathlib import Path

import requests
import pandas as pd
import matplotlib.pyplot as plt
from dotenv import load_dotenv

# ========= 設定 =========
# .envから読み取り（なければデフォルト）
load_dotenv()
BASE = os.getenv("BASE_URL", "http://127.0.0.1:8000")  # FastAPIが動いているURL
API_KEY = os.getenv("API_KEY", "devkey-123")
DAYS = int(os.getenv("SUMMARY_DAYS", "7"))
TZ = os.getenv("SUMMARY_TZ", "Asia/Tokyo")

OUTDIR = Path("out")
OUTDIR.mkdir(parents=True, exist_ok=True)

def fetch_summary():
    url = f"{BASE}/summary"
    params = {"days": DAYS, "tz": TZ}
    headers = {"X-API-Key": API_KEY}
    r = requests.get(url, params=params, headers=headers, timeout=10)
    r.raise_for_status()
    return r.json()

def plot_bar(counts, title, outpath):
    plt.figure()
    # 並び順を固定
    order = ["楽しい", "悲しい", "怒り", "不安", "しんどい", "中立"]
    x = [k for k in order if k in counts]
    y = [counts[k] for k in x]
    plt.bar(x, y)
    plt.title(title)
    plt.ylabel("count")
    plt.tight_layout()
    plt.savefig(outpath)
    plt.close()

def plot_daily_trend(df, outpath):
    # 日次×感情のピボット（存在しない列は0で補完）
    order = ["楽しい", "悲しい", "怒り", "不安", "しんどい", "中立"]
    if "date" not in df.columns:
        return
    # 欠損列を0で追加
    for col in order:
        if col not in df.columns:
            df[col] = 0
    df2 = df[["date"] + order].copy()
    df2["date"] = pd.to_datetime(df2["date"])
    df2 = df2.sort_values("date")

    # 1枚にまとめて折れ線（色は指定しない）
    plt.figure()
    for col in order:
        plt.plot(df2["date"], df2[col], label=col)
    plt.title("Daily Emotion Trend")
    plt.xlabel("date")
    plt.ylabel("count")
    plt.legend()
    plt.tight_layout()
    plt.savefig(outpath)
    plt.close()

def main():
    print(f"[INFO] GET {BASE}/summary?days={DAYS}&tz={TZ}")
    data = fetch_summary()
    # 返却例を out/summary_raw.json に保存
    raw_path = OUTDIR / "summary_raw.json"
    raw_path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[INFO] Saved raw JSON -> {raw_path}")

    # 出力形式に合わせて整形（例：{summary: [...], totals: {...}} を想定）
    # 1) 総数バー
    totals = data.get("totals") or data.get("total") or {}
    # 予防：キーが英語/日本語混在でも拾えるよう best-effort で集約
    rename = {
        "joy":"楽しい","sad":"悲しい","anger":"怒り","anxiety":"不安","tired":"しんどい","neutral":"中立"
    }
    totals_jp = {}
    for k,v in totals.items():
        kj = rename.get(k, k)
        totals_jp[kj] = v

    bar_path = OUTDIR / "summary_last7_bar.png"
    plot_bar(totals_jp, f"Emotion Distribution (last {DAYS} days)", bar_path)
    print(f"[INFO] Saved bar chart -> {bar_path}")

    # 2) 日毎トレンド（summary配列に date と emotion別countが入っている想定）
    # 例：{"date":"2025-10-06","楽しい":3,"悲しい":1,...}
    summary_rows = data.get("summary") or data.get("daily") or []
    if isinstance(summary_rows, list) and summary_rows:
        df = pd.DataFrame(summary_rows)
        # 英語キーが来た場合の置換
        df = df.rename(columns=rename)
        trend_path = OUTDIR / "summary_daily_trend.png"
        plot_daily_trend(df, trend_path)
        print(f"[INFO] Saved daily trend -> {trend_path}")
    else:
        print("[WARN] No daily rows for trend plot. Skipped.")

    print("[DONE] Visualizations are in the 'out' folder.")

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print("[ERROR]", e)
        sys.exit(1)
