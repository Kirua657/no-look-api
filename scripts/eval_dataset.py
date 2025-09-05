import os, sys, csv
from pathlib import Path

os.environ.setdefault("NOLOOK_DISABLE_OPENAI", "1")

try:
    from genai import main as appmod
except Exception:
    print("genai/main.py を import できません。パスを確認してください。", file=sys.stderr)
    sys.exit(1)

CLASSES = appmod.EMOTION_KEYS
IDX = {c: i for i, c in enumerate(CLASSES)}

def load_rows(path: Path):
    rows = []
    with path.open(encoding="utf-8", newline="") as f:
        for row in csv.DictReader(f):
            rows.append((row["text"], row["expected"]))
    return rows

def eval_csv(path: Path):
    rows = load_rows(path)
    cm = [[0]*len(CLASSES) for _ in CLASSES]
    correct = 0
    for text, exp in rows:
        pred, _, _ = appmod.classify_emotion(text)
        cm[IDX.get(exp, IDX["中立"])][IDX[pred]] += 1
        if pred == exp:
            correct += 1
    n = len(rows)
    acc = correct / n if n else 0.0
    print(f"n={n}  accuracy={acc:.3f}")
    print(",".join(["exp\\pred"] + CLASSES))
    for i, c in enumerate(CLASSES):
        print(",".join([c] + [str(cm[i][j]) for j in range(len(CLASSES))]))

if __name__ == "__main__":
    p = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("data/devset.csv")
    eval_csv(p)
