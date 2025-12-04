import pandas as pd, json, re, emoji, os

os.makedirs("data_clean", exist_ok=True)
def clean(text: str) -> str:
    text = re.sub(r"http\S+", "<URL>", text)
    text = emoji.demojize(text, delimiters=(" ", " "))
    return text.lower().strip()

for split in ("train", "validation", "test"):
    df = pd.read_json(f"data_raw/{split}.jsonl", lines=True)
    df["text"] = df["text"].apply(clean)
    df = df[df["text"].str.len() > 2]          # drop empty
    df.to_parquet(f"data_clean/{split}.parquet")

print("✅ Clean done.")