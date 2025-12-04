from datasets import load_dataset
import os

os.makedirs("data_raw", exist_ok=True)
ds = load_dataset("tweet_eval", "sentiment")
for split in ("train", "validation", "test"):
    ds[split].to_json(f"data_raw/{split}.jsonl")
print("✅ Download done:", {k: len(v) for k, v in ds.items()})