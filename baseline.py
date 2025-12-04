import os, torch, evaluate
from datasets import load_from_disk
from datasets import Dataset
from transformers import (AutoTokenizer, AutoModelForSequenceClassification,
                          TrainingArguments, Trainer)
import pandas as pd


# 1. data
train_df = pd.read_parquet("data_clean/train.parquet")
val_df   = pd.read_parquet("data_clean/validation.parquet")
train_ds = Dataset.from_pandas(train_df)
val_ds   = Dataset.from_pandas(val_df)

# 2. tokenizer
tok = AutoTokenizer.from_pretrained("roberta-base")
def tokenize(batch):
    return tok(batch["text"], padding='max_length', truncation=True, max_length=128)
train_ds = train_ds.map(tokenize, batched=True)
val_ds   = val_ds.map(tokenize, batched=True)

# 3. model
model = AutoModelForSequenceClassification.from_pretrained("roberta-base", num_labels=3)

# 4. metrics
metric = evaluate.load("f1")
def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = logits.argmax(axis=-1)
    return metric.compute(predictions=preds, references=labels, average="macro")

# 5. trainer
args = TrainingArguments(
    output_dir="ckpt_cloud",
    per_device_train_batch_size=64,
    per_device_eval_batch_size=64,
    num_train_epochs=3,
    eval_strategy="epoch",
    learning_rate=2e-5,
    weight_decay=0.01,
    warmup_ratio=0.1,
    logging_steps=100,
    report_to="none",
    save_strategy="no",
    seed=3407,
    logging_dir="runs_cloud",
)
trainer = Trainer(
    model=model,
    args=args,
    train_dataset=train_ds,
    eval_dataset=val_ds,
    compute_metrics=compute_metrics,
)

# 6. train & evaluate
trainer.train()
eval_res = trainer.evaluate()
print("Final macro-F1:", eval_res["eval_f1"])