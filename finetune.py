import os, torch, pandas as pd
from datasets import load_dataset, Dataset
from transformers import (
    RobertaTokenizer, RobertaForMaskedLM, RobertaForSequenceClassification,
    TrainingArguments, Trainer, DataCollatorForLanguageModeling
)

# 1. 继续 MLM（无标签 1 epoch）
def continue_mlm():
    raw_ds = load_dataset("json", data_files="mlm_corpus.jsonl", split="train")
    tok = RobertaTokenizer.from_pretrained("roberta-base")

    def tokenize(ex):
        return tok(ex["text"], truncation=True, max_length=128)
    tokenized = raw_ds.map(tokenize, batched=True, remove_columns=["text"])

    data_collator = DataCollatorForLanguageModeling(tokenizer=tok, mlm=True, mlm_probability=0.15)
    model = RobertaForMaskedLM.from_pretrained("roberta-base")

    args = TrainingArguments(
        output_dir="ckpt_mlm", per_device_train_batch_size=64, num_train_epochs=1,
        learning_rate=1e-4, weight_decay=0.01, warmup_steps=1000, logging_steps=100,
        save_strategy="epoch", report_to="none", seed=3407,
    )
    trainer = Trainer(model=model, args=args, train_dataset=tokenized, data_collator=data_collator)
    trainer.train()
    trainer.save_model("ckpt_mlm_final")


# 2. Fine-tune 分类（三分类）
def finetune_cls():
    train_df = pd.read_parquet("data_clean/train.parquet")
    val_df = pd.read_parquet("data_clean/validation.parquet")
    train_ds = Dataset.from_pandas(train_df)
    val_ds = Dataset.from_pandas(val_df)

    tok = RobertaTokenizer.from_pretrained("roberta-base")

    def tokenize(batch):
        return tok(batch["text"], padding='max_length', truncation=True, max_length=128)
    train_ds = train_ds.map(tokenize, batched=True)
    val_ds = val_ds.map(tokenize, batched=True)

    model = RobertaForSequenceClassification.from_pretrained("ckpt_mlm_final", num_labels=3)

    args = TrainingArguments(
        output_dir="ckpt_final", per_device_train_batch_size=64, per_device_eval_batch_size=64,
        num_train_epochs=3, eval_strategy="epoch", learning_rate=2e-5, weight_decay=0.01,
        warmup_ratio=0.1, logging_steps=100, report_to="none", save_strategy="no", seed=3407,
    )
    trainer = Trainer(
        model=model, args=args, train_dataset=train_ds, eval_dataset=val_ds,
        compute_metrics=lambda p: {"f1": (p.predictions.argmax(-1) == p.label_ids).mean()},
    )
    trainer.train()
    eval_res = trainer.evaluate()
    print("Final macro-F1:", eval_res["eval_f1"])


# 3. 一条龙执行
if __name__ == "__main__":
    print("===== 1. 继续 MLM =====")
    continue_mlm()
    print("===== 2. Fine-tune 分类 =====")
    finetune_cls()