from datasets import load_dataset, Dataset
from transformers import RobertaTokenizer, RobertaForMaskedLM, DataCollatorForLanguageModeling, Trainer, TrainingArguments

# 1. 数据：无标签推文
raw_ds = load_dataset("json", data_files="mlm_corpus.jsonl", split="train")

# 2. tokenizer
tok = RobertaTokenizer.from_pretrained("roberta-base")

def tokenize(ex):
    return tok(ex["text"], truncation=True, max_length=128)
tokenized = raw_ds.map(tokenize, batched=True, remove_columns=["text"])

# 3. data collator
data_collator = DataCollatorForLanguageModeling(tokenizer=tok, mlm=True, mlm_probability=0.15)

# 4. 模型
model = RobertaForMaskedLM.from_pretrained("roberta-base")

# 5. 训练参数
args = TrainingArguments(
    output_dir="ckpt_mlm",
    per_device_train_batch_size=64,
    num_train_epochs=1,          # 1 epoch ≈ 10k steps
    learning_rate=1e-4,          # MLM 常用 lr
    weight_decay=0.01,
    warmup_steps=1000,
    logging_steps=100,
    save_strategy="steps",
    save_steps=5000,
    report_to="none",
    seed=3407,
)

# 6. 训练
trainer = Trainer(
    model=model,
    args=args,
    train_dataset=tokenized,
    data_collator=data_collator,
)
trainer.train()
trainer.save_model("ckpt_mlm_final")