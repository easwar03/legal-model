from datasets import load_dataset
from transformers import (AutoTokenizer, DataCollatorWithPadding, TrainingArguments, AutoModelForSequenceClassification, Trainer)
import numpy as np
import evaluate


def tokenize_function(example):
    return tokenizer(example["sentence"], truncation=True)


def compute_metrics(eval_predictions):
    metrics = evaluate.load("glue", "sst2")
    logits, labels = eval_predictions
    prediction = np.argmax(logits, axis=-1)
    return metrics.compute(predictions=prediction, references=labels)


raw_datasets = load_dataset("glue", "sst2")
checkpoint = "siebert/sentiment-roberta-large-english"
tokenizer = AutoTokenizer.from_pretrained(checkpoint)

small_train_dataset = raw_datasets["train"].shuffle(seed=42).select([i for i in list(range(5000))])
small_test_dataset = raw_datasets["test"].shuffle(seed=42).select([i for i in list(range(500))])
small_validation_dataset = raw_datasets["validation"].shuffle(seed=42).select([i for i in list(range(500))])

tokenized_train = small_train_dataset.map(tokenize_function, batched=True)
tokenized_test = small_test_dataset.map(tokenize_function, batched=True)
tokenized_validation = small_validation_dataset.map(tokenize_function, batched=True)
data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

training_args = TrainingArguments("test-trainer",)

model = AutoModelForSequenceClassification.from_pretrained(checkpoint, num_labels=2)

trainer = Trainer(
    model,
    training_args,
    train_dataset=tokenized_train,
    eval_dataset=tokenized_validation,
    data_collator=data_collator,
    tokenizer=tokenizer,
    compute_metrics=compute_metrics,
)

predictions = trainer.predict(tokenized_validation)
print(predictions.predictions.shape, predictions.label_ids.shape)

preds = np.argmax(predictions.predictions, axis=-1)

metric = evaluate.load("glue", "sst2")
metric.compute(predictions=preds, references=predictions.label_ids)

trainer.train()

model.save_pretrained("saved_model")
tokenizer.save_pretrained("saved_model")
