import os

os.environ["HF_HUB_DOWNLOAD_TIMEOUT"] = "300"
os.environ["HF_HUB_REQUEST_TIMEOUT"] = "300"
import time
import pandas as pd
import numpy as np
from datasets import Dataset
import evaluate
from transformers import (
    AutoTokenizer,
    AutoModelForTokenClassification,
    TrainingArguments,
    Trainer,
)
import torch

DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "..", "data", "labeled")
DATA_DIR = os.path.abspath(DATA_DIR)
CONLL_FILE = os.path.join(DATA_DIR, "relabeled_data_20250622_232809.conll")
OUTPUT_BASE_DIR = os.path.join(os.path.dirname(__file__), "..", "..", "models")
OUTPUT_BASE_DIR = os.path.abspath(OUTPUT_BASE_DIR)
RESULTS_FILE = os.path.join(OUTPUT_BASE_DIR, "model_comparison_results.csv")
os.makedirs(OUTPUT_BASE_DIR, exist_ok=True)

MODELS = [
    {
        "name": "xlm-roberta-base",
        "output_dir": os.path.join(OUTPUT_BASE_DIR, "xlm_roberta_ner"),
    },
    {
        "name": "bert-base-multilingual-cased",
        "output_dir": os.path.join(OUTPUT_BASE_DIR, "mbert_ner"),
    },
    {
        "name": "distilbert-base-multilingual-cased",
        "output_dir": os.path.join(OUTPUT_BASE_DIR, "distilbert_ner"),
    },
]

label_list = ["O", "B-Product", "I-Product", "B-LOC", "I-LOC", "B-PRICE", "I-PRICE"]
label2id = {label: idx for idx, label in enumerate(label_list)}
id2label = {idx: label for idx, label in enumerate(label_list)}


def load_conll(file_path):
    sentences, labels = [], []
    current_sentence, current_labels = [], []
    with open(file_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    token, label = line.split()
                    if label not in label2id:
                        continue
                    current_sentence.append(token)
                    current_labels.append(label2id[label])
                except ValueError:
                    continue
            else:
                if current_sentence:
                    sentences.append(current_sentence)
                    labels.append(current_labels)
                current_sentence, current_labels = [], []
    if current_sentence:
        sentences.append(current_sentence)
        labels.append(current_labels)
    if not sentences:
        raise ValueError("No valid data loaded from CoNLL file.")
    return {"tokens": sentences, "ner_tags": labels}


data = load_conll(CONLL_FILE)
dataset = Dataset.from_dict(data)
dataset = dataset.train_test_split(test_size=0.2, seed=42)


def tokenize_and_align_labels(examples, tokenizer):
    tokenized_inputs = tokenizer(
        examples["tokens"],
        truncation=True,
        is_split_into_words=True,
        padding="max_length",
        max_length=128,
        return_tensors="pt",
    )
    labels = []
    for i, label in enumerate(examples["ner_tags"]):
        word_ids = tokenized_inputs.word_ids(batch_index=i)
        aligned_labels = [-100] * len(word_ids)
        for j, word_id in enumerate(word_ids):
            if word_id is not None and word_id < len(label):
                aligned_labels[j] = label[word_id]
        labels.append(aligned_labels)
    tokenized_inputs["labels"] = labels
    return tokenized_inputs


def measure_inference_time(
    model, tokenizer, dataset, device="cuda" if torch.cuda.is_available() else "cpu"
):
    model.eval()
    model.to(device)
    inputs = tokenizer(
        dataset["tokens"],
        truncation=True,
        is_split_into_words=True,
        padding="max_length",
        max_length=128,
        return_tensors="pt",
    )
    input_ids = inputs["input_ids"].to(device)
    attention_mask = inputs["attention_mask"].to(device)
    start_time = time.time()
    with torch.no_grad():
        outputs = model(input_ids, attention_mask=attention_mask)
    end_time = time.time()
    num_samples = len(dataset["tokens"])
    inference_time = (end_time - start_time) / num_samples
    return inference_time


metric = evaluate.load("seqeval")


def compute_metrics(p):
    predictions, labels = p
    predictions = np.argmax(predictions, axis=2)
    true_labels = [[label_list[l] for l in label if l != -100] for label in labels]
    pred_labels = [
        [label_list[p] for (p, l) in zip(pred, label) if l != -100]
        for pred, label in zip(predictions, labels)
    ]
    results = metric.compute(
        predictions=pred_labels, references=true_labels, zero_division=0
    )
    return {
        "precision": results["overall_precision"],
        "recall": results["overall_recall"],
        "f1": results["overall_f1"],
        "accuracy": results["overall_accuracy"],
    }


results = []

for model_info in MODELS:
    model_name = model_info["name"]
    output_dir = model_info["output_dir"]
    os.makedirs(output_dir, exist_ok=True)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForTokenClassification.from_pretrained(
        model_name, num_labels=len(label_list), id2label=id2label, label2id=label2id
    )
    tokenized_dataset = dataset.map(
        lambda examples: tokenize_and_align_labels(examples, tokenizer), batched=True
    )
    training_args = TrainingArguments(
        output_dir=output_dir,
        eval_strategy="epoch",
        save_strategy="epoch",
        learning_rate=2e-5,
        per_device_train_batch_size=8,
        per_device_eval_batch_size=16,
        num_train_epochs=3,
        weight_decay=0.01,
        load_best_model_at_end=True,
        metric_for_best_model="f1",
        logging_dir=os.path.join(output_dir, "logs"),
        logging_steps=10,
    )
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_dataset["train"],
        eval_dataset=tokenized_dataset["test"],
        tokenizer=tokenizer,
        compute_metrics=compute_metrics,
    )
    trainer.train()
    eval_results = trainer.evaluate()
    inference_time = measure_inference_time(model, tokenizer, tokenized_dataset["test"])
    model_size_mb = sum(
        os.path.getsize(os.path.join(output_dir, f)) / (1024 * 1024)
        for f in os.listdir(output_dir)
        if f.endswith(".bin")
    )
    results.append(
        {
            "Model": model_name,
            "F1-Score": eval_results["eval_f1"],
            "Precision": eval_results["eval_precision"],
            "Recall": eval_results["eval_recall"],
            "Accuracy": eval_results["eval_accuracy"],
            "Inference Time (s/sample)": inference_time,
            "Model Size (MB)": model_size_mb,
        }
    )
    trainer.save_model(output_dir)
    tokenizer.save_pretrained(output_dir)

results_df = pd.DataFrame(results)
print("\nModel Comparison Results:")
print(results_df.to_string())
results_df.to_csv(RESULTS_FILE, index=False, encoding="utf-8")
best_model = results_df.loc[results_df["F1-Score"].idxmax()]
print(f"\nBest Model: {best_model['Model']}")
print(f"F1-Score: {best_model['F1-Score']:.4f}")
print(f"Inference Time: {best_model['Inference Time (s/sample)']:.4f} s/sample")
print(f"Model Size: {best_model['Model Size (MB)']:.2f} MB")
