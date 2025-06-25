import os
import numpy as np
import pandas as pd
import torch
from transformers import (
    AutoTokenizer,
    AutoModelForTokenClassification,
    pipeline,
)
import shap
from lime.lime_text import LimeTextExplainer
from datasets import Dataset

MODEL_DIR = os.path.join(
    os.path.dirname(__file__), "..", "..", "models", "xlm_roberta_ner"
)
MODEL_DIR = os.path.abspath(MODEL_DIR)
DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "..", "data", "labeled")
DATA_DIR = os.path.abspath(DATA_DIR)
CONLL_FILE = os.path.join(DATA_DIR, "relabeled_data_20250622_232809.conll")
OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "..", "..", "outputs")
OUTPUT_DIR = os.path.abspath(OUTPUT_DIR)
os.makedirs(OUTPUT_DIR, exist_ok=True)

label_list = ["O", "B-Product", "I-Product", "B-LOC", "I-LOC", "B-PRICE", "I-PRICE"]


def load_conll(file_path):
    sentences, labels = [], []
    current_sentence, current_labels = [], []
    with open(file_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    token, label = line.split()
                    current_sentence.append(token)
                    current_labels.append(label)
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
    return {"tokens": sentences, "labels": labels}


data = Dataset.from_dict(load_conll(CONLL_FILE))
test_samples = data.shuffle(seed=42).select(range(5))

tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR)
model = AutoModelForTokenClassification.from_pretrained(MODEL_DIR)
ner_pipeline = pipeline(
    "ner", model=model, tokenizer=tokenizer, aggregation_strategy="simple"
)


def predict_proba(texts):
    if isinstance(texts, str):
        texts = [texts]
    max_len = 128
    fixed_output_len = max_len * len(label_list)
    probs_list = []
    for text in texts:
        tokens = text.split()
        num_tokens = min(len(tokens), max_len)
        if num_tokens == 0:
            probs_list.append(np.zeros(fixed_output_len))
            continue
        inputs = tokenizer(
            tokens,
            return_tensors="pt",
            truncation=True,
            padding="max_length",
            max_length=max_len,
            is_split_into_words=True,
        )
        with torch.no_grad():
            outputs = model(**inputs)
        probs = torch.softmax(outputs.logits, dim=-1).squeeze(0).numpy()
        word_ids = inputs.word_ids()
        token_probs = []
        current_word = None
        current_probs = []
        for i, word_id in enumerate(word_ids):
            if word_id is None:
                continue
            if word_id != current_word and current_word is not None:
                token_probs.append(np.mean(current_probs, axis=0))
                current_probs = []
            current_probs.append(probs[i])
            current_word = word_id
        if current_probs:
            token_probs.append(np.mean(current_probs, axis=0))
        word_probs = np.zeros((max_len, len(label_list)))
        for i, prob in enumerate(token_probs):
            if i < num_tokens:
                word_probs[i] = prob
        probs_list.append(word_probs.flatten()[:fixed_output_len])
    return np.array(probs_list)


explainer_shap = shap.Explainer(
    predict_proba, masker=shap.maskers.Text(tokenizer, mask_token=tokenizer.pad_token)
)
shap_values_list = []
for sample in test_samples["tokens"]:
    text = " ".join(sample)
    shap_values = explainer_shap([text])
    shap_values_list.append(shap_values)

lime_explainer = LimeTextExplainer(class_names=label_list)
lime_explanations = []
for sample in test_samples["tokens"]:
    text = " ".join(sample)
    try:
        exp = lime_explainer.explain_instance(
            text, predict_proba, num_features=5, num_samples=100
        )
        lime_explanations.append(exp)
    except Exception as e:
        print(
            f"Failed to generate LIME explanation for sample: {text}. Error: {str(e)}"
        )
        lime_explanations.append(None)

for idx, (sample, shap_vals, lime_exp) in enumerate(
    zip(test_samples["tokens"], shap_values_list, lime_explanations)
):
    print(f"\nSample {idx + 1}: {' '.join(sample)}")
    print("\nSHAP Explanation:")
    num_tokens = len(sample)
    for i, label in enumerate(label_list):
        print(f"Label {label}:")
        try:
            shap_vals_label = shap_vals.values[0][:num_tokens, i]
        except (IndexError, ValueError):
            shap_vals_label = np.zeros(num_tokens)
        for token, value in zip(sample, shap_vals_label):
            print(f"  {token}: {value:.4f}")
    print("\nLIME Explanation:")
    if lime_exp is not None:
        lime_exp.save_to_file(os.path.join(OUTPUT_DIR, f"lime_sample_{idx + 1}.html"))
        print(f"[Saved to outputs/lime_sample_{idx + 1}.html]")
    else:
        print("LIME explanation failed for this sample.")
    with open(
        os.path.join(OUTPUT_DIR, f"shap_sample_{idx + 1}.txt"), "w", encoding="utf-8"
    ) as f:
        for i, label in enumerate(label_list):
            f.write(f"Label {label}:\n")
            try:
                shap_vals_label = shap_vals.values[0][:num_tokens, i]
            except (IndexError, ValueError):
                shap_vals_label = np.zeros(num_tokens)
            for token, value in zip(sample, shap_vals_label):
                f.write(f"  {token}: {value:.4f}\n")
