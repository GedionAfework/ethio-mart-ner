import pandas as pd
import regex as re
import os
from datetime import datetime


def normalize_amharic(text):
    if not isinstance(text, str):
        return ""
    text = re.sub(r"\s+", " ", text.strip())
    text = re.sub(r"[^\u1200-\u137F0-9\s]", "", text)
    return text


def tokenize_amharic(text):
    if not isinstance(text, str):
        return []
    tokens = text.split()
    return tokens


def preprocess_data(input_file, output_dir):
    data = pd.read_csv(input_file)

    data["Cleaned_Message"] = data["Message Text"].apply(normalize_amharic)

    data["Tokens"] = data["Cleaned_Message"].apply(tokenize_amharic)

    os.makedirs(output_dir, exist_ok=True)
    output_file = os.path.join(
        output_dir, f'processed_data_{datetime.now().strftime("%Y%m%d_%H%M%S")}.csv'
    )
    data.to_csv(output_file, index=False, encoding="utf-8")

    print(f"Processed data saved to {output_file}")


if __name__ == "__main__":
    input_file = "data/raw/telegram_data_20250622_224738.csv"
    output_dir = "data/processed"
    preprocess_data(input_file, output_dir)
