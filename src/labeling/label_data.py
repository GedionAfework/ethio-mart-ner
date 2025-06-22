import pandas as pd
import os
from datetime import datetime


def load_and_sample_data(input_file, n_samples=50):
    df = pd.read_csv(input_file)
    sampled_df = df[df["Tokens"].notna()].sample(
        n=min(n_samples, len(df)), random_state=42
    )
    messages = sampled_df["Tokens"].apply(eval).tolist()
    return messages


def save_conll(messages, output_file):
    with open(output_file, "w", encoding="utf-8") as f:
        for message in messages:
            for token in message:
                f.write(f"{token} O\n")
            f.write("\n")


def main():
    input_file = "data/processed/processed_data_20250622_231744.csv"
    output_dir = "data/labeled"
    os.makedirs(output_dir, exist_ok=True)
    output_file = os.path.join(
        output_dir, f'labeled_data_{datetime.now().strftime("%Y%m%d_%H%M%S")}.conll'
    )

    messages = load_and_sample_data(input_file, n_samples=50)
    save_conll(messages, output_file)
    print(f"CoNLL file saved to {output_file}. Please manually label the tokens.")


if __name__ == "__main__":
    main()
