import os
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from collections import defaultdict
import re
from transformers import AutoTokenizer, AutoModelForTokenClassification, pipeline
import logging

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "..", "data", "labeled")
DATA_DIR = os.path.abspath(DATA_DIR)
CONLL_FILE = os.path.join(DATA_DIR, "relabeled_data_20250622_232809.conll")
MODEL_DIR = os.path.join(
    os.path.dirname(__file__), "..", "..", "models", "xlm_roberta_ner"
)
MODEL_DIR = os.path.abspath(MODEL_DIR)
OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "..", "..", "outputs")
OUTPUT_DIR = os.path.abspath(OUTPUT_DIR)
os.makedirs(OUTPUT_DIR, exist_ok=True)
SCORECARD_FILE = os.path.join(OUTPUT_DIR, "vendor_scorecard.csv")


def load_conll(file_path):
    vendors = defaultdict(list)
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            current_sentence, current_labels = [], []
            current_vendor = "Unknown"
            current_views = 0
            current_timestamp = datetime.now()
            line_count = 0
            for line in f:
                line_count += 1
                line = line.strip()
                if not line:
                    if current_sentence:
                        vendors[current_vendor].append(
                            {
                                "text": " ".join(current_sentence),
                                "tokens": current_sentence,
                                "labels": current_labels,
                                "views": current_views,
                                "timestamp": current_timestamp,
                            }
                        )
                        logger.info(
                            f"Processed post for vendor {current_vendor} at line {line_count}"
                        )
                    current_sentence, current_labels = [], []
                    continue
                if line.startswith("#"):
                    if line.startswith("# vendor:"):
                        current_vendor = line[9:].strip() or "Unknown"
                    elif line.startswith("# views:"):
                        try:
                            current_views = int(line[8:].strip())
                        except ValueError:
                            logger.warning(
                                f"Invalid views format at line {line_count}: {line}"
                            )
                            current_views = 0
                    elif line.startswith("# timestamp:"):
                        try:
                            current_timestamp = datetime.strptime(
                                line[12:].strip(), "%Y-%m-%d %H:%M:%S"
                            )
                        except ValueError:
                            logger.warning(
                                f"Invalid timestamp format at line {line_count}: {line}"
                            )
                            current_timestamp = datetime.now()
                else:
                    try:
                        token, label = line.split()
                        current_sentence.append(token)
                        current_labels.append(label)
                    except ValueError:
                        logger.warning(
                            f"Invalid token/label format at line {line_count}: {line}"
                        )
                        continue
            if current_sentence:
                vendors[current_vendor].append(
                    {
                        "text": " ".join(current_sentence),
                        "tokens": current_sentence,
                        "labels": current_labels,
                        "views": current_views,
                        "timestamp": current_timestamp,
                    }
                )
                logger.info(
                    f"Processed final post for vendor {current_vendor} at line {line_count}"
                )
    except FileNotFoundError:
        logger.error(f"CoNLL file not found at {file_path}")
    if not vendors:
        logger.error("No vendor data loaded. Check CoNLL file format.")
    else:
        logger.info(f"Loaded data for {len(vendors)} vendors.")
    return vendors


def extract_entities(posts, tokenizer, model):
    ner_pipeline = pipeline(
        "ner", model=model, tokenizer=tokenizer, aggregation_strategy="simple"
    )
    for post in posts:
        try:
            ner_results = ner_pipeline(post["text"])
            price = None
            product = None
            for entity in ner_results:
                if entity["entity_group"] in ["B-PRICE", "I-PRICE"]:
                    price_text = (
                        entity["word"].replace(",", "").replace("ብር", "").strip()
                    )
                    try:
                        price = float(re.sub(r"[^\d.]", "", price_text))
                    except ValueError:
                        continue
                elif entity["entity_group"] in ["B-Product", "I-Product"]:
                    product = (
                        entity["word"]
                        if not product
                        else product + " " + entity["word"]
                    )
            post["price"] = price
            post["product"] = product
        except Exception as e:
            logger.warning(
                f"Failed to extract entities for post: {post['text']}. Error: {str(e)}"
            )
            post["price"] = None
            post["product"] = None
    return posts


def calculate_metrics(posts):
    if not posts:
        return 0, 0, None, 0
    timestamps = [post["timestamp"] for post in posts]
    views = [post["views"] for post in posts]
    prices = [post["price"] for post in posts if post["price"] is not None]
    earliest_date = min(timestamps)
    latest_date = max(timestamps)
    days_span = max(1, (latest_date - earliest_date).days + 1)
    weeks_span = days_span / 7
    posting_freq = len(posts) / weeks_span
    avg_views = np.mean(views) if views else 0
    avg_price = np.mean(prices) if prices else 0
    max_views_idx = np.argmax(views) if views else None
    top_post = posts[max_views_idx] if max_views_idx is not None else None
    return posting_freq, avg_views, top_post, avg_price


def lending_score(avg_views, posting_freq):
    normalized_views = min(avg_views / 1000, 10)
    normalized_freq = min(posting_freq, 10)
    score = (normalized_views * 0.5) + (normalized_freq * 0.5)
    return round(score, 2)


def main():
    tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR)
    model = AutoModelForTokenClassification.from_pretrained(MODEL_DIR)
    vendors = load_conll(CONLL_FILE)
    vendor_metrics = {}
    for vendor, posts in vendors.items():
        if posts:
            posts = extract_entities(posts, tokenizer, model)
            posting_freq, avg_views, top_post, avg_price = calculate_metrics(posts)
            vendor_metrics[vendor] = {
                "avg_views": avg_views,
                "posting_freq": posting_freq,
                "avg_price": avg_price,
                "lending_score": lending_score(avg_views, posting_freq),
                "top_post_text": top_post["text"] if top_post else "None",
                "top_post_product": (
                    top_post["product"] if top_post and top_post["product"] else "N/A"
                ),
                "top_post_price": (
                    top_post["price"] if top_post and top_post["price"] else "N/A"
                ),
                "top_post_views": top_post["views"] if top_post else 0,
            }
            logger.info(
                f"Calculated metrics for vendor {vendor}: Posts/Week={posting_freq:.2f}, Avg Views={avg_views:.2f}, Avg Price={avg_price:.2f}"
            )
    if not vendor_metrics:
        logger.error("No vendor metrics calculated. Exiting.")
        return
    df = pd.DataFrame.from_dict(vendor_metrics, orient="index")[
        [
            "avg_views",
            "posting_freq",
            "avg_price",
            "lending_score",
            "top_post_text",
            "top_post_product",
            "top_post_price",
            "top_post_views",
        ]
    ]
    df.to_csv(SCORECARD_FILE, index_label="Vendor")
    print(f"Vendor scorecard saved: {SCORECARD_FILE}")
    for vendor, metrics in vendor_metrics.items():
        print(f"\nVendor: {vendor}")
        print(f"Average Views/Post: {metrics['avg_views']:.2f}")
        print(f"Posts/Week: {metrics['posting_freq']:.2f}")
        print(f"Average Price (ETB): {metrics['avg_price']:.2f}")
        print(f"Lending Score: {metrics['lending_score']:.2f}")
        print(f"Top Post: {metrics['top_post_text']}")
        print(f"Top Post Product: {metrics['top_post_product']}")
        print(f"Top Post Price: {metrics['top_post_price']}")
        print(f"Top Post Views: {metrics['top_post_views']}")


if __name__ == "__main__":
    main()
