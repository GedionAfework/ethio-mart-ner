# EthioMart NER Project

A project to develop a Named Entity Recognition (NER) system for extracting product names, prices, and locations from Amharic text in Ethiopian Telegram e-commerce channels.

## Setup

1.  Clone the repository:
    ```bash
    git clone https://github.com/GedionAfework/ethio-mart-ner.git
    cd ethio-mart-ner
    ```
2.  Create and activate a virtual environment:
    ```bash
    python -m venv .venv
    source venv/bin/activate  # On Windows: .venv\Scripts\activate
    ```
3.  Install dependencies:
    ```bash
    pip install -r requirements.txt
    ```
4.  Set up environment variables in `.env`:
    ```text
    TG_API_ID=your_api_id
    TG_API_HASH=your_api_hash
    PHONE=your_phone_number
    ```

## Structure

- `data/`: Stores raw, processed, and labeled data.
- `src/`: Contains source code for ingestion, preprocessing, and labeling.
- `tests/`: Contains unit tests.
- `.github/workflows/`: Contains CI/CD configuration.
- `requirements.txt`: Lists project dependencies.

## Tasks

- **Task 1**: Data ingestion and preprocessing from Telegram channels (branch: `task-1`).
- **Task 2**: Labeling data in CoNLL format for NER (branch: `task-2`).

## Running Task 1 (Data Ingestion and Preprocessing)

1.  **Data Ingestion**:
    ```bash
    python src/ingestion/telegram_scraper.py
    ```
    - Fetches messages from specified Telegram channels.
    - Saves data to `data/raw/`.
2.  **Data Preprocessing**:
    ```bash
    python src/preprocessing/preprocess.py
    ```
    - Cleans and tokenizes Amharic text.
    - Saves processed data to `data/processed/`.

## Running Task 2 (Data Labeling)

1.  **Data Labeling**:
    ```bash
    python src/labeling/label_data.py
    ```
    - Generates a CoNLL file with sampled messages.
    - Manually label the file and save as `data/labeled/labeled_data_final.conll`.

## CI/CD Workflow

- A GitHub Actions workflow (`.github/workflows/ci.yml`) runs on push/pull to `main`:
  - Lints code with `flake8`.
  - Runs tests with `pytest` for preprocessing and labeling.
- To add tests, update `tests/test_pipeline.py`.
- Ensure all dependencies are listed in `requirements.txt`.
