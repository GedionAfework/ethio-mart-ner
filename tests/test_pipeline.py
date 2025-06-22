import pytest
import pandas as pd
import os
import tempfile
from src.preprocessing.preprocess import normalize_amharic, tokenize_amharic
from src.labeling.label_data import load_and_sample_data, save_conll


def test_normalize_amharic():
    assert normalize_amharic("ሰላላ!!! test 123") == "ሰላላ 123"
    assert normalize_amharic(None) == ""
    assert normalize_amharic("አማርኛ   text  with  spaces") == "አማርኛ text with spaces"


def test_tokenize_amharic():
    assert tokenize_amharic("ሰላላ አማርኛ 123") == ["ሰላላ", "አማርኛ", "123"]
    assert tokenize_amharic("") == []
    assert tokenize_amharic(None) == []


def test_preprocess_and_label():
    data = pd.DataFrame(
        {
            "Message Text": ["ሰላላ አማርኛ 123"],
            "Channel Title": ["Test Channel"],
            "Channel Username": ["@test"],
            "Message ID": [1],
            "Timestamp": ["2023-01-01"],
            "Views": [100],
            "Media Path": [None],
        }
    )
    with tempfile.NamedTemporaryFile(
        mode="w", delete=False, encoding="utf-8", suffix=".csv"
    ) as tmp_file:
        data.to_csv(tmp_file.name, index=False)

    messages = load_and_sample_data(tmp_file.name, n_samples=1)
    assert len(messages) == 1
    assert isinstance(messages[0], list)

    with tempfile.NamedTemporaryFile(
        mode="w", delete=False, encoding="utf-8", suffix=".conll"
    ) as conll_file:
        save_conll(messages, conll_file.name)
        with open(conll_file.name, "r", encoding="utf-8") as f:
            content = f.read()
            assert "ሰላላ O\n" in content
            assert "\n\n" in content

    os.remove(tmp_file.name)
    os.remove(conll_file.name)
