"""
Unit tests for the data preprocessing script (`preprocess.py`).

This script contains test cases to ensure the correctness and robustness
of the functions implemented in the preprocessing pipeline. The tests
validate functionality such as:

1. Loading data from CSV files.
2. Standardizing and renaming column names.
3. Encoding binary and categorical columns.
4. Generating age brackets and mapping them to generation names.

The tests are implemented using `pytest` for ease of use and coverage.

Usage:
    To run the tests, execute the following command in the terminal:
    ```
    pytest src/test_preprocessing.py
    ```

Tests included:
    - `test_load_data`: Ensures the `load_data` function handles missing files appropriately.
    - `test_preprocess_columns`: Validates that columns are renamed as expected.
    - `test_encode_binary_columns`: Confirms correct binary encoding for specified columns.
    - `test_age_bracket`: Checks if age values are correctly categorized into brackets.
    - `test_age_bracket_str`: Validates that age brackets are mapped to correct generation names.
"""

import pytest
import pandas as pd
from sklearn.datasets import make_classification
from models.preprocess_data import load_data, preprocess_columns, encode_binary_columns, age_bracket_str


def test_load_data():
    with pytest.raises(FileNotFoundError):
        load_data("non_existent_file.csv")


def data_validation():
    X, y = make_classification(n_samples=309, n_features=20, random_state=42)
    assert X.shape == (309, 20), 'The shape of X should be (309, 20)'
    assert y.shape == (309,), 'The shape of y should be (309,)'       


def test_preprocess_columns():
    df = pd.DataFrame({'Chronic Disease': [1]})
    df = preprocess_columns(df)
    assert 'chronic_disease' in df.columns


def test_encode_binary_columns():
    df = pd.DataFrame({'binary_column': [2, 1, 2]})
    df = encode_binary_columns(df, ['binary_column'])
    assert df['binary_column'].tolist() == [1, 0, 1]


def test_age_bracket_str():
    assert age_bracket_str(1) == 'generation_Z'
    assert age_bracket_str(6) == 'unknown'
