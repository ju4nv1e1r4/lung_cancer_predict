import pytest
from unittest.mock import patch
from sklearn.metrics import precision_score
from sklearn.ensemble import AdaBoostClassifier
from imblearn.under_sampling import RandomUnderSampler
from train import parse_arg
import pandas as pd
from sklearn.model_selection import train_test_split


@pytest.fixture
def mock_data():
    df = pd.read_csv('data/processed/preprocessed.csv')
    X = df.drop(columns=['lung_cancer', 'gender', 'age', 'gen_flag', 'generation'])
    y = df['lung_cancer']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)
    rus = RandomUnderSampler(random_state=42)
    X_res, y_res = rus.fit_resample(X_train, y_train)
    return X_res, X_test, y_res, y_test


def test_parse_arg():
    test_args = [
        "train.py", "--n-estimators", "50", 
        "--learning-rate", "0.1", 
        "--algorithm", "SAMME"
    ]

    with patch('sys.argv', test_args):
        args = parse_arg()
        assert isinstance(args.n_estimators, int)
        assert args.learning_rate > 0
        assert args.algorithm in ['SAMME', 'SAMME.R']


def test_model_training(mock_data):
    X_res, X_test, y_res, y_test = mock_data
    model = AdaBoostClassifier(n_estimators=100, learning_rate=0.5, algorithm='SAMME.R')
    model.fit(X_res, y_res)
    y_pred = model.predict(X_test)
    precision = precision_score(y_test, y_pred)
    assert precision > 0.75
