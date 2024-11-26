import pytest
from unittest.mock import patch
from sklearn.datasets import make_classification
from sklearn.metrics import recall_score
from sklearn.ensemble import AdaBoostClassifier
from imblearn.under_sampling import RandomUnderSampler
from models.train_model import parse_arg
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
        assert args.learning_rate > 0, 'Hyperparameter "learning_rate" should be positive.'
        assert args.algorithm in ['SAMME', 'SAMME.R'], 'Hyperparameter "algorithm" should be "SAMME OR SAMME.R".'


def test_fit_hasattr():
    X, y = make_classification(n_samples=309, n_features=20, random_state=42)
    model = AdaBoostClassifier(random_state=42)
    fit_ = model.fit(X, y)
    assert hasattr(fit_, "coef_"), 'Model should have attributes after training.'


def test_model_training(mock_data):
    X_res, X_test, y_res, y_test = mock_data
    model = AdaBoostClassifier(n_estimators=100, learning_rate=0.5, algorithm='SAMME')
    model.fit(X_res, y_res)
    y_pred = model.predict(X_test)
    precision = recall_score(y_test, y_pred)
    assert precision > 0.75, f'Precision must be better than 75%. Actual precision is {precision}'


def test_model_prediction():
    X, y = make_classification(n_samples=309, n_features=20, random_state=42)
    model = AdaBoostClassifier(random_state=42).fit(X, y)
    y_pred = model.predict(X)
    assert set(y_pred) <= {0, 1}, 'Predictions should 0 or 1'
