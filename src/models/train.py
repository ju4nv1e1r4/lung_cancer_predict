import pandas as pd
import numpy as np

import argparse
import mlflow

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score

def parse_arg():
    parser = argparse.ArgumentParser(description='Lung Cancer Model Predictor')

    parser.add_argument(
        '--max-depth',
        type=int,
        default=5,
        help="The maximum depth of the tree."
    )

    parser.add_argument(
        '--n-estimators',
        type=int,
        default=100,
        help="The number of trees in the forest."
    )

    parser.add_argument(
        '--random-state',
        type=int,
        default=42,
        help="""
    Controls both the randomness of the bootstrapping of the samples used when 
    building trees (if bootstrap=True) and the sampling of the features to consider 
    when looking for the best split at each node (if max_features < n_features).
    """
    )
    return parser.parse_args()


df = pd.read_csv('../../data/processed/processed_data_lung_cancer.csv')
X = df.drop('lung_cancer', axis=1)
y = df['lung_cancer']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)


def main():
    args = parse_arg()

    rf_params = {
        'max_depth': args.max_depth,
        'n_estimators': args.n_estimators,
        'random_state': args.random_state
    }

    mlflow.set_tracking_uri('http://127.0.0.1:5000')
    mlflow.set_experiment('lung-cancer-predict-model')

    with mlflow.start_run(run_name='Random Forest Classifier'):
        mlflow.sklearn.autolog()
        random_forest = RandomForestClassifier(**rf_params)

        random_forest.fit(X_train, y_train)
        mlflow.sklearn.log_model(random_forest,'Random Forest Classifier',)
        y_pred_test = random_forest.predict(X_test)

        rf_accuracy = accuracy_score(y_test, y_pred_test)
        rf_f1score = f1_score(y_test, y_pred_test, average='binary')
        mlflow.log_metric('Random Forest Classifier: Accuracy', rf_accuracy)
        mlflow.log_metric('Random Forest Classifier: F1 Score', rf_f1score)


if __name__ == '__main__':
    main()