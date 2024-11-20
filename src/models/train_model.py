"""
This script trains an AdaBoostClassifier model to predict lung cancer probability
based on preprocessed data. It includes the following steps:

1. Parse command-line arguments for model hyperparameters.
2. Load and preprocess data.
3. Perform random undersampling to handle data imbalance.
4. Train the AdaBoostClassifier model using scikit-learn.
5. Log metrics (accuracy, F1 score, precision, recall) and parameters to MLflow.
6. Log structured events and metrics using structlog.

Command-line Arguments:
- --n-estimators: Number of estimators for the AdaBoost model (default: 100).
- --learning-rate: Learning rate for the AdaBoost model (default: 0.5).
- --algorithm: Boosting algorithm to use (default: 'SAMME').

Dependencies:
- pandas
- sklearn
- imblearn
- mlflow
- structlog

"""

import pandas as pd

import argparse
import structlog
import mlflow

from sklearn.model_selection import train_test_split
from imblearn.under_sampling import RandomUnderSampler
from sklearn.ensemble import AdaBoostClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

structlog.configure(
    processors=[
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.processors.JSONRenderer()
    ]
)
logger = structlog.get_logger()

n=0
n+=1

def parse_arg():
    parser = argparse.ArgumentParser(description='Lung Cancer Model Predictor')

    parser.add_argument(
        '--n-estimators',
        type=int,
        default=100,
        help="The maximum number of estimators at which boosting is terminated."
    )

    parser.add_argument(
        '--learning-rate',
        type=float,
        default=.5,
        help="""
        Weight applied to each classifier at each boosting iteration.
        A higher learning rate increases the contribution of each classifier. 
        There is a trade-off between the learning_rate and n_estimators parameters.
        """
    )

    parser.add_argument(
        '--algorithm',
        type=str,
        default='SAMME',
        help="Use the SAMME discrete boosting algorithm."
    )
    parser.add_argument(
        '--run-name',
        type=str,
        default='Run-Classifier-{}'.format(n),
        help="Offers the option to rename the run name of experiment"
    )

    return parser.parse_args()


df = pd.read_csv('data/processed/preprocessed.csv')

X = df.drop(columns=['lung_cancer', 'gender', 'age', 'gen_flag', 'generation'])
y = df['lung_cancer']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.3, random_state=42)
rus = RandomUnderSampler(random_state=42)
X_res, y_res = rus.fit_resample(X_train, y_train)

def main():
    args = parse_arg()

    params = {
        'n_estimators': args.n_estimators,
        'learning_rate': args.learning_rate,
        'algorithm': args.algorithm
    }

    logger.info("Training started", params=params)

    mlflow.set_tracking_uri('http://127.0.0.1:5000')
    mlflow.set_experiment('AdaBoost-prototype')

    with mlflow.start_run(run_name=args.run_name):
        mlflow.sklearn.autolog()
        model = AdaBoostClassifier(**params)
        model.fit(X_res, y_res)

        logger.info("Model Trained", model='AdaBoostClassifier', params=params)

        mlflow.sklearn.log_model(model,'AdaBoostClassifier',)
        y_pred = model.predict(X_test)

        accuracy = accuracy_score(y_test, y_pred)
        f1score = f1_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred)
        recall = recall_score(y_test, y_pred)

        metrics = {
            "accuracy": accuracy,
            "f1_score": f1score,
            "precision": precision,
            "recall": recall
        }

        logger.info("Metrics calculated", metrics=metrics)

        print('>> Accuracy: {}'.format(accuracy))
        print('>> F1 Score: {}'.format(f1score))
        print('>> Precision: {}'.format(precision))
        print('>> Recall: {}'.format(recall))

        mlflow.log_metric('Accuracy Score', accuracy)
        mlflow.log_metric('F1 Score', f1score)
        mlflow.log_metric('Precision Score', precision)
        mlflow.log_metric('Recall Score', recall)


if __name__ == '__main__':
    main()
