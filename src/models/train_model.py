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
import logging
import mlflow

from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

structlog.configure(
    processors=[
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.processors.JSONRenderer()
    ],
    logger_factory=structlog.stdlib.LoggerFactory(),
    wrapper_class=structlog.stdlib.BoundLogger,
    cache_logger_on_first_use=True,
)

logging.basicConfig(
    filename="reports/logs/training_logs.json",
    level=logging.INFO,
    format="%(message)s"
)

logger = structlog.get_logger()

def parse_arg():
    parser = argparse.ArgumentParser(description='Lung Cancer Model Predictor')

    parser.add_argument(
        '--random-state',
        type=int,
        default=42,
        help='Random control.')

    parser.add_argument(
        '--splitter',
        type=str,
        default='best',
        help="The maximum number of estimators at which boosting is terminated."
    )

    parser.add_argument(
        '--max-features',
        type=str,
        default='sqrt',
        help="Use the SAMME discrete boosting algorithm."
    )

    parser.add_argument(
        '--max-depth',
        type=int,
        default=None,
        help="Use the SAMME discrete boosting algorithm."
    )

    parser.add_argument(
        '--criterion',
        type=str,
        default='gini',
        help="Use the SAMME discrete boosting algorithm."
    )

    parser.add_argument(
        '--run-name',
        type=str,
        default='Train-Classifier',
        help="Offers the option to rename the run name of experiment"
    )

    return parser.parse_args()


df = pd.read_csv('data/processed/preprocessed.csv')

X = df.drop(columns=['lung_cancer', 'gender', 'age', 'gen_flag', 'generation'])
y = df['lung_cancer']
args = parse_arg()
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.35, random_state=args.random_state)

def main():
    args = parse_arg()

    params = {'splitter': args.splitter,
              'max_features': args.max_features,
              'max_depth': args.max_depth,
              'criterion': args.criterion
    }

    logger.info("Training started", params=params)

    mlflow.set_tracking_uri('http://127.0.0.1:5000')
    mlflow.set_experiment('DT-6040')

    with mlflow.start_run(run_name=args.run_name):
        mlflow.sklearn.autolog()
        model = DecisionTreeClassifier(**params)
        try:
            model.fit(X_train, y_train)
            logger.info("Model training completed successfully")
        except Exception as e:
            logger.error("Error during model training", error=str(e))
            raise

        logger.info("Model Trained", model='AdaBoostClassifier', params=params)

        mlflow.sklearn.log_model(model, 'AdaBoostClassifier')
        y_pred = model.predict(X_test)

        accuracy = accuracy_score(y_test, y_pred)
        f1score = f1_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred)
        recall = recall_score(y_test, y_pred)
        cm = confusion_matrix(y_test, y_pred)

        tp,  fn , fp, tn = cm[0, 0], cm[0, 1], cm[1, 0], cm[1, 1]

        tpr = tp / (tp + fn)
        fpr = fp / (fp + tn)

        metrics = {
            "accuracy": accuracy,
            "f1_score": f1score,
            "precision": precision,
            "recall": recall,
            'TPR': tpr,
            'FPR': fpr
        }

        logger.info("Metrics calculated", metrics=metrics)

        print('>> Accuracy: {}'.format(accuracy))
        print('>> F1 Score: {}'.format(f1score))
        print('>> Precision: {}'.format(precision))
        print('>> Recall: {}'.format(recall))
        print('>> FPR: {}'.format(fpr))
        print('>> TPR: {}'.format(tpr))

        mlflow.log_metric('Accuracy Score', accuracy)
        mlflow.log_metric('F1 Score', f1score)
        mlflow.log_metric('Precision Score', precision)
        mlflow.log_metric('Recall Score', recall)
        mlflow.log_metric('TPR', tpr)
        mlflow.log_metric('FPR', fpr)


if __name__ == '__main__':
    main()
