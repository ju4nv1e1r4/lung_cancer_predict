import mlflow
mlflow.set_tracking_uri('http://127.0.0.1:5000')
logged_model = 'mlflow-artifacts:/613581331439773821/e3f72759f04c47f8b0df660f796cf8d3/artifacts/model'

# Load model as a PyFuncModel.
loaded_model = mlflow.pyfunc.load_model(logged_model)

# Predict on a Pandas DataFrame.
import pandas as pd

data = pd.read_csv('../../data/processed/data_predict.csv')
predicted = loaded_model.predict(data)

data['predicted'] = predicted
data.to_csv('../../reports/diagnostic.csv')