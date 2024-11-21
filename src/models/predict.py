import mlflow
import pandas as pd

logged_model = 'src/models/mlartifacts/210001136261627048/a4c15c5934734298994e3ffe71cfdaf5/artifacts/model'

model = mlflow.pyfunc.load_model(logged_model)

df = pd.read_csv('data/processed/preprocessed.csv')
df = df.drop(columns=['lung_cancer', 'gender', 'age', 'gen_flag', 'generation', 'lung_cancer'])

predicted = model.predict(df)

df['predicted'] = predicted
df.to_csv('reports/predicted_data.csv', index=False)
