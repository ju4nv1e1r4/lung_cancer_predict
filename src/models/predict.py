import mlflow
import pandas as pd

logged_model = 'src/models/mlartifacts/210001136261627048/2dbfe78657a34a92993e6da8428f9d02/artifacts/model'

model = mlflow.pyfunc.load_model(logged_model)

df = pd.read_csv('data/processed/preprocessed.csv')
df = df.drop(columns=['lung_cancer', 'gender', 'age', 'gen_flag', 'generation', 'lung_cancer'])

predicted = model.predict(df)

df['predicted'] = predicted
df.to_csv('reports/predicted_data.csv', index=False)
