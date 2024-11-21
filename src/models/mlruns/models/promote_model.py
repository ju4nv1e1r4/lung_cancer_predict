from mlflow.tracking import MlflowClient

client = MlflowClient()

model_name = 'AdaBoost-Precision95'
version = 1

client.transition_model_version_stage(
    name=model_name,
    version=version,
    stage='Staging',
    archive_existing_versions=True,
)
