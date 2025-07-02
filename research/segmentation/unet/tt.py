import mlflow
import os

from pathlib import Path

uri = "http://172.30.1.149:5000"
mlflow.set_tracking_uri(uri)
mlflow.set_experiment("New_TestExperiment7")
with mlflow.start_run(run_name="Test") as run:
    mlflow.log_param("test", "true")
    print("uri : ", mlflow.get_artifact_uri())
    artifacts = "outputs/checkpoints"
    print(os.path.exists(artifacts))
    mlflow.log_artifacts(local_dir=artifacts)
