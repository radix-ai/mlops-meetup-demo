"""Register the model from the latest MLFlow run in BentoML."""
import logging

import bentoml
import mlflow

from mlops_demo.configuration import get_tracking_uri

logging.basicConfig(level=logging.INFO)
mlflow.set_tracking_uri(get_tracking_uri())

runs = mlflow.search_runs(experiment_ids="0", max_results=1)
RUN_ID = runs["run_id"][0]
MODEL_NAME = "iris_model"

# REGISTER MODEL IN BENTOML
model_uri = f"runs:/{RUN_ID}/{MODEL_NAME}"
bento_model = bentoml.mlflow.import_model(MODEL_NAME, model_uri)
logging.info("\nModel imported to BentoML: %s" % bento_model)
