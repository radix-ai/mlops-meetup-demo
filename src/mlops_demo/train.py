"""Code to train a new model."""
import logging

import bentoml  # type: ignore[syntax]
import mlflow
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

from mlops_demo.configuration import configure_mlflow
from mlops_demo.data import load_data, split_data
from mlops_demo.utils import draw_confusion_matrix

logging.basicConfig(level=logging.INFO)

# MLFLOW SETUP
run_id = configure_mlflow()

# PARAMS
C = 1
logging.info(f"Parameters: C={C}")
mlflow.log_params({"C": C})

# LOAD DATA
data = load_data()
x_train, y_train, x_test, y_test = split_data(data, split_ratio=0.5)

# TRAIN
model: LogisticRegression = LogisticRegression(random_state=0, C=C, solver="liblinear").fit(
    x_train, y_train
)
y_train_pred = model.predict(x_train)
train_acc = accuracy_score(y_train, y_train_pred)

logging.info(f"Training Accuracy: {train_acc:.3f}")
mlflow.log_metric("training_accuracy", train_acc)

# SCORE
y_test_pred = model.predict(x_test)
test_acc = accuracy_score(y_test, y_test_pred)
logging.info(f"Test Accuracy: {test_acc:.3f}")
mlflow.log_metric("test_accuarcy", test_acc)

confusion_matrix = draw_confusion_matrix(y_test, y_test_pred)
mlflow.log_image(confusion_matrix, "confusion_matrix.png")

model_name = "iris_model"
mlflow.sklearn.log_model(model, model_name, registered_model_name=model_name)

# # REGISTER MODEL IN BENTOML
model_uri = f"runs:/{run_id}/{model_name}"
bento_model = bentoml.mlflow.import_model(model_name, model_uri)
logging.info("\nModel imported to BentoML: %s" % bento_model)
