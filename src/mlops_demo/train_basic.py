"""Code to train a new model without MLFlow and BentoML."""
import logging

from joblib import dump
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix

from mlops_demo.data import load_data, split_data

logging.basicConfig(level=logging.INFO)

# PARAMS
C = 1
logging.info(f"Parameters: C={C}")

# LOAD DATA
data = load_data()
x_train, y_train, x_test, y_test = split_data(data, split_ratio=0.5)

# TRAIN
model: LogisticRegression = LogisticRegression(random_state=0, C=C, solver="liblinear").fit(
    x_train, y_train
)

# SCORE
# TRAIN ACCURACY
y_train_pred = model.predict(x_train)
train_acc = accuracy_score(y_train, y_train_pred)
logging.info(f"Training Accuracy: {train_acc:.3f}")

# TEST ACCURACY
y_test_pred = model.predict(x_test)
test_acc = accuracy_score(y_test, y_test_pred)
logging.info(f"Test Accuracy: {test_acc:.3f}")

# Confusion Matrix
logging.info(f"Confusion Matrix\n{confusion_matrix(y_test, y_test_pred)}")

model_name = "iris_model"
dump(model, f"{model_name}.pickle")
