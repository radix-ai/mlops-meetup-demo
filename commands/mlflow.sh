
mlflow server \
    --backend-store-uri sqlite:///mlflow.db \
    --default-artifact-root file:///app/mlflow/ \
    --host 127.0.0.1:5050