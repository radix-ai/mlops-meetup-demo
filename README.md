# MLOps Meetup Demo

This repo demonstrates how to take the basic model training code in `src/mlops_demo/train_basic.py` and add MLOps tools to it to track experiments and package the resulting model into a deployable docker container. This is done using [MLFlow](https://www.mlflow.org/docs/latest/tracking.html) and [BentoML](https://www.bentoml.com/).

# Open this demo in your browser

1. Open [https://github.com/radix-ai/mlops-meetup-demo](https://github.com/radix-ai/mlops-meetup-demo) in your browser.
2. Click on _Code_ and select _Create codespace_ to start a Dev Container with [GitHub Codespaces](https://github.com/features/codespaces).

# Tasks


## Start a MLFlow Tracking server

To track runs with MLFlow we need to run an MLFLow tracking server. The following command runs a local server.
```
mlflow server \
    --backend-store-uri sqlite:///mlflow.db \
    --default-artifact-root file:///app/mlflow/ \
    --host 127.0.0.1
```

## Train a new model

With the MLFlow server running run the following.

```
poetry run python src/mlops_demo/train.py
```

## Register latest model in BentoML
Register th latest model from MLFlow in BentoML so that it can be used by the BentoML service.

```
python /app/src/mlops_demo/register_model.py
```

## Run the model in a local BentoML Service
We can run the service we defined in `src/mlops_demo/service.py` and the `iris_model` we registered in `train.py` as follows.

```
bentoml serve src.mlops_demo.service:svc --reload --port=8000
```

## Build a deployment ready docker image with BentoML

This uses the service defined in `src/mlops_demo/service.py` and the `iris_model` we registered in `train.py` to build a deployable docker image. Additional config is defined in `bentofile.yaml`.

1. Build a bento.
```
bentoml build
```
2. Turn it into docker container.
```
bentoml containerize iris_service:latest
```