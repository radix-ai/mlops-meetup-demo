"""BentML Service for iris model."""
import bentoml
import numpy as np
from bentoml.io import JSON
from pydantic import BaseModel

DECODING = {0: "setosa", 1: "versicolor", 2: "virginica"}

iris_clf_runner = bentoml.mlflow.get("iris_model:latest").to_runner()

svc = bentoml.Service("iris_service", runners=[iris_clf_runner])


class Iris(BaseModel):
    """Service input."""

    sepal_length: float
    sepal_width: float
    petal_length: float
    petal_width: float


class Response(BaseModel):
    """Service result."""

    iris_type: str


@svc.api(input=JSON(pydantic_model=Iris), output=JSON(pydantic_model=Response))
def classify(iris: Iris) -> Response:
    """Classify an iris based on sepal_length, sepal_width, petal_length and petal_width.

    Parameters
    ----------
    iris
        The sepal_length, sepal_width, petal_length and petal_width of an Iris

    Returns
    -------
    Response
        Response containing the iris_type.
    """
    input_array = np.array(list(iris.dict().values())).reshape(1, -1)
    result = iris_clf_runner.predict.run(input_array)
    return {"iris_type": DECODING[int(result)]}
