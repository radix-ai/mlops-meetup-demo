"""Example code to prepare data."""
from typing import Tuple

import numpy.typing as npt
import pandas as pd

ENCODING = {"setosa": 0, "versicolor": 1, "virginica": 2}


def load_data(data_path="data/iris.csv") -> pd.DataFrame:
    """Load data.

    Parameters
    ----------
    data_path
        Path to data to load.

    Returns
    -------
    DataFrame
        Loaded Data
    """
    return pd.read_csv(data_path)


def split_data(
    data: pd.DataFrame, split_ratio: float
) -> Tuple[npt.NDArray, npt.NDArray, npt.NDArray, npt.NDArray]:
    """Node for splitting the classical Iris data set into training and test sets.

    Test and train sets are each split into features and labels.
    The split ratio parameter is taken from conf/project/parameters.yml.
    The data and the parameters will be loaded and provided to your function
    automatically when the pipeline is executed and it is time to run this node.

    Parameters
    ----------
    data : pd.DataFrame
        Input dataframe
    split_ratio : float
        Ratio used to split train and test data.

    Returns
    -------
    Tuple
        train_x: pd.Dataframe - training dataset input features,
        train_y: pd.Dataframe - training dataset target labels,
        test_x: pd.Dataframe - test dataset input features,
        test_y: pd.Dataframe - test dataset target labels,
    """
    data.columns = [
        "sepal_length",
        "sepal_width",
        "petal_length",
        "petal_width",
        "target",
    ]

    # Shuffle all the data
    data = data.sample(frac=1).reset_index(drop=True)

    # Split to training and testing data
    n = data.shape[0]
    n_test = int(n * split_ratio)
    training_data = data.iloc[n_test:, :].reset_index(drop=True)
    test_data = data.iloc[:n_test, :].reset_index(drop=True)

    # Split the data to features and labels
    train_data_x = training_data.loc[:, "sepal_length":"petal_width"]  # type: ignore[misc]
    train_data_y = training_data["target"].apply(lambda x: ENCODING[x])
    test_data_x = test_data.loc[:, "sepal_length":"petal_width"]  # type: ignore[misc]
    test_data_y = test_data["target"].apply(lambda x: ENCODING[x])

    # When returning many variables, it is a good practice to give them names:
    return (
        train_data_x.to_numpy(),
        train_data_y.to_numpy().squeeze(),
        test_data_x.to_numpy(),
        test_data_y.to_numpy().squeeze(),
    )
