"""Plotting Utils."""
import io

import matplotlib.pyplot as plt
import numpy.typing as npt
import seaborn as sns
from matplotlib.figure import Figure
from PIL import Image
from sklearn.metrics import confusion_matrix


def fig2img(fig: Figure) -> Image.Image:
    """Convert a Matplotlib figure to a PIL Image and return it.

    Parameters
    ----------
    fig
        Matplotlib figure.

    Returns
    -------
    Image.Image
        PIL image of the plot.
    """
    buf = io.BytesIO()
    fig.savefig(buf)
    buf.seek(0)
    img = Image.open(buf)
    return img


def draw_confusion_matrix(y_true: npt.NDArray, y_pred: npt.NDArray) -> Image.Image:
    """Draw a plot of a confusion matrix.

    Parameters
    ----------
    y_true : array-like of shape (n_samples,)
        Ground truth (correct) target values.

    y_pred : array-like of shape (n_samples,)
        Estimated targets as returned by a classifier.

    Returns
    -------
    Image.Image
        PIL image of the confusion matrix.
    """
    cm = confusion_matrix(y_true, y_pred)
    fig, axs = plt.subplots(1, 1, figsize=(5, 5))
    sns.heatmap(cm, ax=axs, annot=True, cmap="Blues")
    plt.ylabel("Actual label")
    plt.xlabel("Predicted label")
    plt.title("Confusion Matrix")
    return fig2img(fig)
