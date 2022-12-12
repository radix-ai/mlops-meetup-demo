"""Setup configuration for mlfow."""
import mlflow

TRACKING_URI = "http://127.0.0.1:5000"


def get_tracking_uri() -> str:
    """Get the MLFlow tracking URI."""
    return TRACKING_URI


def configure_mlflow_autolog() -> str:
    """Configure MLFlow with autologging enabled.

    Returns
    -------
    str
        Run id of the started mlflow run.
    """
    mlflow.set_tracking_uri(get_tracking_uri())
    mlflow.start_run()
    mlflow.sklearn.autolog()  # registered_model_name="iris"
    return mlflow.active_run().info.run_id


def configure_mlflow() -> str:
    """Configure MLFlow.

    Returns
    -------
    str
        Run id of the started mlflow run.
    """
    mlflow.set_tracking_uri(get_tracking_uri())
    mlflow.start_run()
    return mlflow.active_run().info.run_id
