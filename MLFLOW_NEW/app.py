import os
import warnings
import sys
import logging
from urllib.parse import urlparse

import pandas as pd
import numpy as np

from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.linear_model import ElasticNet

import mlflow
from mlflow.models import infer_signature
import mlflow.sklearn


# -------------------------------
# Logging setup
# -------------------------------
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# -------------------------------
# Evaluation metrics
# -------------------------------
def eval_metrics(actual, pred):
    rmse = np.sqrt(mean_squared_error(actual, pred))
    mae = mean_absolute_error(actual, pred)
    r2 = r2_score(actual, pred)
    return rmse, mae, r2


# -------------------------------
# Load dataset
# -------------------------------
def load_data(url):
    try:
        data = pd.read_csv(url, sep=";")
        logger.info("Dataset loaded successfully")
        return data
    except Exception as e:
        logger.exception("Error loading dataset: %s", e)
        raise


# -------------------------------
# Main pipeline
# -------------------------------
def main(alpha=0.5, l1_ratio=0.5):

    warnings.filterwarnings("ignore")
    np.random.seed(42)

    # Set experiment
    mlflow.set_experiment("wine-quality-exp")

    # Dataset
    csv_url = "https://raw.githubusercontent.com/mlflow/mlflow/master/tests/datasets/winequality-red.csv"
    data = load_data(csv_url)

    # Split data
    train, test = train_test_split(data, test_size=0.25, random_state=42)

    train_x = train.drop(["quality"], axis=1)
    test_x = test.drop(["quality"], axis=1)

    train_y = train["quality"]
    test_y = test["quality"]

    with mlflow.start_run():

        # -------------------------------
        # Tags (MLOps best practice)
        # -------------------------------
        mlflow.set_tag("project", "wine-quality")
        mlflow.set_tag("model_type", "ElasticNet")
        mlflow.set_tag("developer", "bunny")

        # -------------------------------
        # Model training
        # -------------------------------
        model = ElasticNet(alpha=alpha, l1_ratio=l1_ratio, random_state=42)
        model.fit(train_x, train_y)

        # -------------------------------
        # Predictions
        # -------------------------------
        predictions = model.predict(test_x)

        # -------------------------------
        # Metrics
        # -------------------------------
        rmse, mae, r2 = eval_metrics(test_y, predictions)

        logger.info(f"RMSE: {rmse}")
        logger.info(f"MAE: {mae}")
        logger.info(f"R2: {r2}")

        # -------------------------------
        # MLflow logging
        # -------------------------------
        mlflow.log_param("alpha", alpha)
        mlflow.log_param("l1_ratio", l1_ratio)
        mlflow.log_param("data_url", csv_url)

        mlflow.log_metric("rmse", rmse)
        mlflow.log_metric("mae", mae)
        mlflow.log_metric("r2", r2)

        # -------------------------------
        # Model signature + example
        # -------------------------------
        signature = infer_signature(train_x, model.predict(train_x))
        input_example = train_x.iloc[:5]

        # -------------------------------
        # Model logging / registry
        # -------------------------------
        tracking_scheme = urlparse(mlflow.get_tracking_uri()).scheme

        if tracking_scheme != "file":
            mlflow.sklearn.log_model(
                sk_model=model,
                name="model",
                signature=signature,
                input_example=input_example,
                registered_model_name="ElasticnetWineModel"
            )
        else:
            mlflow.sklearn.log_model(
                sk_model=model,
                name="model",
                signature=signature,
                input_example=input_example
            )

        logger.info("Model logged successfully!")


# -------------------------------
# Entry point
# -------------------------------
if __name__ == "__main__":

    alpha = float(sys.argv[1]) if len(sys.argv) > 1 else 0.5
    l1_ratio = float(sys.argv[2]) if len(sys.argv) > 2 else 0.5

    main(alpha, l1_ratio)