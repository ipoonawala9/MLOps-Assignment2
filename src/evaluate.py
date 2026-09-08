import json

import pandas as pd
import joblib
import yaml

from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score
)


DATA_FILE = "data/processed.csv"
MODEL_FILE = "models/model.pkl"
PARAMS_FILE = "params.yaml"
METRICS_FILE = "metrics/metrics.json"


def evaluate_model():

    # Load parameters
    with open(PARAMS_FILE, "r") as file:
        params = yaml.safe_load(file)

    test_size = params["train"]["test_size"]
    random_state = params["train"]["random_state"]

    # Load processed dataset
    df = pd.read_csv(DATA_FILE)

    X = df.drop("target", axis=1)
    y = df["target"]

    # Recreate the same train/test split
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=random_state,
        stratify=y
    )

    # Load trained model
    model = joblib.load(MODEL_FILE)

    # Generate predictions
    predictions = model.predict(X_test)

    # Calculate metrics
    accuracy = accuracy_score(y_test, predictions)

    precision = precision_score(
        y_test,
        predictions,
        average="weighted"
    )

    recall = recall_score(
        y_test,
        predictions,
        average="weighted"
    )

    f1 = f1_score(
        y_test,
        predictions,
        average="weighted"
    )

    metrics = {
        "accuracy": round(accuracy, 4),
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "f1_score": round(f1, 4)
    }

    # Save metrics
    with open(METRICS_FILE, "w") as file:
        json.dump(metrics, file, indent=4)

    print("Evaluation completed!")
    print(json.dumps(metrics, indent=4))


if __name__ == "__main__":
    evaluate_model()