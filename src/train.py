import pandas as pd
import yaml
import joblib

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier


DATA_FILE = "data/processed.csv"
MODEL_FILE = "models/model.pkl"
PARAMS_FILE = "params.yaml"


def train_model():

    # Load parameters
    with open(PARAMS_FILE, "r") as file:
        params = yaml.safe_load(file)

    test_size = params["train"]["test_size"]
    random_state = params["train"]["random_state"]

    n_estimators = params["model"]["n_estimators"]
    max_depth = params["model"]["max_depth"]

    # Load processed dataset
    df = pd.read_csv(DATA_FILE)

    X = df.drop("target", axis=1)
    y = df["target"]

    # Split dataset
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=random_state,
        stratify=y
    )

    # Create Random Forest model
    model = RandomForestClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        random_state=random_state
    )

    # Train model
    model.fit(X_train, y_train)

    # Save model
    joblib.dump(model, MODEL_FILE)

    print("Model trained successfully!")
    print(f"n_estimators: {n_estimators}")
    print(f"max_depth: {max_depth}")
    print(f"Model saved to: {MODEL_FILE}")


if __name__ == "__main__":
    train_model()